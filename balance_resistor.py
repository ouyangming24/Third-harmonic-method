"""
数控调平电阻串口协议

下发: AT+RES.SP=1.234/
回读示例:
  +OK.
  +SP(R)=1.234
  +PV(R)=1.233
  +UMax(V)=0.8
  +RLimit(R)=0.000
  +TAmb(C)=21.73

说明: +UMax(V) 是 1ω 信号幅值上限（不是锁相/信号发生器输出上限）。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import serial
    from serial.tools import list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    list_ports = None
    SERIAL_AVAILABLE = False


@dataclass
class BalanceResistorStatus:
    ok: bool = False
    sp_ohm: Optional[float] = None
    pv_ohm: Optional[float] = None
    umax_v: Optional[float] = None
    rlimit_ohm: Optional[float] = None
    tamb_c: Optional[float] = None
    raw_lines: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            'sp_ohm': self.sp_ohm,
            'pv_ohm': self.pv_ohm,
            'umax_v': self.umax_v,
            'rlimit_ohm': self.rlimit_ohm,
            'tamb_c': self.tamb_c,
        }


_KEY_PATTERNS = {
    'sp_ohm': re.compile(r'\+SP\(R\)\s*=\s*([+-]?\d+(?:\.\d+)?)', re.I),
    'pv_ohm': re.compile(r'\+PV\(R\)\s*=\s*([+-]?\d+(?:\.\d+)?)', re.I),
    'umax_v': re.compile(r'\+UMax\(V\)\s*=\s*([+-]?\d+(?:\.\d+)?)', re.I),
    'rlimit_ohm': re.compile(r'\+RLimit\(R\)\s*=\s*([+-]?\d+(?:\.\d+)?)', re.I),
    'tamb_c': re.compile(r'\+TAmb\(C\)\s*=\s*([+-]?\d+(?:\.\d+)?)', re.I),
}


def parse_balance_response(text: str) -> BalanceResistorStatus:
    """解析设备回读文本。"""
    lines = [ln.strip() for ln in text.replace('\r\n', '\n').replace('\r', '\n').split('\n') if ln.strip()]
    st = BalanceResistorStatus(raw_lines=lines)
    joined = '\n'.join(lines)
    st.ok = any(ln.upper().startswith('+OK') for ln in lines) or ('+OK' in joined.upper())
    for key, pat in _KEY_PATTERNS.items():
        m = pat.search(joined)
        if m:
            setattr(st, key, float(m.group(1)))
    return st


def list_serial_ports() -> List[str]:
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in list_ports.comports()]


class BalanceResistorController:
    """数控调平电阻串口控制器。"""

    def __init__(self):
        self._ser = None
        self.last_status: Optional[BalanceResistorStatus] = None

    @property
    def connected(self) -> bool:
        return self._ser is not None and getattr(self._ser, 'is_open', False)

    def connect(self, port: str, baudrate: int = 9600, timeout: float = 1.0) -> None:
        if not SERIAL_AVAILABLE:
            raise RuntimeError("未安装 pyserial，请先执行: pip install pyserial")
        self.disconnect()
        self._ser = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
            write_timeout=timeout,
        )
        time.sleep(0.1)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def disconnect(self) -> None:
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    self._ser.close()
            except Exception:
                pass
        self._ser = None

    def _read_all(self, settle_s: float = 0.15, idle_s: float = 0.12) -> str:
        """写入后读取回包：先短暂等待，再读到空闲。"""
        time.sleep(settle_s)
        chunks: List[bytes] = []
        idle_deadline = time.time() + idle_s
        hard_deadline = time.time() + max(1.5, float(self._ser.timeout or 1.0) + 0.5)
        while time.time() < hard_deadline:
            waiting = getattr(self._ser, 'in_waiting', 0) or 0
            if waiting:
                chunks.append(self._ser.read(waiting))
                idle_deadline = time.time() + idle_s
                continue
            if chunks and time.time() >= idle_deadline:
                break
            time.sleep(0.02)
        return b''.join(chunks).decode(errors='replace')

    def set_resistance(self, ohms: float) -> BalanceResistorStatus:
        if not self.connected:
            raise RuntimeError("调平电阻未连接")
        if ohms < 0:
            raise ValueError("电阻值不能为负")
        cmd = f"AT+RES.SP={ohms:.6g}/"
        self._ser.reset_input_buffer()
        payload = (cmd + "\r\n").encode('ascii', errors='ignore')
        self._ser.write(payload)
        self._ser.flush()
        raw = self._read_all()
        # 若几乎无回包，再试一次不带 CRLF（部分固件只认 '/'）
        if not raw.strip():
            self._ser.reset_input_buffer()
            self._ser.write(cmd.encode('ascii', errors='ignore'))
            self._ser.flush()
            raw = self._read_all()
        st = parse_balance_response(raw)
        self.last_status = st
        if not st.raw_lines and not raw.strip():
            raise TimeoutError("下发后无收到设备回读，请检查串口/波特率/接线")
        return st
