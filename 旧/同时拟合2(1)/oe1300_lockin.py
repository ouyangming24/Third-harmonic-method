# -*- coding: utf-8 -*-
"""
OE1300 锁相放大器远程控制（ASCII 指令，见《MODEL OE1300》手册 Rev 1.0.4）

- 串口: 115200 8N1，命令以回车结束
- 网口: TCP Client 连接仪器，默认 192.168.1.1:10001

与 ZurichLockInControl 保持相同对外方法名，便于 GUI 切换。
"""
from __future__ import annotations

import re
import socket
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# 可选串口
try:
    import serial
except ImportError:
    serial = None


def _zi_order_to_ofsl(order_time: int) -> int:
    """苏黎世 GUI 的 order(阶数) 近似映射到 OE1300 OFSL 索引 (0=6dB/oct .. 7=48dB/oct)"""
    mapping = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7}
    return mapping.get(int(order_time), 3)


class OE1300LockInControl:
    """
    OE1300 锁相：内部参考、解调器1=基波、解调器3=3次谐波（HARM 0,1 与 HARM 2,3），
    使用 SNAP? 同时读取 X/Y（基波）与 XD3/YD3（第三解调通道，对应 3ω）。

    若仪器默认解调器顺序不同，请修改 snap_indices 或 demod_harm_pairs。
    """

    # SNAP/OUTP 参数号（与手册 SNAP 表一致：基波 X=0,Y=1；第三解调通道 X=8,Y=9）
    snap_x1, snap_y1, snap_x3, snap_y3 = 0, 1, 8, 9

    # (解调器索引 0~6, 谐波次数) —— 解调器1→1f，解调器3→3f
    demod_harm_pairs: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 1), (2, 3))

    def __init__(
        self,
        connection: str = "tcp",
        serial_port: Optional[str] = None,
        tcp_host: str = "192.168.1.1",
        tcp_port: int = 10001,
        baudrate: int = 115200,
        timeout: float = 3.0,
    ):
        self.connection = connection.lower().strip()
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._ser = None
        self._input_range_index: Optional[int] = None  # 若设置则 configure_signal_inputs 下发 IRNG

        if self.connection == "tcp":
            self._sock = socket.create_connection((tcp_host, tcp_port), timeout=timeout)
            self._sock.settimeout(timeout)
            print(f"OE1300 TCP 已连接 {tcp_host}:{tcp_port}")
        elif self.connection == "serial":
            if serial is None:
                raise ImportError("串口模式需要安装 pyserial: pip install pyserial")
            if not serial_port:
                raise ValueError("串口模式请指定 serial_port，例如 COM3")
            self._ser = serial.Serial(
                serial_port, baudrate, timeout=timeout, write_timeout=timeout
            )
            print(f"OE1300 串口已打开 {serial_port} @ {baudrate}")
        else:
            raise ValueError("connection 应为 'tcp' 或 'serial'")

    def _write_raw(self, data: bytes) -> None:
        if self._sock:
            self._sock.sendall(data)
        else:
            self._ser.write(data)

    def _read_line(self) -> str:
        if self._sock:
            buf = b""
            while True:
                ch = self._sock.recv(1)
                if not ch:
                    break
                buf += ch
                if ch in (b"\n", b"\r"):
                    if ch == b"\r":
                        nxt = self._sock.recv(1)
                        if nxt:
                            buf += nxt
                    break
            return buf.decode(errors="replace").strip()
        return self._ser.readline().decode(errors="replace").strip()

    def _command(self, line: str, expect_reply: bool = True) -> str:
        """发送一行命令（自动加 \\r），可选读取一行应答。"""
        cmd = line.strip()
        if not cmd.endswith("\r"):
            cmd = cmd + "\r"
        self._write_raw(cmd.encode("ascii", errors="ignore"))
        if not expect_reply:
            return ""
        return self._read_line()

    def _query_snap_values(self, indices: Tuple[int, ...]) -> List[float]:
        """SNAP?i,j,k,... 返回逗号分隔浮点数"""
        args = ",".join(str(i) for i in indices)
        resp = self._command(f"SNAP?{args}")
        if not resp:
            raise RuntimeError("SNAP? 无返回")
        parts = [p.strip() for p in resp.split(",")]
        out: List[float] = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", p)
                if m:
                    out.append(float(m.group()))
                else:
                    out.append(float("nan"))
        return out

    def set_input_range_index(self, index: Optional[int]) -> None:
        """设置 IRNG 档位索引（手册表）；None 表示不通过指令改量程。"""
        self._input_range_index = index

    def configure_signal_inputs(
        self,
        input_channel=0,
        range=1.0,
        offset=0.0,
        ac=False,
        diff=False,
    ):
        """
        input_channel 保留与 Zurich 接口一致；OE1300 单路输入。
        ISRC: 0=A 单端, 1=A-B 差分, 2=电流
        ICPL: 0=AC, 1=DC
        IRNG: 由量程索引设置（若需精确档位请用手册 IRNG 表查 i）
        """
        isrc = 1 if diff else 0
        icpl = 0 if ac else 1
        self._command(f"ISRC {isrc}", expect_reply=False)
        time.sleep(0.02)
        self._command(f"ICPL {icpl}", expect_reply=False)
        time.sleep(0.02)
        # IRNG 为离散档位；未确认索引前不自动下发，避免误档（请面板设置或与手册核对后可传 input_range_index）
        irng = getattr(self, "_input_range_index", None)
        if irng is not None:
            self._command(f"IRNG {int(irng)}", expect_reply=False)
            time.sleep(0.02)
            print(f"OE1300 输入: ISRC={isrc}, ICPL={icpl}, IRNG={irng}")
        else:
            print(
                f"OE1300 输入: ISRC={isrc}, ICPL={icpl} (IRNG 未通过指令设置，请确认面板量程)"
            )

    def configure_signal_outputs(
        self,
        output_channel=0,
        amplitude=1.5,
        on=True,
        range=10.0,
        offset=0.0,
    ):
        """内部参考 + 正弦输出：FMOD 1, SWVT 1, SLVL (Vrms)"""
        self._command("FMOD 1", expect_reply=False)
        time.sleep(0.05)
        if on:
            self._command("SWVT 1", expect_reply=False)
        else:
            self._command("SWVT 0", expect_reply=False)
        time.sleep(0.05)
        amp = max(100e-6, min(5.0, float(amplitude)))
        self._command(f"SLVL {amp:.6f}", expect_reply=False)
        time.sleep(0.05)
        print(f"OE1300 输出: 内部参考, SWVT={1 if on else 0}, SLVL={amp} Vrms")

    def configure_lockin(self, freq, order_time, time_constant):
        """设置频率、时间常数、滤波阶数；配置两路谐波解调 1f 与 3f"""
        f = float(freq)
        self._command(f"FREQ {f:.6f}", expect_reply=False)
        time.sleep(0.05)

        tc = float(time_constant)
        tc = max(1e-6, min(3000.0, tc))
        self._command(f"OFLT {tc:.6g}", expect_reply=False)
        time.sleep(0.05)

        ofsl = _zi_order_to_ofsl(order_time)
        self._command(f"OFSL {ofsl}", expect_reply=False)
        time.sleep(0.05)

        for demod_idx, harm in self.demod_harm_pairs:
            self._command(f"DMOD {demod_idx},0", expect_reply=False)
            time.sleep(0.02)
            self._command(f"HARM {demod_idx},{int(harm)}", expect_reply=False)
            time.sleep(0.02)

        print(
            f"OE1300 锁相: FREQ={f} Hz, OFLT={tc} s, OFSL={ofsl}, "
            f"HARM {self.demod_harm_pairs}"
        )

    def read_demod_data(self, duration=1.0, sample_rate=1e3, n_samples: int = 20):
        """
        在 duration 内多次 SNAP 取平均，返回与 Zurich 相同结构：
        x1,y1,x3,y3 均为长度 n_samples 的列表（元素可为标量重复）
        """
        n = max(1, int(n_samples))
        interval = max(duration / n, 0.05)
        x1l, y1l, x3l, y3l = [], [], [], []
        idx = (
            self.snap_x1,
            self.snap_y1,
            self.snap_x3,
            self.snap_y3,
        )
        for k in range(n):
            vals = self._query_snap_values(idx)
            if len(vals) < 4:
                raise RuntimeError(f"SNAP 返回值不足4个: {vals}")
            x1l.append(vals[0])
            y1l.append(vals[1])
            x3l.append(vals[2])
            y3l.append(vals[3])
            if k < n - 1:
                time.sleep(interval)
        return x1l, y1l, x3l, y3l

    @staticmethod
    def calculate_amplitude_phase(x, y):
        if len(x) == 0 or len(y) == 0:
            print("No valid data for amplitude and phase calculation.")
            return None, None, None, None
        x_mean = float(np.mean(np.array(x, dtype=float)))
        y_mean = float(np.mean(np.array(y, dtype=float)))
        amplitude = float(np.sqrt(x_mean**2 + y_mean**2))
        phase_deg = float(np.degrees(np.arctan2(y_mean, x_mean)))
        return amplitude, phase_deg, x_mean, y_mean

    def save_results_to_excel(
        self,
        frequency,
        output_voltage,
        amplitude1,
        phase1,
        amplitude3,
        phase3,
        x3_mean,
        y3_mean,
        filename="oe1300_3omega.xlsx",
    ):
        amplitude1 = round(amplitude1, 9) if amplitude1 is not None else None
        amplitude3 = round(amplitude3, 9) if amplitude3 is not None else None
        x3_mean = round(x3_mean, 9) if x3_mean is not None else None
        y3_mean = round(y3_mean, 9) if y3_mean is not None else None
        try:
            df = pd.read_excel(filename)
        except FileNotFoundError:
            df = pd.DataFrame(
                columns=[
                    "Frequency",
                    "V3_real",
                    "V3_imag",
                    "Output Voltage",
                    "Amplitude_1f",
                    "Phase_1f",
                    "Amplitude_3f",
                    "Phase_3f",
                ]
            )
        new_data = pd.DataFrame(
            {
                "Frequency": [frequency],
                "V3_real": [x3_mean],
                "V3_imag": [y3_mean],
                "Output Voltage": [output_voltage],
                "Amplitude_1f": [amplitude1],
                "Phase_1f": [phase1],
                "Amplitude_3f": [amplitude3],
                "Phase_3f": [phase3],
            }
        )
        df = pd.concat([df, new_data], ignore_index=True)
        df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")

    def wait_for_integration(self, time_constant, duration_multiplier=5):
        wait_time = float(time_constant) * float(duration_multiplier)
        wait_time = max(wait_time, 0.2)
        print(f"OE1300 等待稳定 {wait_time:.2f} s ...")
        time.sleep(wait_time)

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._ser:
            try:
                self._ser.close()
            except OSError:
                pass
            self._ser = None
        print("OE1300 已断开")

    def idn(self) -> str:
        try:
            # 手册: *IDN?（四位命令）
            return self._command("*IDN?")
        except Exception as e:
            return f"<IDN error: {e}>"


__all__ = ["OE1300LockInControl"]
