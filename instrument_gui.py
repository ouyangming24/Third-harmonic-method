"""
温阻联测仪 — 宇电温度(RS485) + HY2516 电阻(SCPI) 合一界面

温度：AFC=1 → aibus / 无校验；PV=0x4A，PID 可读写
电阻：TRG 触发新测量（非 FETCh? 缓存）
TCR：多点 (T,R) 线性拟合 α = (1/R₀)·dR/dT，单位 1/°C 与 ppm/°C

依赖：pip install pyserial matplotlib
运行：python instrument_gui.py
"""
from __future__ import annotations

import csv
import math
import socket
import struct
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    import numpy as np

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    FigureCanvasTkAgg = None  # type: ignore
    Figure = None  # type: ignore
    np = None  # type: ignore
# —— 宇电寄存器 ——
REG_SV, REG_CTRL, REG_P, REG_I, REG_D = 0x00, 0x06, 0x07, 0x08, 0x09
REG_CTI, REG_DPT, REG_AT, REG_PV = 0x0A, 0x0C, 0x1D, 0x4A
REG_AMAN, REG_MV, REG_SRUN, REG_SPR, REG_ACT = 0x18, 0x1A, 0x1B, 0x2A, 0x22
CTRL_NAMES = {0: "ONOFF", 1: "APID", 2: "nPID", 3: "PoP", 4: "SoP"}
AT_NAMES = {0: "OFF", 1: "on", 2: "FoFF", 3: "AAt"}
AMAN_NAMES = {0: "MAN", 1: "Auto", 2: "FSV", 3: "FAut"}
SRUN_NAMES = {0: "run", 1: "StoP", 2: "HoLd"}
ACT_NAMES = {0: "rE反作用加热", 1: "dr正作用", 2: "rEbA", 3: "drbA"}


# ======================== 公共工具 ========================

def crc16_modbus(data: bytes) -> bytes:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if (crc & 1) else (crc >> 1)
    return struct.pack("<H", crc)


def dpt_decimals(dpt: int) -> int:
    return dpt - 127 if dpt > 127 else max(dpt, 0)


def apply_dpt(raw: int, dpt: int) -> float:
    n = dpt_decimals(dpt)
    return raw / (10 ** n) if n else float(raw)


def to_raw(value: float, dpt: int) -> int:
    n = dpt_decimals(dpt)
    return int(round(value * (10 ** n))) if n else int(round(value))


def clamp_i16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


def format_ohm(ohm: float) -> str:
    a = abs(ohm)
    if a < 1e-3:
        return f"{ohm * 1e6:.3f} µΩ"
    if a < 1:
        return f"{ohm * 1e3:.4f} mΩ"
    if a < 1e3:
        return f"{ohm:.6g} Ω"
    return f"{ohm / 1e3:.4f} kΩ"


def parse_meas(text: str) -> float:
    if not text:
        raise TimeoutError("仪器无返回测量数据")
    return float(text.split(",")[0].strip())


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """最小二乘 y = a + b·x，返回 (a, b, R²)。"""
    n = len(xs)
    if n < 2:
        raise ValueError("至少需要 2 个点")
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-18:
        raise ValueError("温度跨度太小，无法拟合")
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    y_mean = sy / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else 1.0
    return a, b, r2


def compute_tcr(
    temps: list[float],
    ohms: list[float],
    t_ref: float = 25.0,
) -> dict:
    """
    线性模型 R = a + b·T
    α(Tref) = b / R(Tref)，R(Tref)=a+b·Tref
    同时给出两点首末公式作对照。
    """
    if len(temps) != len(ohms) or len(temps) < 2:
        raise ValueError("需要至少 2 组 (T,R)")
    a, b, r2 = linear_fit(temps, ohms)
    r_ref = a + b * t_ref
    if abs(r_ref) < 1e-30:
        raise ValueError("参考温度处电阻接近 0，无法计算 TCR")
    alpha = b / r_ref  # 1/°C
    # 两点：首末
    t1, t2 = temps[0], temps[-1]
    r1, r2_ohm = ohms[0], ohms[-1]
    alpha_2pt = (r2_ohm - r1) / (r1 * (t2 - t1)) if abs(r1 * (t2 - t1)) > 1e-30 else float("nan")
    return {
        "a": a,
        "b": b,
        "r2": r2,
        "t_ref": t_ref,
        "r_ref": r_ref,
        "alpha": alpha,
        "alpha_ppm": alpha * 1e6,
        "alpha_2pt": alpha_2pt,
        "alpha_2pt_ppm": alpha_2pt * 1e6 if math.isfinite(alpha_2pt) else float("nan"),
        "t_min": min(temps),
        "t_max": max(temps),
        "n": len(temps),
    }


# ======================== HY2516 电阻 ========================

def scpi_exchange(ip: str, port: int, cmd: str, timeout: float = 5.0, expect_reply: bool = True) -> str:
    with socket.create_connection((ip, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall((cmd + "\n").encode("ascii"))
        if not expect_reply:
            time.sleep(0.05)
            return ""
        try:
            raw = s.recv(4096)
        except socket.timeout as e:
            raise TimeoutError(f"{cmd} 超时无数据") from e
    return raw.decode("ascii", errors="ignore").strip()


def prepare_for_trigger(ip: str, port: int) -> None:
    try:
        scpi_exchange(ip, port, "TRIGger:SOURce EXT", timeout=2.0, expect_reply=False)
    except OSError:
        pass
    time.sleep(0.05)


def read_resistance_live(ip: str, port: int = 502, timeout: float = 5.0) -> tuple[float, str]:
    prepare_for_trigger(ip, port)
    last_err = ""
    for cmd in ("TRG", "TRIGger:IMMediate"):
        try:
            text = scpi_exchange(ip, port, cmd, timeout=timeout)
            return parse_meas(text), f"{cmd} → {text}"
        except Exception as e:
            last_err = f"{cmd}: {e}"
    try:
        scpi_exchange(ip, port, "TRIGger:SOURce INT", timeout=2.0, expect_reply=False)
        time.sleep(0.3)
        text = scpi_exchange(ip, port, "FETCh?", timeout=timeout)
        return parse_meas(text), f"FETCh?(INT后备) → {text}"
    except Exception as e:
        raise RuntimeError(f"实时读取失败。{last_err} | FETCh?: {e}") from e


# ======================== 宇电温度 ========================

class YudianTemp:
    def __init__(
        self,
        port: str,
        baud: int = 9600,
        addr: int = 1,
        protocol: str = "aibus",
        parity: str = "N",
        rs485: str = "none",
        timeout: float = 0.8,
    ):
        if serial is None:
            raise RuntimeError("请先安装: pip install pyserial")
        self.port = port
        self.baud = baud
        self.addr = addr
        self.protocol = protocol
        self.parity = parity.upper()
        self.rs485 = rs485
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._dpt: Optional[int] = None
        self._tx_high = True
        self._lock = threading.Lock()

    def open(self) -> None:
        self.close()
        parity_map = {
            "N": serial.PARITY_NONE,
            "E": serial.PARITY_EVEN,
            "O": serial.PARITY_ODD,
        }
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baud,
            bytesize=8,
            parity=parity_map.get(self.parity, serial.PARITY_NONE),
            stopbits=1,
            timeout=self.timeout,
            write_timeout=1.0,
        )
        try:
            self._ser.dtr = True
        except Exception:
            pass
        if self.rs485 == "rts_hi":
            self._tx_high = True
            self._rx_dir()
        elif self.rs485 == "rts_lo":
            self._tx_high = False
            self._rx_dir()
        else:
            try:
                self._ser.rts = True
            except Exception:
                pass
        time.sleep(0.08)
        self._ser.reset_input_buffer()
        self._dpt = None

    def close(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None

    def _ensure_open(self) -> None:
        if not self._ser or not self._ser.is_open:
            self.open()

    def _tx_dir(self) -> None:
        if self._ser and self.rs485 != "none":
            try:
                self._ser.rts = self._tx_high
            except Exception:
                pass

    def _rx_dir(self) -> None:
        if self._ser and self.rs485 != "none":
            try:
                self._ser.rts = not self._tx_high
            except Exception:
                pass

    def _exchange(self, frame: bytes, n: int) -> bytes:
        assert self._ser is not None
        self._ser.reset_input_buffer()
        if self.rs485 != "none":
            self._tx_dir()
            time.sleep(0.002)
        self._ser.write(frame)
        self._ser.flush()
        time.sleep(max(0.04, len(frame) * 11 / max(self.baud, 1) + 0.02))
        if self.rs485 != "none":
            self._rx_dir()
        deadline = time.time() + self.timeout
        resp = b""
        while len(resp) < n and time.time() < deadline:
            chunk = self._ser.read(n - len(resp))
            if chunk:
                resp += chunk
            else:
                time.sleep(0.01)
        return resp

    def ensure_dpt(self) -> int:
        if self._dpt is None:
            self._dpt = self.read_raw(REG_DPT)
        return self._dpt

    def read_raw(self, reg: int) -> int:
        with self._lock:
            self._ensure_open()
            if self.protocol == "aibus":
                return self._aibus_read(reg)["param"]
            return self._modbus_regs(reg, 1)[0]

    def write_raw(self, reg: int, value: int) -> None:
        value = clamp_i16(value)
        with self._lock:
            self._ensure_open()
            if self.protocol == "aibus":
                self._aibus_write(reg, value)
            else:
                self._modbus_write(reg, value)

    def read_temperature(self) -> float:
        dpt = self.ensure_dpt()
        if self.protocol == "aibus":
            with self._lock:
                self._ensure_open()
                info = self._aibus_read(REG_PV)
            return apply_dpt(info["pv"], dpt)
        return apply_dpt(self.read_raw(REG_PV), dpt)

    def read_pid(self) -> dict:
        dpt = self.ensure_dpt()
        return {
            "sv": apply_dpt(self.read_raw(REG_SV), dpt),
            "p": apply_dpt(self.read_raw(REG_P), dpt),
            "i": float(self.read_raw(REG_I)),
            "d": float(self.read_raw(REG_D)),
            "cti": float(self.read_raw(REG_CTI)),
            "ctrl": int(self.read_raw(REG_CTRL)),
            "at": int(self.read_raw(REG_AT)),
            "dpt": dpt,
        }

    def write_pid(
        self,
        *,
        sv: Optional[float] = None,
        p: Optional[float] = None,
        i: Optional[float] = None,
        d: Optional[float] = None,
        cti: Optional[float] = None,
        ctrl: Optional[int] = None,
    ) -> None:
        dpt = self.ensure_dpt()
        if sv is not None:
            self.write_raw(REG_SV, to_raw(sv, dpt))
        if p is not None:
            self.write_raw(REG_P, to_raw(p, dpt))
        if i is not None:
            self.write_raw(REG_I, clamp_i16(round(i)))
        if d is not None:
            self.write_raw(REG_D, clamp_i16(round(d)))
        if cti is not None:
            self.write_raw(REG_CTI, clamp_i16(round(cti)))
        if ctrl is not None:
            self.write_raw(REG_CTRL, int(ctrl))

    def start_autotune(self) -> None:
        self.write_raw(REG_AT, 1)

    def stop_autotune(self) -> None:
        self.write_raw(REG_AT, 0)

    def read_control_status(self) -> dict:
        """读易导致「改 SV 发呆」的状态：手自动 / 运行 / 升速限制 / 作用方向 / MV。"""
        dpt = self.ensure_dpt()
        return {
            "pv": self.read_temperature(),
            "sv": apply_dpt(self.read_raw(REG_SV), dpt),
            "aman": int(self.read_raw(REG_AMAN)),
            "srun": int(self.read_raw(REG_SRUN)),
            "spr": apply_dpt(self.read_raw(REG_SPR), dpt),
            "act": int(self.read_raw(REG_ACT)),
            "mv": int(self.read_raw(REG_MV)),
            "ctrl": int(self.read_raw(REG_CTRL)),
        }

    def control_unstick(self, new_sv: Optional[float] = None, clear_spr: bool = True) -> dict:
        """
        解除「改 SV 后长时间不加热」的常见卡滞：
        1) 强制 run + Auto
        2) 可选清零 SPr（升温速率限制）
        3) 手动→自动切换以清积分饱和（类似断电复位积分状态）
        4) 再写入新 SV（若给出）
        """
        dpt = self.ensure_dpt()
        # 运行 / 自动
        self.write_raw(REG_SRUN, 0)  # run
        self.write_raw(REG_AMAN, 1)  # Auto
        if clear_spr:
            self.write_raw(REG_SPR, 0)
        # 手自动切换清积分
        self.write_raw(REG_AMAN, 0)  # MAN
        time.sleep(0.15)
        self.write_raw(REG_AMAN, 1)  # Auto
        time.sleep(0.1)
        if new_sv is not None:
            self.write_raw(REG_SV, to_raw(new_sv, dpt))
        return self.read_control_status()

    def _modbus_regs(self, start: int, count: int) -> list[int]:
        pdu = bytes(
            [
                self.addr & 0xFF,
                0x03,
                (start >> 8) & 0xFF,
                start & 0xFF,
                (count >> 8) & 0xFF,
                count & 0xFF,
            ]
        )
        frame = pdu + crc16_modbus(pdu)
        resp = self._exchange(frame, 5 + count * 2)
        if len(resp) < 5 + count * 2:
            raise TimeoutError(f"无响应 TX={frame.hex(' ')} RX={resp.hex(' ') or '(空)'}")
        if resp[0] != (self.addr & 0xFF):
            raise RuntimeError(f"站号不匹配 RX={resp.hex(' ')}")
        if resp[1] & 0x80:
            raise RuntimeError(f"异常码 0x{resp[2]:02X}")
        if resp[1] != 0x03:
            raise RuntimeError(f"功能码异常 RX={resp.hex(' ')}")
        data = resp[3 : 3 + count * 2]
        out = []
        for i in range(0, len(data), 2):
            u = struct.unpack(">H", data[i : i + 2])[0]
            out.append(struct.unpack(">h", struct.pack(">H", u))[0])
        return out

    def _modbus_write(self, reg: int, value: int) -> None:
        u = value & 0xFFFF
        pdu = bytes(
            [
                self.addr & 0xFF,
                0x06,
                (reg >> 8) & 0xFF,
                reg & 0xFF,
                (u >> 8) & 0xFF,
                u & 0xFF,
            ]
        )
        frame = pdu + crc16_modbus(pdu)
        resp = self._exchange(frame, 8)
        if len(resp) < 8:
            raise TimeoutError(f"写无响应 TX={frame.hex(' ')} RX={resp.hex(' ') or '(空)'}")
        if resp[1] & 0x80:
            raise RuntimeError(f"写异常码 0x{resp[2]:02X}")

    def _aibus_read(self, param: int) -> dict:
        addr_b = (0x80 + self.addr) & 0xFF
        cs = (param * 256 + 0x52 + self.addr) & 0xFFFF
        frame = bytes(
            [addr_b, addr_b, 0x52, param & 0xFF, 0x00, 0x00, cs & 0xFF, (cs >> 8) & 0xFF]
        )
        resp = self._exchange(frame, 10)
        if len(resp) != 10:
            raise TimeoutError(f"无完整响应 TX={frame.hex(' ')} RX={resp.hex(' ') or '(空)'}")
        return {
            "pv": struct.unpack("<h", resp[0:2])[0],
            "sv": struct.unpack("<h", resp[2:4])[0],
            "param": struct.unpack("<h", resp[6:8])[0],
        }

    def _aibus_write(self, param: int, value: int) -> dict:
        addr_b = (0x80 + self.addr) & 0xFF
        v = value & 0xFFFF
        cs = (param * 256 + 0x43 + self.addr + v) & 0xFFFF
        frame = bytes(
            [
                addr_b,
                addr_b,
                0x43,
                param & 0xFF,
                v & 0xFF,
                (v >> 8) & 0xFF,
                cs & 0xFF,
                (cs >> 8) & 0xFF,
            ]
        )
        resp = self._exchange(frame, 10)
        if len(resp) != 10:
            raise TimeoutError(f"写无完整响应 TX={frame.hex(' ')} RX={resp.hex(' ') or '(空)'}")
        return {
            "pv": struct.unpack("<h", resp[0:2])[0],
            "sv": struct.unpack("<h", resp[2:4])[0],
            "param": struct.unpack("<h", resp[6:8])[0],
        }


# ======================== 联测 GUI ========================

class InstrumentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("温阻联测仪 — 宇电温度 + HY2516 电阻")
        self.geometry("1100x900")
        self.configure(bg="#1a2332")

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._client: Optional[YudianTemp] = None
        self._ok_cfg: Optional[dict] = None
        self._last_ohm: Optional[float] = None
        self.enable_temp = tk.BooleanVar(value=True)
        self.enable_ohm = tk.BooleanVar(value=True)
        self._tcr_points: list[dict] = []  # {t, r, ts}
        self._tcr_stop = threading.Event()
        self._tcr_thread: Optional[threading.Thread] = None
        self._tcr_running = False
        # 实时曲线缓冲
        self._live_t0: Optional[float] = None
        self._live_times: list[float] = []
        self._live_temps: list[float] = []
        self._live_ohms: list[float] = []
        self._plot_last_draw = 0.0
        self._mpl_canvas = None
        self._ax_tr = None
        self._ax_rt = None
        self._ax_r2 = None
        self._line_t = None
        self._line_r = None
        self._line_rt = None
        self._scatter_pts = None
        self._fit_line = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.refresh_ports()

    def _style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("T.TLabel", background="#1a2332", foreground="#e8eef7", font=("Segoe UI Semibold", 15))
        style.configure("H.TLabel", background="#1a2332", foreground="#8fa3bf", font=("Segoe UI", 9))
        style.configure("V.TLabel", background="#243044", foreground="#f4f7fb", font=("Consolas", 28, "bold"))
        style.configure("S.TLabel", background="#1a2332", foreground="#a8bdd6", font=("Segoe UI", 9))
        style.configure("C.TFrame", background="#243044")
        style.configure("R.TFrame", background="#1a2332")
        style.configure("F.TLabel", background="#243044", foreground="#c5d4e8", font=("Segoe UI", 9))
        style.configure("Tip.TLabel", background="#243044", foreground="#9fb4ce", font=("Segoe UI", 8))
        style.configure("TNotebook", background="#1a2332")
        style.configure("TNotebook.Tab", padding=[10, 4])
        style.configure("TCheckbutton", background="#243044", foreground="#c5d4e8")

    def _build(self) -> None:
        self._style()
        root = ttk.Frame(self, style="R.TFrame", padding=14)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="温阻联测仪", style="T.TLabel").pack(anchor="w")
        ttk.Label(
            root,
            text="温度=宇电 RS485 · 电阻=HY2516 · TCR=多点拟合 α(ppm/°C)",
            style="H.TLabel",
        ).pack(anchor="w", pady=(2, 8))

        # 双通道大字显示
        dash = ttk.Frame(root, style="C.TFrame", padding=14)
        dash.pack(fill=tk.X)
        left = ttk.Frame(dash, style="C.TFrame")
        mid = ttk.Frame(dash, style="C.TFrame")
        right = ttk.Frame(dash, style="C.TFrame")
        left.pack(side=tk.LEFT, expand=True, fill=tk.X)
        mid.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(12, 0))
        right.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(12, 0))

        self.temp_var = tk.StringVar(value="—")
        self.ohm_var = tk.StringVar(value="—")
        self.tcr_dash_var = tk.StringVar(value="—")
        ttk.Label(left, text="温度 °C", style="F.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.temp_var, style="V.TLabel").pack(anchor="w")
        self.cfg_var = tk.StringVar(value="温度: 未连接")
        ttk.Label(left, textvariable=self.cfg_var, style="Tip.TLabel").pack(anchor="w")
        self.sv_live_var = tk.StringVar(value="SV: —")
        ttk.Label(left, textvariable=self.sv_live_var, style="Tip.TLabel").pack(anchor="w")

        ttk.Label(mid, text="电阻", style="F.TLabel").pack(anchor="w")
        ttk.Label(mid, textvariable=self.ohm_var, style="V.TLabel").pack(anchor="w")
        self.ohm_cfg_var = tk.StringVar(value="电阻: 未连接")
        ttk.Label(mid, textvariable=self.ohm_cfg_var, style="Tip.TLabel").pack(anchor="w")

        ttk.Label(right, text="TCR α", style="F.TLabel").pack(anchor="w")
        ttk.Label(right, textvariable=self.tcr_dash_var, style="V.TLabel").pack(anchor="w")
        self.tcr_info_var = tk.StringVar(value="TCR: 未计算")
        ttk.Label(right, textvariable=self.tcr_info_var, style="Tip.TLabel").pack(anchor="w")

        # 联测控制条
        bar = ttk.Frame(root, style="C.TFrame", padding=10)
        bar.pack(fill=tk.X, pady=8)
        ttk.Checkbutton(bar, text="采温度", variable=self.enable_temp).pack(side=tk.LEFT)
        ttk.Checkbutton(bar, text="采电阻", variable=self.enable_ohm).pack(side=tk.LEFT, padx=8)
        ttk.Label(bar, text="间隔(秒)", style="F.TLabel").pack(side=tk.LEFT, padx=(12, 4))
        self.interval_var = tk.StringVar(value="1.0")
        ttk.Entry(bar, textvariable=self.interval_var, width=8).pack(side=tk.LEFT)
        self.start_btn = ttk.Button(bar, text="联测开始", command=self.start)
        self.stop_btn = ttk.Button(bar, text="停止", command=self.stop, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(16, 4))
        self.stop_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="各读一次", command=self.read_both_once).pack(side=tk.LEFT, padx=8)

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        tab_t = ttk.Frame(nb, style="C.TFrame", padding=10)
        tab_r = ttk.Frame(nb, style="C.TFrame", padding=10)
        tab_tcr = ttk.Frame(nb, style="C.TFrame", padding=10)
        nb.add(tab_t, text="  温度 / PID  ")
        nb.add(tab_r, text="  电阻 HY2516  ")
        nb.add(tab_tcr, text="  TCR 电阻温度系数  ")

        self._build_temp_tab(tab_t)
        self._build_ohm_tab(tab_r)
        self._build_tcr_tab(tab_tcr)

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.status_var, style="S.TLabel").pack(anchor="w", pady=(6, 0))
        self.log = tk.Text(root, height=5, bg="#121a26", fg="#9fb4ce", relief=tk.FLAT, font=("Consolas", 9))
        self.log.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

    def _build_temp_tab(self, parent: ttk.Frame) -> None:
        cfg = ttk.Frame(parent, style="C.TFrame")
        cfg.pack(fill=tk.X)

        self.com_var = tk.StringVar(value="")
        self.baud_var = tk.StringVar(value="9600")
        self.addr_var = tk.StringVar(value="1")
        self.parity_var = tk.StringVar(value="N")
        self.proto_var = tk.StringVar(value="aibus")

        ttk.Label(cfg, text="COM", style="F.TLabel").grid(row=0, column=0, sticky="w", pady=2)
        self.com_box = ttk.Combobox(cfg, textvariable=self.com_var, width=14)
        self.com_box.grid(row=0, column=1, sticky="we", padx=6, pady=2)
        ttk.Button(cfg, text="刷新", width=6, command=self.refresh_ports).grid(row=0, column=2)

        ttk.Label(cfg, text="波特率", style="F.TLabel").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(
            cfg, textvariable=self.baud_var, values=["4800", "9600", "19200", "28800"], width=14
        ).grid(row=1, column=1, sticky="we", padx=6, pady=2)

        ttk.Label(cfg, text="Addr", style="F.TLabel").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(cfg, textvariable=self.addr_var, width=16).grid(row=2, column=1, sticky="we", padx=6, pady=2)

        ttk.Label(cfg, text="校验", style="F.TLabel").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Combobox(cfg, textvariable=self.parity_var, values=["N", "E"], state="readonly", width=14).grid(
            row=3, column=1, sticky="we", padx=6, pady=2
        )

        ttk.Label(cfg, text="协议", style="F.TLabel").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Combobox(
            cfg, textvariable=self.proto_var, values=["auto", "modbus", "aibus"], state="readonly", width=14
        ).grid(row=4, column=1, sticky="we", padx=6, pady=2)
        cfg.columnconfigure(1, weight=1)

        btns = ttk.Frame(parent, style="C.TFrame")
        btns.pack(fill=tk.X, pady=6)
        self.probe_btn = ttk.Button(btns, text="自动探测", command=self.auto_probe)
        self.temp_once_btn = ttk.Button(btns, text="读温度", command=self.read_temp_once)
        self.probe_btn.pack(side=tk.LEFT)
        self.temp_once_btn.pack(side=tk.LEFT, padx=6)

        pid = ttk.Frame(parent, style="C.TFrame")
        pid.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(pid, text="PID / 设定", style="F.TLabel").grid(row=0, column=0, columnspan=3, sticky="w")

        self.sv_var = tk.StringVar()
        self.p_var = tk.StringVar()
        self.i_var = tk.StringVar()
        self.d_var = tk.StringVar()
        self.cti_var = tk.StringVar()
        self.ctrl_var = tk.StringVar(value="1")
        self.at_var = tk.StringVar(value="—")

        for row, label, var in [
            (1, "SV °C", self.sv_var),
            (2, "P 比例带", self.p_var),
            (3, "I 积分(秒)", self.i_var),
            (4, "d 微分(0.1s)", self.d_var),
            (5, "CtI(0.1s)", self.cti_var),
        ]:
            ttk.Label(pid, text=label, style="F.TLabel").grid(row=row, column=0, sticky="w", pady=1)
            ttk.Entry(pid, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=6, pady=1)

        ttk.Label(pid, text="CtrL", style="F.TLabel").grid(row=6, column=0, sticky="w", pady=1)
        ttk.Combobox(
            pid, textvariable=self.ctrl_var, values=["0", "1", "2", "3", "4"], width=10, state="readonly"
        ).grid(row=6, column=1, sticky="w", padx=6, pady=1)
        ttk.Label(pid, text="推荐 1=APID", style="Tip.TLabel").grid(row=6, column=2, sticky="w")

        ttk.Label(pid, text="At", style="F.TLabel").grid(row=7, column=0, sticky="w")
        ttk.Label(pid, textvariable=self.at_var, style="F.TLabel").grid(row=7, column=1, sticky="w", padx=6)

        self.ctrl_status_var = tk.StringVar(value="状态: —")
        ttk.Label(pid, textvariable=self.ctrl_status_var, style="Tip.TLabel", wraplength=680).grid(
            row=8, column=0, columnspan=3, sticky="w", pady=2
        )

        ttk.Label(
            pid,
            text="25~45°C：SV 设常用点→自整定。若改高 SV 仍降温很久：多半积分饱和或 SPr 限速，点「解除卡住」。",
            style="Tip.TLabel",
            wraplength=680,
        ).grid(row=9, column=0, columnspan=3, sticky="w", pady=4)

        pb = ttk.Frame(pid, style="C.TFrame")
        pb.grid(row=10, column=0, columnspan=3, sticky="w")
        ttk.Button(pb, text="读取 PID", command=self.read_pid).pack(side=tk.LEFT)
        ttk.Button(pb, text="写入 PID", command=self.write_pid).pack(side=tk.LEFT, padx=4)
        ttk.Button(pb, text="只写 SV", command=self.write_sv_only).pack(side=tk.LEFT)
        ttk.Button(pb, text="写SV并解除卡住", command=self.write_sv_unstick).pack(side=tk.LEFT, padx=4)
        ttk.Button(pb, text="启动自整定", command=self.start_at).pack(side=tk.LEFT, padx=4)
        ttk.Button(pb, text="停止自整定", command=self.stop_at).pack(side=tk.LEFT)
        pb2 = ttk.Frame(pid, style="C.TFrame")
        pb2.grid(row=11, column=0, columnspan=3, sticky="w", pady=(4, 0))
        ttk.Button(pb2, text="读控制状态", command=self.read_ctrl_status).pack(side=tk.LEFT)
        ttk.Button(pb2, text="解除卡住(手↔自)", command=self.do_unstick).pack(side=tk.LEFT, padx=4)

    def _build_ohm_tab(self, parent: ttk.Frame) -> None:
        cfg = ttk.Frame(parent, style="C.TFrame")
        cfg.pack(fill=tk.X)
        self.ip_var = tk.StringVar(value="192.168.30.36")
        self.port_var = tk.StringVar(value="502")

        ttk.Label(cfg, text="IP", style="F.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(cfg, textvariable=self.ip_var, width=22).grid(row=0, column=1, sticky="we", padx=6, pady=3)
        ttk.Label(cfg, text="端口", style="F.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(cfg, textvariable=self.port_var, width=22).grid(row=1, column=1, sticky="we", padx=6, pady=3)
        cfg.columnconfigure(1, weight=1)

        ttk.Label(
            parent,
            text="每次用 TRG 触发新测量（不是 FETCh? 缓存）。",
            style="Tip.TLabel",
        ).pack(anchor="w", pady=6)

        btns = ttk.Frame(parent, style="C.TFrame")
        btns.pack(fill=tk.X)
        self.idn_btn = ttk.Button(btns, text="识别仪器", command=self.do_idn)
        self.ohm_once_btn = ttk.Button(btns, text="读电阻", command=self.read_ohm_once)
        self.idn_btn.pack(side=tk.LEFT)
        self.ohm_once_btn.pack(side=tk.LEFT, padx=6)

    def _build_tcr_tab(self, parent: ttk.Frame) -> None:
        tip = (
            "采点条件：温度进入容差并保持设定秒数，且最近「R稳定秒」内电阻变化不超过「R容差(相对)」。"
        )
        ttk.Label(parent, text=tip, style="Tip.TLabel", wraplength=900).pack(anchor="w")

        cfg = ttk.Frame(parent, style="C.TFrame")
        cfg.pack(fill=tk.X, pady=4)

        self.tcr_tref_var = tk.StringVar(value="25")
        self.tcr_targets_var = tk.StringVar(value="25,30,35,40,45")
        self.tcr_tol_var = tk.StringVar(value="0.2")
        self.tcr_hold_var = tk.StringVar(value="20")
        self.tcr_r_hold_var = tk.StringVar(value="30")
        self.tcr_r_tol_var = tk.StringVar(value="0.0005")
        self.tcr_timeout_var = tk.StringVar(value="600")
        self.tcr_result_var = tk.StringVar(value="尚未计算")

        # 两列压缩参数区，给图+表留高度
        left_cfg = ttk.Frame(cfg, style="C.TFrame")
        right_cfg = ttk.Frame(cfg, style="C.TFrame")
        left_cfg.pack(side=tk.LEFT, fill=tk.X, expand=True)
        right_cfg.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(12, 0))

        for row, label, var in [
            (0, "Tref °C", self.tcr_tref_var),
            (1, "扫描目标", self.tcr_targets_var),
            (2, "容差 °C", self.tcr_tol_var),
            (3, "T保持 秒", self.tcr_hold_var),
        ]:
            ttk.Label(left_cfg, text=label, style="F.TLabel").grid(row=row, column=0, sticky="w", pady=1)
            ttk.Entry(left_cfg, textvariable=var, width=28).grid(row=row, column=1, sticky="we", padx=6, pady=1)
        left_cfg.columnconfigure(1, weight=1)

        for row, label, var in [
            (0, "R稳定 秒", self.tcr_r_hold_var),
            (1, "R容差(相对)", self.tcr_r_tol_var),
            (2, "超时 秒", self.tcr_timeout_var),
        ]:
            ttk.Label(right_cfg, text=label, style="F.TLabel").grid(row=row, column=0, sticky="w", pady=1)
            ttk.Entry(right_cfg, textvariable=var, width=14).grid(row=row, column=1, sticky="we", padx=6, pady=1)
        right_cfg.columnconfigure(1, weight=1)
        ttk.Label(
            right_cfg,
            text="默认30s；0.0005≈0.05%",
            style="Tip.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        btns = ttk.Frame(parent, style="C.TFrame")
        btns.pack(fill=tk.X, pady=4)
        ttk.Button(btns, text="采当前点", command=self.tcr_add_point).pack(side=tk.LEFT)
        ttk.Button(btns, text="计算 TCR", command=self.tcr_compute).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="清空数据", command=self.tcr_clear).pack(side=tk.LEFT)
        ttk.Button(btns, text="导出 CSV", command=self.tcr_export).pack(side=tk.LEFT, padx=4)
        self.tcr_sweep_btn = ttk.Button(btns, text="自动扫描", command=self.tcr_start_sweep)
        self.tcr_stop_btn = ttk.Button(btns, text="停止扫描", command=self.tcr_stop_sweep, state=tk.DISABLED)
        self.tcr_sweep_btn.pack(side=tk.LEFT, padx=(12, 4))
        self.tcr_stop_btn.pack(side=tk.LEFT)

        ttk.Label(parent, textvariable=self.tcr_result_var, style="F.TLabel", wraplength=900).pack(
            anchor="w", pady=(2, 4)
        )

        # 左曲线 + 右紧凑采样表
        body = ttk.Frame(parent, style="C.TFrame")
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=4, minsize=420)
        body.columnconfigure(1, weight=1, minsize=200)
        body.rowconfigure(0, weight=1, minsize=220)

        plot_frame = ttk.Frame(body, style="C.TFrame")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._init_tcr_plot(plot_frame)

        table_frame = ttk.Frame(body, style="C.TFrame")
        table_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(table_frame, text="采样点", style="F.TLabel").pack(anchor="w")
        tree_wrap = ttk.Frame(table_frame, style="C.TFrame")
        tree_wrap.pack(fill=tk.BOTH, expand=True)
        cols = ("n", "T", "R")
        self.tcr_tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", height=10)
        self.tcr_tree.heading("n", text="#")
        self.tcr_tree.heading("T", text="T")
        self.tcr_tree.heading("R", text="R")
        self.tcr_tree.column("n", width=28, anchor="center", stretch=False)
        self.tcr_tree.column("T", width=58, anchor="e", stretch=True)
        self.tcr_tree.column("R", width=78, anchor="e", stretch=True)
        scroll = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self.tcr_tree.yview)
        self.tcr_tree.configure(yscrollcommand=scroll.set)
        self.tcr_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _init_tcr_plot(self, parent: ttk.Frame) -> None:
        if not HAS_MPL:
            ttk.Label(
                parent,
                text="未安装 matplotlib，无法显示曲线。请执行: pip install matplotlib",
                style="Tip.TLabel",
            ).pack(anchor="w")
            return
        fig = Figure(figsize=(7.2, 2.8), dpi=100, facecolor="#243044")
        fig.subplots_adjust(left=0.09, right=0.91, top=0.88, bottom=0.18, wspace=0.30)
        self._ax_tr = fig.add_subplot(121)
        self._ax_rt = fig.add_subplot(122)
        self._ax_r2 = self._ax_tr.twinx()
        for ax in (self._ax_tr, self._ax_rt, self._ax_r2):
            ax.set_facecolor("#1a2332")
            ax.tick_params(colors="#c5d4e8", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#5a6e88")
            ax.title.set_color("#e8eef7")
            ax.xaxis.label.set_color("#a8bdd6")
            ax.yaxis.label.set_color("#a8bdd6")

        self._ax_tr.set_title("T / R vs time", fontsize=10)
        self._ax_tr.set_xlabel("t (s)")
        self._ax_tr.set_ylabel("T (°C)", color="#6ec6ff")
        self._ax_r2.set_ylabel("R (Ω)", color="#ffb86c")
        self._ax_rt.set_title("R vs T", fontsize=10)
        self._ax_rt.set_xlabel("T (°C)")
        self._ax_rt.set_ylabel("R (Ω)")

        (self._line_t,) = self._ax_tr.plot([], [], color="#6ec6ff", lw=1.4, label="T")
        (self._line_r,) = self._ax_r2.plot([], [], color="#ffb86c", lw=1.4, label="R")
        (self._line_rt,) = self._ax_rt.plot([], [], color="#9ad67a", lw=1.0, alpha=0.7)
        self._scatter_pts = self._ax_rt.scatter([], [], c="#ff6b6b", s=36, zorder=5, label="points")
        (self._fit_line,) = self._ax_rt.plot([], [], color="#e8eef7", ls="--", lw=1.2, label="fit")

        self._mpl_canvas = FigureCanvasTkAgg(fig, master=parent)
        widget = self._mpl_canvas.get_tk_widget()
        widget.configure(height=280)
        widget.pack(fill=tk.BOTH, expand=True)
        self._mpl_canvas.draw_idle()

    def _tcr_plot_reset(self) -> None:
        self._live_t0 = time.time()
        self._live_times.clear()
        self._live_temps.clear()
        self._live_ohms.clear()
        self._plot_last_draw = 0.0
        if not HAS_MPL or self._line_t is None:
            return
        self._line_t.set_data([], [])
        self._line_r.set_data([], [])
        self._line_rt.set_data([], [])
        self._fit_line.set_data([], [])
        if np is not None:
            self._scatter_pts.set_offsets(np.empty((0, 2)))
        self._ax_tr.relim()
        self._ax_tr.autoscale_view()
        self._ax_r2.relim()
        self._ax_r2.autoscale_view()
        self._ax_rt.relim()
        self._ax_rt.autoscale_view()
        self._mpl_canvas.draw_idle()

    def _tcr_plot_push(self, t: float, r: float, force: bool = False) -> None:
        """追加实时点并节流刷新曲线（须在主线程调用）。"""
        if self._live_t0 is None:
            self._live_t0 = time.time()
        elapsed = time.time() - self._live_t0
        self._live_times.append(elapsed)
        self._live_temps.append(t)
        self._live_ohms.append(r)
        # 限制点数，避免长时间扫描卡顿
        max_n = 2500
        if len(self._live_times) > max_n:
            cut = len(self._live_times) - max_n
            del self._live_times[:cut]
            del self._live_temps[:cut]
            del self._live_ohms[:cut]
        now = time.time()
        if not force and (now - self._plot_last_draw) < 0.35:
            return
        self._plot_last_draw = now
        self._tcr_plot_redraw()

    def _tcr_plot_redraw(self) -> None:
        if not HAS_MPL or self._line_t is None:
            return
        xs = self._live_times
        ts = self._live_temps
        rs = self._live_ohms
        self._line_t.set_data(xs, ts)
        self._line_r.set_data(xs, rs)
        self._line_rt.set_data(ts, rs)
        if self._tcr_points:
            pts = np.array([[p["t"], p["r"]] for p in self._tcr_points], dtype=float)
            self._scatter_pts.set_offsets(pts)
        else:
            self._scatter_pts.set_offsets(np.empty((0, 2)))
        self._ax_tr.relim()
        self._ax_tr.autoscale_view()
        self._ax_r2.relim()
        self._ax_r2.autoscale_view()
        self._ax_rt.relim()
        self._ax_rt.autoscale_view()
        self._mpl_canvas.draw_idle()

    def _tcr_plot_show_fit(self, a: float, b: float, t_min: float, t_max: float) -> None:
        if not HAS_MPL or self._fit_line is None:
            return
        if t_max <= t_min:
            t_max = t_min + 1.0
        xs = [t_min, t_max]
        ys = [a + b * x for x in xs]
        self._fit_line.set_data(xs, ys)
        self._mpl_canvas.draw_idle()

    # —— 日志 / 串口 ——
    def _log(self, msg: str) -> None:
        self.log.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.see(tk.END)

    def refresh_ports(self) -> None:
        ports = []
        if list_ports is not None:
            items = list(list_ports.comports())
            ports = [p.device for p in items]
            detail = ", ".join(f"{p.device}({p.description})" for p in items) or "无"
            self._log(f"串口: {detail}")
        self.com_box["values"] = ports
        if ports and not self.com_var.get():
            prefer = ports[0]
            for p in list_ports.comports() if list_ports else []:
                if "PL2303" in (p.description or "") or "USB" in (p.description or "").upper():
                    prefer = p.device
                    break
            self.com_var.set(prefer)

    def _ohm_host(self) -> tuple[str, int]:
        return self.ip_var.get().strip(), int(self.port_var.get())

    def _base_kwargs(self) -> dict:
        com = self.com_var.get().strip()
        if not com:
            raise RuntimeError("请选择 COM 口")
        return {
            "port": com,
            "baud": int(self.baud_var.get()),
            "addr": int(self.addr_var.get()),
            "parity": self.parity_var.get(),
        }

    def _client_from_cfg(self) -> YudianTemp:
        if self._ok_cfg:
            c = self._ok_cfg
            return YudianTemp(
                port=c["port"],
                baud=c["baud"],
                addr=c["addr"],
                protocol=c["protocol"],
                parity=c["parity"],
                rs485=c["rs485"],
            )
        base = self._base_kwargs()
        proto = self.proto_var.get()
        if proto == "auto":
            proto = "aibus"
        return YudianTemp(
            port=base["port"],
            baud=base["baud"],
            addr=base["addr"],
            protocol=proto,
            parity=base["parity"],
            rs485="none",
        )

    def _get_live_or_temp_client(self) -> tuple[YudianTemp, bool]:
        if self._running and self._client is not None:
            return self._client, False
        return self._client_from_cfg(), True

    def _remember(self, client: YudianTemp, temp: Optional[float] = None) -> None:
        self._ok_cfg = {
            "port": client.port,
            "baud": client.baud,
            "addr": client.addr,
            "parity": client.parity,
            "protocol": client.protocol,
            "rs485": client.rs485,
            "temp": temp,
        }
        self.cfg_var.set(
            f"温度: {client.baud}bps / {client.protocol} / {client.rs485} / Addr={client.addr}"
        )

    # —— 温度操作 ——
    def auto_probe(self) -> None:
        if serial is None:
            messagebox.showerror("缺少依赖", "pip install pyserial")
            return
        try:
            base = self._base_kwargs()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._log("======== 温度自动探测 ========")
        bauds = [base["baud"]]
        for b in (9600, 19200, 28800, 4800):
            if b not in bauds:
                bauds.append(b)
        proto_ui = self.proto_var.get()
        protos = ["aibus", "modbus"] if proto_ui == "auto" else [proto_ui]
        found = None
        for baud in bauds:
            for proto in protos:
                for rsm in ("none", "rts_hi", "rts_lo"):
                    client = YudianTemp(
                        port=base["port"],
                        baud=baud,
                        addr=base["addr"],
                        protocol=proto,
                        parity=base["parity"],
                        rs485=rsm,
                    )
                    try:
                        self._log(f"试 {baud}/{proto}/{rsm} ...")
                        client.open()
                        t = client.read_temperature()
                        found = {
                            "port": base["port"],
                            "baud": baud,
                            "addr": base["addr"],
                            "parity": base["parity"],
                            "protocol": proto,
                            "rs485": rsm,
                            "temp": t,
                        }
                        self._log(f"成功 → {t:.3f} °C")
                        client.close()
                        break
                    except Exception as e:
                        self._log(f"  失败: {e}")
                        client.close()
                if found:
                    break
            if found:
                break

        if not found:
            self._ok_cfg = None
            self.cfg_var.set("温度: 探测失败")
            messagebox.showwarning("探测失败", "温度通道无应答，检查 A/B、Addr/AFC、COM。")
            return
        self._ok_cfg = found
        self.baud_var.set(str(found["baud"]))
        self.proto_var.set(found["protocol"])
        self.temp_var.set(f"{found['temp']:.3f}")
        self.cfg_var.set(
            f"温度: {found['baud']}bps / {found['protocol']} / {found['rs485']} / Addr={found['addr']}"
        )
        self.status_var.set(f"温度探测成功  {found['temp']:.3f} °C")

    def read_temp_once(self) -> None:
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            t = client.read_temperature()
            self.temp_var.set(f"{t:.3f}")
            self._log(f"温度 = {t:.3f} °C")
            self._remember(client, t)
            self.status_var.set(f"温度 OK  {t:.3f} °C")
        except Exception as e:
            self._log(f"读温度失败: {e}")
            messagebox.showerror("读温度失败", str(e))
        finally:
            if owned:
                client.close()

    def _fill_pid_ui(self, info: dict) -> None:
        def fmt(x: float) -> str:
            s = f"{x:.3f}".rstrip("0").rstrip(".")
            return s if s else "0"

        self.sv_var.set(fmt(info["sv"]))
        self.p_var.set(fmt(info["p"]))
        self.i_var.set(str(int(info["i"])) if info["i"] == int(info["i"]) else str(info["i"]))
        self.d_var.set(str(int(info["d"])) if info["d"] == int(info["d"]) else str(info["d"]))
        self.cti_var.set(str(int(info["cti"])) if info["cti"] == int(info["cti"]) else str(info["cti"]))
        self.ctrl_var.set(str(info["ctrl"]))
        self.at_var.set(f"{info['at']} ({AT_NAMES.get(info['at'], '?')})")
        self.sv_live_var.set(
            f"SV: {info['sv']:.3f} °C | CtrL={CTRL_NAMES.get(info['ctrl'], info['ctrl'])}"
        )

    def read_pid(self) -> None:
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            info = client.read_pid()
            self._fill_pid_ui(info)
            self._remember(client)
            self._log(
                f"PID: SV={info['sv']:.3f} P={info['p']:.3f} I={info['i']} "
                f"d={info['d']} CtI={info['cti']} CtrL={info['ctrl']} At={info['at']}"
            )
            self.status_var.set("PID 已读取")
        except Exception as e:
            self._log(f"读 PID 失败: {e}")
            messagebox.showerror("读 PID 失败", str(e))
        finally:
            if owned:
                client.close()

    def _parse_pid_fields(self) -> dict:
        return {
            "sv": float(self.sv_var.get()),
            "p": float(self.p_var.get()),
            "i": float(self.i_var.get()),
            "d": float(self.d_var.get()),
            "cti": float(self.cti_var.get()),
            "ctrl": int(self.ctrl_var.get()),
        }

    def write_pid(self) -> None:
        try:
            vals = self._parse_pid_fields()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return
        if not messagebox.askyesno("确认", f"写入 PID？\n{vals}"):
            return
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            client.write_pid(**vals)
            self._fill_pid_ui(client.read_pid())
            self._remember(client)
            self._log(f"已写入 PID: {vals}")
            self.status_var.set("PID 已写入")
        except Exception as e:
            self._log(f"写 PID 失败: {e}")
            messagebox.showerror("写 PID 失败", str(e))
        finally:
            if owned:
                client.close()

    def write_sv_only(self) -> None:
        try:
            sv = float(self.sv_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "请填写有效 SV")
            return
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            client.write_pid(sv=sv)
            self.sv_live_var.set(f"SV: {sv:.3f} °C")
            self._log(f"已写 SV={sv}")
            self._remember(client)
        except Exception as e:
            messagebox.showerror("写 SV 失败", str(e))
        finally:
            if owned:
                client.close()

    def _show_ctrl_status(self, st: dict) -> None:
        txt = (
            f"状态: PV={st['pv']:.3f} SV={st['sv']:.3f} | "
            f"AMAn={AMAN_NAMES.get(st['aman'], st['aman'])} "
            f"Srun={SRUN_NAMES.get(st['srun'], st['srun'])} "
            f"SPr={st['spr']:g} "
            f"Act={ACT_NAMES.get(st['act'], st['act'])} "
            f"MV={st['mv']} "
            f"CtrL={CTRL_NAMES.get(st['ctrl'], st['ctrl'])}"
        )
        self.ctrl_status_var.set(txt)
        self.sv_live_var.set(f"SV: {st['sv']:.3f} °C | MV={st['mv']}")
        self._log(txt)

    def read_ctrl_status(self) -> None:
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            st = client.read_control_status()
            self._show_ctrl_status(st)
            self._remember(client)
            tips = []
            if st["aman"] != 1:
                tips.append("当前不是 Auto，自动控温可能未生效")
            if st["srun"] != 0:
                tips.append("Srun 非 run（停止/保持）")
            if st["spr"] and st["spr"] > 0:
                tips.append(f"SPr={st['spr']:g} 限制升温速率，改 SV 后会慢慢爬，像发呆")
            if st["act"] == 1:
                tips.append("Act=正作用，加热系统通常应为反作用(0)")
            if tips:
                messagebox.showwarning("可能原因", "\n".join(tips))
            else:
                self.status_var.set("控制状态已读取")
        except Exception as e:
            self._log(f"读控制状态失败: {e}")
            messagebox.showerror("失败", str(e))
        finally:
            if owned:
                client.close()

    def do_unstick(self) -> None:
        if not messagebox.askyesno(
            "解除卡住",
            "将执行：Srun=run、清 SPr、手动↔自动（清积分），不改 SV。\n继续？",
        ):
            return
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            st = client.control_unstick(new_sv=None, clear_spr=True)
            self._show_ctrl_status(st)
            self._remember(client)
            self.status_var.set("已解除卡住")
            self._log("已执行控温复位（手↔自）")
            messagebox.showinfo("完成", "已复位。若仍不加热，请看 Act/接线/输出模块。")
        except Exception as e:
            self._log(f"解除卡住失败: {e}")
            messagebox.showerror("失败", str(e))
        finally:
            if owned:
                client.close()

    def write_sv_unstick(self) -> None:
        try:
            sv = float(self.sv_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "请填写有效 SV")
            return
        if not messagebox.askyesno(
            "写 SV 并解除卡住",
            f"先复位积分/SPr，再写入 SV={sv}\n"
            "（解决：原先 SV 低于 PV 很久，再改高 SV 仍长时间不加热）\n继续？",
        ):
            return
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            st = client.control_unstick(new_sv=sv, clear_spr=True)
            self.sv_var.set(f"{sv:g}")
            self._show_ctrl_status(st)
            self._remember(client)
            self.status_var.set(f"SV={sv} 已写入并复位")
            self._log(f"写 SV={sv} 并解除卡住完成")
        except Exception as e:
            self._log(f"写SV解除卡住失败: {e}")
            messagebox.showerror("失败", str(e))
        finally:
            if owned:
                client.close()

    def start_at(self) -> None:
        if not messagebox.askyesno("自整定", "启动 At=1？建议先设好 SV。温度会波动。"):
            return
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            if self.sv_var.get().strip():
                try:
                    client.write_pid(sv=float(self.sv_var.get()))
                except ValueError:
                    pass
            client.start_autotune()
            self._fill_pid_ui(client.read_pid())
            self._log("已启动自整定")
            self.status_var.set("自整定进行中…")
        except Exception as e:
            messagebox.showerror("失败", str(e))
        finally:
            if owned:
                client.close()

    def stop_at(self) -> None:
        client, owned = self._get_live_or_temp_client()
        try:
            if owned:
                client.open()
            client.stop_autotune()
            self._fill_pid_ui(client.read_pid())
            self._log("已停止自整定")
        except Exception as e:
            messagebox.showerror("失败", str(e))
        finally:
            if owned:
                client.close()

    # —— 电阻操作 ——
    def do_idn(self) -> None:
        try:
            ip, port = self._ohm_host()
            text = scpi_exchange(ip, port, "*IDN?")
            self._log(f"*IDN? → {text}")
            self.ohm_cfg_var.set(f"电阻: {text[:48]}")
            self.status_var.set(text)
        except Exception as e:
            self._log(f"*IDN? 失败: {e}")
            messagebox.showerror("识别失败", str(e))

    def read_ohm_once(self) -> None:
        try:
            ip, port = self._ohm_host()
            ohm, detail = read_resistance_live(ip, port)
            text = format_ohm(ohm)
            self.ohm_var.set(text)
            self.ohm_cfg_var.set(f"电阻: {ip}:{port}")
            self._log(detail)
            self._log(f"电阻 = {text}")
            self.status_var.set(f"电阻 OK  {text}")
        except Exception as e:
            self._log(f"读电阻失败: {e}")
            messagebox.showerror("读电阻失败", str(e))

    def read_both_once(self) -> None:
        if self.enable_temp.get():
            try:
                self.read_temp_once()
            except Exception:
                pass
        if self.enable_ohm.get():
            try:
                self.read_ohm_once()
            except Exception:
                pass
        if not self.enable_temp.get() and not self.enable_ohm.get():
            messagebox.showinfo("提示", "请至少勾选「采温度」或「采电阻」")

    # —— TCR ——
    def _tcr_refresh_tree(self) -> None:
        for i in self.tcr_tree.get_children():
            self.tcr_tree.delete(i)
        for i, p in enumerate(self._tcr_points, 1):
            self.tcr_tree.insert(
                "",
                tk.END,
                values=(i, f"{p['t']:.2f}", f"{p['r']:.5g}"),
            )
        kids = self.tcr_tree.get_children()
        if kids:
            self.tcr_tree.see(kids[-1])

    def _tcr_append(self, t: float, r: float) -> None:
        ts = time.strftime("%H:%M:%S")
        self._tcr_points.append({"t": t, "r": r, "ts": ts})
        self._tcr_refresh_tree()
        self._tcr_plot_push(t, r, force=True)
        self._log(f"TCR 点#{len(self._tcr_points)}: T={t:.4f} °C  R={r:.8g} Ω")

    def _read_tr_pair(self, client: Optional[YudianTemp] = None) -> tuple[float, float]:
        """同时读温度与电阻。client 若给出则复用（已 open）。"""
        owned = False
        if client is None:
            client = self._client_from_cfg()
            client.open()
            owned = True
        try:
            t = client.read_temperature()
            ip, port = self._ohm_host()
            r, _ = read_resistance_live(ip, port)
            self.temp_var.set(f"{t:.3f}")
            self.ohm_var.set(format_ohm(r))
            return t, r
        finally:
            if owned:
                client.close()

    def tcr_add_point(self) -> None:
        try:
            if self._running and self._client is not None:
                t, r = self._read_tr_pair(self._client)
            else:
                t, r = self._read_tr_pair()
            self._tcr_append(t, r)
            self.status_var.set(f"已采点 T={t:.3f} R={format_ohm(r)}")
        except Exception as e:
            self._log(f"采点失败: {e}")
            messagebox.showerror("采点失败", str(e))

    def tcr_clear(self) -> None:
        if self._tcr_points and not messagebox.askyesno("确认", "清空全部 TCR 数据点？"):
            return
        self._tcr_points.clear()
        self._tcr_refresh_tree()
        self._tcr_plot_reset()
        self.tcr_dash_var.set("—")
        self.tcr_info_var.set("TCR: 未计算")
        self.tcr_result_var.set("尚未计算")
        self._log("TCR 数据已清空")

    def tcr_compute(self) -> None:
        if len(self._tcr_points) < 2:
            messagebox.showinfo("提示", "至少需要 2 个数据点")
            return
        try:
            t_ref = float(self.tcr_tref_var.get())
            temps = [p["t"] for p in self._tcr_points]
            ohms = [p["r"] for p in self._tcr_points]
            res = compute_tcr(temps, ohms, t_ref)
        except Exception as e:
            messagebox.showerror("计算失败", str(e))
            return
        self.tcr_dash_var.set(f"{res['alpha_ppm']:.2f}")
        self.tcr_info_var.set(
            f"ppm/°C @ {res['t_ref']:g}°C  R²={res['r2']:.4f}  n={res['n']}"
        )
        self.tcr_result_var.set(
            f"α = {res['alpha']:.6e} /°C  =  {res['alpha_ppm']:.2f} ppm/°C\n"
            f"拟合 R = {res['a']:.6g} + {res['b']:.6g}·T   R²={res['r2']:.5f}\n"
            f"R(Tref={res['t_ref']:g}) = {res['r_ref']:.8g} Ω   "
            f"范围 {res['t_min']:.2f}~{res['t_max']:.2f} °C\n"
            f"两点对照 α = {res['alpha_2pt_ppm']:.2f} ppm/°C（首末点）"
        )
        self._log(
            f"TCR: α={res['alpha_ppm']:.2f} ppm/°C @ {res['t_ref']:g}°C, "
            f"R²={res['r2']:.5f}, n={res['n']}"
        )
        self.status_var.set(f"TCR = {res['alpha_ppm']:.2f} ppm/°C")
        self._tcr_plot_show_fit(res["a"], res["b"], res["t_min"], res["t_max"])
        self._tcr_plot_redraw()

    def tcr_export(self) -> None:
        if not self._tcr_points:
            messagebox.showinfo("提示", "无数据可导出")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("全部", "*.*")],
            initialfile=f"tcr_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        )
        if not path:
            return
        try:
            t_ref = float(self.tcr_tref_var.get())
        except ValueError:
            t_ref = 25.0
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["index", "T_C", "R_ohm", "time"])
            for i, p in enumerate(self._tcr_points, 1):
                w.writerow([i, f"{p['t']:.6f}", f"{p['r']:.10g}", p["ts"]])
            if len(self._tcr_points) >= 2:
                try:
                    res = compute_tcr(
                        [p["t"] for p in self._tcr_points],
                        [p["r"] for p in self._tcr_points],
                        t_ref,
                    )
                    w.writerow([])
                    w.writerow(["alpha_1_per_C", res["alpha"]])
                    w.writerow(["alpha_ppm_per_C", res["alpha_ppm"]])
                    w.writerow(["Tref_C", res["t_ref"]])
                    w.writerow(["R_at_Tref", res["r_ref"]])
                    w.writerow(["fit_a", res["a"]])
                    w.writerow(["fit_b", res["b"]])
                    w.writerow(["R_squared", res["r2"]])
                    w.writerow(["alpha_2point_ppm", res["alpha_2pt_ppm"]])
                except Exception:
                    pass
        self._log(f"已导出 {path}")
        messagebox.showinfo("导出完成", path)

    def tcr_start_sweep(self) -> None:
        if self._tcr_running or self._running:
            messagebox.showinfo("提示", "请先停止联测/扫描，再启动自动扫描")
            return
        client: Optional[YudianTemp] = None
        try:
            targets = [float(x.strip()) for x in self.tcr_targets_var.get().split(",") if x.strip()]
            tol = float(self.tcr_tol_var.get())
            hold = float(self.tcr_hold_var.get())
            r_hold = float(self.tcr_r_hold_var.get())
            r_tol = float(self.tcr_r_tol_var.get())
            timeout = float(self.tcr_timeout_var.get())
            if len(targets) < 2:
                raise ValueError("至少 2 个目标温度")
            if tol <= 0 or hold < 0 or r_hold <= 0 or r_tol <= 0 or timeout <= 0:
                raise ValueError("容差/保持/R稳定/超时参数无效")
            self._ohm_host()
            client = self._client_from_cfg()
            client.open()
            client.read_temperature()
        except Exception as e:
            if client is not None:
                client.close()
            messagebox.showerror("无法启动扫描", str(e))
            return

        clear = messagebox.askyesnocancel(
            "自动扫描",
            f"目标: {targets}\n"
            f"T: 容差±{tol}°C 保持{hold}s\n"
            f"R: 最近{r_hold}s 内变化≤{r_tol:g}(相对)\n"
            f"单点超时 {timeout}s\n\n"
            "是=清空后扫描  否=追加扫描  取消=不开始",
        )
        if clear is None:
            client.close()
            return
        if clear:
            self._tcr_points.clear()
            self._tcr_refresh_tree()

        self._tcr_plot_reset()
        self._client = client
        self._remember(client)
        self._tcr_stop.clear()
        self._tcr_running = True
        self.tcr_sweep_btn.configure(state=tk.DISABLED)
        self.tcr_stop_btn.configure(state=tk.NORMAL)
        self.start_btn.configure(state=tk.DISABLED)
        self._log(
            f"TCR 自动扫描开始: {targets} | T保持={hold}s R稳定={r_hold}s R容差={r_tol:g}"
        )
        self._tcr_thread = threading.Thread(
            target=self._tcr_sweep_loop,
            args=(targets, tol, hold, r_hold, r_tol, timeout),
            daemon=True,
        )
        self._tcr_thread.start()

    def tcr_stop_sweep(self) -> None:
        self._tcr_stop.set()
        self._log("正在停止 TCR 扫描…")

    def _tcr_sweep_finish(self) -> None:
        self._tcr_running = False
        self.tcr_sweep_btn.configure(state=tk.NORMAL)
        self.tcr_stop_btn.configure(state=tk.DISABLED)
        self.start_btn.configure(state=tk.NORMAL)
        if self._client:
            self._client.close()
            self._client = None
        if len(self._tcr_points) >= 2:
            self.tcr_compute()
        self.status_var.set("TCR 扫描结束")
        self._log("TCR 扫描结束")

    @staticmethod
    def _r_window_stable(
        hist: list[tuple[float, float]],
        r_hold: float,
        r_tol: float,
        now: float,
    ) -> tuple[bool, float]:
        """
        最近 r_hold 秒内电阻相对峰峰值 ≤ r_tol 则稳定。
        返回 (是否稳定, 窗口内相对变化量；数据不足为 inf)。
        """
        window = [(ts, rv) for ts, rv in hist if now - ts <= r_hold]
        if len(window) < 2:
            return False, float("inf")
        if now - window[0][0] < r_hold * 0.95:
            return False, float("inf")
        vals = [rv for _, rv in window]
        r_min, r_max = min(vals), max(vals)
        r_mean = sum(vals) / len(vals)
        base = max(abs(r_mean), 1e-12)
        rel = (r_max - r_min) / base
        return rel <= r_tol, rel

    def _tcr_sweep_loop(
        self,
        targets: list[float],
        tol: float,
        hold: float,
        r_hold: float,
        r_tol: float,
        timeout: float,
    ) -> None:
        client = self._client
        assert client is not None
        try:
            for sv in targets:
                if self._tcr_stop.is_set():
                    break
                self.after(0, lambda s=sv: self._log(f"→ 设定 SV={s:g} °C，等待 T/R 双稳定…"))
                self.after(0, lambda s=sv: self.status_var.set(f"扫描 SV={s:g} °C"))
                try:
                    client.control_unstick(new_sv=sv, clear_spr=True)
                    self.after(0, lambda s=sv: self.sv_live_var.set(f"SV: {s:.3f} °C"))
                except Exception as e:
                    self.after(0, lambda err=str(e): self._log(f"写 SV 失败: {err}"))
                    continue

                deadline = time.time() + timeout
                t_stable_since: Optional[float] = None
                r_hist: list[tuple[float, float]] = []
                while time.time() < deadline and not self._tcr_stop.is_set():
                    now = time.time()
                    t: Optional[float] = None
                    r: Optional[float] = None
                    try:
                        t = client.read_temperature()
                        self.after(0, lambda x=t: self.temp_var.set(f"{x:.3f}"))
                    except Exception as e:
                        self.after(0, lambda err=str(e): self._log(f"读温失败: {err}"))
                    try:
                        ip, port = self._ohm_host()
                        r, _ = read_resistance_live(ip, port)
                        self.after(0, lambda o=r: self.ohm_var.set(format_ohm(o)))
                    except Exception as e:
                        self.after(0, lambda err=str(e): self._log(f"读电阻失败: {err}"))
                    if t is not None and r is not None:
                        self.after(0, lambda tt=t, rr=r: self._tcr_plot_push(tt, rr))
                    if r is not None:
                        r_hist.append((now, r))
                        # 只保留略长于窗口的历史
                        cut = now - r_hold - 5.0
                        r_hist = [(ts, rv) for ts, rv in r_hist if ts >= cut]

                    if t is None:
                        time.sleep(1.0)
                        continue

                    if abs(t - sv) <= tol:
                        if t_stable_since is None:
                            t_stable_since = now
                    else:
                        t_stable_since = None

                    t_ok = t_stable_since is not None and (now - t_stable_since) >= hold
                    r_ok, r_rel = self._r_window_stable(r_hist, r_hold, r_tol, now)

                    t_left = (
                        hold - (now - t_stable_since) if t_stable_since is not None else hold
                    )
                    if r_ok:
                        r_left = 0.0
                    elif len(r_hist) >= 2:
                        r_left = max(0.0, r_hold - (now - r_hist[0][0]))
                    else:
                        r_left = r_hold

                    r_txt = f"{r_rel:.2e}" if math.isfinite(r_rel) else "—"
                    self.after(
                        0,
                        lambda x=t, s=sv, tl=t_left, rl=r_left, rt=r_txt, to=t_ok, ro=r_ok: self.status_var.set(
                            f"SV={s:g} PV={x:.3f}  "
                            f"T再稳{max(0, tl):.0f}s{'✓' if to else ''}  "
                            f"RΔ={rt} 再{max(0, rl):.0f}s{'✓' if ro else ''}"
                        ),
                    )

                    if t_ok and r_ok:
                        break
                    time.sleep(0.3)
                else:
                    if not self._tcr_stop.is_set():
                        self.after(0, lambda s=sv: self._log(f"SV={s:g} 超时未达 T/R 双稳定，跳过"))
                    continue

                if self._tcr_stop.is_set():
                    break
                # 采平均：连续读几次电阻与温度
                ts, rs = [], []
                for _ in range(3):
                    if self._tcr_stop.is_set():
                        break
                    try:
                        t = client.read_temperature()
                        ip, port = self._ohm_host()
                        r, _ = read_resistance_live(ip, port)
                        ts.append(t)
                        rs.append(r)
                        self.after(0, lambda x=t: self.temp_var.set(f"{x:.3f}"))
                        self.after(0, lambda o=r: self.ohm_var.set(format_ohm(o)))
                        self.after(0, lambda tt=t, rr=r: self._tcr_plot_push(tt, rr, True))
                    except Exception as e:
                        self.after(0, lambda err=str(e): self._log(f"采点读失败: {err}"))
                    time.sleep(0.5)
                if ts and rs:
                    t_avg = sum(ts) / len(ts)
                    r_avg = sum(rs) / len(rs)
                    self.after(0, lambda ta=t_avg, ra=r_avg: self._tcr_append(ta, ra))
        finally:
            self.after(0, self._tcr_sweep_finish)

    # —— 联测循环 ——
    def start(self) -> None:
        if self._running:
            return
        if self._tcr_running:
            messagebox.showinfo("提示", "请先停止 TCR 扫描")
            return
        if not self.enable_temp.get() and not self.enable_ohm.get():
            messagebox.showinfo("提示", "请至少勾选一路通道")
            return
        client: Optional[YudianTemp] = None
        try:
            float(self.interval_var.get())
            if self.enable_ohm.get():
                self._ohm_host()
            if self.enable_temp.get():
                client = self._client_from_cfg()
                client.open()
                t = client.read_temperature()
                self.temp_var.set(f"{t:.3f}")
                self._remember(client, t)
            if self.enable_ohm.get():
                ip, port = self._ohm_host()
                ohm, _ = read_resistance_live(ip, port)
                self.ohm_var.set(format_ohm(ohm))
                self.ohm_cfg_var.set(f"电阻: {ip}:{port}")
        except Exception as e:
            if client is not None:
                client.close()
            messagebox.showerror("无法启动", str(e))
            return

        self._client = client
        self._stop.clear()
        self._running = True
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.probe_btn.configure(state=tk.DISABLED)
        self.temp_once_btn.configure(state=tk.DISABLED)
        self.ohm_once_btn.configure(state=tk.DISABLED)
        self.idn_btn.configure(state=tk.DISABLED)
        self.tcr_sweep_btn.configure(state=tk.DISABLED)
        self._log("联测开始")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.probe_btn.configure(state=tk.NORMAL)
        self.temp_once_btn.configure(state=tk.NORMAL)
        self.ohm_once_btn.configure(state=tk.NORMAL)
        self.idn_btn.configure(state=tk.NORMAL)
        if not self._tcr_running:
            self.tcr_sweep_btn.configure(state=tk.NORMAL)
        if self._client and not self._tcr_running:
            self._client.close()
            self._client = None
        self.status_var.set("已停止")
        self._log("已停止")

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            parts = []
            if self.enable_temp.get() and self._client is not None:
                try:
                    temp = self._client.read_temperature()
                    self.after(0, lambda t=temp: self._on_temp(t))
                    parts.append(f"{temp:.3f}°C")
                except Exception as e:
                    self.after(0, lambda err=str(e): self._on_err(f"温度: {err}"))
            if self.enable_ohm.get():
                try:
                    ip, port = self._ohm_host()
                    ohm, detail = read_resistance_live(ip, port)
                    text = format_ohm(ohm)
                    self.after(0, lambda o=ohm, tx=text, d=detail: self._on_ohm(o, tx, d))
                    parts.append(text)
                except Exception as e:
                    self.after(0, lambda err=str(e): self._on_err(f"电阻: {err}"))
            if parts:
                msg = "  |  ".join(parts)
                self.after(0, lambda m=msg: self.status_var.set(f"{time.strftime('%H:%M:%S')}  {m}"))
            try:
                interval = float(self.interval_var.get())
            except ValueError:
                interval = 1.0
            time.sleep(max(0.0, interval - (time.time() - t0)))

    def _on_temp(self, temp: float) -> None:
        self.temp_var.set(f"{temp:.3f}")

    def _on_ohm(self, ohm: float, text: str, detail: str) -> None:
        self.ohm_var.set(text)
        if self._last_ohm is None or abs(self._last_ohm - ohm) > 1e-9:
            self._log(detail)
            self._last_ohm = ohm

    def _on_err(self, err: str) -> None:
        self.status_var.set(f"失败: {err}")
        self._log(f"失败: {err}")

    def _on_close(self) -> None:
        self._stop.set()
        self._tcr_stop.set()
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self.destroy()


if __name__ == "__main__":
    InstrumentApp().mainloop()
