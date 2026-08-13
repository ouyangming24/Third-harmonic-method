"""
自动调平闭环：数控电阻 + 锁相 1ω + 差分开关。

流程（与实验手工步骤对应）:
  1. 适中频率 / 适中输出幅值，打开差分
  2. 用有符号 X 的 PI 控制自动步长，使差分 1ω → 0
  3. 关闭差分
  4. 调节锁相输出幅值，使单端 1ω 达到目标加热幅值（不超过 UMax）
  5. 再开差分；若差分 1ω 仍够小则结束，否则缩小最大ΔR 后精调
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from balance_resistor import BalanceResistorController
from oe1300_lockin import OE1300LockInControl

# 与测量页一致
LOCKIN_TC_FACTOR = 5.0


@dataclass
class AutoBalanceConfig:
    # 锁相连接
    lockin_backend: str = "zurich"  # zurich | oe1300_tcp | oe1300_serial
    device_id: str = "dev3791"
    zi_host: str = "192.168.59.62"
    zi_port: int = 8004
    oe_tcp_host: str = "192.168.1.1"
    oe_tcp_port: int = 10001
    oe_serial_port: str = "COM3"
    oe_input_range_index: Optional[int] = None

    # 调平工况
    balance_freq_hz: float = 635.0
    coarse_output_v: float = 2.0
    target_v1_heat_v: float = 0.200  # 单端 1ω 目标（加热幅值）
    r_start_ohm: float = 1.0
    r_min_ohm: float = 0.01
    r_max_ohm: float = 1000.0
    # PI 调平：最大单步 ΔR（界面可设）；实际步长由误差/灵敏度自动决定
    max_delta_r_ohm: float = 0.05
    min_step_ohm: float = 0.0005
    pi_probe_ohm: float = 0.02  # 估计 dX/dR 的试探步长
    pi_ki_factor: float = 0.12  # 积分项相对比例项的权重
    pi_noise_floor_v: float = 5e-6  # |X| 低于此视为噪声，减弱积分
    # 兼容旧字段名
    coarse_step_ohm: float = 0.05
    diff_ok_v: float = 300e-6  # 差分 1ω 可接受上限
    rebalance_v: float = 800e-6  # 重开差分后超过此值则重调
    max_outer_loops: int = 4
    max_resist_evals: int = 80
    umax_v: Optional[float] = None  # 1ω 信号上限（非锁相输出上限）
    settle_multiplier: int = 4
    zurich_read_duration: float = 0.5
    oe_snap_n_samples: int = 4
    oe_snap_duration: float = 0.6

    def __post_init__(self) -> None:
        # 若只设了旧名 coarse_step，同步到 max_delta
        if self.max_delta_r_ohm == 0.05 and self.coarse_step_ohm != 0.05:
            self.max_delta_r_ohm = float(self.coarse_step_ohm)
        elif self.coarse_step_ohm == 0.05 and self.max_delta_r_ohm != 0.05:
            self.coarse_step_ohm = float(self.max_delta_r_ohm)
        else:
            self.max_delta_r_ohm = float(self.max_delta_r_ohm or self.coarse_step_ohm)
            self.coarse_step_ohm = float(self.max_delta_r_ohm)


StatusCallback = Callable[[dict], None]


def _clip_r(r: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(r)))


def _configure_outputs(lockin, amplitude: float, on: bool = True) -> None:
    """Zurich 第 4 参名 range_val，OE 为 range；用位置参数兼容两者。"""
    lockin.configure_signal_outputs(0, float(amplitude), on, 10.0, 0.0)


def _connect_lockin(cfg: AutoBalanceConfig):
    backend = (cfg.lockin_backend or "zurich").lower()
    if backend == "zurich":
        from 苏黎世锁相适配3 import ZurichLockInControl
        return ZurichLockInControl(
            cfg.device_id, server_host=cfg.zi_host, server_port=int(cfg.zi_port)
        )
    if backend == "oe1300_tcp":
        li = OE1300LockInControl(
            connection="tcp",
            tcp_host=cfg.oe_tcp_host,
            tcp_port=int(cfg.oe_tcp_port),
        )
        if cfg.oe_input_range_index is not None:
            li.set_input_range_index(int(cfg.oe_input_range_index))
        return li
    if backend == "oe1300_serial":
        li = OE1300LockInControl(
            connection="serial",
            serial_port=cfg.oe_serial_port,
        )
        if cfg.oe_input_range_index is not None:
            li.set_input_range_index(int(cfg.oe_input_range_index))
        return li
    raise ValueError(f"未知锁相类型: {backend}")


class AutoBalanceWorker:
    """在工作线程内调用的调平逻辑（可单测）。"""

    def __init__(
        self,
        cfg: AutoBalanceConfig,
        balance: BalanceResistorController,
        emit_status: Optional[StatusCallback] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ):
        self.cfg = cfg
        self.balance = balance
        self.emit_status = emit_status or (lambda _d: None)
        self.should_stop = should_stop or (lambda: False)
        self.lockin = None
        self.backend = (cfg.lockin_backend or "zurich").lower()
        self.diff_on = False
        self.r_ohm = float(cfg.r_start_ohm)
        self.output_v = float(cfg.coarse_output_v)
        self.v1_amp = None
        self.v1_x = None
        self.v1_phase = None
        self._resist_evals = 0
        self.final_output_v = None
        self.final_r_ohm = None
        self.final_v1_diff = None

    def _check_stop(self) -> None:
        if self.should_stop():
            raise InterruptedError("用户停止自动调平")

    def _status(self, phase: str, message: str = "") -> None:
        self.emit_status(
            {
                "phase": phase,
                "message": message,
                "resistance_ohm": self.r_ohm,
                "diff_on": self.diff_on,
                "v1_amp": self.v1_amp,
                "v1_x": self.v1_x,
                "v1_phase": self.v1_phase,
                "output_v": self.output_v,
                "umax_v": self.cfg.umax_v,
            }
        )

    def _output_vmax(self) -> float:
        """锁相输出幅值上限：苏黎世 10 V，OE1300 5 Vrms。"""
        return 10.0 if self.backend == "zurich" else 5.0

    def _clamp_output(self, v: float) -> float:
        """仅钳制仪器输出能力；UMax 不限制锁相输出。"""
        v = max(1e-4, float(v))
        return min(v, self._output_vmax())

    def _effective_v1_target(self) -> float:
        """加热定标用的 1ω 目标；若有 UMax 则不得超过。"""
        target = float(self.cfg.target_v1_heat_v)
        umax = self.cfg.umax_v
        if umax is not None and umax > 0 and target > umax + 1e-12:
            self._status(
                "warn",
                f"目标1ω {target:.4f} V 超过 UMax={umax:.4f} V，按 UMax 定标",
            )
            return float(umax)
        return target

    def _v1_over_umax(self, amp: Optional[float] = None) -> bool:
        umax = self.cfg.umax_v
        if umax is None or umax <= 0:
            return False
        a = self.v1_amp if amp is None else amp
        return a is not None and float(a) > float(umax) + 1e-12

    def _set_diff(self, on: bool) -> None:
        self._check_stop()
        self.lockin.configure_signal_inputs(
            input_channel=0, range=1.0, ac=False, diff=bool(on)
        )
        self.diff_on = bool(on)
        time.sleep(0.05)
        self._status("diff", f"差分模式: {'开' if on else '关'}")

    def _set_output(self, amp: float) -> None:
        self._check_stop()
        amp = self._clamp_output(amp)
        _configure_outputs(self.lockin, amp, on=True)
        self.output_v = amp
        self._status("output", f"输出幅值 → {amp:.4f} V")

    def _set_resistance(self, ohms: float) -> float:
        self._check_stop()
        ohms = _clip_r(ohms, self.cfg.r_min_ohm, self.cfg.r_max_ohm)
        st = self.balance.set_resistance(ohms)
        self._resist_evals += 1
        if st.sp_ohm is not None:
            self.r_ohm = float(st.sp_ohm)
        else:
            self.r_ohm = ohms
        if st.umax_v is not None:
            self.cfg.umax_v = float(st.umax_v)
        self._status("resist", f"设定电阻 → {self.r_ohm:.6g} Ω")
        return self.r_ohm

    def _settle(self) -> None:
        freq = max(1e-6, float(self.cfg.balance_freq_hz))
        tc = LOCKIN_TC_FACTOR / freq
        self.lockin.wait_for_integration(
            tc, duration_multiplier=int(self.cfg.settle_multiplier)
        )

    def _read_v1(self, settle: bool = True) -> float:
        self._check_stop()
        if settle:
            self._settle()
        if self.backend == "zurich":
            x1, y1, _, _ = self.lockin.read_demod_data(
                duration=float(self.cfg.zurich_read_duration)
            )
        else:
            x1, y1, _, _ = self.lockin.read_demod_data(
                duration=max(0.05, float(self.cfg.oe_snap_duration)),
                n_samples=max(1, int(self.cfg.oe_snap_n_samples)),
            )
        amp, phase, x_mean, _ = self.lockin.calculate_amplitude_phase(x1, y1)
        if amp is None:
            raise RuntimeError("未能读取 1ω 解调数据")
        self.v1_amp = float(amp)
        self.v1_x = float(x_mean) if x_mean is not None else None
        self.v1_phase = float(phase) if phase is not None else None
        note = f"1ω={self.v1_amp * 1e6:.1f} μV"
        if self.v1_x is not None:
            note += f", X={self.v1_x * 1e6:.1f} μV"
        if self._v1_over_umax(self.v1_amp):
            note += f" ⚠超过 UMax={self.cfg.umax_v:.4f} V"
        self._status("read", note)
        return self.v1_amp

    def _estimate_dx_dr(self, r: float, max_delta: float) -> float:
        """试探步估计灵敏度 sens=dX/dR；失败则抛错。"""
        cfg = self.cfg
        probe = min(
            abs(float(cfg.pi_probe_ohm)),
            abs(max_delta),
            max(cfg.min_step_ohm * 10, 0.005),
        )
        if probe < cfg.min_step_ohm:
            probe = cfg.min_step_ohm

        self._set_resistance(r)
        self._read_v1(settle=True)
        e0 = float(self.v1_x if self.v1_x is not None else 0.0)

        for direction in (1.0, -1.0):
            self._check_stop()
            r_try = _clip_r(r + direction * probe, cfg.r_min_ohm, cfg.r_max_ohm)
            actual = r_try - r
            if abs(actual) < 0.5 * cfg.min_step_ohm:
                continue
            self._set_resistance(r_try)
            self._read_v1(settle=True)
            e1 = float(self.v1_x if self.v1_x is not None else 0.0)
            # 回到原点
            self._set_resistance(r)
            self._read_v1(settle=True)
            sens = (e1 - e0) / actual
            if abs(sens) >= max(1e-6, float(cfg.pi_noise_floor_v) / max(abs(actual), 1e-9)):
                self._status(
                    "search",
                    f"灵敏度 dX/dR ≈ {sens:.4g} V/Ω (试探 {actual:+.4g} Ω)",
                )
                return sens

        raise RuntimeError(
            "无法估计 dX/dR：试探后 X 几乎不变。请检查差分是否开启、接线或增大试探/最大ΔR"
        )

    def _minimize_diff_v1(self, max_delta: Optional[float] = None, target: float = 300e-6) -> float:
        """
        用有符号 X 的 PI 控制自动决定步长，使差分 1ω → 0。
        ΔR = -(e + ki*∫e) / sens，|ΔR| 受 max_delta 限制。
        """
        cfg = self.cfg
        max_delta = float(max_delta if max_delta is not None else cfg.max_delta_r_ohm)
        max_delta = max(cfg.min_step_ohm, abs(max_delta))
        ki = float(cfg.pi_ki_factor)
        noise = float(cfg.pi_noise_floor_v)

        r = self._set_resistance(self.r_ohm)
        amp = self._read_v1(settle=True)
        e = float(self.v1_x if self.v1_x is not None else 0.0)
        self._status(
            "search",
            f"PI调平起点 R={r:.6g} Ω, |V1|={amp * 1e6:.1f} μV, X={e * 1e6:.1f} μV",
        )
        if amp <= target:
            return amp

        sens = self._estimate_dx_dr(r, max_delta)
        # 重新读当前点（估计过程会回到 r）
        amp = self._read_v1(settle=True)
        e = float(self.v1_x if self.v1_x is not None else 0.0)
        best_amp, best_r = amp, r
        i_accum = 0.0
        gain_scale = 1.0  # 过冲时自动衰减
        stagnant = 0
        prev_e = e

        while self._resist_evals < cfg.max_resist_evals:
            self._check_stop()
            amp = float(self.v1_amp)
            e = float(self.v1_x if self.v1_x is not None else 0.0)
            if amp < best_amp:
                best_amp, best_r = amp, r
            if amp <= target:
                break

            # 积分（噪声门限内不累积）
            if abs(e) > noise:
                i_accum += e
            else:
                i_accum *= 0.5

            # 抗饱和：限制积分折合的电阻修正
            i_term = ki * i_accum
            max_i = abs(sens) * max_delta * 3.0
            if abs(i_term) > max_i and abs(sens) > 0:
                i_accum = (max_i if i_term > 0 else -max_i) / ki
                i_term = ki * i_accum

            raw = -(e + i_term) / sens * gain_scale
            # 限幅单步
            if abs(raw) > max_delta:
                raw = max_delta if raw > 0 else -max_delta
            if abs(raw) < cfg.min_step_ohm:
                # 误差仍大但算出的步长过小 → 灵敏度可能偏大，略放大
                if amp > target * 1.5:
                    gain_scale = min(4.0, gain_scale * 1.25)
                    stagnant += 1
                    if stagnant >= 3:
                        # 重新标定灵敏度
                        try:
                            sens = self._estimate_dx_dr(r, max_delta)
                            gain_scale = 1.0
                            i_accum = 0.0
                            stagnant = 0
                        except RuntimeError:
                            break
                    self._read_v1(settle=True)
                    continue
                # 已接近，步长太小则结束
                if amp <= target * 2 or stagnant >= 2:
                    break
                stagnant += 1
                if stagnant >= 4:
                    break
                continue

            stagnant = 0
            r_next = _clip_r(r + raw, cfg.r_min_ohm, cfg.r_max_ohm)
            if abs(r_next - r) < 0.5 * cfg.min_step_ohm:
                break

            self._set_resistance(r_next)
            amp_new = self._read_v1(settle=True)
            e_new = float(self.v1_x if self.v1_x is not None else 0.0)

            # 过冲检测：幅值明显变差且符号翻转 → 衰减增益并回退半步
            if amp_new > amp * 1.25 and (e * e_new < 0):
                gain_scale *= 0.5
                i_accum *= 0.3
                r_back = _clip_r(r + 0.5 * (r_next - r), cfg.r_min_ohm, cfg.r_max_ohm)
                self._set_resistance(r_back)
                amp_new = self._read_v1(settle=True)
                e_new = float(self.v1_x if self.v1_x is not None else 0.0)
                self._status(
                    "search",
                    f"过冲，增益×0.5 → {gain_scale:.3g}; R={r_back:.6g}",
                )
                r = r_back
            else:
                r = r_next

            self._status(
                "search",
                f"PI: ΔR={raw:+.5g} Ω → R={r:.6g}, |V1|={amp_new * 1e6:.1f} μV, "
                f"X={e_new * 1e6:.1f} μV",
            )
            # 符号持续同向且收敛慢时可略增增益
            if e_new * prev_e > 0 and amp_new < amp * 0.9:
                gain_scale = min(2.0, gain_scale * 1.05)
            prev_e = e_new

            if amp_new < best_amp:
                best_amp, best_r = amp_new, r

        # 回到最优电阻
        if abs(best_r - self.r_ohm) > cfg.min_step_ohm * 0.5 or best_amp < float(self.v1_amp or 1e9):
            self._set_resistance(best_r)
            best_amp = self._read_v1(settle=True)
        self.r_ohm = best_r
        self._status(
            "search",
            f"PI调平结束: R={best_r:.6g} Ω, |V1|={best_amp * 1e6:.1f} μV",
        )
        return best_amp

    def _scale_output_to_target_v1(self) -> float:
        """差分关闭时，按比例调节输出使单端 1ω ≈ 目标值（且不超过 UMax）。"""
        target = self._effective_v1_target()
        vout = self.output_v
        last_amp = None
        for i in range(4):
            self._check_stop()
            amp = self._read_v1(settle=True)
            last_amp = amp
            if amp < 1e-9:
                raise RuntimeError("单端 1ω 过小，无法按比例定标输出幅值")
            # 已超过 UMax：先按比例压回
            umax = self.cfg.umax_v
            if umax is not None and umax > 0 and amp > umax + 1e-12:
                vout = self._clamp_output(vout * (umax / amp) * 0.98)
                self._set_output(vout)
                self._status(
                    "heat",
                    f"1ω={amp:.4f} V 超过 UMax={umax:.4f} V，降低输出 → {vout:.4f} V",
                )
                continue
            ratio = target / amp
            # 已足够接近则停止
            if abs(ratio - 1.0) < 0.02:
                break
            vout = self._clamp_output(vout * ratio)
            self._set_output(vout)
            self._status(
                "heat",
                f"定标输出第 {i + 1} 次: V1={amp * 1e3:.3f} mV → 目标 {target * 1e3:.1f} mV",
            )
        last_amp = self._read_v1(settle=True)
        if self._v1_over_umax(last_amp):
            raise RuntimeError(
                f"定标后单端1ω={last_amp:.4f} V 仍超过 UMax={self.cfg.umax_v:.4f} V，请降低目标1ω或粗调输出"
            )
        return last_amp

    def run(self) -> dict:
        cfg = self.cfg
        if not self.balance.connected:
            raise RuntimeError("请先连接数控调平电阻")

        coarse = self._clamp_output(cfg.coarse_output_v)
        if cfg.umax_v is not None and cfg.target_v1_heat_v > cfg.umax_v + 1e-9:
            self._status(
                "warn",
                f"目标1ω {cfg.target_v1_heat_v:.4f} V 超过 UMax={cfg.umax_v:.4f} V "
                f"（UMax 限制的是 1ω 信号，不是锁相输出）",
            )

        self.lockin = _connect_lockin(cfg)
        try:
            freq = float(cfg.balance_freq_hz)
            tc = LOCKIN_TC_FACTOR / max(freq, 1e-6)
            self._status("init", f"连接锁相，f={freq:g} Hz，粗调输出 {coarse:.3f} V")
            self._set_diff(True)
            self._set_output(coarse)
            self.lockin.configure_lockin(freq=freq, order_time=4, time_constant=tc)
            time.sleep(0.1)

            self.r_ohm = float(cfg.r_start_ohm)
            max_delta = float(cfg.max_delta_r_ohm)
            best_diff = None

            for outer in range(1, int(cfg.max_outer_loops) + 1):
                self._check_stop()
                self._status("loop", f"==== 外循环 {outer}/{cfg.max_outer_loops} ====")

                # 2) 差分开，PI 调电阻
                self._set_diff(True)
                best_diff = self._minimize_diff_v1(
                    max_delta=max_delta, target=cfg.diff_ok_v
                )
                self._status(
                    "balanced",
                    f"差分调平完成: R={self.r_ohm:.6g} Ω, V1={best_diff * 1e6:.1f} μV",
                )

                # 3) 关差分
                self._set_diff(False)

                # 4) 调输出到目标加热 1ω（受 UMax 限制）
                heat_v1 = self._scale_output_to_target_v1()
                self._status(
                    "heat_done",
                    f"加热定标: 输出={self.output_v:.4f} V, 单端1ω={heat_v1 * 1e3:.3f} mV",
                )

                # 5) 再开差分检查
                self._set_diff(True)
                check_v1 = self._read_v1(settle=True)
                self.final_v1_diff = check_v1
                self.final_r_ohm = self.r_ohm
                self.final_output_v = self.output_v

                if check_v1 <= cfg.diff_ok_v:
                    self._status(
                        "done",
                        f"调平成功: 差分1ω={check_v1 * 1e6:.1f} μV ≤ {cfg.diff_ok_v * 1e6:.0f} μV",
                    )
                    return {
                        "ok": True,
                        "r_ohm": self.r_ohm,
                        "output_v": self.output_v,
                        "v1_diff": check_v1,
                        "v1_heat": heat_v1,
                        "message": "自动调平完成，可开始测量",
                    }

                if check_v1 <= cfg.rebalance_v and outer >= cfg.max_outer_loops:
                    break

                self._status(
                    "rebalance",
                    f"差分1ω={check_v1 * 1e6:.1f} μV 偏大，缩小最大ΔR 后精调…",
                )
                max_delta = max(cfg.min_step_ohm * 5, max_delta * 0.4)

            msg = (
                f"调平结束（未完全达标）: R={self.r_ohm:.6g} Ω, "
                f"输出={self.output_v:.4f} V, 差分1ω={self.final_v1_diff * 1e6:.1f} μV"
            )
            self._status("done", msg)
            return {
                "ok": bool(self.final_v1_diff is not None and self.final_v1_diff <= cfg.rebalance_v),
                "r_ohm": self.r_ohm,
                "output_v": self.output_v,
                "v1_diff": self.final_v1_diff,
                "message": msg,
            }
        finally:
            if self.lockin is not None:
                try:
                    self.lockin.disconnect()
                except Exception:
                    pass
                self.lockin = None


class AutoBalanceThread(QThread):
    """GUI 用自动调平线程。"""

    status_update = pyqtSignal(dict)
    finished_ok = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg: Optional[AutoBalanceConfig] = None
        self.balance: Optional[BalanceResistorController] = None
        self._stop = False

    def setup(self, cfg: AutoBalanceConfig, balance: BalanceResistorController) -> None:
        self.cfg = cfg
        self.balance = balance
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            if self.cfg is None or self.balance is None:
                raise RuntimeError("自动调平未 setup")
            worker = AutoBalanceWorker(
                self.cfg,
                self.balance,
                emit_status=lambda d: self.status_update.emit(d),
                should_stop=lambda: self._stop,
            )
            result = worker.run()
            self.finished_ok.emit(result)
        except InterruptedError as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            self.error_occurred.emit(str(e))
