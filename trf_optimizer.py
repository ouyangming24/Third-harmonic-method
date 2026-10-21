"""信赖域反射优化器（与正演模型、GUI 解耦）。

两个实现共用同一套归一化参数空间：
  * custom — 自写 Coleman–Li（针对 3ω 信号量级改过收敛判据）
  * scipy  — scipy.optimize.least_squares(method='trf')，同样先按初值归一化，
             并关闭绝对 gtol，避免 V3 ~ 1e-6 时在起点误判收敛

本模块只接收残差函数 fun(x_scaled) -> ndarray，不导入 Olson / PyQt。
"""

from __future__ import annotations

import numpy as np

SOLVER_CUSTOM = "custom"
SOLVER_SCIPY = "scipy"
SOLVER_LABELS = {
    SOLVER_CUSTOM: "自写 Coleman–Li",
    SOLVER_SCIPY: "SciPy least_squares",
}

_FD_REL_STEP = 1e-6


class FitCancelled(Exception):
    """优化过程中用户中止。"""


def build_scaling(p0, lo, hi):
    """按初值把物理参数缩放到 O(1)，等价 MATLAB TypicalX。"""
    p0 = np.asarray(p0, dtype=float).copy()
    lo = np.asarray(lo, dtype=float).copy()
    hi = np.asarray(hi, dtype=float).copy()

    for i in range(p0.size):
        if not np.isfinite(lo[i]):
            lo[i] = -np.inf
        if not np.isfinite(hi[i]):
            hi[i] = np.inf
        if lo[i] >= hi[i]:
            mid = p0[i] if np.isfinite(p0[i]) else 0.0
            span = abs(mid) * 0.5 if abs(mid) > 0 else 1.0
            lo[i], hi[i] = mid - span, mid + span
        p0[i] = float(np.clip(p0[i], lo[i], hi[i]))

    scale = np.ones_like(p0)
    for i in range(p0.size):
        if abs(p0[i]) > 0:
            scale[i] = abs(p0[i])
        elif np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i] > lo[i]:
            scale[i] = hi[i] - lo[i]
    return p0, lo, hi, scale


def make_fd_jacobian(fun, lo, hi):
    """归一化空间前向差分雅可比，相对步长 1e-6。"""

    def jac(x, r0):
        J = np.zeros((r0.size, x.size))
        for k in range(x.size):
            h = _FD_REL_STEP * max(abs(x[k]), 1.0)
            if np.isfinite(hi[k]) and x[k] + h > hi[k]:
                h = -h
            x_p = x.copy()
            x_p[k] = np.clip(x[k] + h, lo[k], hi[k])
            actual = x_p[k] - x[k]
            if actual == 0:
                continue
            J[:, k] = (fun(x_p) - r0) / actual
        return J

    return jac


def _solve_trust_region(J, r, delta):
    try:
        U, s, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(J.shape[1])
    if s.size == 0:
        return np.zeros(J.shape[1])

    uf = U.T @ r
    s_cut = s[0] * 1e-12 if s[0] > 0 else 0.0
    inv = np.where(s > s_cut, 1.0 / np.maximum(s, 1e-300), 0.0)
    p_gn = -(Vt.T @ (uf * inv))
    if np.all(np.isfinite(p_gn)) and np.linalg.norm(p_gn) <= delta:
        return p_gn

    def _p_of(lam):
        return -(Vt.T @ (s * uf / (s ** 2 + lam)))

    lam_hi = max(s[0] ** 2, 1.0)
    for _ in range(200):
        if np.linalg.norm(_p_of(lam_hi)) <= delta:
            break
        lam_hi *= 10.0
    lam_lo = 0.0
    for _ in range(80):
        lam_mid = 0.5 * (lam_lo + lam_hi) if lam_lo > 0 else lam_hi * 0.5
        if np.linalg.norm(_p_of(lam_mid)) > delta:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
    p = _p_of(lam_hi)
    return p if np.all(np.isfinite(p)) else np.zeros(J.shape[1])


def _step_to_bound(x, step, lo, hi):
    alpha = np.full(x.size, np.inf)
    for i in range(x.size):
        if step[i] > 0 and np.isfinite(hi[i]):
            alpha[i] = (hi[i] - x[i]) / step[i]
        elif step[i] < 0 and np.isfinite(lo[i]):
            alpha[i] = (lo[i] - x[i]) / step[i]
    alpha = np.maximum(alpha, 0.0)
    a_min = float(np.min(alpha)) if alpha.size else np.inf
    hits = np.isclose(alpha, a_min) & np.isfinite(alpha)
    return a_min, hits


def _model_gain(J, r, step):
    js = J @ step
    return -(js @ (2.0 * r + js))


def _coleman_li_scaling(x, g, lo, hi):
    v = np.ones_like(x)
    dv = np.zeros_like(x)
    for i in range(x.size):
        if g[i] < 0 and np.isfinite(hi[i]):
            v[i] = hi[i] - x[i]
            dv[i] = -1.0
        elif g[i] > 0 and np.isfinite(lo[i]):
            v[i] = x[i] - lo[i]
            dv[i] = 1.0
    return np.maximum(np.abs(v), 1e-300), dv


def _raise_if_cancelled(should_cancel):
    if should_cancel is not None and should_cancel():
        print("拟合已由用户中止。")
        raise FitCancelled()


def custom_trf(fun, jac, x0, lo, hi, *, max_iter=30, tol=1e-6, delta0=0.5,
               should_cancel=None):
    """自写 Coleman–Li 信赖域反射。x 已在归一化空间。"""
    x = np.asarray(x0, dtype=float).copy()
    r = fun(x)
    if not np.all(np.isfinite(r)):
        raise RuntimeError("初值处正演结果非有限值，请检查初值与边界设置。")
    cost = float(r @ r)
    delta = float(max(delta0, 1e-3))
    theta = 0.995
    status = "已达到最大迭代次数"
    n_fev = 1
    terminate = False

    for it in range(int(max_iter)):
        _raise_if_cancelled(should_cancel)

        J = jac(x, r)
        n_fev += x.size
        if not np.all(np.isfinite(J)):
            status = "雅可比出现非有限值，提前停止"
            break

        g = J.T @ r
        v, dv = _coleman_li_scaling(x, g, lo, hi)
        d = np.sqrt(v)

        opt = float(np.max(np.abs(g * v))) / max(cost, 1e-300) if g.size else 0.0
        if opt < tol:
            status = "一阶最优性满足"
            break

        J_h = J * d
        diag_h = g * dv
        pos = diag_h > 0
        if np.any(pos):
            aug = np.zeros((int(np.sum(pos)), x.size))
            aug[np.arange(int(np.sum(pos))), np.where(pos)[0]] = np.sqrt(diag_h[pos])
            J_sub = np.vstack([J_h, aug])
            r_sub = np.concatenate([r, np.zeros(aug.shape[0])])
        else:
            J_sub, r_sub = J_h, r

        accepted = False
        for _ in range(12):
            _raise_if_cancelled(should_cancel)

            p_h = _solve_trust_region(J_sub, r_sub, delta)
            step = d * p_h

            candidates = []
            a_bound, hits = _step_to_bound(x, step, lo, hi)
            if a_bound >= 1.0:
                candidates.append(step)
            else:
                clipped = step * (theta * a_bound)
                candidates.append(clipped)

                x_edge = x + clipped
                refl = step.copy()
                refl[hits] *= -1.0
                a_refl, _ = _step_to_bound(x_edge, refl, lo, hi)
                j_refl = J @ refl
                denom = float(j_refl @ j_refl)
                if denom > 0:
                    base = r + J @ clipped
                    t_opt = -float(j_refl @ base) / denom
                    p_norm = max(float(np.linalg.norm(p_h)), 1e-300)
                    t_cap = min(theta * a_refl, delta / p_norm - theta * a_bound)
                    t = float(np.clip(t_opt, 0.0, max(t_cap, 0.0)))
                    if t > 0:
                        candidates.append(clipped + t * refl)

                grad_dir = -(v * g)
                a_grad, _ = _step_to_bound(x, grad_dir, lo, hi)
                j_grad = J @ grad_dir
                dg = float(j_grad @ j_grad)
                if dg > 0:
                    t_opt = -float(j_grad @ r) / dg
                    t = float(np.clip(t_opt, 0.0, theta * a_grad))
                    if t > 0:
                        candidates.append(t * grad_dir)

            best_step, best_gain = None, 0.0
            for cand in candidates:
                gain = _model_gain(J, r, cand)
                if gain > best_gain:
                    best_step, best_gain = cand, gain

            if best_step is None or best_gain <= 0:
                status = "二次模型无下降方向"
                terminate = True
                break

            x_new = np.clip(x + best_step, lo, hi)
            r_new = fun(x_new)
            n_fev += 1
            if not np.all(np.isfinite(r_new)):
                delta *= 0.25
                continue

            cost_new = float(r_new @ r_new)
            actual = cost - cost_new
            ratio = actual / best_gain if best_gain > 0 else -1.0

            step_norm = float(np.linalg.norm(best_step / np.maximum(d, 1e-300)))
            if ratio < 0.25:
                delta = max(0.25 * step_norm, 1e-12)
            elif ratio > 0.75 and step_norm >= 0.99 * delta:
                delta = min(2.0 * delta, 1e3)

            if ratio > 0:
                rel = actual / max(cost, 1e-300)
                print(f"迭代 {it + 1}/{max_iter}: 残差平方和 = {cost_new:.6e}, "
                      f"ρ = {ratio:.3f}, Δ = {delta:.3e}")
                x, r, cost = x_new, r_new, cost_new
                accepted = True
                if rel < tol:
                    status = "残差下降小于容差，已收敛"
                    terminate = True
                elif float(np.linalg.norm(best_step)) < tol * (tol + float(np.linalg.norm(x))):
                    status = "参数步长小于容差，已收敛"
                    terminate = True
                break

            if delta <= 1e-12:
                status = "信赖域半径过小，无法继续下降"
                terminate = True
                break

        if terminate:
            break
        if not accepted:
            status = "信赖域内找不到可接受的步长"
            break

    return {
        "x": x, "cost": cost, "status": status, "nfev": n_fev,
        "solver": SOLVER_CUSTOM,
    }


def scipy_trf(fun, x0, lo, hi, *, max_iter=30, tol=1e-6, should_cancel=None):
    """SciPy TRF。x 已在归一化空间；gtol=0 避免绝对梯度误停。"""
    import warnings
    from scipy.optimize import least_squares

    x0 = np.asarray(x0, dtype=float)
    n_fev_box = {"n": 0}

    def wrapped(x):
        _raise_if_cancelled(should_cancel)
        n_fev_box["n"] += 1
        r = np.asarray(fun(x), dtype=float).ravel()
        if not np.all(np.isfinite(r)):
            return np.full_like(r, 1e30)
        return r

    def callback(_intermediate):
        if should_cancel is not None and should_cancel():
            print("拟合已由用户中止。")
            raise StopIteration

    n = max(int(x0.size), 1)
    max_nfev = max(int(max_iter) * (n + 1) * 3, n * 20)

    kwargs = dict(
        jac="2-point",
        bounds=(lo, hi),
        method="trf",
        ftol=float(tol),
        xtol=float(tol),
        gtol=0.0,
        x_scale=1.0,
        diff_step=_FD_REL_STEP,
        max_nfev=max_nfev,
        verbose=2,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Setting `gtol` below the machine epsilon")
            res = least_squares(wrapped, x0, callback=callback, **kwargs)
    except TypeError:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Setting `gtol` below the machine epsilon")
            res = least_squares(wrapped, x0, **kwargs)

    if int(getattr(res, "status", 0)) == -2:
        raise FitCancelled()
    rss = 2.0 * float(res.cost)
    return {
        "x": np.asarray(res.x, dtype=float),
        "cost": rss,
        "status": str(res.message).strip(),
        "nfev": int(getattr(res, "nfev", n_fev_box["n"])),
        "solver": SOLVER_SCIPY,
    }


def minimize_trf(fun, x0, lo, hi, *, solver=SOLVER_CUSTOM, max_iter=30,
                 tol=1e-6, delta0=0.5, should_cancel=None):
    """在归一化空间求解。返回 dict: x, cost, status, nfev, solver。"""
    solver = (solver or SOLVER_CUSTOM).lower()
    if solver == SOLVER_SCIPY:
        return scipy_trf(
            fun, x0, lo, hi,
            max_iter=max_iter, tol=tol, should_cancel=should_cancel,
        )
    if solver != SOLVER_CUSTOM:
        raise ValueError(f"未知求解器: {solver!r}，可选 {SOLVER_CUSTOM!r} / {SOLVER_SCIPY!r}")
    jac = make_fd_jacobian(fun, lo, hi)
    return custom_trf(
        fun, jac, x0, lo, hi,
        max_iter=max_iter, tol=tol, delta0=delta0, should_cancel=should_cancel,
    )


def report_result(result, p_fit, lo, hi, labels, delta0):
    name = SOLVER_LABELS.get(result.get("solver"), result.get("solver", "?"))
    print(f"优化器: {name}")
    if result.get("solver") == SOLVER_CUSTOM:
        print(f"  参数按初值归一化；初始信赖域半径 Δ0 = {delta0:g}（取自界面 ζ）")
    else:
        print("  参数按初值归一化后交给 SciPy TRF；已关闭绝对 gtol，ζ 不参与 SciPy 半径")
    print(f"  结束原因: {result['status']}；正演调用 {result['nfev']} 次，"
          f"残差平方和 = {result['cost']:.6e}")
    at_lo = np.isclose(p_fit, lo, rtol=1e-6, atol=0.0) & np.isfinite(lo)
    at_hi = np.isclose(p_fit, hi, rtol=1e-6, atol=0.0) & np.isfinite(hi)
    hit = [labels[i] for i in range(len(p_fit)) if at_lo[i] or at_hi[i]]
    if hit:
        print("  警告: 以下参数触及边界: " + ", ".join(hit))
