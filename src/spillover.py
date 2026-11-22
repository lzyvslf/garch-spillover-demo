"""Diebold-Yilmaz (2012) 波动率溢出指数（广义方差分解，不依赖变量排序）。

方法：
1. 对收益率序列拟合 VAR(p)，p 由 AIC 在 1..max_lag 里选；
2. 用广义脉冲响应（Pesaran & Shin, 1998）做方差分解；
3. 输出总溢出指数 TOI、净溢出矩阵、滚动窗口 TOI。

参考：Diebold & Yilmaz (2012), Journal of International Economics 83(3), 582-596.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from pathlib import Path

H_STEP = 10        # 预测步长，DY 原文取 10
ROLL_WIN = 200     # 滚动窗口长度
ROLL_STEP = 10     # 每 10 个交易日推进一次（原版逐日，这里为控制运行时间）


def aic_order_select(data: pd.DataFrame, max_lag: int = 8) -> int:
    best_aic, best_lag = np.inf, 1
    for lag in range(1, max_lag + 1):
        try:
            fitted = VAR(data).fit(maxlags=lag, ic=None)
            if fitted.aic < best_aic:
                best_aic, best_lag = fitted.aic, lag
        except Exception:
            continue
    return best_lag


def generalized_decomposition(A: np.ndarray, omega: np.ndarray, h: int) -> np.ndarray:
    """广义方差分解比例矩阵（Pesaran & Shin 1998）：theta_ij = (sum_m (A_m Σ)_ij^2) / σ_jj。

    行=受影响方，列=冲击来源方；每行归一到 1。
    """
    k = omega.shape[0]
    sigma_diag = np.diag(omega).reshape(1, k)        # σ_jj，按列广播
    theta = np.zeros((h + 1, k, k))
    for m in range(h):
        psi_m = A[m] @ omega                          # (k, k)
        theta[m] = (psi_m ** 2) / sigma_diag
    cum = theta.sum(axis=0)
    return cum / cum.sum(axis=1, keepdims=True)


def total_index(decomp: np.ndarray) -> float:
    return float((decomp.sum() - np.trace(decomp)) / decomp.sum() * 100)


def impulse_matrices(fitted, h: int) -> np.ndarray:
    """取正交化前的脉冲响应系数矩阵，兼容 statsmodels 0.14 / 0.15 两套 API。"""
    if hasattr(fitted, "irf"):                      # statsmodels >= 0.15
        return np.asarray(fitted.irf(h).irfs, dtype=float)
    return fitted.impulse_responses(steps=h, orthogonalized=False).values   # 0.14


def compute_spillover(returns: pd.DataFrame, out_dir: Path):
    data = returns.dropna()
    lag = aic_order_select(data)
    fitted = VAR(data).fit(maxlags=lag, ic=None)
    A = impulse_matrices(fitted, H_STEP)
    decomp = generalized_decomposition(A, fitted.sigma_u, H_STEP)

    tbl = pd.DataFrame(decomp * 100, index=data.columns, columns=data.columns).round(2)
    from_others = ((decomp.sum(axis=1) - np.diag(decomp)) * 100).round(2)
    to_others = ((decomp.sum(axis=0) - np.diag(decomp)) * 100).round(2)
    tbl["FROM其他(%)"] = from_others
    tbl["TO其他(%)"] = to_others
    tbl["NET(%)"] = (to_others - from_others).round(2)
    tbl.to_csv(out_dir / "spillover_table.csv", float_format="%.2f", lineterminator="\n")

    toi = total_index(decomp)
    meta = pd.DataFrame({
        "var_lag": [lag],
        "h_step": [H_STEP],
        "toi_pct": [round(toi, 2)],
        "n_obs": [len(data)],
        "roll_win": [ROLL_WIN],
        "roll_step": [ROLL_STEP],
    })
    meta.to_csv(out_dir / "spillover_meta.csv", index=False, lineterminator="\n")
    return tbl, toi, lag


def rolling_toi(returns: pd.DataFrame) -> pd.Series:
    data = returns.dropna()
    idx, vals = [], []
    for end in range(ROLL_WIN, len(data) + 1, ROLL_STEP):
        win = data.iloc[end - ROLL_WIN:end]
        try:
            lag = aic_order_select(win, max_lag=4)
            fitted = VAR(win).fit(maxlags=lag, ic=None)
            A = impulse_matrices(fitted, H_STEP)
            vals.append(round(total_index(generalized_decomposition(A, fitted.sigma_u, H_STEP)), 4))
            idx.append(win.index[-1])
        except Exception:
            continue
    s = pd.Series(vals, index=pd.DatetimeIndex(idx, name="date"), name="toi_pct")
    return s
