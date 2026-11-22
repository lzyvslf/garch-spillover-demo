"""GARCH(1,1) 建模：每个资产单独拟合，输出参数表与条件波动率序列。

判读要点（RUN.md 里会再写一遍）：
- alpha + beta 越接近 1，波动冲击衰减越慢、风险记忆越长；
- beta > 0.9 属于典型的高持续性，符合金融资产经验规律；
- 超过 0.995 提示接近 IGARCH，需人工复核。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path

PERSISTENCE_WARN = 0.995


def fit_all(returns: pd.DataFrame, out_dir: Path):
    rows = []
    vol = {}
    for asset in returns.columns:
        r = returns[asset].dropna()
        try:
            am = arch_model(r, mean="AR(1)", lags=1, vol="GARCH", p=1, q=1, dist="t")
            res = am.fit(disp="off", show_warning=False)
        except Exception:
            am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            res = am.fit(disp="off", show_warning=False)
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        persist = alpha + beta
        mu_key = next((k for k in ("mu", "Const") if k in res.params.index), None)
        rows.append({
            "asset": asset,
            "omega": round(omega, 6),
            "alpha_1": round(alpha, 4),
            "beta_1": round(beta, 4),
            "persistence": round(persist, 4),
            "mu": round(float(res.params[mu_key]), 4) if mu_key else None,
            "loglik": round(float(res.loglikelihood), 2),
            "aic": round(float(res.aic), 2),
            "bic": round(float(res.bic), 2),
            "flag": "接近IGARCH" if persist > PERSISTENCE_WARN else "OK",
        })
        vol[asset] = pd.Series(res.conditional_volatility, index=r.index)
    table = pd.DataFrame(rows).set_index("asset")
    table.to_csv(out_dir / "garch_table.csv", float_format="%.4f", lineterminator="\n")
    vol_df = pd.DataFrame(vol)
    vol_df = (vol_df / 100 * np.sqrt(252) * 100).round(4)   # 年化百分比
    vol_df.to_csv(out_dir / "cond_vol_annualized.csv", float_format="%.4f", lineterminator="\n")
    return table, vol_df
