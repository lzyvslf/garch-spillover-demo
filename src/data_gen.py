"""合成多资产收益率数据（GARCH(1,1) 过程 + 跨资产冲击传导）。

设计意图：
- 用 GARCH(1,1) 生成条件方差，保证数据里真的有波动聚集，后面的 GARCH 拟合才有东西可抓；
- 资产之间引入同期冲击传导，使 Diebold-Yilmaz 溢出指数非平凡；
- 固定随机种子 + 固定换行符 + 固定小数位，保证同机同版本逐字节可复现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ASSETS = ["EQ_CN", "EQ_US", "CRUDE", "GOLD"]
N_DAYS = 2000          # 约 8 年交易日
SEED = 20261122        # 固定种子，勿改（改了所有哈希都会变）
SCALE = {"EQ_CN": 0.018, "EQ_US": 0.014, "CRUDE": 0.024, "GOLD": 0.012}
PARAMS = {
    "EQ_CN": (0.02, 0.08, 0.90),
    "EQ_US": (0.01, 0.06, 0.92),
    "CRUDE": (0.03, 0.10, 0.88),
    "GOLD": (0.01, 0.05, 0.93),
}


def _garch11_paths(n: int, omega: float, alpha: float, beta: float, rng) -> np.ndarray:
    """模拟单条 GARCH(1,1) 收益率路径。"""
    h = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    eps = np.zeros(n)
    eps[0] = np.sqrt(h[0]) * rng.standard_normal()
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        eps[t] = np.sqrt(h[t]) * rng.standard_normal()
    return eps


def generate(out_path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    raw = {a: _garch11_paths(N_DAYS, *PARAMS[a], rng) for a in ASSETS}
    df = pd.DataFrame(raw)
    # 跨资产同期冲击：原油对 A 股正向、对黄金负向
    shock = df["CRUDE"].to_numpy()
    df["EQ_CN"] = 0.85 * df["EQ_CN"] + 0.35 * shock
    df["GOLD"] = 0.90 * df["GOLD"] - 0.20 * shock
    for a in ASSETS:
        df[a] = df[a] * SCALE[a] * 100          # 转百分比
    df = df.round(4)
    df.index = pd.bdate_range("2018-01-02", periods=N_DAYS, name="date")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, float_format="%.4f", lineterminator="\n", encoding="utf-8")
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    df = generate(root / "data" / "synthetic_returns.csv")
    print(df.tail())
    print(f"已生成 {len(df)} 行 x {df.shape[1]} 列")
