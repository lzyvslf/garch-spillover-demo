"""一键运行：合成数据 -> GARCH(1,1) -> Diebold-Yilmaz 溢出 -> 出图 -> 写清单。

用法：
    python run_all.py

产出：
    data/synthetic_returns.csv
    output/garch_table.csv, cond_vol_annualized.csv
    output/spillover_table.csv, spillover_meta.csv, toi_timeseries.csv
    output/spillover_heatmap.png, vol_timeseries.png, toi_timeseries.png
    RUN.md, manifest.json, checksums.txt, verify.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data_gen import SEED, N_DAYS, generate           # noqa: E402
from src.garch_model import fit_all                        # noqa: E402
from src.spillover import compute_spillover, rolling_toi   # noqa: E402
from src.plotting import heatmap, toi_line, vol_lines      # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "output"

VERIFY_PY = '''"""verify.py — 自查脚本：验文件哈希，再按容差比对关键数字。

用法：python verify.py    最后一行是 VERDICT=PASS / VERDICT=FAIL
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOL = 1e-6
fails = 0


def say(ok, what, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + what + ((" " + extra) if extra else ""))
    if not ok:
        fails += 1


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def close(a, b):
    return abs(float(a) - float(b)) <= TOL


def main():
    mf = ROOT / "manifest.json"
    if not mf.exists():
        print("FAIL 缺少 manifest.json")
        print("VERDICT=FAIL")
        return 1
    m = json.loads(mf.read_text(encoding="utf-8"))

    bad = [f["name"] for f in m["files"]
           if not (ROOT / f["name"]).exists() or sha(ROOT / f["name"]) != f["sha256"]]
    say(not bad, "文件哈希 %d 个" % len(m["files"]), ("不符: " + ", ".join(bad)) if bad else "全部一致")

    import csv
    def read_csv(rel):
        p = ROOT / rel
        if not p.exists():
            return None
        with p.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    data = read_csv("data/synthetic_returns.csv")
    say(data is not None and len(data) == m["n_obs"], "样本量", "%s 行" % (len(data) if data else "缺文件"))

    meta = read_csv("output/spillover_meta.csv")
    if meta:
        row = meta[0]
        say(int(row["var_lag"]) == m["var_lag"], "VAR 滞后阶", "%s / 记录 %s" % (row["var_lag"], m["var_lag"]))
        say(close(row["toi_pct"], m["toi_pct"]), "总溢出指数 TOI", "%s%% / 记录 %s%%" % (row["toi_pct"], m["toi_pct"]))
    else:
        say(False, "spillover_meta.csv", "缺文件")

    g = read_csv("output/garch_table.csv")
    if g:
        worst = 0.0
        for row in g:
            a = row["asset"]
            if a in m["persistence"]:
                d = abs(float(row["persistence"]) - float(m["persistence"][a]))
                worst = max(worst, d)
                say(d <= TOL, "持续性 %s" % a, "实际 %s / 记录 %s" % (row["persistence"], m["persistence"][a]))
        say(worst <= TOL, "最大偏差", "%.2e（容差 %.0e）" % (worst, TOL))
    else:
        say(False, "garch_table.csv", "缺文件")

    print("VERDICT=" + ("PASS" if fails == 0 else "FAIL"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
'''


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def env_fingerprint() -> dict:
    fp = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("numpy", "pandas", "scipy", "arch", "statsmodels", "matplotlib"):
        try:
            fp[m] = __import__(m).__version__
        except Exception:
            fp[m] = None
    return fp


RUN_TEMPLATE = """# GARCH + Diebold-Yilmaz 溢出分析 · 可复现交付包（免费最小版）

样本 __N_OBS__ 个交易日 ｜ VAR 滞后阶 __LAG__ ｜ 总溢出指数 TOI __TOI__% ｜ 生成 __AT__
随机种子 __SEED__（固定，勿改）｜ 环境指纹见 manifest.json 的 env 字段

## 一、一分钟跑通

```
pip install -r requirements.txt
python run_all.py
python verify.py
```

最后一行应为 `VERDICT=PASS`。从合成数据一路跑到图，不需要数据账号、不联网。

## 二、这个包在证明什么

1. 数据是合成但结构是真的：用 GARCH(1,1) 生成条件方差，序列里确实有波动聚集，后面的 GARCH 拟合不是对着白噪声做无用功；资产间引入同期冲击传导，溢出指数才非平凡。
2. 每一步都可核查：参数表、分解矩阵、滚动序列全部落成 CSV，不只给你一张图。
3. 校验与复现分开说清：`checksums.txt` 验的是"收到的包没被改过"（`sha256sum -c` 一条命令，不用 Python）；`verify.py` 验的是"数字对得上"，按 1e-6 容差比对 manifest 里记录的 TOI、滞后阶、各资产持续性。

## 三、包里有什么

| 文件 | 是什么 |
|---|---|
| `RUN.md` | 本说明 |
| `requirements.txt` | 依赖与锁定版本（换版本前先看第四节） |
| `run_all.py` | 一键入口 |
| `verify.py` | 自查：验哈希 + 比数字 + 报最大偏差 |
| `src/data_gen.py` | 合成数据（GARCH 过程 + 冲击传导） |
| `src/garch_model.py` | 每资产拟合 GARCH(1,1)，AR(1) 均值 + t 分布 |
| `src/spillover.py` | VAR + 广义方差分解（Diebold-Yilmaz 2012） |
| `src/plotting.py` | 热力图、条件波动率、滚动 TOI 三张图 |
| `data/synthetic_returns.csv` | 4 资产 × __N_OBS__ 天，单位：百分比 |
| `output/garch_table.csv` | omega / alpha / beta / 持续性 / loglik / AIC / BIC |
| `output/cond_vol_annualized.csv` | 年化条件波动率序列 |
| `output/spillover_table.csv` | 4×4 分解矩阵 + FROM / TO / NET |
| `output/spillover_meta.csv` | 滞后阶、预测步长、TOI、样本量 |
| `output/toi_timeseries.csv` | 滚动 TOI（窗口 200，每 10 天推进） |
| `output/*.png` | 三张图 |
| `manifest.json` | 文件哈希 + 关键数字 + 环境指纹 |
| `checksums.txt` | sha256sum 格式清单 |

## 四、怎么读结果

- 持续性 alpha+beta 越接近 1，波动冲击衰减越慢、风险记忆越长；beta>0.9 是金融资产的经验常态。
- TOI 越高，四个资产的波动互相解释得越多、系统性联动越强。
- NET 为正表示净溢出方（传出去的多），为负表示净接收方。
- 热力图行是受影响方、列是冲击来源方，行内相加为 100%。

## 五、复现的边界（必须说清）

- 同机同版本重跑：逐字节一致，`checksums.txt` 全 OK。
- 换机器 / 换 BLAS / 换包版本：数值优化可能在末位有极小差异，`verify.py` 用 1e-6 容差比对并打印最大偏差；偏差超标先比对 manifest 的 `env` 字段，多半是版本不一致，而不是数据错。
- 想完全对齐，就按 `requirements.txt` 装同一套版本。

## 六、边界与红线

- 全部为合成数据，不含真实行情或客户数据；用于方法与流程演示，不构成任何投资建议。
- 这是"能跑通、能自己复现、能对着学"的分析包，不是替你写论文或替你出结论的代写服务。
"""


def write_run_md(n_obs, lag, toi, seed, at):
    txt = RUN_TEMPLATE
    for k, v in (("__N_OBS__", str(n_obs)), ("__LAG__", str(lag)), ("__TOI__", f"{toi:.2f}"),
                 ("__SEED__", str(seed)), ("__AT__", at)):
        txt = txt.replace(k, v)
    p = ROOT / "RUN.md"
    p.write_text(txt, encoding="utf-8", newline="\n")
    return p


def main() -> int:
    OUT.mkdir(exist_ok=True)
    print("[1/6] 生成合成数据 ...")
    returns = generate(DATA / "synthetic_returns.csv")

    print("[2/6] 拟合 GARCH(1,1) ...")
    garch_tbl, vol = fit_all(returns, OUT)
    print(garch_tbl[["alpha_1", "beta_1", "persistence", "flag"]].to_string())

    print("[3/6] Diebold-Yilmaz 溢出指数 ...")
    spill, toi, lag = compute_spillover(returns, OUT)
    print(spill[["FROM其他(%)", "TO其他(%)", "NET(%)"]].to_string())

    print("[4/6] 滚动总溢出指数（耗时较长）...")
    toi_series = rolling_toi(returns)
    toi_series.to_csv(OUT / "toi_timeseries.csv", float_format="%.4f", lineterminator="\n")

    print("[5/6] 出图 ...")
    heatmap(spill, OUT / "spillover_heatmap.png")
    vol_lines(vol, OUT / "vol_timeseries.png")
    toi_line(toi_series, OUT / "toi_timeseries.png")

    print("[6/6] 写清单与说明 ...")
    at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    (ROOT / "verify.py").write_text(VERIFY_PY, encoding="utf-8", newline="\n")
    write_run_md(len(returns), lag, toi, SEED, at)

    # 清单只覆盖"数据与结果"与稳定的源码文件；RUN.md / manifest.json 含生成时间，逐次会变，故不入清单。
    tracked = []
    for rel in ("data/synthetic_returns.csv",
                "output/garch_table.csv", "output/cond_vol_annualized.csv",
                "output/spillover_table.csv", "output/spillover_meta.csv",
                "output/toi_timeseries.csv",
                "output/spillover_heatmap.png", "output/vol_timeseries.png",
                "output/toi_timeseries.png", "verify.py", "run_all.py",
                "requirements.txt", "src/data_gen.py", "src/garch_model.py",
                "src/spillover.py", "src/plotting.py"):
        p = ROOT / rel
        if p.exists():
            tracked.append({"name": rel, "bytes": p.stat().st_size, "sha256": sha256_of(p)})

    manifest = {
        "pack": "garch-spillover-demo",
        "seed": SEED,
        "n_days_plan": N_DAYS,
        "n_obs": len(returns),
        "var_lag": lag,
        "h_step": 10,
        "toi_pct": round(toi, 2),
        "persistence": {a: float(garch_tbl.loc[a, "persistence"]) for a in garch_tbl.index},
        "env": env_fingerprint(),
        "generated_at": at,
        "files": tracked,
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    lines = [f"{f['sha256']}  {f['name']}" for f in sorted(tracked, key=lambda x: x["name"])]
    (ROOT / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    print(f"\n完成。样本 {len(returns)} 天，VAR 滞后阶 {lag}，TOI {toi:.2f}%，"
          f"滚动点 {len(toi_series)} 个，文件 {len(tracked)} 个。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
