# GARCH + Diebold-Yilmaz 溢出分析 · 可复现交付包（免费最小版）

样本 2000 个交易日 ｜ VAR 滞后阶 1 ｜ 总溢出指数 TOI 15.37% ｜ 生成 2026-11-22 18:44:24
随机种子 20261122（固定，勿改）｜ 环境指纹见 manifest.json 的 env 字段

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
| `data/synthetic_returns.csv` | 4 资产 × 2000 天，单位：百分比 |
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
