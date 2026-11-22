"""verify.py — 自查脚本：验文件哈希，再按容差比对关键数字。

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
