"""Daily-bar simulation of the first-board and V3 opening selection."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import time
import akshare as ak
import pandas as pd

MAIN = r"(600|601|603|605|000|001|002|003)\d{3}"

def choose(rows):
    order = lambda row: (row["gap"], -row["amount"], row["code"])
    first = sorted((r for r in rows if r["first"]), key=order)
    v3 = sorted((r for r in rows if r["v3"]), key=order)
    picks = first[:1]
    used = {r["code"] for r in picks}
    return picks + [r for r in v3 if r["code"] not in used][:1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buy-date", default=datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat())
    parser.add_argument('--preselect', action='store_true', help='将指定日期作为盘后信号日')
    args = parser.parse_args()
    buy = pd.Timestamp(args.buy_date)
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    calendar = pd.to_datetime(ak.calendar_trade_date_hist_sina()["trade_date"]).sort_values()
    if buy > calendar.iloc[-1]:
        raise RuntimeError("交易日历尚未覆盖目标日期")
    if buy not in set(calendar):
        print("休市")
        return
    if buy > today:
        raise ValueError("目标日期在未来")
    signal = buy if args.preselect else calendar[calendar < buy].iloc[-1]
    previous = calendar[calendar < signal].iloc[-1]
    fmt = lambda day: day.strftime("%Y%m%d")
    pool = ak.stock_zt_pool_em(date=fmt(signal))
    if pool.empty:
        raise RuntimeError("涨停池为空，需核查数据源")
    pool = pool[pool["代码"].astype(str).str.fullmatch(MAIN) & ~pool["名称"].str.contains("ST|退", case=False)]
    start = fmt(signal - pd.Timedelta(days=140))

    def load(row):
        code = str(row["代码"])
        try:
            def history(adjust, end):
                try:
                    frame = ak.stock_zh_a_hist_em(symbol=code, start_date=start, end_date=end, adjust=adjust, timeout=6)
                    if not frame.empty:
                        return frame
                except Exception as error:
                    print(f"{code} 东方财富日线失败，改用新浪: {error}", flush=True)
                symbol = ("sh" if code.startswith("6") else "sz") + code
                for attempt in range(2):
                    try:
                        frame = ak.stock_zh_a_daily_sina(symbol=symbol, start_date=start, end_date=end, adjust=adjust)
                        if frame.empty:
                            raise ValueError("新浪日线为空")
                        return frame.rename(columns={"date": "日期", "open": "开盘", "close": "收盘",
                                                     "high": "最高", "low": "最低", "amount": "成交额"})
                    except Exception:
                        if attempt == 1:
                            raise
                        time.sleep(1)

            raw = history("", fmt(buy))
            adj = history("qfq", fmt(signal))
            raw = raw.set_index(pd.to_datetime(raw["日期"])).sort_index()
            adj = adj.set_index(pd.to_datetime(adj["日期"])).sort_index()
            if signal not in raw.index or signal not in adj.index:
                raise ValueError("缺少信号日日线")
            bars = raw.loc[:signal]
            if len(bars) < 20 or (signal - bars.index[0]).days < 60:
                return None
            if bars["成交额"].tail(20).mean() < 30_000_000:
                return None
            closes = adj.loc[:signal, "收盘"]
            record = {
                "code": code, "name": row["名称"], "amount": float(bars.at[signal, "成交额"]),
                "first": int(row["连板数"]) == 1 and previous in bars.index,
                "v3": len(closes) >= 61 and bars.at[signal, "最高"] != bars.at[signal, "最低"],
                "gap": None, "open": None,
            }
            if not args.preselect and buy in raw.index:
                close = float(bars.at[signal, "收盘"])
                open_price = float(raw.at[buy, "开盘"])
                record.update(gap=open_price / close - 1, open=open_price)
            if record["v3"]:
                for n in (20, 60):
                    record[f"ret{n}"] = closes.iloc[-1] / closes.iloc[-n-1] - 1
            return record
        except Exception as error:
            raise RuntimeError(f"{code} 行情读取失败") from error

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = [r for r in workers.map(load, pool.to_dict("records")) if r is not None]
    v3 = [r for r in rows if r["v3"]]
    if v3:
        momentum = pd.DataFrame(v3)[["ret20", "ret60"]]
        scores = ((momentum - momentum.mean()) / momentum.std(ddof=0).replace(0, float("nan"))).fillna(0).sum(axis=1)
        for row, score in zip(v3, scores):
            row["score"] = score
    if args.preselect:
        next_dates = calendar[calendar > signal]
        if next_dates.empty:
            raise RuntimeError("交易日历缺少下一交易日")
        target = next_dates.iloc[0].date()
        lines = [f"# {signal.date()} 盘后预选", "", f"适用开盘日 {target}。开盘按缺口排序，首板优先，V3 去重补位。", ""]
        for label, key in (("首板", "first"), ("V3", "v3")):
            candidates = sorted((r for r in rows if r[key]), key=lambda r: (-r["amount"], r["code"]))
            lines += [f"## {label}候选（{len(candidates)} 只）", "", "| 代码 | 股票 | 信号日成交额 |", "| --- | --- | ---: |"]
            lines.extend(f"| {r['code']} | {r['name']} | {r['amount']:,.0f} |" for r in candidates)
            lines.append("")
        path = Path("reports/preselect") / f"{signal.date()}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    eligible = [r for r in rows if r["gap"] is not None and -0.099 < r["gap"] < 0 and r["open"] <= 100]
    if rows and all(r["open"] is None for r in rows):
        raise RuntimeError("目标日开盘行情尚未就绪")
    picks = choose(eligible)
    lines = [f"# {buy.date()} 首板 × V3 开盘模拟", "", f"信号日 {signal.date()}。使用日线开盘价和信号日收盘价计算缺口。", "",
             "| 股票 | 路线 | 开盘价 | 缺口 | 模拟股数 | 计划卖出日 |", "| --- | --- | ---: | ---: | ---: | --- |"]
    next_dates = calendar[calendar > buy]
    sell = next_dates.iloc[0].date() if not next_dates.empty else "下一交易日"
    for i, row in enumerate(picks):
        route = "首板" if i == 0 and row["first"] else "V3"
        shares = int(10000 / row["open"] / 100) * 100
        lines.append(f"| {row['code']} {row['name']} | {route} | {row['open']:.2f} | {row['gap']:.2%} | {shares} | {sell} 收盘 |")
    lines += ["", f"有开盘价 {sum(r['open'] is not None for r in rows)}/{len(rows)} 只，缺价票未参与排序。"]
    if not picks:
        lines.append("无合格模拟信号。")
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/{buy.date()}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
