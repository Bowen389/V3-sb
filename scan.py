"""Daily-bar simulation of the first-board and V3 opening selection."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo
import argparse
import json
import time
import akshare as ak
import pandas as pd

MAIN = r"(600|601|603|605|000|001|002|003)\d{3}"
SINA_LOCK = Lock()

def sina_history(**kwargs):
    # Sina's decoder embeds V8; concurrent first initialization can abort Python.
    with SINA_LOCK:
        return ak.stock_zh_a_hist_sina(**kwargs)

def history(code, adjust, start, end, required):
    """Only accept a source/cache after validating the requested session."""
    cache = Path('data/prices') / f'{code}-{adjust or "raw"}.csv'
    def validate(frame):
        frame = frame.rename(columns={'date': '日期', 'open': '开盘', 'close': '收盘',
                                      'high': '最高', 'low': '最低', 'amount': '成交额'})
        needed = ['日期', '收盘'] if adjust else ['日期', '开盘', '收盘', '最高', '最低', '成交额']
        if not set(needed).issubset(frame.columns):
            raise ValueError('行情字段不完整')
        frame = frame.copy()
        frame['日期'] = pd.to_datetime(frame['日期']).dt.normalize()
        frame = frame.drop_duplicates('日期').sort_values('日期')
        frame = frame[(frame['日期'] >= pd.Timestamp(start)) & (frame['日期'] <= pd.Timestamp(end))]
        for col in needed[1:]:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
        if pd.Timestamp(required) not in set(frame['日期']):
            raise ValueError(f'缺少目标日日线 {required}')
        if frame[needed[1:]].isna().any().any() or (frame['收盘'] <= 0).any():
            raise ValueError('行情含无效数值')
        if len(frame) < 20 or (pd.Timestamp(required) - frame['日期'].min()).days < 60:
            raise ValueError('历史覆盖不足，不能判定资格')
        return frame
    if cache.exists():
        try:
            return validate(pd.read_csv(cache))
        except Exception:
            pass
    symbol = ('sh' if code.startswith('6') else 'sz') + code
    sources = [lambda: ak.stock_zh_a_hist_em(symbol=code, start_date=start, end_date=end, adjust=adjust),
               lambda: sina_history(symbol=symbol, start_date=start, end_date=end, adjust=adjust)]
    if adjust:
        sources.append(lambda: ak.stock_zh_a_hist_tx(symbol=symbol, start_date=start, end_date=end, adjust=adjust))
    errors = []
    for source in sources:
        try:
            frame = validate(source())
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            return frame
        except Exception as error:
            errors.append(str(error))
    raise RuntimeError('；'.join(errors))

def choose(rows):
    order = lambda row: (row["gap"], -row["amount"], row["code"])
    first = sorted((r for r in rows if r["first"]), key=order)
    v3 = sorted((r for r in rows if r["v3"]), key=order)
    picks = first[:1]
    used = {r["code"] for r in picks}
    return picks + [r for r in v3 if r["code"] not in used][:1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buy-date")
    parser.add_argument('--preselect', action='store_true', help='将指定日期作为盘后信号日')
    args = parser.parse_args()
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    from exchange_calendars import get_calendar
    sessions = get_calendar('XSHG').sessions
    calendar = pd.Series(pd.to_datetime(sessions).tz_localize(None))
    cutoff = pd.Timestamp(now.date()) - pd.Timedelta(days=int(now.hour < 15))
    buy = pd.Timestamp(args.buy_date) if args.buy_date else calendar[calendar <= cutoff].iloc[-1]
    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
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
    pool_cache = Path('data/pools') / f'{signal.date()}.csv'
    for attempt in range(3):
        try:
            pool = ak.stock_zt_pool_em(date=fmt(signal))
            if pool.empty or not {'代码', '名称', '连板数'}.issubset(pool.columns):
                raise ValueError('涨停池为空或字段不完整')
            pool_cache.parent.mkdir(parents=True, exist_ok=True)
            pool.to_csv(pool_cache, index=False)
            break
        except Exception:
            if pool_cache.exists():
                pool = pd.read_csv(pool_cache, dtype={'代码': str})
                break
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    if pool.empty:
        raise RuntimeError("涨停池为空，需核查数据源")
    pool['代码'] = pool['代码'].astype(str).str.zfill(6)
    pool = pool[pool["代码"].str.fullmatch(MAIN) & ~pool["名称"].str.contains("ST|退", case=False)]
    start = fmt(signal - pd.Timedelta(days=140))

    failures = {}
    def load(row):
        code = str(row["代码"])
        try:
            raw = history(code, '', start, fmt(buy), signal if args.preselect else buy)
            adj = history(code, 'qfq', start, fmt(signal), signal)
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
            failures[code] = str(error)
            print(f'{code} 待补行情: {error}', flush=True)
            return None

    with ThreadPoolExecutor(max_workers=4) as workers:
        rows = [r for r in workers.map(load, pool.to_dict("records")) if r is not None]
    if failures:
        time.sleep(3)
        retry = pool[pool['代码'].astype(str).isin(failures)].to_dict('records')
        failures.clear()
        with ThreadPoolExecutor(max_workers=2) as workers:
            rows.extend(r for r in workers.map(load, retry) if r is not None)
    quality = {'date': str(buy.date()), 'stage': 'preselect' if args.preselect else 'simulation',
               'status': 'partial' if failures else 'complete', 'pool_count': len(pool),
               'resolved_count': len(pool) - len(failures), 'errors': failures,
               'updated_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()}
    quality_path = Path('reports/quality') / f'{buy.date()}-{quality["stage"]}.json'
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding='utf-8')
    notice = f'数据状态：{quality["status"]}；已核验 {quality["resolved_count"]}/{len(pool)}；缺失：{", ".join(failures) or "无"}。'
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
        lines += [notice, '缺失未补齐时，本报告仅为部分候选，不能锁定最终排名。', ''] if failures else [notice, '']
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
    if failures:
        Path(f'reports/{buy.date()}.md').write_text(f'# {buy.date()} 开盘模拟待补齐\n\n{notice}\n\n候选池不完整，暂不生成最终买入名单。\n', encoding='utf-8')
        return
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
