"""
backtest.py -- does the Overheat Score actually warn before losses?

Put this file next to stock_bubble_detector.py and train_bubble_model.py, then:

    python backtest.py
    python backtest.py --extra-test-tickers      # also test on stocks that later fell hard

Design (what makes this a real test, not a demo):
  * TIME SPLIT. The model is trained ONLY on snapshots dated on or before --cutoff
    (default 2022-12-31) and is then scored on dates AFTER the cutoff. No test date
    is inside the training window, so there is no look-ahead.
  * OUTCOME. For every test date we measure what happened next: the worst loss from
    that day's close over the next 126 (about 6 months) and 252 trading days, and
    whether it fell 20% or more.
  * BASELINES. The score is compared to simple rules a person could use anyway:
    RSI, price/200-day average, plain volatility, and the ungated Isolation Forest (to show whether
    the directional gate helps). Rankings are compared at the SAME flag rate.
  * UNCERTAINTY. A bootstrap over calendar months gives a 95% interval on the AUC
    difference, because tickers move together and windows overlap.

Outputs go to backtest_out/: summary.md, summary.csv, observations.csv,
band_losses.png, roc.png. Needs internet (Yahoo Finance), about 10 minutes.
Educational use; not financial advice.
"""
import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stock_bubble_detector as sbd
import train_bubble_model as tbm

logging.getLogger("stock_bubble_detector").setLevel(logging.WARNING)
logging.getLogger("train_bubble_model").setLevel(logging.WARNING)

# Stocks that later fell hard. TEST ONLY: never used for training. Still-listed
# survivors, so this only partly addresses survivorship bias (see summary.md).
EXTRA_TEST_TICKERS = ['ZM', 'PTON', 'COIN', 'RIVN', 'GME', 'AMC', 'CVNA', 'DOCU', 'MRNA',
                      'SNAP', 'LCID', 'HOOD', 'RBLX', 'AFRM', 'UPST', 'BYND', 'TDOC', 'ROKU',
                      'FSLY', 'SPCE']
BANDS = [(0, 'Healthy'), (10, 'Low'), (30, 'Moderate'), (50, 'High'), (70, 'Critical')]
BAND_COLORS = ['#2E7D32', '#9CCB3B', '#F2C230', '#EE7B1F', '#B3261E']


# ------------------------------------------------------------------ data
def get_prices(ticker, start, end, cache_dir):
    path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    df = sbd.load_stock_data(ticker, start, end)
    if df is None or df.empty:
        return None
    df = df[['Close']].dropna().copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_pickle(path)
    return df


# ----------------------------------------------------------------- model
def build_train_rows(prices, tickers, cutoff, step):
    rows = []
    for t in tickers:
        if t not in prices:
            continue
        d = prices[t][prices[t].index <= cutoff]
        if len(d) < tbm.MIN_HISTORY_DAYS:
            continue
        for i in range(tbm.MIN_HISTORY_DAYS, len(d), step):
            f = sbd.compute_feature_vector(d.iloc[:i + 1])
            if f is not None:
                rows.append(f)
    return rows


def fit_bundle(rows, contamination):
    x = np.array([[r[n] for n in sbd.FEATURE_NAMES] for r in rows])
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=42)
    model.fit(xs)
    lo, hi = np.percentile(model.score_samples(xs), [1, 99])
    return {'model': model, 'scaler': scaler, 'score_low': float(lo), 'score_high': float(hi)}


def score_features(b, f):
    """Same math as run_bubble_analysis(). Returns (ungated, gated) scores."""
    x = b['scaler'].transform(np.array([[f[n] for n in sbd.FEATURE_NAMES]]))
    raw = float(b['model'].score_samples(x)[0])
    s = float(np.clip(100 * (b['score_high'] - raw) / (b['score_high'] - b['score_low']), 0, 100))
    z = {n: (f[n] - b['scaler'].mean_[i]) / b['scaler'].scale_[i]
         for i, n in enumerate(sbd.FEATURE_NAMES)}
    gated = min(s, 15.0) if z['rsi'] + z['ma_ratio'] + z['accel'] <= 0 else s
    return s, gated


def self_check(bundle, prices, tickers, start_idx_fn):
    """Confirm this script's scoring matches the project's run_bubble_analysis()."""
    old = sbd._MODEL_BUNDLE
    sbd._MODEL_BUNDLE = bundle
    worst, n = 0.0, 0
    try:
        for t in tickers[:6]:
            df = prices[t]
            i = start_idx_fn(df)
            if i is None or i >= len(df):
                continue
            snap = df.iloc[:i + 1]
            f = sbd.compute_feature_vector(snap)
            if f is None:
                continue
            ref = sbd.run_bubble_analysis(t, snap)['bubble_score']
            mine = score_features(bundle, f)[1]
            worst = max(worst, abs(ref - mine))
            n += 1
    finally:
        sbd._MODEL_BUNDLE = old
    if n == 0 or worst > 0.06:
        raise RuntimeError(f"Scoring mismatch vs run_bubble_analysis (max diff {worst}, n={n}).")
    print(f"  self-check OK: matches run_bubble_analysis() on {n} snapshots (max diff {worst:.3f})")


# ------------------------------------------------------------------ test
def score_test_dates(prices, groups, bundle, cutoff, test_step, horizons):
    recs = []
    for t, df in prices.items():
        close = df['Close'].values
        idx = df.index
        first = max(tbm.MIN_HISTORY_DAYS, int(idx.searchsorted(cutoff, side='right')))
        for i in range(first, len(df) - min(horizons), test_step):
            f = sbd.compute_feature_vector(df.iloc[:i + 1])
            if f is None:
                continue
            raw, gated = score_features(bundle, f)
            rec = {'ticker': t, 'group': groups[t], 'date': idx[i],
                   'month': idx[i].strftime('%Y-%m'), 'rsi': f['rsi'],
                   'ma_ratio': f['ma_ratio'], 'volatility': f['volatility'],
                   'raw_anomaly': raw, 'overheat': gated}
            for h in horizons:
                if i + h < len(df):
                    path = close[i + 1:i + h + 1] / close[i]
                    rec[f'fwd_ret_{h}'] = path[-1] - 1
                    rec[f'fwd_worst_{h}'] = min(path.min() - 1, 0.0)
                else:
                    rec[f'fwd_ret_{h}'] = np.nan
                    rec[f'fwd_worst_{h}'] = np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


# --------------------------------------------------------------- summary
SIGNALS = {'Overheat Score': 'overheat', 'Ungated anomaly': 'raw_anomaly',
           'RSI': 'rsi', 'Price/200MA': 'ma_ratio', 'Volatility': 'volatility'}


def auc(y, x):
    return roc_auc_score(y, x) if 0 < y.sum() < len(y) else np.nan


def summarize(d, h, crash_thr, top_frac, label):
    col = f'fwd_worst_{h}'
    d = d.dropna(subset=[col])
    if len(d) == 0:
        return [], d
    y = (d[col] <= -crash_thr).astype(int).values
    base = y.mean()
    rows = []

    def add(kind, name, flag, a=np.nan):
        n = int(flag.sum())
        r = {'subset': label, 'horizon_days': h, 'kind': kind, 'signal': name,
             'n_obs': len(d), 'base_crash_rate': base, 'auc': a,
             'n_flagged': n, 'pct_flagged': n / len(d)}
        if n > 0:
            r['crash_rate_flagged'] = y[flag].mean()
            r['lift'] = r['crash_rate_flagged'] / base if base > 0 else np.nan
            r['mean_worst_loss_flagged'] = d[col].values[flag].mean()
            r['mean_fwd_return_flagged'] = d[f'fwd_ret_{h}'].values[flag].mean()
        if n < len(d):
            r['mean_worst_loss_unflagged'] = d[col].values[~flag].mean()
        rows.append(r)

    for name, c in SIGNALS.items():
        x = d[c].values
        add(f'top {int(top_frac * 100)}% by score', name, x >= np.quantile(x, 1 - top_frac), auc(y, x))
    add('fixed rule', 'Overheat Score >= 50', d['overheat'].values >= 50)
    add('fixed rule', 'RSI > 70', d['rsi'].values > 70)
    add('fixed rule', 'Price/200MA > 1.3', d['ma_ratio'].values > 1.3)
    add('fixed rule', 'RSI > 70 and Price/200MA > 1.3',
        (d['rsi'].values > 70) & (d['ma_ratio'].values > 1.3))
    return rows, d


def bootstrap_auc_diff(d, h, crash_thr, n_boot, seed=0):
    col = f'fwd_worst_{h}'
    d = d.dropna(subset=[col]).copy()
    d['y'] = (d[col] <= -crash_thr).astype(int)
    months = d['month'].unique()
    by_month = {m: g for m, g in d.groupby('month')}
    rng = np.random.default_rng(seed)
    out = {b: [] for b in ('RSI', 'Price/200MA', 'Volatility', 'Ungated anomaly')}
    for _ in range(n_boot):
        pick = rng.choice(months, size=len(months), replace=True)
        s = pd.concat([by_month[m] for m in pick])
        y = s['y'].values
        if not 0 < y.sum() < len(y):
            continue
        a0 = roc_auc_score(y, s['overheat'])
        out['RSI'].append(a0 - roc_auc_score(y, s['rsi']))
        out['Price/200MA'].append(a0 - roc_auc_score(y, s['ma_ratio']))
        out['Volatility'].append(a0 - roc_auc_score(y, s['volatility']))
        out['Ungated anomaly'].append(a0 - roc_auc_score(y, s['raw_anomaly']))
    res = {}
    for k, v in out.items():
        if len(v) >= 20:
            res[k] = (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return res


def verdict_line(name, ci):
    m, lo, hi = ci
    if lo > 0:
        txt = "Overheat Score ranks BETTER"
    elif hi < 0:
        txt = "Overheat Score ranks WORSE"
    else:
        txt = "no demonstrable difference"
    return f"- vs {name}: AUC difference {m:+.3f} (95% interval {lo:+.3f} to {hi:+.3f}) -> {txt}"


# ----------------------------------------------------------------- plots
def plot_bands(d, h, path):
    col = f'fwd_worst_{h}'
    d = d.dropna(subset=[col]).copy()
    d['band'] = pd.cut(d['overheat'], [-1, 9.999, 29.999, 49.999, 69.999, 101],
                       labels=[b[1] for b in BANDS])
    g = d.groupby('band', observed=False)[col].agg(['mean', 'count'])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(g)), g['mean'] * 100, color=BAND_COLORS, edgecolor='#1A1A1A')
    for i, (m, n) in enumerate(zip(g['mean'], g['count'])):
        if n > 0:
            ax.text(i, m * 100 - 0.6, f"{m * 100:.1f}%\nn={n}", ha='center', va='top', fontsize=12)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, fontsize=13)
    ax.set_ylabel(f"Average worst loss in next {h} trading days (%)", fontsize=12)
    ax.set_title("Out-of-sample: later losses by Overheat band", fontsize=14, fontweight='bold')
    ax.axhline(0, color='#1A1A1A', lw=1)
    ax.margins(y=0.2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(d, h, crash_thr, path):
    col = f'fwd_worst_{h}'
    d = d.dropna(subset=[col])
    y = (d[col] <= -crash_thr).astype(int).values
    if not 0 < y.sum() < len(y):
        return
    colors = {'Overheat Score': '#005035', 'Ungated anomaly': '#8A1538',
              'RSI': '#1F3A5F', 'Price/200MA': '#EE7B1F', 'Volatility': '#6A1B9A'}
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, c in SIGNALS.items():
        fpr, tpr, _ = roc_curve(y, d[c].values)
        ax.plot(fpr, tpr, lw=2.5 if name == 'Overheat Score' else 1.8, color=colors[name],
                label=f"{name} (AUC {roc_auc_score(y, d[c].values):.2f})")
    ax.plot([0, 1], [0, 1], color='#999999', ls='--', lw=1)
    ax.set_xlabel("False alarm rate", fontsize=12)
    ax.set_ylabel(f"Share of >{int(crash_thr * 100)}% drops caught", fontsize=12)
    ax.set_title(f"Out-of-sample, next {h} trading days", fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=False)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Out-of-time backtest of the Overheat Score.")
    ap.add_argument('--cutoff', default='2022-12-31', help="train on data up to this date")
    ap.add_argument('--start', default='2014-01-01', help="first date of price history")
    ap.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'))
    ap.add_argument('--train-step', type=int, default=tbm.DEFAULT_SNAPSHOT_STEP_DAYS)
    ap.add_argument('--test-step', type=int, default=21, help="trading days between test dates")
    ap.add_argument('--horizons', type=int, nargs='+', default=[126, 252])
    ap.add_argument('--crash-threshold', type=float, default=0.20)
    ap.add_argument('--top-frac', type=float, default=0.10)
    ap.add_argument('--bootstrap', type=int, default=500)
    ap.add_argument('--extra-test-tickers', action='store_true')
    ap.add_argument('--outdir', default='backtest_out')
    ap.add_argument('--cache', default='.price_cache')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    cutoff = pd.Timestamp(args.cutoff)
    hs = sorted(args.horizons)

    core = list(tbm.TICKER_UNIVERSE)
    extra = [t for t in EXTRA_TEST_TICKERS if t not in core] if args.extra_test_tickers else []
    print(f"1/5 Downloading prices ({len(core)} core + {len(extra)} extra tickers)...")
    prices, groups = {}, {}
    for t in core + extra:
        df = get_prices(t, args.start, args.end, args.cache)
        if df is None or len(df) < tbm.MIN_HISTORY_DAYS + 30:
            print(f"  skipping {t}: no usable data")
            continue
        prices[t] = df
        groups[t] = 'core' if t in core else 'extra'
    n_core = sum(1 for g in groups.values() if g == 'core')
    if n_core < 0.8 * len(core):
        print(f"  WARNING: only {n_core}/{len(core)} core tickers downloaded; results are weaker.")
    if n_core == 0:
        sys.exit("No core data downloaded. Check your internet connection.")

    print(f"2/5 Training on snapshots dated on or before {args.cutoff} only...")
    rows = build_train_rows(prices, [t for t in core if t in prices], cutoff, args.train_step)
    if len(rows) < 100:
        sys.exit(f"Only {len(rows)} training rows; use an earlier --start or later --cutoff.")
    bundle = fit_bundle(rows, tbm.DEFAULT_CONTAMINATION)
    print(f"  {len(rows)} training snapshots from {n_core} tickers")

    def first_test_idx(df):
        return max(tbm.MIN_HISTORY_DAYS, int(df.index.searchsorted(cutoff, side='right')))
    self_check(bundle, prices, [t for t in core if t in prices], first_test_idx)

    print("3/5 Scoring test dates after the cutoff...")
    obs = score_test_dates(prices, groups, bundle, cutoff, args.test_step, hs)
    if obs.empty:
        sys.exit("No test observations; check --cutoff and --end.")
    obs.to_csv(os.path.join(args.outdir, 'observations.csv'), index=False)
    print(f"  {len(obs)} scored observations across {obs['ticker'].nunique()} tickers "
          f"({obs['date'].min().date()} to {obs['date'].max().date()})")

    print("4/5 Summarizing...")
    subsets = [('all tickers', obs)]
    if extra:
        subsets.append(('core tickers only', obs[obs.group == 'core']))
        subsets.append(('extra (later-fell) tickers only', obs[obs.group == 'extra']))
    all_rows, md = [], []
    md.append(f"# Overheat Score backtest\n")
    md.append(f"Trained on snapshots through **{args.cutoff}**; tested on **{obs['date'].min().date()} "
              f"to {obs['date'].max().date()}** ({len(obs)} scored ticker-dates, "
              f"{obs['ticker'].nunique()} tickers). 'Crash' = worst loss of {int(args.crash_threshold * 100)}% "
              f"or more from that day's close within the horizon.\n")
    verdicts = []
    for label, sub in subsets:
        for h in hs:
            rows_, used = summarize(sub, h, args.crash_threshold, args.top_frac, label)
            if not rows_:
                continue
            all_rows += rows_
            t = pd.DataFrame(rows_)
            md.append(f"\n## {label}, next {h} trading days "
                      f"(n={len(used)}, base crash rate {t['base_crash_rate'].iloc[0]:.1%})\n")
            show = t[['kind', 'signal', 'auc', 'pct_flagged', 'crash_rate_flagged', 'lift',
                      'mean_worst_loss_flagged', 'mean_worst_loss_unflagged']].copy()
            for c in ('pct_flagged', 'crash_rate_flagged', 'mean_worst_loss_flagged',
                      'mean_worst_loss_unflagged'):
                show[c] = show[c].map(lambda v: '' if pd.isna(v) else f"{v:.1%}")
            show['auc'] = show['auc'].map(lambda v: '' if pd.isna(v) else f"{v:.3f}")
            show['lift'] = show['lift'].map(lambda v: '' if pd.isna(v) else f"{v:.2f}x")
            md.append(show.to_markdown(index=False) if hasattr(show, 'to_markdown') and _has_tabulate()
                      else "```\n" + show.to_string(index=False) + "\n```")
            if label == 'all tickers':
                ci = bootstrap_auc_diff(used, h, args.crash_threshold, args.bootstrap)
                if ci:
                    md.append(f"\nBootstrap over calendar months, horizon {h}:")
                    for k, v in ci.items():
                        md.append(verdict_line(k, v))
                        if h == hs[0]:
                            verdicts.append(verdict_line(k, v))
    pd.DataFrame(all_rows).to_csv(os.path.join(args.outdir, 'summary.csv'), index=False)

    md.append("\n## Read this before quoting any number\n"
              "- Tickers move together and test windows overlap, so the effective sample is far smaller than the row count; trust the bootstrap intervals, not the raw counts.\n"
              "- The core universe is 63 stocks that are large and liquid today (survivors). The extra tickers are also still listed, so survivorship bias is reduced, not removed.\n"
              "- One split, one market regime (2023 onward). A different cutoff can change the answer; try --cutoff 2021-12-31 and 2023-12-31.\n"
              "- Volatility is included as a baseline on purpose: volatile stocks fall further by construction, so a score that only tracks volatility adds nothing. Beating RSI but not Volatility is not a win.\n"
              "- The 95% intervals are optimistic (overlapping windows, many comparisons); treat 'BETTER' as a hint to dig deeper, not proof.\n"
              "- AUC 0.5 means no ranking skill. A high score is not a probability of a crash. Not financial advice.\n")
    with open(os.path.join(args.outdir, 'summary.md'), 'w') as fh:
        fh.write("\n".join(md))

    print("5/5 Drawing plots...")
    plot_bands(obs, hs[0], os.path.join(args.outdir, 'band_losses.png'))
    plot_roc(obs, hs[0], args.crash_threshold, os.path.join(args.outdir, 'roc.png'))

    print("\n" + "=" * 70)
    print(f"HEADLINE (all tickers, next {hs[0]} trading days)")
    head = pd.DataFrame([r for r in all_rows if r['subset'] == 'all tickers' and r['horizon_days'] == hs[0]])
    print(head[['kind', 'signal', 'auc', 'pct_flagged', 'crash_rate_flagged', 'lift']].to_string(index=False))
    for v in verdicts:
        print(v)
    print(f"\nFull write-up: {os.path.join(args.outdir, 'summary.md')}")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == '__main__':
    main()
