"""
make_poster_figures.py -- build every poster figure from the Overheat Score project.

Put this file in the same folder as stock_bubble_detector.py and
train_bubble_model.py, then run:

    python make_poster_figures.py
    python make_poster_figures.py --poster Overheat_Score_Poster_v2.pptx   # also fills the poster

What it does (needs internet for Yahoo Finance, ~5-10 minutes):
  1. Trains the Isolation Forest with the project's own training code.
  2. Scores the four KNOWN_TEST_CASES with the project's own run_bubble_analysis().
  3. Writes 4 PNGs, sized in inches to match the poster slots 1:1 (so font sizes
     here are the printed point sizes):
        stock_charts.png   2x2 price + 200-day average + RSI for the four cases
        scores.png         Overheat Score per case, colored by risk band
        zscores.png        feature z-scores per case
        feature_map.png    RSI vs price/200-day average for all training
                           snapshots, colored by score, with the four cases starred
  4. Writes case_scores.csv with the real numbers (use them in your poster text).
  5. With --poster, drops the PNGs into the frames of the poster .pptx
     (needs: pip install python-pptx) and saves <name>_filled.pptx.

Nothing here is simulated: every number comes from Yahoo Finance data and the
model trained in step 1. Not financial advice.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import stock_bubble_detector as sbd
import train_bubble_model as tbm

# Slot sizes in inches (width, height) -- must match the frames in the poster.
SLOTS = {
    "stock_charts": (12.2, 9.2),
    "feature_map": (5.9, 5.9),
    "scores": (8.65, 3.5),
    "zscores": (8.65, 3.3),
}

GREEN, MAROON, INK, NAVY, RED = "#005035", "#8A1538", "#1A1A1A", "#1F3A5F", "#C62828"
BANDS = [  # (lower bound, name, color) -- same cut-offs as assign_risk_level()
    (0, "Healthy", "#2E7D32"),
    (10, "Low", "#9CCB3B"),
    (30, "Moderate", "#F2C230"),
    (50, "High", "#EE7B1F"),
    (70, "Critical", "#B3261E"),
]
SHORT_FEATURES = ["RSI", "Price/200MA", "Volatility", "Accel.", "Drawdown"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#555555",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def band_of(score):
    name, color = BANDS[0][1], BANDS[0][2]
    for lower, n, c in BANDS:
        if score >= lower:
            name, color = n, c
    return name, color


def score_from_features(bundle, feats):
    """Same math as run_bubble_analysis(), for training-set snapshots (dicts)."""
    x = np.array([[feats[n] for n in sbd.FEATURE_NAMES]])
    xs = bundle["scaler"].transform(x)
    raw = float(bundle["model"].score_samples(xs)[0])
    lo, hi = bundle["score_low"], bundle["score_high"]
    s = float(np.clip(100 * (hi - raw) / (hi - lo), 0, 100))
    z = {n: (feats[n] - bundle["scaler"].mean_[i]) / bundle["scaler"].scale_[i]
         for i, n in enumerate(sbd.FEATURE_NAMES)}
    if z["rsi"] + z["ma_ratio"] + z["accel"] <= 0:
        s = min(s, 15.0)
    return s, z


def load_cases(years_buffer=2):
    cases = []
    for ticker, as_of, expected, note in tbm.KNOWN_TEST_CASES:
        dt = datetime.strptime(as_of, "%Y-%m-%d")
        start = (dt - timedelta(days=365 * years_buffer)).strftime("%Y-%m-%d")
        data = sbd.load_stock_data(ticker, start, as_of)
        if data is None or len(data) < tbm.MIN_HISTORY_DAYS:
            print(f"  skipping {ticker} {as_of}: not enough data")
            continue
        res = sbd.run_bubble_analysis(ticker, data)
        if res["bubble_score"] is None:
            print(f"  skipping {ticker} {as_of}: {res['risk_level']}")
            continue
        cases.append(dict(ticker=ticker, as_of=as_of, expected=expected,
                          score=res["bubble_score"], z=res["sub_scores"], data=data))
    return cases


# ----------------------------------------------------------------- figures
def fig_stock_charts(cases, path):
    w, h = SLOTS["stock_charts"]
    ncols, nrows = 2, (len(cases) + 1) // 2
    fig = plt.figure(figsize=(w, h))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.40, wspace=0.17,
                             left=0.075, right=0.99, top=0.925, bottom=0.055)
    for k, c in enumerate(cases):
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
        ax_p = fig.add_subplot(inner[0])
        ax_r = fig.add_subplot(inner[1], sharex=ax_p)
        d = c["data"]
        px = d["Close"]
        ma200 = sbd.get_moving_average(d, 200)
        ax_p.plot(px.index, px, color=NAVY, lw=2.2)
        ax_p.plot(ma200.index, ma200, color=RED, lw=2.2, ls="-.")
        ratio = sbd.get_price_to_200ma_ratio(d)
        if ratio is not None:
            ax_p.fill_between(px.index, px, ma200,
                              where=((ratio > 1.30) & px.notna() & ma200.notna()).values,
                              color=RED, alpha=0.18, lw=0)
        name, color = band_of(c["score"])
        ax_p.set_title(f"{c['ticker']}  {c['as_of']}", loc="left", fontsize=20,
                       fontweight="bold", pad=24)
        ax_p.text(0, 1.04, f"Score {c['score']:.0f} ({name})  |  expected {c['expected']}",
                  transform=ax_p.transAxes, fontsize=14.5, fontweight="bold",
                  color=color if name not in ("Low", "Moderate") else INK, va="bottom")
        ax_p.tick_params(labelsize=15)
        ax_p.set_ylabel("USD", fontsize=15)
        plt.setp(ax_p.get_xticklabels(), visible=False)

        rsi = sbd.get_rsi(d)
        ax_r.plot(rsi.index, rsi, color="#6A1B9A", lw=1.8)
        ax_r.axhline(70, color=RED, ls="--", lw=1.4)
        ax_r.fill_between(rsi.index, 70, rsi, where=(rsi > 70).values, color=RED, alpha=0.25)
        ax_r.set_ylim(0, 100)
        ax_r.set_yticks([30, 70])
        ax_r.set_ylabel("RSI", fontsize=15)
        ax_r.tick_params(labelsize=15)
        ax_r.xaxis.set_major_locator(mdates.MonthLocator(interval=8))
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_scores(cases, path):
    w, h = SLOTS["scores"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.95, bottom=0.30)
    xs = np.arange(len(cases))
    for lower, _, _ in BANDS[1:]:
        ax.axhline(lower, color="#999999", ls=":", lw=1.2, zorder=0)
    bars = ax.bar(xs, [c["score"] for c in cases], width=0.62,
                  color=[band_of(c["score"])[1] for c in cases], edgecolor=INK, lw=1.2)
    for b, c in zip(bars, cases):
        ax.text(b.get_x() + b.get_width() / 2, c["score"] + 2, f"{c['score']:.0f}",
                ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c['ticker']}\n{c['as_of']}\nexpect {c['expected']}" for c in cases],
                       fontsize=14)
    ax.set_yticks([0, 30, 50, 70, 100])
    ax.set_ylim(0, 112)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylabel("Overheat Score", fontsize=15)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_zscores(cases, path):
    w, h = SLOTS["zscores"]
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.84, bottom=0.17)
    palette = {"overheated": ["#8A1538", "#D8708A"], "calm": ["#005035", "#7FB59E"]}
    used = {"overheated": 0, "calm": 0}
    n = len(cases)
    width = 0.8 / n
    allz = []
    for i, c in enumerate(cases):
        col = palette.get(c["expected"], ["#555555"])[used.get(c["expected"], 0) % 2]
        used[c["expected"]] = used.get(c["expected"], 0) + 1
        vals = [c["z"][f] for f in sbd.FEATURE_NAMES]
        allz += vals
        ax.bar(np.arange(5) + (i - (n - 1) / 2) * width, vals, width * 0.92,
               color=col, label=c["ticker"])
    for y in (-2, 2):
        ax.axhline(y, color="#777777", ls="--", lw=1.2)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(SHORT_FEATURES, fontsize=14)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(min(-2.6, min(allz) - 0.3), max(2.6, max(allz) + 0.3))
    ax.set_ylabel("z-score", fontsize=15)
    ax.legend(ncol=n, fontsize=14, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.0), handlelength=1.2, columnspacing=1.4)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_feature_map(rows, bundle, cases, path):
    w, h = SLOTS["feature_map"]
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0.15, 0.30, 0.82, 0.66])
    cax = fig.add_axes([0.15, 0.10, 0.82, 0.035])
    rsi = np.array([r["rsi"] for r in rows])
    ma = np.array([r["ma_ratio"] for r in rows])
    sc = np.array([score_from_features(bundle, r)[0] for r in rows])
    order = np.argsort(sc)
    pts = ax.scatter(rsi[order], ma[order], c=sc[order], cmap="YlOrRd", vmin=0, vmax=100,
                     s=60, alpha=0.85, edgecolors="none")
    feats = [sbd.compute_feature_vector(c["data"]) for c in cases]
    for f in feats:
        ax.scatter([f["rsi"]], [f["ma_ratio"]], marker="*", s=900, color="white",
                   edgecolors=INK, linewidths=1.8, zorder=5)
    ax.margins(x=0.10, y=0.10)
    xmin, xmax = ax.get_xlim()
    for c, f in zip(cases, feats):
        near_right = f["rsi"] > xmin + 0.72 * (xmax - xmin)
        ax.annotate(c["ticker"], (f["rsi"], f["ma_ratio"]),
                    xytext=(-14, 10) if near_right else (14, 10),
                    ha="right" if near_right else "left",
                    textcoords="offset points", fontsize=15, fontweight="bold", zorder=6)
    ax.set_xlabel("RSI (14-day)", fontsize=16)
    ax.set_ylabel("Price / 200-day average", fontsize=16)
    ax.tick_params(labelsize=14)
    cb = fig.colorbar(pts, cax=cax, orientation="horizontal")
    cb.set_label("Overheat Score of each training snapshot", fontsize=14)
    cb.ax.tick_params(labelsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ poster
def fill_poster(pptx_in, figdir):
    from pptx import Presentation
    mapping = {"fig_stock_charts": "stock_charts.png", "fig_scores": "scores.png",
               "fig_zscores": "zscores.png", "fig_feature_map": "feature_map.png"}
    prs = Presentation(pptx_in)
    slide = prs.slides[0]
    placed = 0
    for shape in list(slide.shapes):
        png = mapping.get(shape.name)
        if png and os.path.exists(os.path.join(figdir, png)):
            l, t, wd, ht = shape.left, shape.top, shape.width, shape.height
            shape._element.getparent().remove(shape._element)
            slide.shapes.add_picture(os.path.join(figdir, png), l, t, wd, ht)
            placed += 1
    out = os.path.splitext(pptx_in)[0] + "_filled.pptx"
    prs.save(out)
    print(f"Placed {placed} figures -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Build Overheat Score poster figures.")
    ap.add_argument("--outdir", default="poster_figures")
    ap.add_argument("--years", type=int, default=tbm.DEFAULT_YEARS_OF_HISTORY)
    ap.add_argument("--poster", help="poster .pptx whose figure frames should be filled")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("1/4 Building training snapshots and training the model (5-10 min)...")
    rows = tbm.build_training_rows(years=args.years)
    bundle = tbm.train_and_save(rows)
    sbd.reload_model()

    print("2/4 Scoring the known test cases...")
    cases = load_cases()
    if not cases:
        sys.exit("No test case could be scored -- check your internet connection.")

    print("3/4 Drawing figures...")
    fig_stock_charts(cases, os.path.join(args.outdir, "stock_charts.png"))
    fig_scores(cases, os.path.join(args.outdir, "scores.png"))
    fig_zscores(cases, os.path.join(args.outdir, "zscores.png"))
    fig_feature_map(rows, bundle, cases, os.path.join(args.outdir, "feature_map.png"))

    table = pd.DataFrame([dict(ticker=c["ticker"], as_of=c["as_of"], expected=c["expected"],
                               score=c["score"], band=band_of(c["score"])[0],
                               **{f"z_{k}": v for k, v in c["z"].items()}) for c in cases])
    table.to_csv(os.path.join(args.outdir, "case_scores.csv"), index=False)
    print("\n" + table[["ticker", "as_of", "expected", "score", "band"]].to_string(index=False))
    print(f"\nTraining snapshots: {len(rows)}  |  figures in: {args.outdir}/")

    if args.poster:
        print("4/4 Filling the poster...")
        fill_poster(args.poster, args.outdir)
    else:
        print("4/4 (skipped: no --poster given)")


if __name__ == "__main__":
    main()
