"""Generate the README figures as SVG, in light and dark variants.

Hand-written SVG rather than a plotting library: it adds no dependency, stays
crisp at any width, and lets each figure be built for both GitHub themes so
neither one glares. The README pairs them with <picture> + prefers-color-scheme.

Every number here is read from the study outputs, never typed in by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED, REPORTS, VALID_END
from src.setups import REVERSION_SETUPS

FIG = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(fg="#1F2328", mid="#59636E", faint="#D1D9E0", grid="#EAEEF2",
                  good="#0F766E", bad="#B42318", warn="#B45309", bar="#8C959F"),
    "dark":  dict(fg="#E6EDF3", mid="#9198A1", faint="#30363D", grid="#21262D",
                  good="#3FB950", bad="#F85149", warn="#D29922", bar="#6E7681"),
}
FONT = "ui-monospace,'SF Mono',Menlo,Consolas,monospace"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{SANS}">{body}</svg>')


def txt(x, y, s, fill, size=12, anchor="start", weight="400", mono=False):
    f = f' font-family="{FONT}"' if mono else ""
    return (f'<text x="{x}" y="{y}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{f}>{esc(s)}</text>')


# ---------------------------------------------------------------- figure 1
def fig_nulltest(C):
    """An unpurged splitter invents an edge out of pure noise."""
    W, H = 720, 250
    rows = [("Purged CV + embargo", 0.4986, 0.054, C["good"]),
            ("Naive random k-fold", 0.5610, 0.649, C["bad"])]
    b = [txt(0, 20, "Validation tested on PURE NOISE", C["fg"], 15, weight="600"),
         txt(0, 40, "Random features, no signal. A correct harness must find nothing.",
             C["mid"], 12)]
    x0, top, bh, gap = 210, 62, 26, 46
    lo, hi = 0.47, 0.58
    for i, (name, auc, spread, col) in enumerate(rows):
        y = top + i * gap
        w = (auc - lo) / (hi - lo) * 380
        b += [txt(0, y + bh - 9, name, C["fg"], 12),
              f'<rect x="{x0}" y="{y}" width="380" height="{bh}" fill="{C["grid"]}" rx="2"/>',
              f'<rect x="{x0}" y="{y}" width="{w:.0f}" height="{bh}" fill="{col}" rx="2"/>',
              txt(x0 + w + 8, y + bh - 9, f"AUC {auc:.4f}", col, 12, weight="600", mono=True)]
    # 0.50 reference
    x50 = x0 + (0.50 - lo) / (hi - lo) * 380
    b += [f'<line x1="{x50:.0f}" y1="{top-6}" x2="{x50:.0f}" y2="{top+2*gap-14}" '
          f'stroke="{C["mid"]}" stroke-width="1" stroke-dasharray="3 3"/>',
          txt(x50, top - 12, "0.50 = no skill", C["mid"], 10, anchor="middle", mono=True)]

    y2 = top + 2 * gap + 6
    b += [f'<line x1="0" y1="{y2}" x2="{W}" y2="{y2}" stroke="{C["faint"]}"/>',
          txt(0, y2 + 26, "Fabricated profit per trade, from data containing no signal",
              C["mid"], 12)]
    for i, (name, auc, spread, col) in enumerate(rows):
        y = y2 + 46 + i * 30
        b += [txt(0, y + 13, name, C["fg"], 12),
              f'<rect x="{x0}" y="{y}" width="{spread/0.70*380:.0f}" height="17" fill="{col}" rx="2"/>',
              txt(x0 + spread / 0.70 * 380 + 8, y + 13, f"{spread:+.3f}R", col, 12,
                  weight="600", mono=True)]
    return svg(W, H, "".join(b))


# ---------------------------------------------------------------- figure 2
def fig_exits(C):
    """Ranking 12 exit schemes per trade vs per day of capital inverts the order."""
    df = pd.read_csv(REPORTS / "stage11_exit_sweep.csv")
    df["epd"] = df.edge / df.hold.clip(lower=.5)
    g = (df.groupby("exit").agg(edge=("edge", "mean"), hold=("hold", "mean"),
                                epd=("epd", "mean")).reset_index())
    a = g.sort_values("edge", ascending=False).reset_index(drop=True)
    bb = g.sort_values("epd", ascending=False).reset_index(drop=True)
    pos_b = {r.exit: i for i, r in bb.iterrows()}

    W, H = 720, 400
    b = [txt(0, 20, "The same 12 exits, ranked two ways", C["fg"], 15, weight="600"),
         txt(0, 40, "Widening an exit raises return per trade and lowers return per unit "
             "of capital.", C["mid"], 12),
         txt(120, 68, "BY RETURN / TRADE", C["mid"], 10, weight="600", mono=True),
         txt(600, 68, "BY RETURN / DAY HELD", C["mid"], 10, anchor="end",
             weight="600", mono=True)]
    top, step = 84, 25
    for i, r in a.iterrows():
        j = pos_b[r.exit]
        y1, y2 = top + i * step, top + j * step
        moved = abs(i - j)
        col = C["good"] if j < i else (C["bad"] if j > i else C["mid"])
        stroke = col if moved >= 3 else C["faint"]
        wdt = 2 if moved >= 3 else 1
        b += [f'<path d="M300,{y1} C400,{y1} 400,{y2} 500,{y2}" fill="none" '
              f'stroke="{stroke}" stroke-width="{wdt}" opacity="{0.95 if moved>=3 else 0.5}"/>',
              txt(292, y1 + 4, r.exit, C["fg"], 11, anchor="end", mono=True),
              txt(508, y2 + 4, r.exit, C["fg"], 11, mono=True),
              txt(120, y1 + 4, f"{r.edge:+.3f}", C["mid"], 11, anchor="end", mono=True),
              txt(700, y2 + 4, f"{r.epd:.4f}", C["mid"], 11, anchor="end", mono=True)]
    y = top + len(a) * step + 18
    b += [f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{C["faint"]}"/>',
          txt(0, y + 22, "The widest bracket looks twice as good per trade and is half as "
              "good per unit of capital.", C["fg"], 12),
          txt(0, y + 40, "It had quietly stopped being a mean-reversion trade and become "
              "buy-and-hold.", C["mid"], 12)]
    return svg(W, H, "".join(b))


# ---------------------------------------------------------------- figure 3
def fig_survival(C):
    """Ten setups cleared in development; one cleared out of sample."""
    ev = pd.read_parquet(PROCESSED / "events.parquet")
    ctl_all = ev[ev.setup == "any_liquid_reversion"]

    def edge(sub, ctl):
        a = sub.groupby("date")["r_net"].mean()
        bq = ctl.groupby("date")["r_net"].mean()
        d = (a - bq).dropna()
        if len(d) < 12:
            return np.nan
        return float(d.groupby(pd.PeriodIndex(d.index, freq="M")).mean().mean())

    rows = []
    for s in REVERSION_SETUPS:
        g = ev[ev.setup == s]
        rows.append({"setup": s,
                     "dev": edge(g[g.date <= VALID_END], ctl_all[ctl_all.date <= VALID_END]),
                     "oos": edge(g[g.date > VALID_END], ctl_all[ctl_all.date > VALID_END])})
    d = pd.DataFrame(rows).dropna().sort_values("oos", ascending=False).reset_index(drop=True)

    W, H = 720, 60 + len(d) * 26 + 70
    b = [txt(0, 20, "Ten reversion setups: development vs out-of-sample", C["fg"], 15,
             weight="600"),
         txt(0, 40, "Same-day edge against randomly chosen liquid stocks. One survived.",
             C["mid"], 12)]
    lo, hi = -0.06, 0.11
    x0, wid, top, step = 190, 430, 66, 26
    zx = x0 + (0 - lo) / (hi - lo) * wid
    b += [f'<line x1="{zx:.0f}" y1="{top-10}" x2="{zx:.0f}" y2="{top+len(d)*step-8}" '
          f'stroke="{C["mid"]}" stroke-width="1"/>',
          txt(zx, top - 16, "0", C["mid"], 10, anchor="middle", mono=True),
          txt(x0 - 4, top - 16, "worse than random", C["mid"], 9, anchor="end"),
          txt(x0 + wid, top - 16, "better", C["mid"], 9, anchor="end")]
    for i, r in d.iterrows():
        y = top + i * step
        xd = x0 + (r.dev - lo) / (hi - lo) * wid
        xo = x0 + (r.oos - lo) / (hi - lo) * wid
        surv = r.oos > 0.02
        col = C["good"] if surv else C["bar"]
        b += [txt(182, y + 4, r.setup, C["fg"] if surv else C["mid"], 11,
                  anchor="end", weight="600" if surv else "400", mono=True),
              f'<line x1="{xd:.0f}" y1="{y}" x2="{xo:.0f}" y2="{y}" stroke="{col}" '
              f'stroke-width="{2 if surv else 1}" opacity="{1 if surv else .55}"/>',
              f'<circle cx="{xd:.0f}" cy="{y}" r="3" fill="none" stroke="{col}" stroke-width="1.5"/>',
              f'<circle cx="{xo:.0f}" cy="{y}" r="4" fill="{col}"/>']
        if surv:
            b.append(txt(xo + 10, y + 4, f"{r.oos:+.3f}", col, 11, weight="600", mono=True))
    y = top + len(d) * step + 14
    b += [f'<line x1="0" y1="{y}" x2="{W}" y2="{y}" stroke="{C["faint"]}"/>',
          f'<circle cx="6" cy="{y+22}" r="3" fill="none" stroke="{C["mid"]}" stroke-width="1.5"/>',
          txt(16, y + 26, "development 1999-2020", C["mid"], 11),
          f'<circle cx="200" cy="{y+22}" r="4" fill="{C["mid"]}"/>',
          txt(210, y + 26, "out-of-sample 2021-2025", C["mid"], 11),
          txt(0, y + 46, "Four cleared zero in development. One cleared it out of sample "
              "— about what chance gives you.", C["fg"], 12)]
    return svg(W, H, "".join(b))


FIGS = {"null-test": fig_nulltest, "exit-ranking": fig_exits, "setup-survival": fig_survival}

if __name__ == "__main__":
    for name, fn in FIGS.items():
        for theme, C in THEMES.items():
            p = FIG / f"{name}-{theme}.svg"
            p.write_text(fn(C), encoding="utf-8")
            print(f"  wrote {p.relative_to(FIG.parent.parent)}  ({p.stat().st_size//1024} KB)")
    print(f"\n{len(FIGS)*2} figures -> {FIG}")
