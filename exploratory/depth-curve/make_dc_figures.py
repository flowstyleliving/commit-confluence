#!/usr/bin/env python
"""DC paper figure builder.

Renders figures for the Depth Curves (DC) paper from ALREADY-SCORED artifacts.

DISCIPLINE RAILS (see DC_DATA_CONTRACT.md for the full data contract):
  * This module RENDERS scored values. It never re-scores, re-fits or re-selects.
    Every figure asserts its plotted values against the scored JSON before saving;
    an assertion failure is a hard crash, never a warning.
  * E5 (terminal dip) is a single-look registered WEAKEN 8/12 against a frozen
    >=10/12 bar plus family bars that were missed. Captions never round that up.
  * E6 (cliff onset) is NOT TESTED -- gatekept behind an E5 CONFIRM that did not
    occur. Grid-A cliff numbers are DISCOVERY ONLY.
  * Grid A = discovery, grid B = held-out confirmation. Never one shared axis
    without a labelled split.
  * Whole lane is torch/Modal nf4, comparable=false: never pooled with sealed MLX.
  * Mistral-Medium-3.5 is FP8-origin -- flag wherever it appears.
  * js_no_bos returns None under total BOS-sink collapse (gemma-3-27b): render as
    explicitly undefined, never zero, never silently dropped.

Usage:  python make_dc_figures.py --figure money [--check]
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = "/Users/msrk/Documents/the_GOAT/wiki/paper/dc-figures"
PRIMARY = "final_js_no_bos"

# The ACE panel winner marginal for anli_r1/Llama-3.3-70B. EXTERNAL to this lane.
# Source: furnace-guard/artifacts/modal_profiles_ext/profiles_ext/anli_r1/
#         Llama-3.3-70B-Instruct.profile.json .primary_full_panel.winner_marginal.auroc
# It is the IN-SAMPLE full-panel marginal (like-for-like against the in-sample
# depth curve). The deployed OOB median for the same winner is 0.7954
# [0.6995, 0.8728], winner_stability 0.511 -- stated in the caption so no reader
# mistakes 0.897 for beating a bootstrapped bound.
PANEL_WINNER_AUROC = 0.8156
PANEL_PROFILE = ("/Users/msrk/Documents/furnace-guard/artifacts/modal_profiles_ext/"
                 "profiles_ext/anli_r1/Llama-3.3-70B-Instruct.profile.json")

# Panel rungs are mid=N//2, N-2, N-1 (PRE_REGISTRATION.md line 9), 0-indexed.
def rungs(n):
    return [n // 2, n - 2, n - 1]

def load_grid_a():
    with open(os.path.join(LANE, "RESULTS.json")) as f:
        return json.load(f)

def assert_close(name, got, want, tol=0.0):
    """Hard-fail comparison. tol=0.0 means exact equality is required."""
    got, want = float(got), float(want)
    ok = (got == want) if tol == 0.0 else bool(np.isclose(got, want, atol=tol))
    if not ok:
        raise AssertionError(
            "PLOTTED VALUE DISAGREES WITH SCORED ARTIFACT: %s plotted=%r scored=%r (tol=%r)"
            % (name, got, want, tol))

def figure_money(check_only=False):
    """FIG2 -- the blind-spot money figure.

    Llama-3.3-70B / anli_r1 full per-block curve, shuffled envelope shaded, the
    three ACE panel rungs marked, the panel winner drawn as a reference line, and
    the mid-stack peak annotated. The visual argument is: three rungs, all
    outside the band that carries the signal.
    """
    res = load_grid_a()
    key = "anli_r1/Llama-3.3-70B-Instruct"
    cell = res["cells"][key]

    curve = np.asarray(cell["curves_signfree_auroc"][PRIMARY], dtype=float)
    env = np.asarray(cell["envelope_q97_5_primary"], dtype=float)
    n = int(cell["n_layers"])
    lstar = int(cell["lstar"])
    peak = float(cell["peak_auroc"])
    x = np.arange(n)

    # ---- assertion contract: every plotted quantity is traced to the artifact ----
    assert curve.shape == (n,), "curve length %d != n_layers %d" % (len(curve), n)
    assert env.shape == (n,), "envelope length %d != n_layers %d" % (len(env), n)
    assert_close("peak_auroc", curve[lstar], peak)
    assert_close("argmax(curve) == lstar", int(np.argmax(curve)), lstar)
    n_over = int((curve > env).sum())
    assert_close("blocks over envelope", n_over, 44)
    assert_close("envelope ceiling", env.max(), 0.6145, tol=5e-5)
    assert_close("peak block", lstar, 48)
    assert_close("peak value", peak, 0.8975, tol=5e-5)
    r = rungs(n)
    assert r == [40, 78, 79], "rungs %r != [40, 78, 79]" % r
    # the headline claim itself, asserted rather than asserted-in-prose:
    assert peak > PANEL_WINNER_AUROC, "peak does not exceed the panel winner"
    for b in r:
        assert curve[b] < peak, "rung %d is not below the peak" % b
    # and the external reference must match its source profile on disk
    if os.path.exists(PANEL_PROFILE):
        with open(PANEL_PROFILE) as f:
            prof = json.load(f)
        assert_close("panel winner marginal",
                     prof["primary_full_panel"]["winner_marginal"]["auroc"],
                     PANEL_WINNER_AUROC, tol=5e-5)
    else:
        raise AssertionError("panel profile not found: %s" % PANEL_PROFILE)

    if check_only:
        print("money: all assertions PASS (n=%d lstar=%d peak=%.4f over_env=%d)"
              % (n, lstar, peak, n_over))
        return

    ink, band, accent, rung_c = "#1a1a1a", "#c9c9c9", "#b2182b", "#2166ac"
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    ax.fill_between(x, 0.5, env, color=band, alpha=0.55, lw=0,
                    label="shuffled-label envelope (97.5%)")
    ax.axhline(0.5, color="#999999", lw=0.8, ls=":", zorder=1)
    ax.plot(x, curve, color=ink, lw=1.7, zorder=3,
            label="per-block sign-free AUROC (%s)" % PRIMARY)

    ax.axhline(PANEL_WINNER_AUROC, color=accent, lw=1.3, ls="--", zorder=2,
               label="ACE panel winner, in-sample (%.4f)" % PANEL_WINNER_AUROC)

    for i, b in enumerate(r):
        ax.axvline(b, color=rung_c, lw=1.0, ls="-", alpha=0.45, zorder=2)
        ax.plot([b], [curve[b]], marker="o", ms=6, mfc="white", mec=rung_c,
                mew=1.8, zorder=5,
                label="ACE panel rungs (blocks %s)" % ",".join(map(str, r)) if i == 0 else None)

    ax.plot([lstar], [peak], marker="*", ms=15, color=accent, zorder=6)
    ax.annotate("peak %.3f @ block %d" % (peak, lstar),
                xy=(lstar, peak + 0.005), xytext=(lstar + 7, 0.930),
                fontsize=9, color=accent, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=accent, lw=1.0,
                                shrinkA=0, shrinkB=3))

    ax.set_xlabel("decoder block (0-indexed, N=%d)" % n)
    ax.set_ylabel("sign-free AUROC")
    ax.set_xlim(-1, n)
    ax.set_ylim(0.42, 0.95)
    ax.set_title("Llama-3.3-70B-Instruct / ANLI R1 — the three-rung panel straddles the band",
                 fontsize=10.5, pad=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.95)
    ax.text(0.985, 0.03,
            "%d/%d blocks above envelope   ·   grid A (discovery)   ·   torch/Modal nf4, not byte-comparable with sealed MLX panels"
            % (n_over, n),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.4, color="#666666")

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIGDIR, "fig2_money_llama33_anli.%s" % ext)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)

FIGURES = {"money": figure_money}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", choices=sorted(FIGURES) + ["all"], default="all")
    ap.add_argument("--check", action="store_true",
                    help="run every assertion without writing files")
    a = ap.parse_args()
    names = sorted(FIGURES) if a.figure == "all" else [a.figure]
    for nm in names:
        FIGURES[nm](check_only=a.check)


# =============================================================================
# Remaining artifacts. Added 2026-08-24 under MK decision "A": grid-B panels
# carry NO shuffled-label envelope, because no envelope was ever registered for
# grid-B slugs and computing one would be a new Monte-Carlo statistic on
# confirmatory cells. Grid-B curves ARE recomputed, using the sealed scorer's
# own signfree_auroc_matrix -- arithmetic the scorer already defines.
# =============================================================================
import sys
sys.path.insert(0, LANE)
from depth_score import signfree_auroc_matrix  # sealed scorer, imported not reimplemented
# Grid B has its own schema (1.1-gridB) and its own validating loader, which also
# re-checks the frozen data sha, the pinned HF revision and both gates. Use it
# rather than a hand-rolled read.
from score_grid_b import load_cell_b, load_task_labels

GRID_B_JSON = os.path.join(LANE, "GRID_B_RESULTS.json")
RESCORE_JSON = os.path.join(LANE, "RESCORE_GRID_A.json")
NPZ_A = os.path.join(LANE, "npz", "depth_curve")
NPZ_B = os.path.join(LANE, "npz", "depth_grid_b")
FP8_CELL = "Mistral-Medium-3.5-128B"          # FP8-origin, flagged in every panel
JS_DOMAIN_CELL = "gemma-3-27b-it"             # js_no_bos undefined under BOS-sink collapse

def load_json(p):
    with open(p) as f:
        return json.load(f)

def short(key):
    task, slug = key.split("/", 1)
    return "%s / %s" % (slug.replace("-Instruct", "").replace("-it", ""),
                        "ANLI" if task.startswith("anli") else "Halu")

def figure_forest(check_only=False):
    """FIG3 -- terminal-dip forest plot.

    Both grids are shown on ONE axis because both are measured with the SAME
    cross-fitted estimator (grid A via RESCORE_GRID_A.json, not its registered
    E4 boolean). Grid A is hollow = discovery; grid B is filled = held-out
    confirmation. The registered outcome is E5 WEAKEN 8/12 against a frozen
    >=10/12 bar plus family bars that were missed -- NOT a confirmation.
    """
    rs, gb = load_json(RESCORE_JSON), load_json(GRID_B_JSON)
    rows = []
    for k, c in rs["cells"].items():
        e5 = c["e5"]
        if e5.get("defined"):
            rows.append((short(k), e5["delta"], e5["boot_ci_5_95_conditional"],
                         "A", e5.get("success"), k))
    for k, c in gb["cells"].items():
        e5 = c["e5"]
        if c.get("failed") or not e5.get("defined"):
            rows.append((short(k), None, None, "B", False, k))
        else:
            rows.append((short(k), e5["delta"], e5["boot_ci_5_95_conditional"],
                         "B", e5.get("success"), k))

    pooled = gb["bootstrap_aggregates"]["pooled"]["ci_5_95_alldef"]
    e5 = gb["E5"]
    assert_close("E5 count_of_12", e5["count_of_12"], 8)
    assert e5["decision"] == "WEAKEN", "E5 decision is %r, not WEAKEN" % e5["decision"]
    assert e5["guards"]["count_ge_10"] is False and e5["guards"]["families_ge_3of4"] is False
    assert_close("pooled CI lo", pooled[0], 0.12276666666666666, tol=1e-12)
    assert_close("pooled CI hi", pooled[1], 0.2012333333333333, tol=1e-12)
    n_defined_b = sum(1 for r in rows if r[3] == "B" and r[1] is not None)
    assert_close("grid-B evaluable", n_defined_b, gb["n_evaluable_of_12"])
    n_undef = sum(1 for r in rows if r[1] is None)
    assert_close("grid-B aborted cells", n_undef, 3)
    for nm, d, ci, g, ok, k in rows:                       # every point inside its own CI
        if d is not None:
            assert ci[0] <= d <= ci[1], "delta outside its CI for %s" % k

    if check_only:
        print("forest: assertions PASS (%d grid-A, %d grid-B defined, %d undefined)"
              % (8, n_defined_b, n_undef))
        return

    rows.sort(key=lambda r: (r[3], -(r[1] if r[1] is not None else -9)))
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ypos = np.arange(len(rows))[::-1]
    for y, (nm, d, ci, g, ok, k) in zip(ypos, rows):
        if d is None:
            ax.plot([0.0], [y], marker="x", ms=8, color="#999999", mew=1.8)
            why = ("js_no_bos undefined (BOS-sink)" if JS_DOMAIN_CELL in k
                   else "operational abort")
            ax.text(0.015, y, "  undefined — %s" % why, va="center",
                    fontsize=7.0, color="#777777")
            continue
        face = "#2166ac" if g == "B" else "white"
        ax.plot([ci[0], ci[1]], [y, y], color="#555555", lw=1.2, zorder=2)
        ax.plot([d], [y], marker="o", ms=7, mfc=face, mec="#2166ac", mew=1.6, zorder=3)
    ax.axvline(0.0, color="#999999", lw=0.9, ls=":")
    ax.axvspan(pooled[0], pooled[1], color="#b2182b", alpha=0.10, lw=0, zorder=1)
    ax.axvline(pooled[0], color="#b2182b", lw=0.9, ls="--")
    ax.axvline(pooled[1], color="#b2182b", lw=0.9, ls="--")
    labels = [("%s%s" % (nm, "  [FP8]" if FP8_CELL in k else "")) for nm, d, ci, g, ok, k in rows]
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlabel(r"cross-fitted terminal-dip magnitude  $\Delta_{cf}$")
    ax.set_title("Terminal-block give-back: hollow = grid A (discovery), filled = grid B (confirmation)",
                 fontsize=10, pad=9)
    ax.text(0.5, -0.115,
            "Registered outcome: E5 WEAKEN 8/12 — every evaluable grid-B cell but one shows the dip "
            "(p$_{grid}$=0.0005), yet the frozen bar of $\\geq$10/12 plus family bars was NOT met.\n"
            "Shaded band = pooled grid-B dip CI [0.123, 0.201]. Grid-A values are the cross-fitted "
            "rescore, not the registered grid-A E4 boolean.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig3_terminal_dip_forest")

def figure_peakfrac(check_only=False):
    """FIG4 -- peak fraction vs depth. The 'no rule established' figure.

    NO bootstrap CIs: peak_cf carries no interval on either grid, and grid A's
    .lstar_boot_ci belongs to a DIFFERENT estimator (the in-sample argmax), so
    splicing it onto peak_cf points would silently mix estimators. Instead the
    honest spread of the five cross-fitted training-fold peaks is drawn.
    """
    gb, rs = load_json(GRID_B_JSON), load_json(RESCORE_JSON)
    panel = gb["E1pp_panel"]
    pts = []
    for grid, sect, src in (("A", panel["grid_a_separate"], rs["cells"]),
                            ("B", panel["grid_b"], gb["cells"])):
        for k, v in sect.items():
            folds = src.get(k, {}).get("e5", {}).get("fold_peaks")
            pts.append((k, grid, int(v["N"]), float(v["frac"]), float(v["peak_cf"]), folds))

    for k, grid, N, frac, pk, folds in pts:               # frac must equal peak_cf / N
        assert_close("frac == peak_cf/N for %s" % k, round(pk / N, 4), frac, tol=5e-5)
        if folds:
            assert_close("peak_cf == median(fold_peaks) for %s" % k,
                         float(np.median(folds)), pk, tol=1e-9)
    assert_close("grid-A cells", sum(1 for p in pts if p[1] == "A"), 8)
    assert_close("grid-B cells", sum(1 for p in pts if p[1] == "B"), 9)

    if check_only:
        print("peakfrac: assertions PASS (%d points, no CIs plotted by design)" % len(pts))
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, grid, N, frac, pk, folds in pts:
        face = "#2166ac" if grid == "B" else "white"
        if folds:
            lo, hi = min(folds) / N, max(folds) / N
            ax.plot([N, N], [lo, hi], color="#bbbbbb", lw=1.0, zorder=1)
        ax.plot([N], [frac], marker="s" if FP8_CELL in k else "o", ms=7,
                mfc=face, mec="#2166ac", mew=1.6, zorder=3)
    ax.set_xlabel("depth N (decoder blocks)")
    ax.set_ylabel(r"peak fraction  $\ell^*_{cf}/N$")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Peak location obeys no transferable placement rule", fontsize=10.5, pad=9)
    ax.text(0.5, -0.19,
            "Hollow = grid A, filled = grid B, square = FP8-origin (Mistral-Medium-3.5). Grey bars are the "
            "spread of the five cross-fitted training-fold peaks,\nNOT a bootstrap CI — none is banked for "
            "this estimator. E1: no rule established (neither the absolute nor the relative frozen rule fired).",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig4_peak_fraction")

def figure_cliff(check_only=False):
    """FIG5 -- cliff onset. DISCOVERY ONLY.

    E6 was NOT TESTED: it was gatekept behind an E5 CONFIRM that did not occur.
    These are grid-A registered E2 values and carry no confirmatory status.
    halueval_qa/Llama-3.3-70B's J = 0.0000 is a STRUCTURAL empty-window fallback
    (l*=26 < ceil(0.5N)=40, so the jump-search segment is empty and depth_score
    returns 0.0) -- it is rendered as undefined, never as a measured zero.
    """
    res = load_json(os.path.join(LANE, "RESULTS.json"))
    rows = []
    for k, c in res["cells"].items():
        e2 = c["E2"]
        J, R, shape = e2["max_adjacent_jump"], e2["rise"], e2["shape"]
        structural = (J == 0.0 and c["lstar"] < int(np.ceil(0.5 * c["n_layers"])))
        rows.append((short(k), J, R, shape, structural, k))

    fallback = [r for r in rows if r[4]]
    assert len(fallback) == 1, "expected exactly 1 structural fallback, got %d" % len(fallback)
    assert "Llama-3.3-70B" in fallback[0][5] and "halueval" in fallback[0][5]
    assert_close("grid-A cells", len(rows), 8)
    for nm, J, R, shape, st, k in rows:                    # frozen E2 rule must reproduce the shape
        if st:
            continue
        want = "CLIFF" if (J >= 0.15 and J >= 0.5 * R) else ("GRADUAL" if J <= 0.08 else "MIXED")
        assert want == shape, "E2 shape mismatch for %s: rule says %s, json says %s" % (k, want, shape)

    if check_only:
        print("cliff: assertions PASS (8 cells, 1 structural fallback held out)")
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    mk = {"CLIFF": "o", "GRADUAL": "^", "MIXED": "D"}
    seen = set()
    for nm, J, R, shape, st, k in rows:
        if st:
            continue
        lab = shape if shape not in seen else None
        seen.add(shape)
        ax.plot([R], [J], marker=mk.get(shape, "o"), ms=8, mfc="#2166ac",
                mec="#14456f", mew=1.2, label=lab, zorder=3)
        ax.annotate(nm, (R, J), textcoords="offset points", xytext=(7, -3),
                    fontsize=6.8, color="#444444")
    lim = max(r[2] for r in rows if not r[4]) * 1.15
    xs = np.linspace(0, lim, 50)
    ax.plot(xs, 0.5 * xs, color="#b2182b", lw=1.0, ls="--", label=r"$J = 0.5R$ (cliff bar)")
    ax.axhline(0.15, color="#b2182b", lw=1.0, ls=":", label=r"$J \geq 0.15$ (cliff bar)")
    ax.set_xlabel("total rise  R"); ax.set_ylabel("max adjacent-block jump  J")
    ax.set_xlim(0, lim); ax.set_ylim(0, max(r[1] for r in rows) * 1.25)
    ax.set_title("Cliff onset — DISCOVERY ONLY (E6 was NOT TESTED)", fontsize=10.5, pad=9)
    ax.legend(fontsize=7.4, loc="upper left")
    ax.text(0.5, -0.185,
            "Grid A only. E6 was gatekept behind an E5 CONFIRM that did not occur, so no cliff endpoint was "
            "ever evaluated; these carry no confirmatory status.\nExcluded: halueval / Llama-3.3-70B, whose "
            "J = 0.0000 is a structural empty-window fallback (l*=26 below the search floor), not a measured "
            "absence of a jump.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color="#444444")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig5_cliff_onset")

_LABEL_CACHE = {}

def _labels(task):
    if task not in _LABEL_CACHE:
        _LABEL_CACHE[task] = load_task_labels(task)
    return _LABEL_CACHE[task]

def figure_grid(check_only=False):
    """FIG1 -- the full depth-curve grid.

    Grid A: banked in-sample sign-free curves WITH their registered shuffled-label
    envelope. Grid B: curves recomputed from the npz with the sealed scorer's own
    signfree_auroc_matrix, and deliberately UNSHADED -- no envelope was registered
    for grid-B slugs (MK decision A, 2026-08-24). The two grids are drawn in
    separate labelled blocks and never share a panel.
    """
    res, gb = load_json(os.path.join(LANE, "RESULTS.json")), load_json(GRID_B_JSON)
    panels = []
    for k, c in sorted(res["cells"].items()):
        panels.append((k, "A", np.asarray(c["curves_signfree_auroc"][PRIMARY], float),
                       np.asarray(c["envelope_q97_5_primary"], float)))
    for k in sorted(gb["cells"]):
        task, slug = k.split("/", 1)
        if gb["cells"][k].get("failed"):
            panels.append((k, "B", None, None))
            continue
        y_task = _labels(task)
        prim, L_exp, meta, _p = load_cell_b(NPZ_B, task, slug, y_task)
        curve = signfree_auroc_matrix(prim, y_task)
        panels.append((k, "B", np.asarray(curve, float), None))

    for k, g, cur, env in panels:                          # grid-A curves must match the artifact
        if g == "A":
            assert_close("grid-A peak %s" % k, cur.max(), res["cells"][k]["peak_auroc"], tol=1e-12)
            assert len(cur) == len(env) == res["cells"][k]["n_layers"]
        elif cur is not None:
            assert len(cur) == gb["cells"][k]["L"], "grid-B curve length != registered L for %s" % k
    assert_close("panels", len(panels), 20)
    assert_close("grid-B undefined panels", sum(1 for p in panels if p[2] is None), 3)

    if check_only:
        print("grid: assertions PASS (%d panels; grid-B curves recomputed via sealed scorer, unshaded)"
              % len(panels))
        return

    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.0, 2.1 * nrow), sharey=True)
    for ax, (k, g, cur, env) in zip(axes.ravel(), panels):
        if cur is None:
            ax.text(0.5, 0.5, "no npz\n(cell aborted)", ha="center", va="center",
                    fontsize=7.5, color="#999999", transform=ax.transAxes)
        else:
            x = np.arange(len(cur))
            if env is not None:
                ax.fill_between(x, 0.5, env, color="#c9c9c9", alpha=0.55, lw=0)
            ax.plot(x, cur, color="#1a1a1a" if g == "A" else "#2166ac", lw=1.1)
            ax.axhline(0.5, color="#bbbbbb", lw=0.6, ls=":")
        ax.set_title("[%s] %s%s" % (g, short(k), "  FP8" if FP8_CELL in k else ""),
                     fontsize=7.0, pad=3)
        ax.set_ylim(0.42, 0.95)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    fig.suptitle("Per-block sign-free AUROC, all cells  —  [A] grid A (discovery, shaded = registered "
                 "shuffled envelope)   |   [B] grid B (confirmation, UNSHADED: no envelope registered)",
                 fontsize=9, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    _save(fig, "fig1_depth_curve_grid")

def _save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIGDIR, "%s.%s" % (stem, ext))
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)

FIGURES.update({"forest": figure_forest, "peakfrac": figure_peakfrac,
                "cliff": figure_cliff, "grid": figure_grid})


# ---------------------------------------------------------------------------
# Tables. Emitted as LaTeX (booktabs) plus a plain-text twin for reading.
# ---------------------------------------------------------------------------
DASH = "--"   # a column that was NEVER SCORED for this grid, not a missing value

def _write_table(stem, tex, txt):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext, body in (("tex", tex), ("txt", txt)):
        p = os.path.join(FIGDIR, "%s.%s" % (stem, ext))
        with open(p, "w") as f:
            f.write(body)
        print("wrote", p)

def table_t1(check_only=False):
    """T1 -- per-cell verdict table.

    Grid B legitimately has EMPTY columns: peak AUROC, mid-region median, E2
    rise-shape and peak-location CI were never scored for grid B. They are
    printed as '--' meaning NOT SCORED, and the caption says so. Backfilling any
    of them would be new computation on confirmatory cells.
    """
    res, rs, gb = (load_json(os.path.join(LANE, "RESULTS.json")),
                   load_json(RESCORE_JSON), load_json(GRID_B_JSON))
    rows = []
    for k in sorted(res["cells"]):
        c, e5 = res["cells"][k], rs["cells"][k]["e5"]
        rows.append(dict(cell=short(k), grid="A", N=c["n_layers"], peak_blk=c["lstar"],
                         ci="[%d, %d]" % tuple(c["lstar_boot_ci"]),
                         peak="%.4f" % c["peak_auroc"], mid="%.3f" % c["mid_region_median"],
                         e2=c["E2"]["shape"],
                         dip=("%.4f" % e5["delta"]) if e5.get("defined") else DASH))
    pk = gb["E1pp_panel"]["grid_b"]
    for k in sorted(gb["cells"]):
        c = gb["cells"][k]
        e5 = c["e5"]
        rows.append(dict(cell=short(k), grid="B", N=c["L"],
                         peak_blk=(int(pk[k]["peak_cf"]) if k in pk else DASH),
                         ci=DASH, peak=DASH, mid=DASH, e2=DASH,
                         dip=("%.4f" % e5["delta"]) if e5.get("defined") else "undef."))
    assert_close("T1 rows", len(rows), 20)
    assert sum(1 for r in rows if r["dip"] == "undef.") == 3
    for r in rows:                       # grid B must never claim a column it never scored
        if r["grid"] == "B":
            assert r["peak"] == DASH and r["mid"] == DASH and r["e2"] == DASH and r["ci"] == DASH
    if check_only:
        print("t1: assertions PASS (20 rows; grid-B unscored columns held empty)")
        return

    hdr = ["cell", "grid", "N", "peak blk", "peak CI", "peak AUROC", "mid med", "E2", r"$\Delta_{cf}$"]
    keys = ["cell", "grid", "N", "peak_blk", "ci", "peak", "mid", "e2", "dip"]
    tex = ["\\begin{tabular}{llrrlllll}", "\\toprule", " & ".join(hdr) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(str(r[k]).replace("_", "\\_") for k in keys) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            "% NOTE: '--' in grid-B rows means NOT SCORED for grid B, not missing data.",
            "% Grid A peak blk = in-sample lstar; grid B peak blk = cross-fitted peak_cf (different estimator).",
            "% Delta_cf for grid A is the cross-fitted rescore, NOT the registered grid-A E4 boolean.",
            "% Mistral-Medium-3.5 is FP8-origin (dequantised to bf16)."]
    hdr_txt = [h.replace(r"$\Delta_{cf}$", "dip") for h in hdr]
    w = [max(len(str(r[k])) for r in rows + [dict(zip(keys, hdr_txt))]) for k in keys]
    txt = [" ".join(h.ljust(x) for h, x in zip(hdr_txt, w))]
    txt.append("-" * len(txt[0]))
    for r in rows:
        txt.append(" ".join(str(r[k]).ljust(x) for k, x in zip(keys, w)))
    txt += ["", "'--' = NOT SCORED for that grid (not missing data).",
            "grid A peak blk = in-sample lstar; grid B peak blk = cross-fitted peak_cf.",
            "Mistral-Medium-3.5 is FP8-origin."]
    _write_table("t1_verdict_table", "\n".join(tex) + "\n", "\n".join(txt) + "\n")

def table_t2(check_only=False):
    """T2 -- registered endpoint ledger: every frozen bar against its actual outcome."""
    gb = load_json(GRID_B_JSON)
    E5, E6 = gb["E5"], gb["E6"]
    assert E5["decision"] == "WEAKEN" and E5["count_of_12"] == 8
    assert E6["decision"].startswith("NOT TESTED") and E6["count_of_12"] == 2
    sens = gb["sensitivity"]
    rows = [
        ("E5", "terminal-block give-back",
         r"CONFIRM if $\geq$10/12 AND $\geq$3/4 in each of 3 families AND $p_{grid}<0.05$; "
         r"WEAKEN if 8--9/12; FALSIFY if $\leq$7/12",
         "8/12; families 4/2/2; $p_{grid}$=0.0005",
         "WEAKEN -- count and family bars both missed"),
        ("E6", "cliff onset",
         "gatekept on E5 == CONFIRM; then $\\geq$9/12 AND $\\geq$4/6 per task AND "
         r"$\geq$2/4 per family AND $p_{grid}<0.05$",
         "gate closed (E5 did not CONFIRM); descriptive 2/12",
         "NOT TESTED -- no cliff endpoint was ever evaluated"),
        ("E7", "cross-task peak distance", "descriptive-registered, no bar, no verdict vocabulary",
         "6 models reported", "descriptive"),
        ("E8", "Llama mid-stack band context", "descriptive-registered, no bar",
         "P8 hit (band replicates at 3.1-70B); P9 hit (absent at 8B)", "descriptive"),
        ("E1''", "peak-fraction panel", "descriptive-registered, no bar",
         "no transferable placement rule fired", "descriptive -- no rule established"),
    ]
    if check_only:
        print("t2: assertions PASS (E5 WEAKEN 8/12, E6 NOT TESTED, guards false)")
        return
    tex = ["\\begin{tabular}{llp{5.2cm}p{4.2cm}p{4.4cm}}", "\\toprule",
           "id & endpoint & frozen bar & actual & outcome \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}",
            "% Denominator is ALWAYS 12, including the 3 aborted cells.",
            "%% Sensitivity (registered): leave-Medium-out E5 %s; evaluable-only E5 %s."
            % (sens["leave_medium_out"]["e5_of_10"], sens["evaluable_only"]["e5"]),
            "% Reported as written: a registered miss is a miss."]
    txt = ["REGISTERED ENDPOINT LEDGER (denominator always 12, aborted cells included)", ""]
    for i, e, bar, act, out in rows:
        txt += ["%-5s %s" % (i, e), "      bar    : %s" % bar.replace("$", "").replace("\\geq", ">=").replace("\\leq", "<=").replace("_{grid}", "_grid").replace("--", "-"),
                "      actual : %s" % act.replace("$", "").replace("_{grid}", "_grid"),
                "      OUTCOME: %s" % out, ""]
    txt += ["Sensitivity (registered): leave-Medium-out E5 %s; evaluable-only E5 %s."
            % (sens["leave_medium_out"]["e5_of_10"], sens["evaluable_only"]["e5"]),
            "A registered miss is reported as written."]
    _write_table("t2_endpoint_ledger", "\n".join(tex) + "\n", "\n".join(txt) + "\n")

FIGURES.update({"t1": table_t1, "t2": table_t2})

if __name__ == "__main__":
    main()
