"""Figures and LaTeX tables for the paper, generated from the analysis outputs.

Usage (from the folder that contains 02_probability_axioms/):
    python -X utf8 02_probability_axioms/ccfa-workfiles/figures/paper-displays/source/make_displays.py

Inputs (read only). The paper's Qwen data come from the official Qwen3.8-27B weights (official_qwen/), analysed with
analyze.py:
    02_probability_axioms/official_qwen/results.json   analyze.py output (all tabulated numbers)
    02_probability_axioms/stimuli/items.jsonl          stimuli
    02_probability_axioms/official_qwen/logs/*.jsonl   raw request/response logs (the Jev log is a byte-identical
                                                       copy of logs/jev.jsonl), parsed with analyze.py
Outputs:
    02_probability_axioms/manuscript/figures/*.pdf
    02_probability_axioms/manuscript/tables/*.tex
    02_probability_axioms/official_qwen/results_posthoc.json  descriptive counts used in the text (no inference)
    <this folder>/../build/*.png                        previews for visual QA

Every per-pair value plotted here is taken from analyze.build_tables, and the script asserts that the plotted
pairs reproduce the point estimates in results.json before drawing anything.
"""
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

SRC = Path(__file__).resolve().parent
PROJ = SRC.parents[3]                                   # 02_probability_axioms/
FIG_DIR = PROJ / "manuscript" / "figures"
TAB_DIR = PROJ / "manuscript" / "tables"
BUILD = SRC.parent / "build"
for d in (FIG_DIR, TAB_DIR, BUILD):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------------------------- data
spec = importlib.util.spec_from_file_location("analyze", PROJ / "analyze.py")
an = importlib.util.module_from_spec(spec)
_argv, sys.argv = sys.argv, [str(PROJ / "analyze.py")]
spec.loader.exec_module(an)
sys.argv = _argv

DATA = PROJ / "official_qwen"
RES = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
STIM = an.load_stim()
_, JEV_LOG, _, _ = an._read_logs(DATA / "logs", "jev")
_, QWEN_LOG, _, _ = an._read_logs(DATA / "logs", "qwen_local")
V, _, _ = an.build_tables(STIM, JEV_LOG, QWEN_LOG)
ITEM = {it["id"]: it for it in STIM}


def pairs(va, vb):
    """Negation pairs (item, k) complete in both variants, in stimulus order (the analyze.py pairing rule)."""
    rows = []
    for it in STIM:
        pa, pb = V[va][it["id"]], V[vb][it["id"]]
        for k in range(3):
            if all(q in pa and q in pb for q in (f"S{k}", f"N{k}")):
                rows.append({"id": it["id"], "domain": it["domain"], "label": it["classes"][k],
                             "sa": pa[f"S{k}"], "na": pa[f"N{k}"], "sb": pb[f"S{k}"], "nb": pb[f"N{k}"]})
    return rows


def item_mad(rows, s, n):
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(abs(r[s] + r[n] - 1))
    return {i: float(np.mean(v)) for i, v in by.items()}


def check(x, y, what):
    assert abs(x - y) < 1e-9, f"{what}: plotted {x} != results.json {y}"


P_JQ = pairs("jev", "qwen")
P_JV = pairs("jev", "qwen_verbal")
check(np.mean(list(item_mad(P_JQ, "sa", "na").values())), RES["primary"]["jev"], "primary.jev")
check(np.mean(list(item_mad(P_JQ, "sb", "nb").values())), RES["primary"]["qwen"], "primary.qwen")
VRC = RES["verbal_readout_check"]["jev__vs__qwen_verbal"]["neg_mad"]
check(np.mean(list(item_mad(P_JV, "sb", "nb").values())), VRC["qwen_verbal"], "verbal qwen")
check(np.mean(list(item_mad(P_JV, "sa", "na").values())), VRC["jev"], "verbal jev")
assert len(P_JQ) == RES["primary"]["n_pairs"]


# ----------------------------------------------------------------------------------------------- post-hoc counts
def quadrants(rows, s, n):
    arr = np.array([[r[s], r[n]] for r in rows])
    return {"n_pairs": len(rows),
            "rejects_both": int(((arr[:, 0] < 0.5) & (arr[:, 1] < 0.5)).sum()),
            "accepts_both": int(((arr[:, 0] > 0.5) & (arr[:, 1] > 0.5)).sum()),
            "same_side": int(((arr[:, 0] - 0.5) * (arr[:, 1] - 0.5) > 0).sum()),
            "absdev_gt_0.10": int((np.round(np.abs(arr.sum(1) - 1), 6) > 0.10).sum())}


def pair_count(va, vb, keep=None):
    return sum(1 for r in pairs(va, vb) if keep is None or keep(r))


verbal_vals = [V["qwen_verbal"][i][q] for i in V["qwen_verbal"] for q in V["qwen_verbal"][i]]
jev_noul = [V["jev"][i][q] for i in V["jev"] for q in V["jev"][i] if not q.startswith("CH")]
top5 = Counter(verbal_vals).most_common(5)
neg_labels = set(RES["polarity_robustness"]["excluded_labels"].values())
POSTHOC = {
    "_note": ("Descriptive counts for the paper text, computed from the same parsed pairs as results.json by "
              "make_displays.py. Exploratory; no inference is based on them."),
    "quadrants": {"jev_native": quadrants(P_JQ, "sa", "na"),
                  "qwen_first_token": quadrants(P_JQ, "sb", "nb"),
                  "jev_native_on_verbal_pairs": quadrants(P_JV, "sa", "na"),
                  "qwen_verbalized": quadrants(P_JV, "sb", "nb")},
    "pair_counts": {
        "primary": len(P_JQ),
        "jev_avg_vs_qwen": pair_count("jev_avg", "qwen"),
        "jev_r1_vs_qwen": pair_count("jev_r1", "qwen"),
        "jev_vs_qwen_raw": pair_count("jev", "qwen_raw"),
        "jev_vs_qwen_mass05": pair_count("jev", "qwen_mass05"),
        "jev_vs_qwen_mass08": pair_count("jev", "qwen_mass08"),
        "jev_vs_qwen_verbal": len(P_JV),
        "polarity_robust": pair_count("jev", "qwen", lambda r: r["label"] not in neg_labels),
        "pqa_in_suite": pair_count("jev", "qwen", lambda r: ITEM[r["id"]]["in_jevals_suite"]),
        "pqa_out_of_suite": pair_count("jev", "qwen", lambda r: r["domain"] == "pqa"
                                       and not ITEM[r["id"]]["in_jevals_suite"]),
    },
    "qwen_verbal_values": {"n_values": len(verbal_vals),
                           "top5": [[float(v), c] for v, c in top5],
                           "share_top5": sum(c for _, c in top5) / len(verbal_vals),
                           "share_top2": sum(c for _, c in top5[:2]) / len(verbal_vals),
                           "n_distinct": len(set(verbal_vals))},
    "jev_noul_values": {"n_values": len(jev_noul), "min": float(min(jev_noul)), "max": float(max(jev_noul)),
                        "n_exact_0_or_1": int(sum(v in (0.0, 1.0) for v in jev_noul))},
}
for k, key in (("conf_matched_s", "conf_matched_s"), ("conf_matched_choice", "conf_matched_choice"),
               ("midrange_both", "midrange_both"), ("midrange_both_wide", "midrange_both_wide")):
    POSTHOC["pair_counts"][k] = RES["pair_level"][key]["n_pairs"]


def _label_means():
    out = {}
    for d_, l_ in (("nli", "entailment"), ("nli", "neutral"), ("nli", "contradiction"),
                   ("pqa", "affirmative"), ("pqa", "negative"), ("pqa", "inconclusive")):
        rr = [r for r in P_JQ if r["domain"] == d_ and r["label"] == l_]
        out[f"{d_}:{l_}"] = {"n_pairs": len(rr),
                             "jev_mean_s": float(np.mean([r["sa"] for r in rr])),
                             "jev_mean_n": float(np.mean([r["na"] for r in rr])),
                             "qwen_mean_s": float(np.mean([r["sb"] for r in rr])),
                             "qwen_mean_n": float(np.mean([r["nb"] for r in rr]))}
    return out


def _partition_sums(var):
    sums = [sum(V[var][it["id"]][f"S{k}"] for k in range(3)) for it in STIM
            if all(f"S{k}" in V[var][it["id"]] and f"S{k}" in V["jev" if var != "jev" else "qwen"][it["id"]]
                   for k in range(3))]
    sums = np.round(np.array(sums), 6)
    return {"n_items": int(len(sums)), "n_sum_gt_1": int((sums > 1).sum()), "n_sum_eq_1": int((sums == 1).sum()),
            "n_sum_lt_1": int((sums < 1).sum()), "mean_sum": float(sums.mean())}


def _reject_both_by_conf(rows, s, n):
    out = []
    for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.5001)):
        sel = [r for r in rows if lo <= round(abs(r[s] - 0.5), 6) < hi]
        out.append({"conf_bin": [lo, min(hi, 0.5)], "n_pairs": len(sel),
                    "rejects_both": sum(1 for r in sel if r[s] < 0.5 and r[n] < 0.5),
                    "accepts_both": sum(1 for r in sel if r[s] > 0.5 and r[n] > 0.5),
                    "rejects_both_rate": sum(1 for r in sel if r[s] < 0.5 and r[n] < 0.5) / len(sel)})
    return out


_all_jev_noul = [V[v][i][q] for v in ("jev_r0", "jev_r1", "jev_bundle") for i in V[v] for q in V[v][i]
                 if not q.startswith("CH")]
POSTHOC["label_means"] = _label_means()
POSTHOC["partition_sums"] = {"jev": _partition_sums("jev"), "qwen": _partition_sums("qwen")}
POSTHOC["quadrants_by_confidence"] = {"jev_native": _reject_both_by_conf(P_JQ, "sa", "na"),
                                      "qwen_first_token": _reject_both_by_conf(P_JQ, "sb", "nb")}
POSTHOC["jev_noul_values_all_requests"] = {"n_values": len(_all_jev_noul), "min": float(min(_all_jev_noul)),
                                           "max": float(max(_all_jev_noul)),
                                           "n_exact_0_or_1": int(sum(v in (0.0, 1.0) for v in _all_jev_noul))}
_ent = [it["reference"]["human_entropy"] for it in STIM if it["domain"] == "nli"]
POSTHOC["nli_human_entropy_bits"] = {"median": float(np.median(_ent)), "min": float(min(_ent)),
                                     "max": float(max(_ent))}
check(POSTHOC["partition_sums"]["jev"]["mean_sum"] - 1, RES["secondary_jev_vs_qwen"]["exh_signed"]["jev"],
      "exh_signed jev")
check(POSTHOC["partition_sums"]["qwen"]["mean_sum"] - 1, RES["secondary_jev_vs_qwen"]["exh_signed"]["qwen"],
      "exh_signed qwen")
assert POSTHOC["pair_counts"]["polarity_robust"] == RES["polarity_robustness"]["n_pairs"]

# ---------------------------------------------------------------------------------------------------------------
# Exploratory analyses added after the main analysis (independent review, round 1). Same stratified item bootstrap
# as analyze.py (5,000 resamples, seed 0) unless stated.
# ---------------------------------------------------------------------------------------------------------------
from scipy.stats import spearmanr  # noqa: E402

_DOMS = sorted({it["domain"] for it in STIM})


def _cluster_boot(item_ids, stat, n=an.NBOOT, seed=an.SEED):
    """Item-cluster bootstrap stratified by domain: stat(list of item ids) -> float."""
    r = np.random.default_rng(seed)
    by_dom = {d: [i for i in item_ids if ITEM[i]["domain"] == d] for d in _DOMS}
    out = []
    for _ in range(n):
        pick = []
        for d in _DOMS:
            ids = by_dom[d]
            if ids:
                pick += [ids[j] for j in r.integers(0, len(ids), size=len(ids))]
        out.append(stat(pick))
    return an.ci(np.asarray(out, float))


def form_offsets():
    """Mean probability per question form minus the value any coherent assignment implies when exactly one of three
    labels applies: 1/3 for S_k, 2/3 for N_k and D_k (averaged over the item's three labels). The same targets are
    the base rates, so the offsets are also calibration-in-the-large errors."""
    def off(p, forms):
        tgt = {"S": 1 / 3, "N": 2 / 3, "D": 2 / 3}
        return [float(np.mean([p[f"{f}{k}"] for k in range(3)]) - tgt[f]) for f in forms]
    out = {}
    specs = (("jev", "qwen", "SND", ("jev", "qwen")), ("jev", "qwen_verbal", "SN", ("jev_on_verbal", "qwen_verbal")))
    for va, vb, forms, names in specs:
        rows = []
        for it in STIM:
            pa, pb = an.restrict(V[va][it["id"]], V[vb][it["id"]])
            if all(f"{f}{k}" in pa for f in forms for k in range(3)):
                rows.append((it["domain"], off(pa, forms) + off(pb, forms)))
        cols = [f"{nm}_{f}" for nm in names for f in forms]
        for scope in ("all",) + tuple(_DOMS):
            sel = [(d, v) for d, v in rows if scope == "all" or d == scope]
            pt, bs = an.boot_means(np.array([v for _, v in sel]), [d for d, _ in sel])
            rec = out.setdefault(scope, {})
            rec[f"n_items_{vb}"] = len(sel)
            for j, c in enumerate(cols):
                rec[c], rec[c + "_ci"] = float(pt[j]), an.ci(bs[:, j])
    for scope in out:
        for pref in ("jev", "qwen", "jev_on_verbal", "qwen_verbal"):
            if f"{pref}_S" in out[scope]:
                out[scope][f"{pref}_mean_S"] = out[scope][f"{pref}_S"] + 1 / 3
    a, s = out["all"], RES["secondary_jev_vs_qwen"]
    for sysk in ("jev", "qwen"):                            # exact identities with the metrics in results.json
        check(3 * a[f"{sysk}_S"], s["exh_signed"][sysk], f"3*S offset {sysk}")
        check(a[f"{sysk}_S"] + a[f"{sysk}_N"], s["neg_signed"][sysk], f"S+N offset {sysk}")
        check(a[f"{sysk}_S"] + a[f"{sysk}_D"], s["dcomp_signed"][sysk], f"S+D offset {sysk}")
        check(a[f"{sysk}_D"] - a[f"{sysk}_N"], s["equiv_signed"][sysk], f"D-N offset {sysk}")
    return out


def verbal_rates():
    res, per = an.compare(V, "jev", "qwen_verbal", STIM, metrics=["neg_contra", "neg_mad", "neg_signed"])
    check(res["neg_mad"]["qwen_verbal"], VRC["qwen_verbal"], "verbal MAD via compare")
    q = POSTHOC["quadrants"]
    return {"same_side_rate": res["neg_contra"], "share_absdev_gt_0.10": {
        "qwen_verbal": q["qwen_verbalized"]["absdev_gt_0.10"] / q["qwen_verbalized"]["n_pairs"],
        "jev_on_verbal_pairs": q["jev_native_on_verbal_pairs"]["absdev_gt_0.10"]
        / q["jev_native_on_verbal_pairs"]["n_pairs"], "n_pairs": q["qwen_verbalized"]["n_pairs"]}}


def absdev_correlation():
    by = {}
    for r in P_JQ:
        by.setdefault(r["id"], []).append((abs(r["sa"] + r["na"] - 1), abs(r["sb"] + r["nb"] - 1)))
    ids = list(by)

    def rho(pick):
        xy = [v for i in pick for v in by[i]]
        return spearmanr([a for a, _ in xy], [b for _, b in xy])[0]
    signed = spearmanr([r["sa"] + r["na"] - 1 for r in P_JQ], [r["sb"] + r["nb"] - 1 for r in P_JQ])[0]
    check(signed, RES["pair_level"]["spearman_dev_jev_qwen"]["rho"], "signed dev spearman")
    return {"rho_absdev": float(rho(ids)), "ci_item_cluster": _cluster_boot(ids, rho), "n_pairs": len(P_JQ)}


def h7_difference():
    nli = [it["id"] for it in STIM if it["domain"] == "nli"]
    mj, mq = item_mad(P_JQ, "sa", "na"), item_mad(P_JQ, "sb", "nb")
    ent = {it["id"]: it["reference"]["human_entropy"] for it in STIM if it["domain"] == "nli"}

    def diff(pick):
        e = [ent[i] for i in pick]
        return spearmanr(e, [mq[i] for i in pick])[0] - spearmanr(e, [mj[i] for i in pick])[0]
    rj = spearmanr([ent[i] for i in nli], [mj[i] for i in nli])[0]
    rq = spearmanr([ent[i] for i in nli], [mq[i] for i in nli])[0]
    check(rj, RES["human_ambiguity_nli"]["jev"]["spearman_negmad_vs_entropy"], "H7 jev")
    check(rq, RES["human_ambiguity_nli"]["qwen"]["spearman_negmad_vs_entropy"], "H7 qwen")
    r = np.random.default_rng(an.SEED)
    bs = [diff([nli[j] for j in r.integers(0, len(nli), size=len(nli))]) for _ in range(an.NBOOT)]
    return {"rho_qwen_minus_jev": float(rq - rj), "ci": an.ci(np.asarray(bs)), "n_items": len(nli)}


BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.5001))


def _logit(p, clip):
    p = min(max(p, clip), 1 - clip)
    return math.log(p / (1 - p))


def confidence_bins():
    """Pair-level mean |s+n-1| and |logit s + logit n| per confidence bin |s-0.5|, with item-cluster bootstrap CIs
    (stratified by domain, 5,000 resamples, seed 0), vectorised through per-item bin sums."""
    ids = sorted({r["id"] for r in P_JQ})
    pos = {i: j for j, i in enumerate(ids)}
    out = {}
    for sysk, s, n in (("jev", "sa", "na"), ("qwen", "sb", "nb")):
        cnt = np.zeros((len(ids), 3))
        sab = np.zeros((len(ids), 3))
        slg = np.zeros((len(ids), 3))
        ssg = np.zeros((len(ids), 3))
        for r in P_JQ:
            conf = round(abs(r[s] - 0.5), 6)
            b = next(j for j, (lo, hi) in enumerate(BINS) if lo <= conf < hi)
            i = pos[r["id"]]
            cnt[i, b] += 1
            sab[i, b] += abs(r[s] + r[n] - 1)
            slg[i, b] += abs(_logit(r[s], 0.01) + _logit(r[n], 0.01))
            ssg[i, b] += r[s] + r[n] - 1
        rng_ = np.random.default_rng(an.SEED)
        draws = np.concatenate([np.array([pos[i] for i in ids if ITEM[i]["domain"] == d])[
            rng_.integers(0, sum(ITEM[i]["domain"] == d for i in ids),
                          size=(an.NBOOT, sum(ITEM[i]["domain"] == d for i in ids)))] for d in _DOMS], axis=1)
        bc, ba, bl = cnt[draws].sum(axis=1), sab[draws].sum(axis=1), slg[draws].sum(axis=1)
        bsg = ssg[draws].sum(axis=1)
        res = []
        for j, (lo, hi) in enumerate(BINS):
            res.append({"conf_bin": [lo, min(hi, 0.5)], "n_pairs": int(cnt[:, j].sum()),
                        "absdev": float(sab[:, j].sum() / cnt[:, j].sum()), "absdev_ci": an.ci(ba[:, j] / bc[:, j]),
                        "logitdev": float(slg[:, j].sum() / cnt[:, j].sum()), "logitdev_ci": an.ci(bl[:, j] / bc[:, j]),
                        "signeddev": float(ssg[:, j].sum() / cnt[:, j].sum()), "signeddev_ci": an.ci(bsg[:, j] / bc[:, j])})
            check(res[-1]["absdev"], RES["pair_level"][f"{sysk}_by_confidence"][j]["mean_abs_dev"], f"bin {sysk}{j}")
        out[sysk] = res
    return out


def logit_sensitivity():
    out = {}
    for sysk, s, n in (("jev", "sa", "na"), ("qwen", "sb", "nb")):
        vals = [r[s] for r in P_JQ] + [r[n] for r in P_JQ]
        out[sysk] = {"share_at_clip_bound_0.01": float(np.mean([v <= 0.01 or v >= 0.99 for v in vals])),
                     "mean_distance_from_half": float(np.mean([abs(v - 0.5) for v in vals]))}
    for clip in (0.01, 0.02, 0.05):
        rows = {}
        for r in P_JQ:
            rows.setdefault(r["id"], []).append((abs(_logit(r["sa"], clip) + _logit(r["na"], clip)),
                                                 abs(_logit(r["sb"], clip) + _logit(r["nb"], clip))))
        ids = list(rows)
        M = np.array([[np.mean([a for a, _ in rows[i]]), np.mean([b for _, b in rows[i]])] for i in ids])
        pt, bs = an.boot_means(M, [ITEM[i]["domain"] for i in ids])
        out[f"clip_{clip}"] = {"jev": float(pt[0]), "jev_ci": an.ci(bs[:, 0]), "qwen": float(pt[1]),
                               "qwen_ci": an.ci(bs[:, 1]), "diff": float(pt[0] - pt[1]),
                               "diff_ci": an.ci(bs[:, 0] - bs[:, 1])}
    check(out["clip_0.01"]["qwen"], RES["secondary_jev_vs_qwen"]["neg_logit"]["qwen"], "logit clip 0.01")
    return out


def mass08_identities():
    mets = ["dcomp_mad", "dcomp_signed", "equiv_abs", "equiv_signed", "disj_abs", "disj_signed"]
    res, _ = an.compare(V, "jev", "qwen_mass08", STIM, metrics=mets)
    n_dpairs = 0
    for it in STIM:
        pa, pb = an.restrict(V["jev"][it["id"]], V["qwen_mass08"][it["id"]])
        n_dpairs += sum(1 for k in range(3) if f"S{k}" in pa and f"D{k}" in pa)
    return {"metrics": res, "n_sd_pairs": n_dpairs}


POSTHOC["review_round1"] = {
    "_note": "Exploratory; added after the main analysis.",
    "form_offsets": form_offsets(),
    "verbal_rates": verbal_rates(),
    "absdev_correlation": absdev_correlation(),
    "h7_difference": h7_difference(),
    "confidence_bins": confidence_bins(),
    "logit_sensitivity": logit_sensitivity(),
    "qwen_mass08_identities": mass08_identities(),
}
(DATA / "results_posthoc.json").write_text(json.dumps(POSTHOC, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8")

# ----------------------------------------------------------------------------------------------- style
C_JEV, C_QLP, C_QVB = "#0072B2", "#D55E00", "#CC79A7"
INK, MUTED, GRID = "#222222", "#666666", "#D9D9D9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})
TEXTWIDTH = 6.5


def save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(BUILD / f"{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def fmt_ci(d, key_pt, key_ci, nd=3):
    lo, hi = d[key_ci]
    return f"{d[key_pt]:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


# ----------------------------------------------------------------------------------------------- figure 1
def _pairs_word(n):
    return f"{n} pair" if n == 1 else f"{n} pairs"


def fig_negation_scatter():
    panels = [
        ("(a) Jev, native", P_JQ, "sa", "na", C_JEV,
         fmt_ci(RES["primary"], "jev", "jev_ci"), POSTHOC["quadrants"]["jev_native"]),
        ("(b) Qwen3.8-27B, first-token", P_JQ, "sb", "nb", C_QLP,
         fmt_ci(RES["primary"], "qwen", "qwen_ci"), POSTHOC["quadrants"]["qwen_first_token"]),
        ("(c) Qwen3.8-27B, verbalized", P_JV, "sb", "nb", C_QVB,
         fmt_ci(VRC, "qwen_verbal", "qwen_verbal_ci"), POSTHOC["quadrants"]["qwen_verbalized"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.75), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    area = 2.2                                              # marker area (pt^2) per pair, same in every panel
    for ax, (title, rows, s, n, col, mad, quad) in zip(axes, panels):
        for x0 in (0.0, 0.5):                               # quadrants where both answers point the same way
            ax.add_patch(Rectangle((x0, x0), 0.5, 0.5, facecolor="#F2F2F2", edgecolor="none", zorder=0))
        xs = np.linspace(-0.05, 1.05, 111)
        ax.fill_between(xs, 0.9 - xs, 1.1 - xs, color=GRID, alpha=0.6, lw=0, zorder=1)
        ax.plot([0, 1], [1, 0], color=INK, lw=0.7, zorder=2)
        cnt = Counter((round(r[s], 2), round(r[n], 2)) for r in rows)
        pts = np.array(sorted(([x, y, c] for (x, y), c in cnt.items()), key=lambda t: -t[2]))
        ax.scatter(pts[:, 0], pts[:, 1], s=area * pts[:, 2], c=col, marker="o", alpha=0.5,
                   edgecolors=col, linewidths=0.4, zorder=3)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, 1.06)
        ticks = [0, 0.25, 0.5, 0.75, 1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_aspect("equal")
        pt, ci = mad.split(" [")
        ax.text(0.0, 1.135, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5, color=INK)
        ax.text(0.0, 1.075, f"MAD {pt} [{ci}, {_pairs_word(quad['n_pairs'])}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.2, color=INK)
        ax.text(0.0, 1.015, f"reject both: {quad['rejects_both']}; accept both: {quad['accepts_both']}",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2, color=MUTED)
        ax.set_xlabel("P(yes) for “label is $X_k$”")
    axes[0].set_ylabel("P(yes) for “label is not $X_k$”")
    handles = [Line2D([], [], ls="none", label="pairs per point:")]
    handles += [plt.scatter([], [], s=area * c, c=MUTED, alpha=0.5, edgecolors=MUTED, linewidths=0.4,
                            label=str(c)) for c in (1, 10, 100)]
    handles.append(Line2D([0], [0], color=INK, lw=0.7, label=r"coherent, $s_k+n_k=1$"))
    handles.append(Rectangle((0, 0), 1, 1, facecolor=GRID, alpha=0.6, edgecolor="none",
                             label=r"$|s_k+n_k-1|\leq 0.1$"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.035),
               handletextpad=0.35, columnspacing=1.1, handlelength=1.4)
    save(fig, "fig_negation_scatter")


# ----------------------------------------------------------------------------------------------- figure 2
LABELS = [("nli", "entailment"), ("nli", "neutral"), ("nli", "contradiction"),
          ("pqa", "affirmative"), ("pqa", "negative"), ("pqa", "inconclusive")]


def fig_by_label():
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.25))
    fig.subplots_adjust(wspace=0.28)
    x = np.arange(len(LABELS))
    for ax, metric, ylab in ((axes[0], "neg_mad", r"mean $|s_k+n_k-1|$"),
                             (axes[1], "neg_signed", r"mean $(s_k+n_k-1)$")):
        for off, sysk, col, mk, name in ((-0.13, "jev", C_JEV, "o", "Jev (native)"),
                                         (0.13, "qwen", C_QLP, "s", "Qwen3.8-27B (first-token)")):
            pt = [RES["by_domain_label"][f"{d}:{l}"][metric][sysk] for d, l in LABELS]
            ci = np.array([RES["by_domain_label"][f"{d}:{l}"][metric][f"{sysk}_ci"] for d, l in LABELS])
            ax.errorbar(x + off, pt, yerr=[np.array(pt) - ci[:, 0], ci[:, 1] - np.array(pt)], fmt=mk,
                        color=col, ms=4.2, mfc=col, mec="white", mew=0.4, elinewidth=0.8, capsize=0,
                        label=name, zorder=3)
        ax.axvline(2.5, color=GRID, lw=0.8, zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([l + ("$^\\dagger$" if RES["by_domain_label"][f"{d}:{l}"]["neg_polarity"] else "")
                            for d, l in LABELS], rotation=28, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylab)
        ax.set_xlim(-0.5, 5.5)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylim(0, 0.62)
    axes[1].axhline(0, color=INK, lw=0.6, zorder=1)
    axes[1].set_ylim(-0.62, 0.22)
    for ax, t in zip(axes, ("(a) Violation magnitude", "(b) Signed deviation")):
        ax.set_title(t, loc="left", pad=13)
        ax.text(1.0, 1.01, "NLI", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.5,
                color=MUTED)
        ax.text(4.0, 1.01, "PubMedQA", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=7.5, color=MUTED)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.19))
    save(fig, "fig_by_label")


# ----------------------------------------------------------------------------------------------- figure 3
def fig_confidence_ambiguity():
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    fig.subplots_adjust(wspace=0.42)
    cb = POSTHOC["review_round1"]["confidence_bins"]
    names = [r"$[0,.2)$", r"$[.2,.4)$", r"$[.4,.5]$"]
    x = np.arange(3)
    w = 0.36
    for ax, key, ylab, title, ymax in ((axes[0], "absdev", r"mean $|s_k+n_k-1|$", "(a) Probability scale", 0.44),
                                       (axes[1], "logitdev", r"mean $|\mathrm{logit}\,s_k+\mathrm{logit}\,n_k|$",
                                        "(b) Logit scale", 3.6)):
        for off, sysk, col, hatch, name in ((-w / 2, "jev", C_JEV, None, "Jev (native)"),
                                            (w / 2, "qwen", C_QLP, "////", "Qwen3.8-27B (first-token)")):
            b = cb[sysk]
            vals = np.array([e[key] for e in b])
            cis = np.array([e[key + "_ci"] for e in b])
            ax.bar(x + off, vals, width=w, color=col, alpha=0.85 if hatch is None else 0.25,
                   edgecolor=col, hatch=hatch, lw=0.6, label=name, zorder=2)
            ax.errorbar(x + off, vals, yerr=[vals - cis[:, 0], cis[:, 1] - vals], fmt="none", ecolor=INK,
                        elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=3)
            if key == "absdev":
                for xi, e, hi in zip(x + off, b, cis[:, 1]):
                    ax.text(xi, hi + 0.008, f"{e['n_pairs']}", ha="center", va="bottom", fontsize=6.3,
                            color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel(r"confidence $|s_k-0.5|$")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", pad=3)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.36, -0.17),
               handlelength=1.4)

    ax = axes[2]
    nli = [it for it in STIM if it["domain"] == "nli"]
    mj = item_mad(P_JQ, "sa", "na")
    mq = item_mad(P_JQ, "sb", "nb")
    ent = np.array([it["reference"]["human_entropy"] for it in nli])
    for sysk, mm, col, mk, name in (("jev", mj, C_JEV, "o", "Jev"), ("qwen", mq, C_QLP, "s", "Qwen3.8-27B")):
        y = np.array([mm[it["id"]] for it in nli])
        ax.scatter(ent, y, s=9, c=col, marker=mk, alpha=0.6, edgecolors="white", linewidths=0.3, zorder=3)
        h = RES["human_ambiguity_nli"][sysk]
        order = np.argsort(ent)
        k = 25                                              # running mean over 25 items, display aid only
        run = np.convolve(y[order], np.ones(k) / k, mode="valid")
        ax.plot(ent[order][k // 2: k // 2 + len(run)], run, color=col, lw=1.1, zorder=4)
        ax.text(1.0, 1.0 if sysk == "qwen" else 0.9,
                rf"{name}: $\rho$ = {h['spearman_negmad_vs_entropy']:.2f} [{h['ci'][0]:.2f}, {h['ci'][1]:.2f}]",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color=col)
    ax.set_xlabel("human label entropy (bits)")
    ax.set_ylabel(r"item mean $|s_k+n_k-1|$")
    ax.set_ylim(-0.02, 0.95)
    ax.set_xlim(0.2, 1.6)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("(c) Human disagreement (NLI)", loc="left", pad=3)
    save(fig, "fig_confidence_ambiguity")
    return float(ent.min()), float(ent.max())


# ----------------------------------------------------------------------------------------------- tables
def m(x, nd=3, sign=False):
    """Number in math mode with a true minus sign."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return "$" + s.replace("-", "-") + "$"


def mci(ci, nd=3):
    return "{\\scriptsize$[" + f"{ci[0]:.{nd}f}, {ci[1]:.{nd}f}" + "]$}"


def cell(x, ci=None, nd=3, sign=False):
    return m(x, nd, sign) + ("" if ci is None else " " + mci(ci, nd))


def stack(x, ci, nd=3, sign=False):
    return "\\ci{" + m(x, nd, sign) + "}{" + (f"{ci[0]:.{nd}f}, {ci[1]:.{nd}f}") + "}"


HEADER = ("% Generated by ccfa-workfiles/figures/paper-displays/source/make_displays.py from official_qwen/results.json.\n"
          "% Do not edit by hand; rerun the script.\n"
          "\\providecommand{\\ci}[2]{\\begin{tabular}[t]{@{}c@{}}#1\\\\[-1.5pt]{\\scriptsize$[#2]$}\\end{tabular}}\n")


def write(name, body):
    (TAB_DIR / f"{name}.tex").write_text(HEADER + body, encoding="utf-8")


def tab_negation():
    p, sec, dom = RES["primary"], RES["secondary_jev_vs_qwen"], RES["by_domain"]
    pl = RES["pair_level"]
    rows = []
    for label, sysk in (("Jev (native)", "jev"), ("Qwen3.8-27B (first-token)", "qwen")):
        rows.append(" & ".join([
            label,
            stack(p[sysk], p[f"{sysk}_ci"]),
            stack(dom["nli"]["neg_mad"][sysk], dom["nli"]["neg_mad"][f"{sysk}_ci"]),
            stack(dom["pqa"]["neg_mad"][sysk], dom["pqa"]["neg_mad"][f"{sysk}_ci"]),
            stack(sec["neg_signed"][sysk], sec["neg_signed"][f"{sysk}_ci"], sign=True),
            stack(sec["neg_contra"][sysk], sec["neg_contra"][f"{sysk}_ci"]),
        ]) + " \\\\")
    rows.append(" & ".join([
        "Difference (Jev $-$ Qwen)",
        stack(p["diff"], p["diff_ci"], sign=True),
        stack(dom["nli"]["neg_mad"]["diff"], dom["nli"]["neg_mad"]["diff_ci"], sign=True),
        stack(dom["pqa"]["neg_mad"]["diff"], dom["pqa"]["neg_mad"]["diff_ci"], sign=True),
        stack(sec["neg_signed"]["diff"], sec["neg_signed"]["diff_ci"], sign=True),
        stack(sec["neg_contra"]["diff"], sec["neg_contra"]["diff_ci"], sign=True),
    ]) + " \\\\")
    vr = RES["verbal_readout_check"]["jev__vs__qwen_verbal"]
    vss = POSTHOC["review_round1"]["verbal_rates"]["same_side_rate"]
    verbal = " & ".join([
        "Qwen3.8-27B (verbalized)",
        stack(vr["neg_mad"]["qwen_verbal"], vr["neg_mad"]["qwen_verbal_ci"]),
        "--", "--",
        stack(vr["neg_signed"]["qwen_verbal"], vr["neg_signed"]["qwen_verbal_ci"], sign=True),
        "\\ci{" + m(vss["qwen_verbal"]) + "}{" + f"{vss['qwen_verbal_ci'][0]:.3f}, "
        f"{vss['qwen_verbal_ci'][1]:.3f}" + "}",
    ]) + " \\\\"
    body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            " & \\multicolumn{3}{c}{MAD, mean $|s_k+n_k-1|$} & Signed deviation & Same-side \\\\\n"
            "\\cmidrule(lr){2-4}\n"
            "System & All (160 items) & NLI (100) & PubMedQA (60) & mean $(s_k+n_k-1)$ & rate \\\\\n"
            "\\midrule\n" + "\n".join(rows[:2]) + "\n\\midrule\n" + rows[2] + "\n\\midrule\n" + verbal +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_negation", body)


def tab_robustness():
    pl, rb, pr = RES["pair_level"], RES["robustness"], RES["primary"]
    pc = POSTHOC["pair_counts"]

    def row(name, n_items, n_pairs, j, q, diff, ci, nd=3):
        return " & ".join([name, f"{n_items} ({n_pairs})", m(j, nd), m(q, nd),
                           cell(diff, ci, nd, sign=True)]) + " \\\\"

    vr = RES["verbal_readout_check"]["jev__vs__qwen_verbal"]["neg_mad"]
    po = RES["polarity_robustness"]["neg_mad"]
    ja = rb["jev_avg__vs__qwen"]["neg_mad"]
    lines = [
        row("Main comparison", pr["n_items"], pc["primary"], pr["jev"], pr["qwen"], pr["diff"],
            pr["diff_ci"]),
        row("Jev repeats averaged", ja["n_items"], pc["jev_avg_vs_qwen"],
            ja["jev_avg"], ja["qwen"], ja["diff"], ja["diff_ci"]),
        row("Confidence-matched pairs", pl["conf_matched_s"]["n_items"],
            pl["conf_matched_s"]["n_pairs"], pl["conf_matched_s"]["jev"], pl["conf_matched_s"]["qwen"],
            pl["conf_matched_s"]["diff"], pl["conf_matched_s"]["diff_ci"]),
        row("Qwen verbalized readout", vr["n_items"], pc["jev_vs_qwen_verbal"], vr["jev"],
            vr["qwen_verbal"], vr["diff"], vr["diff_ci"]),
        row("Double-negation labels removed", po["n_items"], pc["polarity_robust"], po["jev"],
            po["qwen"], po["diff"], po["diff_ci"]),
    ]
    rep = [
        row("Choice-matched pairs", pl["conf_matched_choice"]["n_items"],
            pl["conf_matched_choice"]["n_pairs"], pl["conf_matched_choice"]["jev"],
            pl["conf_matched_choice"]["qwen"], pl["conf_matched_choice"]["diff"],
            pl["conf_matched_choice"]["diff_ci"]),
        row("Both $s_k\\in[0.1, 0.9]$", pl["midrange_both"]["n_items"], pl["midrange_both"]["n_pairs"],
            pl["midrange_both"]["jev"], pl["midrange_both"]["qwen"], pl["midrange_both"]["diff"],
            pl["midrange_both"]["diff_ci"]),
        row("Both $s_k\\in[0.05, 0.95]$", pl["midrange_both_wide"]["n_items"],
            pl["midrange_both_wide"]["n_pairs"], pl["midrange_both_wide"]["jev"], pl["midrange_both_wide"]["qwen"],
            pl["midrange_both_wide"]["diff"], pl["midrange_both_wide"]["diff_ci"]),
    ]
    for name, key, sysk, pck in (("Qwen probabilities not rounded", "jev__vs__qwen_raw", "qwen_raw", "jev_vs_qwen_raw"),
                                 ("Qwen readouts with mass $\\ge 0.5$", "jev__vs__qwen_mass05", "qwen_mass05",
                                  "jev_vs_qwen_mass05"),
                                 ("Qwen readouts with mass $\\ge 0.8$", "jev__vs__qwen_mass08", "qwen_mass08",
                                  "jev_vs_qwen_mass08")):
        d = rb[key]["neg_mad"]
        rep.append(row(name, d["n_items"], pc[pck], d["jev"], d[sysk], d["diff"], d["diff_ci"]))
    d = rb["jev_r1__vs__qwen"]["neg_mad"]
    rep.append(row("Jev second repeat instead of first", d["n_items"], pc["jev_r1_vs_qwen"], d["jev_r1"],
                   d["qwen"], d["diff"], d["diff_ci"]))
    d = RES["secondary_jev_vs_qwen"]["neg_logit"]
    rep.append(row("Logit scale", d["n_items"], pc["primary"], d["jev"], d["qwen"], d["diff"], d["diff_ci"], nd=2))
    for name, key, pck in (("PubMedQA items in the Jevals suite", "pqa_in_suite", "pqa_in_suite"),
                           ("PubMedQA items outside the suite", "pqa_out_of_suite", "pqa_out_of_suite")):
        d = RES["jevals_suite_subsets"][key]["neg_mad"]
        rep.append(row(name, d["n_items"], pc[pck], d["jev"], d["qwen"], d["diff"], d["diff_ci"]))
    body = ("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
            "Comparison & Items (pairs) & Jev & Qwen & Difference [95\\% CI] \\\\\n\\midrule\n"
            "\\multicolumn{5}{@{}l}{\\emph{Main comparison and confound checks}} \\\\\n"
            + "\n".join(lines) + "\n\\midrule\n"
            "\\multicolumn{5}{@{}l}{\\emph{Further checks}} \\\\\n"
            + "\n".join(rep) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_robustness", body)


def tab_identities():
    s = RES["secondary_jev_vs_qwen"]
    p = RES["primary"]

    def r(ident, stat, j, jci, q, qci, d, dci, sign=False, nd=3, pct=False):
        f = 100.0 if pct else 1.0
        tail = "\\%" if pct else ""
        return " & ".join([ident, stat,
                           m(f * j, 1 if pct else nd, sign) + tail + " " + mci([f * x for x in jci], 1 if pct else nd),
                           m(f * q, 1 if pct else nd, sign) + tail + " " + mci([f * x for x in qci], 1 if pct else nd),
                           m(f * d, 1 if pct else nd, True) + tail + " " + mci([f * x for x in dci], 1 if pct else nd),
                           ]) + " \\\\"

    L = []
    L.append(r("Negation", "$|s_k+n_k-1|$", p["jev"], p["jev_ci"], p["qwen"], p["qwen_ci"],
               p["diff"], p["diff_ci"]))
    d = s["neg_signed"]
    L.append(r("", "$s_k+n_k-1$", d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"], d["diff"], d["diff_ci"], True))
    for ident, a, b, sa, sb in (
            ("Disjunctive complement", "dcomp_mad", "dcomp_signed", "$|s_k+d_k-1|$", "$s_k+d_k-1$"),
            ("Partition additivity", "exh_abs", "exh_signed", "$|\\sum_k s_k-1|$", "$\\sum_k s_k-1$"),
            ("Disjunction additivity", "disj_abs", "disj_signed", "$|d_k-s_i-s_j|$", "$d_k-s_i-s_j$"),
            ("Unpacking", "equiv_abs", "equiv_signed", "$|d_k-n_k|$", "$d_k-n_k$")):
        L.append("\\addlinespace[2pt]")
        d = s[a]
        L.append(r(ident, sa, d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"], d["diff"], d["diff_ci"]))
        d = s[b]
        L.append(r("", sb, d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"], d["diff"], d["diff_ci"], True))
    L.append("\\addlinespace[2pt]")
    d = s["mono_mag"]
    L.append(r("Monotonicity", "$\\max(0, s_m-q)$", d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"],
               d["diff"], d["diff_ci"]))
    d = s["mono_rate"]
    L.append(r("", "share $>0.05$", d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"], d["diff"], d["diff_ci"],
               pct=True))
    L.append("\\addlinespace[2pt]")
    d = s["xtype_abs"]
    L.append(r("Cross-format", "$|s_k-c_k|$", d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"], d["diff"],
               d["diff_ci"]))
    d = s["xtype_norm_abs"]
    L.append(r("", "$|s_k/\\sum_j s_j-c_k|$", d["jev"], d["jev_ci"], d["qwen"], d["qwen_ci"], d["diff"],
               d["diff_ci"]))
    body = ("\\begin{tabular}{@{}llccc@{}}\n\\toprule\n"
            "Identity & Statistic & Jev & Qwen (first-token) & Difference \\\\\n\\midrule\n"
            + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_identities", body)


def tab_by_label():
    L = []
    for d_, l_ in LABELS:
        e = RES["by_domain_label"][f"{d_}:{l_}"]
        L.append(" & ".join([
            ("NLI" if d_ == "nli" else "PubMedQA") if l_ in ("entailment", "affirmative") else "",
            l_ + ("$^\\dagger$" if e["neg_polarity"] else ""),
            m(e["neg_mad"]["jev"]), m(e["neg_mad"]["qwen"]), cell(e["neg_mad"]["diff"], e["neg_mad"]["diff_ci"],
                                                                   sign=True),
            m(e["neg_signed"]["jev"], sign=True), m(e["neg_signed"]["qwen"], sign=True),
            m(e["equiv_signed"]["jev"], sign=True), m(e["equiv_signed"]["qwen"], sign=True),
            m(e["neg_logit"]["jev"], 2), m(e["neg_logit"]["qwen"], 2),
        ]) + " \\\\")
        if l_ == "contradiction":
            L.append("\\midrule")
    body = ("\\begin{tabular}{@{}llcccccccc@{}}\n\\toprule\n"
            " & & \\multicolumn{3}{c}{$|s_k+n_k-1|$} & \\multicolumn{2}{c}{$s_k+n_k-1$} & "
            "\\multicolumn{2}{c}{$d_k-n_k$} & \\multicolumn{2}{c}{logit MAD} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\n"
            "Domain & Label $X_k$ & Jev & Qwen & Difference [95\\% CI] & Jev & Qwen & Jev & Qwen & Jev & Qwen \\\\\n"
            "\\midrule\n" + "\n".join(L) + "\n\\bottomrule\n\\end{tabular}\n")
    body = body.replace("@{}llcccccccc@{}", "@{}llccccccccc@{}")
    write("tab_by_label", body)


def tab_readout():
    c, lat, mass, nf = RES["calls"], RES["latency_s"], RES["qwen_readout_mass"], RES["noise_floor"]
    L = [
        ("Jev calls (single-question + bundled)", f"{c['jev_records']:,} ({c['jev_ok']:,} succeeded)".replace(",", "{,}")),
        ("Qwen first-token calls", f"{c['qwen_planned_keys']:,} ({c['qwen_keys_parsed']:,} parsed)".replace(",", "{,}")),
        ("Qwen verbalized calls", f"{c['qwen_verbal_planned_keys']} ({c['qwen_verbal_keys_parsed']} parsed)"),
        ("Jev total cost", f"\\${c['jev_cost_usd_all_records']:.4f}"),
        ("Jev latency, median / p95 (s)", f"{lat['jev_median']:.2f} / {lat['jev_p95']:.2f}"),
        (("Qwen server compute per call, mean (s)", f"{lat['qwen_server_s_mean']:.2f}")
         if lat["qwen_server_s_mean"] is not None else
         ("Qwen first-token latency, median / p95 (s)", f"{lat['qwen_median']:.2f} / {lat['qwen_p95']:.2f}")),
        ("Qwen label mass, median: all / S / N / D / choice",
         " / ".join(f"{mass[k]['median']:.3f}" for k in ("all", "form_S", "form_N", "form_D", "form_CH"))),
        ("Qwen label mass $<0.8$: all / S / N / D / choice",
         " / ".join(f"{100 * mass[k]['frac_lt_0.8']:.1f}\\%" for k in ("all", "form_S", "form_N", "form_D", "form_CH"))),
        ("Jev repeat noise: mean $|p^{(1)}-p^{(0)}|$; identical share",
         f"{nf['jev']['mean_abs_repeat_diff']:.4f}; {100 * nf['jev']['frac_identical']:.1f}\\%"),
        ("Jev noise floor $\\mathrm{NF}$ [95\\% CI]",
         f"{nf['jev']['nf_used']:.4f} [{nf['jev']['nf_neg_ci'][0]:.4f}, {nf['jev']['nf_neg_ci'][1]:.4f}]"),
        ("Qwen repeat noise (20 items): mean $|p^{(1)}-p^{(0)}|$; identical share",
         f"{nf['qwen']['mean_abs_repeat_diff']:.4f}; {100 * nf['qwen']['frac_identical']:.1f}\\%"),
        ("Qwen noise floor $\\mathrm{NF}$ [95\\% CI]",
         f"{nf['qwen']['nf_neg']:.4f} [{nf['qwen']['nf_neg_ci'][0]:.4f}, {nf['qwen']['nf_neg_ci'][1]:.4f}]"),
    ]
    body = ("\\begin{tabular}{@{}ll@{}}\n\\toprule\nQuantity & Value \\\\\n\\midrule\n" +
            "\n".join(f"{a} & {b} \\\\" for a, b in L) + "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_readout", body)


def tab_offsets():
    """Post-hoc: mean probability per question form minus its coherent (and base-rate) value, 1/3 or 2/3."""
    fo = POSTHOC["review_round1"]["form_offsets"]
    rows = []
    for label, pref, form, tgt in (("Jev (native)", "jev", "S", "1/3"), ("", "jev", "N", "2/3"),
                                   ("", "jev", "D", "2/3"),
                                   ("Qwen (first-token)", "qwen", "S", "1/3"), ("", "qwen", "N", "2/3"),
                                   ("", "qwen", "D", "2/3"),
                                   ("Qwen (verbalized)", "qwen_verbal", "S", "1/3"),
                                   ("", "qwen_verbal", "N", "2/3")):
        key = f"{pref}_{form}"
        qtext = {"S": "is $X_k$", "N": "is not $X_k$", "D": "is $X_i$ or $X_j$"}[form]
        cells = [cell(fo[sc][key], fo[sc][key + "_ci"], sign=True) for sc in ("all", "nli", "pqa")]
        rows.append(" & ".join([label, f"${form}_k$ ({qtext})", f"${tgt}$"] + cells) + " \\\\")
        if form == "D" or (pref == "qwen_verbal" and form == "N"):
            rows.append("\\addlinespace[2pt]")
    n_v = fo["all"]["n_items_qwen_verbal"]
    body = ("\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n"
            "System & Question form & Target & All & NLI & PubMedQA \\\\\n\\midrule\n" + "\n".join(rows[:-1]) +
            "\n\\bottomrule\n\\end{tabular}\n")
    write("tab_offsets", body)
    return n_v


if __name__ == "__main__":
    fig_negation_scatter()
    fig_by_label()
    ent_range = fig_confidence_ambiguity()
    tab_negation()
    tab_robustness()
    tab_identities()
    tab_by_label()
    tab_readout()
    tab_offsets()
    print("entropy range (bits):", ent_range)
    print("posthoc:", json.dumps(POSTHOC["quadrants"]), json.dumps(POSTHOC["pair_counts"]))
    print("wrote figures to", FIG_DIR, "and tables to", TAB_DIR)
