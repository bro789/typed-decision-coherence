"""02_probability_axioms: analysis from logs ONLY (no API calls). Deterministic. Writes results.json.

Usage (from project root):
  python -X utf8 02_probability_axioms/analyze.py                       # real logs -> results.json
  python -X utf8 02_probability_axioms/analyze.py --logs-dir D --out F  # any logs dir / output file
  python -X utf8 02_probability_axioms/analyze.py --selftest            # fabricated logs in a TEMP dir, checks, deletes

Reads: stimuli/items.jsonl, <logs>/jev.jsonl, <logs>/qwen_local.jsonl (pilot_*.jsonl are never read),
and (bookkeeping only) prereg_snapshot/FREEZE_RECORD.json.

Notation per item (3 exclusive + exhaustive labels X0,X1,X2):
  s_k = P(yes | S_k "label is X_k"),  n_k = P(yes | N_k "label is not X_k"),
  d_k = P(yes | D_k "label is X_i or X_j" (i,j != k)),  c_k = choice probability of X_k.
PRIMARY (H1): neg_mad_i = mean_k |s_k + n_k - 1|; MAD_neg = mean over items (equal weight), stratified
item bootstrap (by domain, 5000 resamples, seed 0). Jev = first repeat r0 (fallback r1); Qwen = logprob
readout rounded to 0.01 (quantisation matched to Jev). Pairs enter only if complete in BOTH systems.
GO/NO-GO (PREREG section 6): GO = F and S and D and P.
  F  parseable-answer rate, Qwen mass, Jev noise floor, paper scale (Qwen time from SERVER-side timings, R2-S4)
  S  S-a (Jev meaningfully incoherent) or S-b (Qwen meaningfully incoherent AND Jev's MAD below Qwen's AND Jev's
     S answers informative). S-b no longer requires Jev's CI upper bound < 0.05: that condition made the rule
     NON-MONOTONE (a Jev result strictly better than one that passes S-a could fail both branches). Whether the
     report may call Jev "coherent" (point and 95% CI upper bound < 0.05, R2-S2) is the report-only field
     decision.jev_coherent_claim.
  D  D1 (gap >= 0.03, ratio >= 1.5, CI excludes 0) and D1_avg (gap survives averaging Jev's two repeats) and
     D2 (confidence-confound guard, R2-M1: level 1 = CONFIDENCE-MATCHED pairs |s_J - s_Q| <= 0.10 (>= 30 items);
     level 2 = logit scale; "not evaluable" never passes) and D3 (verbal readout of Qwen on all 160 items gives the
     same direction, R2-S1)
  P  negative-polarity robustness (R2-M2): without the PQA 'negative' and NLI 'contradiction' pairs the S branch
     that passed still holds and Delta keeps its sign with |Delta| >= 0.03.
All threshold comparisons use values rounded to EPS_ROUND decimals (R2-S5). Mid-range subsets are report-only.
"""
import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _common import stats  # noqa: E402
from _common.logio import latest_by_key, read_jsonl  # noqa: E402

QIDS = ["S0", "N0", "D0", "S1", "N1", "D1", "S2", "N2", "D2", "CH"]
NOUL = [q for q in QIDS if q != "CH"]
REPEAT_QIDS = ["S0", "N0", "S1", "N1", "S2", "N2"]
NBOOT, SEED = 5000, 0
MONO_TOL = 0.05
CLIP = 0.01
# ---- pre-registered decision thresholds (PREREG.md section 6)
TH = {"success_rate": 0.95, "qwen_mass_median": 0.80, "jev_noise_floor_max": 0.02,
      "paper_cost_usd_max": 50.0, "paper_hours_max": 24.0, "paper_jev_calls": 30000, "paper_qwen_calls": 15000,
      "Sa_point": 0.05, "Sa_lo": 0.03, "Sa_noise_mult": 3.0,
      "Sb_qwen_point": 0.05, "Sb_qwen_lo": 0.03, "coherent_claim_below": 0.05,
      "Sb_jev_acc_soft": 0.41, "Sb_qwen_noise_mult": 3.0,
      "nf_identical_frac": 0.95, "nf_prior": 0.007,
      "D_abs_diff": 0.03, "D_ratio": 1.5, "D_ratio_floor": 0.01, "D1_avg_abs_diff": 0.03,
      "D2_match_tol": 0.10, "D2_match_min_items": 30, "D2_match_abs_diff": 0.03,
      "D3_verbal_min_items": 30,
      "P_neg_polarity_labels": {"nli": "contradiction", "pqa": "negative"}, "P_abs_diff": 0.03}
MID_WINDOWS = {"midrange_both": (0.10, 0.90), "midrange_both_wide": (0.05, 0.95)}   # report only since R2-M1
RETRY = "|retry1"                       # run.py re-asks a Jev key ONCE under this suffix if its answer is unparseable
EPS_ROUND = 6                           # every gating statistic is rounded to 6 decimals before comparison (R2-S5)


def R(x):
    """R2-S5: statistics built from 0.01-grid values are rounded before any threshold / sign comparison."""
    return round(float(x), EPS_ROUND)


def load_stim():
    return [json.loads(l) for l in (HERE / "stimuli" / "items.jsonl").read_text(encoding="utf-8").splitlines() if l]


def planned_keys(stim):
    """-> (jev keys, qwen logprob-decide keys, qwen verbal keys). F1 is computed on the first two only."""
    jev = {f"{it['id']}|{q}|r{r}" for r in (0, 1) for it in stim for q in QIDS} | {f"{it['id']}|bundle|r0" for it in stim}
    qwen = {f"{it['id']}|{q}|r0" for it in stim for q in QIDS} | \
           {f"{it['id']}|{q}|r1" for it in stim if it["qwen_repeat"] for q in REPEAT_QIDS}
    verbal = {f"{it['id']}|{q}|verbal" for it in stim for q in REPEAT_QIDS}          # R2-S1: all items
    return jev, qwen, verbal


# ------------------------------------------------------------------------------------------ parsing
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and 0.0 <= v <= 1.0


def parse_jev_answer(ans, qtype, classes):
    """-> float (noul) | dict class->p (choice) | None (format anomaly)"""
    if not isinstance(ans, dict):
        return None
    if qtype == "noul":
        v = ans.get("noul")
        return float(v) if _num(v) else None
    pr = ans.get("probabilities")
    if not isinstance(pr, dict) or not all(c in pr and _num(pr[c]) for c in classes):
        return None
    return {c: float(pr[c]) for c in classes}


def _jev_ans(rec, qid):
    return ((rec.get("response") or {}).get("answers") or {}).get(qid) if rec is not None else None


def build_tables(stim, jev_recs, qwen_recs):
    """Returns P[variant][item_id][qid], parse bookkeeping, and the sets of planned keys with a PARSEABLE answer
    (F1). A Jev key whose answer is unparseable falls back to its '|retry1' record (run.py) if that one parses."""
    V = {v: {it["id"]: {} for it in stim} for v in
         ("jev", "jev_r0", "jev_r1", "jev_avg", "jev_bundle", "qwen", "qwen_raw", "qwen_r1", "qwen_mass05",
          "qwen_mass08", "qwen_verbal")}
    anomalies, fallbacks, retry_used = Counter(), 0, Counter()
    jev_parsed, qwen_parsed, verbal_parsed = set(), set(), set()
    for it in stim:
        iid, cls = it["id"], it["classes"]
        for qid in QIDS:
            qt = it["questions"][qid]["type"]
            got = {}
            for r in (0, 1):
                key = f"{iid}|{qid}|r{r}"
                rec = jev_recs.get(key)
                if rec is None:
                    continue
                val = parse_jev_answer(_jev_ans(rec, "q"), qt, cls)
                if val is None:
                    anomalies["single"] += 1
                    val = parse_jev_answer(_jev_ans(jev_recs.get(key + RETRY), "q"), qt, cls)
                    if val is None:
                        continue
                    retry_used["single"] += 1
                got[r] = val
                jev_parsed.add(key)
                V[f"jev_r{r}"][iid][qid] = val
            if 0 in got:
                V["jev"][iid][qid] = got[0]
            elif 1 in got:
                V["jev"][iid][qid] = got[1]
                fallbacks += 1
            if 0 in got and 1 in got:
                V["jev_avg"][iid][qid] = ((got[0] + got[1]) / 2 if qt == "noul"
                                          else {c: (got[0][c] + got[1][c]) / 2 for c in cls})
            elif got:
                V["jev_avg"][iid][qid] = got.get(0, got.get(1))
        key = f"{iid}|bundle|r0"
        rec = jev_recs.get(key)
        if rec is not None:
            n_ok = 0
            for oq, qid in it["bundle_qids"].items():        # R2-S7: answers come back under the opaque ids
                qt = it["questions"][qid]["type"]
                val = parse_jev_answer(_jev_ans(rec, oq), qt, cls)
                if val is None:
                    anomalies["bundle"] += 1
                    val = parse_jev_answer(_jev_ans(jev_recs.get(key + RETRY), oq), qt, cls)
                    if val is None:
                        continue
                    retry_used["bundle"] += 1
                V["jev_bundle"][iid][qid] = val
                n_ok += 1
            if n_ok == len(QIDS):
                jev_parsed.add(key)
        for r, (vr, vraw) in ((0, ("qwen", "qwen_raw")), (1, ("qwen_r1", None))):
            for qid in QIDS:
                key = f"{iid}|{qid}|r{r}"
                rec = qwen_recs.get(key)
                if rec is None:
                    continue
                qt = it["questions"][qid]["type"]
                if qt == "noul":
                    raw = rec.get("p_yes")
                    if not _num(raw):
                        anomalies["qwen"] += 1
                        continue
                    val, rawv = round(float(raw), 2), float(raw)
                else:
                    pr = rec.get("probs") or {}
                    if not all(c in pr and _num(pr[c]) for c in cls):
                        anomalies["qwen"] += 1
                        continue
                    val, rawv = {c: round(float(pr[c]), 2) for c in cls}, {c: float(pr[c]) for c in cls}
                qwen_parsed.add(key)
                V[vr][iid][qid] = val
                if vraw:
                    V[vraw][iid][qid] = rawv
                    m = rec.get("mass", 0)
                    if m >= 0.5:
                        V["qwen_mass05"][iid][qid] = val
                    if m >= 0.8:
                        V["qwen_mass08"][iid][qid] = val
        for qid in REPEAT_QIDS:                     # verbal (JSON-stated) readout, all items: gating check D3 (R2-S1)
            key = f"{iid}|{qid}|verbal"
            rec = qwen_recs.get(key)
            if rec is None:
                continue
            raw = rec.get("p_yes")
            if _num(raw):
                V["qwen_verbal"][iid][qid] = round(float(raw), 2)
                verbal_parsed.add(key)
            else:
                anomalies["qwen_verbal"] += 1
    return V, {"format_anomalies": dict(anomalies), "jev_r0_missing_used_r1": fallbacks,
               "jev_retry_used": dict(retry_used)}, (jev_parsed, qwen_parsed, verbal_parsed)


# ------------------------------------------------------------------------------------------ item metrics
def _logit(p):
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))


def _argmax_acc(vals, ref_idx):
    m = max(vals)
    tied = [k for k, v in enumerate(vals) if v == m]
    return (1.0 / len(tied)) if ref_idx in tied else 0.0


METRICS = ["neg_mad", "neg_signed", "neg_contra", "neg_logit", "dcomp_mad", "dcomp_signed", "exh_abs", "exh_signed",
           "disj_abs", "disj_signed", "mono_mag", "mono_rate", "equiv_abs", "equiv_signed", "xtype_abs",
           "xtype_norm_abs", "ch_sum_absdev", "s_acc", "s_acc_soft", "ch_acc"]


def _argmax_soft(vals, it):
    """S3: NLI -> human_dist[argmax_k s_k] (ties averaged; expected accuracy under the ChaosNLI human label
    distribution); PQA (no human distribution) -> hard accuracy against final_decision."""
    ref = it["reference"]
    if "human_dist" not in ref:
        return _argmax_acc(vals, it["classes"].index(ref["label"]))
    m = max(vals)
    tied = [k for k, v in enumerate(vals) if v == m]
    return float(np.mean([ref["human_dist"][it["classes"][k]] for k in tied]))


def item_metrics(p, it):
    """p: qid -> value for ONE item/system (already restricted to the common available qids)."""
    out = {m: float("nan") for m in METRICS}
    s = {k: p.get(f"S{k}") for k in range(3)}
    n = {k: p.get(f"N{k}") for k in range(3)}
    d = {k: p.get(f"D{k}") for k in range(3)}
    ch = p.get("CH")
    cls = it["classes"]
    dev = [s[k] + n[k] - 1 for k in range(3) if s[k] is not None and n[k] is not None]
    if dev:
        ks = [k for k in range(3) if s[k] is not None and n[k] is not None]
        out["neg_mad"] = float(np.mean(np.abs(dev)))
        out["neg_signed"] = float(np.mean(dev))
        out["neg_contra"] = float(np.mean([(s[k] - 0.5) * (n[k] - 0.5) > 0 for k in ks]))
        out["neg_logit"] = float(np.mean([abs(_logit(s[k]) + _logit(n[k])) for k in ks]))
    dd = [s[k] + d[k] - 1 for k in range(3) if s[k] is not None and d[k] is not None]
    if dd:
        out["dcomp_mad"] = float(np.mean(np.abs(dd)))
        out["dcomp_signed"] = float(np.mean(dd))
    if all(s[k] is not None for k in range(3)):
        e = sum(s.values()) - 1
        out["exh_abs"], out["exh_signed"] = abs(e), e
        ref = cls.index(it["reference"]["label"])
        out["s_acc"] = _argmax_acc([s[k] for k in range(3)], ref)
        out["s_acc_soft"] = _argmax_soft([s[k] for k in range(3)], it)
    da = []
    for k in range(3):
        i, j = [x for x in range(3) if x != k]
        if d[k] is not None and s[i] is not None and s[j] is not None:
            da.append(d[k] - s[i] - s[j])
    if da:
        out["disj_abs"], out["disj_signed"] = float(np.mean(np.abs(da))), float(np.mean(da))
    viol = []
    for k in range(3):
        for m in range(3):
            if m == k or s[m] is None:
                continue
            for cons in (d[k], n[k]):            # X_m  =>  (X_i or X_j) = D_k  and  X_m => not X_k = N_k
                if cons is not None:
                    viol.append(max(0.0, s[m] - cons))
    if viol:
        out["mono_mag"] = float(np.mean(viol))
        out["mono_rate"] = float(np.mean([round(v, EPS_ROUND) > MONO_TOL for v in viol]))   # S7: 0.01-grid safe
    eq = [d[k] - n[k] for k in range(3) if d[k] is not None and n[k] is not None]
    if eq:
        out["equiv_abs"], out["equiv_signed"] = float(np.mean(np.abs(eq))), float(np.mean(eq))
    if ch is not None:
        out["ch_sum_absdev"] = abs(sum(ch.values()) - 1)
        ref = cls.index(it["reference"]["label"])
        out["ch_acc"] = _argmax_acc([ch[c] for c in cls], ref)
        xt = [abs(s[k] - ch[cls[k]]) for k in range(3) if s[k] is not None]
        if xt:
            out["xtype_abs"] = float(np.mean(xt))
        if all(s[k] is not None for k in range(3)) and sum(s.values()) > 0:
            tot = sum(s.values())
            out["xtype_norm_abs"] = float(np.mean([abs(s[k] / tot - ch[cls[k]]) for k in range(3)]))
    return out


# ------------------------------------------------------------------------------------------ bootstrap
def boot_means(M, doms, n=NBOOT, seed=SEED):
    """M: (N, m) values; stratified (by domain) item bootstrap of column means. Returns (point (m,), bs (n, m))."""
    M = np.asarray(M, float)
    r = np.random.default_rng(seed)
    N = len(M)
    sums = np.zeros((n, M.shape[1]))
    for dom in sorted(set(doms)):
        idx = np.flatnonzero(np.asarray(doms) == dom)
        draw = r.integers(0, len(idx), size=(n, len(idx)))
        sums += M[idx][draw].sum(axis=1)
    return M.mean(axis=0), sums / N


def ci(bs):
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def restrict(pa, pb):
    """Keep only qids available in both systems for this item (symmetric complete-case)."""
    keys = set(pa) & set(pb)
    return {k: pa[k] for k in keys}, {k: pb[k] for k in keys}


def keep_pairs(V, stim, keep, variants=("jev", "qwen")):
    """Copy of V[variants] that keeps, per item, only the S_k / N_k / D_k questions with keep(item, k) True
    (CH dropped). Used for the per-label breakdown and the negative-polarity robustness gate P (R2-M2)."""
    return {v: {it["id"]: {q: x for q, x in V[v][it["id"]].items() if q != "CH" and keep(it, int(q[1]))}
                for it in stim} for v in variants}


def neg_polarity(it, k):
    """R2-M2: True for the label whose meaning is the 'false / no' answer (PQA 'negative', NLI 'contradiction'):
    its N_k question is a double negative."""
    return it["classes"][k] == TH["P_neg_polarity_labels"][it["domain"]]


def compare(V, a, b, stim, metrics=METRICS, domain=None):
    """Per metric: items where both systems have the metric (after symmetric restriction), means, CIs, diff a-b."""
    per = {m: ([], [], []) for m in metrics}
    for it in stim:
        if domain and it["domain"] != domain:
            continue
        pa, pb = restrict(V[a][it["id"]], V[b][it["id"]])
        ma, mb = item_metrics(pa, it), item_metrics(pb, it)
        for m in metrics:
            if not (math.isnan(ma[m]) or math.isnan(mb[m])):
                per[m][0].append(ma[m])
                per[m][1].append(mb[m])
                per[m][2].append(it["domain"])
    out = {}
    for m in metrics:
        xa, xb, dm = per[m]
        if not xa:
            out[m] = {"n_items": 0}
            continue
        pt, bs = boot_means(np.column_stack([xa, xb]), dm)
        out[m] = {"n_items": len(xa), a: float(pt[0]), f"{a}_ci": ci(bs[:, 0]), b: float(pt[1]),
                  f"{b}_ci": ci(bs[:, 1]), "diff": float(pt[0] - pt[1]), "diff_ci": ci(bs[:, 0] - bs[:, 1])}
    return out, per


def noise_floor(V, v0, v1, stim):
    """Repeat noise (identical requests). nf_i = mean_k |(s0-s1)+(n0-n1)|/sqrt(2): the |s+n-1| a perfectly coherent
    system would show from repeat noise alone. Also mean |p0-p1| over all noul qids present in both repeats."""
    nf, ab, ident, dm = [], [], [], []
    for it in stim:
        p0, p1 = V[v0][it["id"]], V[v1][it["id"]]
        ks = [k for k in range(3) if all(q in p0 and q in p1 for q in (f"S{k}", f"N{k}"))]
        if not ks:
            continue
        nf.append(np.mean([abs((p0[f"S{k}"] - p1[f"S{k}"]) + (p0[f"N{k}"] - p1[f"N{k}"])) / math.sqrt(2) for k in ks]))
        qs = [q for q in NOUL if q in p0 and q in p1]
        ab.append(np.mean([abs(p0[q] - p1[q]) for q in qs]))
        ident.append(np.mean([abs(p0[q] - p1[q]) < 1e-9 for q in qs]))
        dm.append(it["domain"])
    if not nf:
        return {"n_items": 0}
    pt, bs = boot_means(np.column_stack([nf, ab, ident]), dm)
    return {"n_items": len(nf), "nf_neg": float(pt[0]), "nf_neg_ci": ci(bs[:, 0]),
            "mean_abs_repeat_diff": float(pt[1]), "mean_abs_repeat_diff_ci": ci(bs[:, 1]),
            "frac_identical": float(pt[2])}


def bundle_vs_single(V, stim):
    """Jev only: mean |p_bundle - p_r0| per item vs repeat noise mean |p_r1 - p_r0| (same noul qids)."""
    xb, xr, dm = [], [], []
    for it in stim:
        pb, p0, p1 = V["jev_bundle"][it["id"]], V["jev_r0"][it["id"]], V["jev_r1"][it["id"]]
        qs = [q for q in NOUL if q in pb and q in p0 and q in p1]
        if not qs:
            continue
        xb.append(np.mean([abs(pb[q] - p0[q]) for q in qs]))
        xr.append(np.mean([abs(p1[q] - p0[q]) for q in qs]))
        dm.append(it["domain"])
    if not xb:
        return {"n_items": 0}
    pt, bs = boot_means(np.column_stack([xb, xr]), dm)
    return {"n_items": len(xb), "bundle_vs_r0": float(pt[0]), "bundle_vs_r0_ci": ci(bs[:, 0]),
            "r1_vs_r0": float(pt[1]), "r1_vs_r0_ci": ci(bs[:, 1]), "diff": float(pt[0] - pt[1]),
            "diff_ci": ci(bs[:, 0] - bs[:, 1])}


def pair_level(V, stim):
    """Common complete (S_k, N_k) pairs of jev and qwen: confidence bins, confidence-matched subsets (D2 level 1,
    R2-M1), report-only mid-range subsets, agreement."""
    rows = []
    for it in stim:
        pj, pq = V["jev"][it["id"]], V["qwen"][it["id"]]
        for k in range(3):
            if all(q in pj and q in pq for q in (f"S{k}", f"N{k}")):
                cj = pj["CH"][it["classes"][k]] if "CH" in pj else None
                cq = pq["CH"][it["classes"][k]] if "CH" in pq else None
                rows.append((it["id"], it["domain"], pj[f"S{k}"], pj[f"N{k}"], pq[f"S{k}"], pq[f"N{k}"], cj, cq))
    out = {"n_pairs": len(rows)}
    if not rows:
        return out
    sj, nj, sq, nq = (np.array([r[i] for r in rows]) for i in (2, 3, 4, 5))
    dj, dq = sj + nj - 1, sq + nq - 1
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.5001)]
    for name, s, dv in (("jev", sj, dj), ("qwen", sq, dq)):
        conf = np.round(np.abs(s - 0.5), EPS_ROUND)                # S7: 0.01-grid safe binning
        adv = np.round(np.abs(dv), EPS_ROUND)
        out[f"{name}_by_confidence"] = [
            {"conf_bin": [lo, min(hi, 0.5)], "n_pairs": int(((conf >= lo) & (conf < hi)).sum()),
             "mean_abs_dev": float(adv[(conf >= lo) & (conf < hi)].mean()) if ((conf >= lo) & (conf < hi)).any() else None}
            for lo, hi in bins]
        out[f"{name}_frac_pairs_absdev_gt_0.10"] = float((adv > 0.10).mean())
        out[f"{name}_frac_pairs_absdev_gt_0.20"] = float((adv > 0.20).mean())
    r, lo, hi = stats.corr_ci(sj, sq, seed=SEED)
    out["spearman_s_jev_qwen"] = {"rho": r, "ci": [lo, hi], "n": len(sj)}
    r, lo, hi = stats.corr_ci(dj, dq, seed=SEED)
    out["spearman_dev_jev_qwen"] = {"rho": r, "ci": [lo, hi], "n": len(dj)}
    # D2 level 1 (R2-M1, GATING): confidence-matched pairs, |s_k^J - s_k^Q| <= 0.10 (both systems equally confident
    # about X_k, so a gap there cannot be "the more confident system looks more coherent")
    tol = TH["D2_match_tol"]
    out["conf_matched_s"] = _pair_subset(rows, lambda w: R(abs(w[2] - w[4])) <= tol, rule=f"|s_J - s_Q| <= {tol}")
    # report only: matched on the separate choice question, and the old mid-range windows (both s_k in [lo, hi])
    out["conf_matched_choice"] = _pair_subset(rows, lambda w: w[6] is not None and w[7] is not None
                                              and R(abs(w[6] - w[7])) <= tol, rule=f"|c_J - c_Q| <= {tol}")
    for name, (lo_, hi_) in MID_WINDOWS.items():
        out[name] = _pair_subset(rows, lambda w, lo_=lo_, hi_=hi_: lo_ <= R(w[2]) <= hi_ and lo_ <= R(w[4]) <= hi_,
                                 rule=f"both s_k in [{lo_}, {hi_}]")
    return out


def _pair_subset(rows, pred, rule):
    """Pairs satisfying pred, |dev| averaged within item first, then items (equal weight); paired bootstrap."""
    sub = {}
    for w in rows:
        if pred(w):
            iid, dom, a, b, c, d = w[:6]
            sub.setdefault(iid, (dom, [], []))
            sub[iid][1].append(abs(a + b - 1))
            sub[iid][2].append(abs(c + d - 1))
    if not sub:
        return {"rule": rule, "n_items": 0, "n_pairs": 0}
    ids = sorted(sub)
    M = np.array([[np.mean(sub[i][1]), np.mean(sub[i][2])] for i in ids])
    pt, bs = boot_means(M, [sub[i][0] for i in ids])
    return {"rule": rule, "n_items": len(ids), "n_pairs": sum(len(sub[i][1]) for i in ids),
            "jev": float(pt[0]), "jev_ci": ci(bs[:, 0]), "qwen": float(pt[1]), "qwen_ci": ci(bs[:, 1]),
            "diff": float(pt[0] - pt[1]), "diff_ci": ci(bs[:, 0] - bs[:, 1])}


# ------------------------------------------------------------------------------------------ bookkeeping
def _read_log(path):
    """-> (all records, key -> last OK record, n unparseable lines). Uses _common.logio; only if that raises on a
    truncated line (S5: process killed mid-write) does it fall back to a tolerant reader that skips and counts it."""
    try:
        return read_jsonl(path), latest_by_key(path), 0
    except json.JSONDecodeError:
        recs, bad = [], 0
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    recs.append(json.loads(ln))
                except json.JSONDecodeError:
                    bad += 1
        latest = {}
        for r in recs:
            if r.get("ok") and "key" in r:
                latest[r["key"]] = r
        return recs, latest, bad


def _read_logs(logs_dir, stem):
    """<stem>.jsonl followed by run.py's continuation files <stem>_c1.jsonl, _c2, ... (written only if a file got a
    truncated line). -> (all records in file order, key -> last OK record, n unparseable lines, file names)."""
    logs_dir = Path(logs_dir)
    files = [p for p in [logs_dir / f"{stem}.jsonl"] + sorted(logs_dir.glob(f"{stem}_c*.jsonl"),
                                                             key=lambda p: int(p.stem.rsplit("_c", 1)[1]))
             if p.exists()]
    recs, latest, bad = [], {}, 0
    for p in files:
        r_, l_, b_ = _read_log(p)
        recs += r_
        latest.update(l_)
        bad += b_
    return recs, latest, bad, [p.name for p in files]


def _cost(r):
    c = ((r.get("response") or {}).get("usage") or {}).get("cost")
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) else None


def _ms(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def bookkeeping(stim, all_jev, all_qwen, jev_recs, qwen_recs, server_recs, V, parsed, bad_lines):
    pj, pq, pv = planned_keys(stim)
    jev_parsed, qwen_parsed, verbal_parsed = parsed
    okj = {r["key"] for r in all_jev if r.get("ok")}
    okq = {r["key"] for r in all_qwen if r.get("ok")}
    retry_keys = {k for k in okj | {r["key"] for r in all_jev} if k.endswith(RETRY)}
    started = [r["started_at"] for r in all_jev + all_qwen if r.get("started_at")]
    latj = [r["latency_s"] for r in all_jev if r.get("ok") and r.get("latency_s") is not None]
    latq = [r["latency_s"] for r in all_qwen if r.get("ok") and r.get("latency_s") is not None
            and r.get("mode") != "verbal"]
    # R2-S4: server-side compute time of the logprob decide calls (llama-server timings, prompt_ms + predicted_ms);
    # unlike the wall-clock latency it excludes queueing behind other experiments on the shared 4-slot server
    srvq = []
    for r in all_qwen:
        t = r.get("timings") if (r.get("ok") and r.get("mode") != "verbal") else None
        if isinstance(t, dict) and _ms(t.get("prompt_ms")) is not None:
            srvq.append((_ms(t["prompt_ms"]) + (_ms(t.get("predicted_ms")) or 0.0)) / 1000.0)
    cost_all = [c for c in (_cost(r) for r in all_jev) if c is not None]
    cost_ok_raw = [_cost(r) for r in all_jev if r.get("ok")]
    cost_ok = [c for c in cost_ok_raw if c is not None]          # S10a: missing cost is NOT counted as 0
    bk = {
        "calls": {
            "jev_records": len(all_jev), "jev_ok": sum(bool(r.get("ok")) for r in all_jev),
            "jev_fail": sum(not r.get("ok") for r in all_jev),
            "qwen_records": len(all_qwen), "qwen_ok": sum(bool(r.get("ok")) for r in all_qwen),
            "qwen_fail": sum(not r.get("ok") for r in all_qwen),
            # F1 (S4): planned keys with a PARSEABLE answer (Jev: incl. the one '|retry1' re-ask) / planned keys
            "jev_planned_keys": len(pj), "jev_keys_parsed": len(pj & jev_parsed),
            "jev_success_rate": len(pj & jev_parsed) / len(pj),
            "qwen_planned_keys": len(pq), "qwen_keys_parsed": len(pq & qwen_parsed),
            "qwen_success_rate": len(pq & qwen_parsed) / len(pq),
            # previous definition (HTTP ok, answer present), reported only
            "jev_keys_ok": len(pj & okj), "jev_http_ok_rate": len(pj & okj) / len(pj),
            "qwen_keys_ok": len(pq & okq), "qwen_http_ok_rate": len(pq & okq) / len(pq),
            "jev_retry_keys_in_log": len(retry_keys),
            "qwen_verbal_planned_keys": len(pv), "qwen_verbal_keys_ok": len(pv & okq),
            "qwen_verbal_keys_parsed": len(pv & verbal_parsed),
            "jev_unplanned_keys_in_log": len({r["key"] for r in all_jev} - pj - retry_keys),
            "qwen_unplanned_keys_in_log": len({r["key"] for r in all_qwen} - pq - pv),
            "log_unparseable_lines": bad_lines,
            "jev_cost_usd_all_records": float(sum(cost_all)), "jev_cost_usd_ok_records": float(sum(cost_ok)),
            "jev_ok_records_missing_cost": sum(c is None for c in cost_ok_raw),
            "first_call": min(started) if started else None, "last_call": max(started) if started else None,
        },
        "latency_s": {
            "jev_median": float(np.median(latj)) if latj else None, "jev_p95": float(np.quantile(latj, .95)) if latj else None,
            "jev_mean": float(np.mean(latj)) if latj else None,
            "qwen_median": float(np.median(latq)) if latq else None, "qwen_p95": float(np.quantile(latq, .95)) if latq else None,
            "qwen_mean": float(np.mean(latq)) if latq else None,
            "qwen_server_s_mean": float(np.mean(srvq)) if srvq else None,
            "qwen_server_s_median": float(np.median(srvq)) if srvq else None, "qwen_server_n": len(srvq)},
        "jev_models": dict(Counter((r.get("response") or {}).get("model") for r in all_jev if r.get("ok"))),
        "jev_http_status": dict(Counter(str(r.get("http_status")) for r in all_jev)),
        "jev_errors": dict(Counter(r.get("error") for r in all_jev if not r.get("ok"))),
        "qwen_errors": dict(Counter(r.get("error") for r in all_qwen if not r.get("ok"))),
        "qwen_server": [r.get("server") for r in server_recs],
    }
    masses = {"all": [], "noul": [], "choice": [], "nli": [], "pqa": [], "form_S": [], "form_N": [], "form_D": [],
              "form_CH": []}
    for k, r in qwen_recs.items():
        if "mass" in r and k in pq:
            masses["all"].append(r["mass"])
            if r.get("kind") in masses:
                masses[r["kind"]].append(r["mass"])
            dom = (r.get("meta") or {}).get("domain")
            if dom in masses:
                masses[dom].append(r["mass"])
            qid = k.split("|")[1]
            masses["form_CH" if qid == "CH" else f"form_{qid[0]}"].append(r["mass"])     # S6: by question form
    bk["qwen_readout_mass"] = {
        g: ({"n": len(m), "median": float(np.median(m)), "p05": float(np.quantile(m, .05)),
             "p25": float(np.quantile(m, .25)), "min": float(np.min(m)), "frac_lt_0.8": float(np.mean(np.array(m) < 0.8)),
             "frac_lt_0.5": float(np.mean(np.array(m) < 0.5))} if m else {"n": 0})
        for g, m in masses.items()}
    bk["n_values"] = {v: sum(len(x) for x in V[v].values()) for v in V}
    fr = HERE / "prereg_snapshot" / "FREEZE_RECORD.json"
    if fr.exists():
        rec = json.loads(fr.read_text(encoding="utf-8"))
        bk["freeze"] = {"frozen_at": rec["frozen_at"], "n_prior_freezes": rec["n_prior_freezes"],
                        "materials": rec["materials"],
                        "frozen_before_first_call": bool(started) and rec["frozen_at"] < min(started)}
    else:
        bk["freeze"] = None
    # paper-scale feasibility (PREREG F4)
    mc = float(np.mean(cost_ok)) if cost_ok else None
    bk["paper_scale"] = {
        "jev_calls": TH["paper_jev_calls"], "qwen_calls": TH["paper_qwen_calls"],
        "jev_mean_cost_per_call_usd": mc,
        "jev_cost_usd": mc * TH["paper_jev_calls"] if mc is not None else None,
        "jev_hours_at_conc4": (np.mean(latj) * TH["paper_jev_calls"] / 4 / 3600) if latj else None,
        # GATING (R2-S4): server-side compute time; the wall-clock version (incl. queueing) is reported only
        "qwen_hours_at_conc4_server": (np.mean(srvq) * TH["paper_qwen_calls"] / 4 / 3600) if srvq else None,
        "qwen_hours_at_conc4_wallclock": (np.mean(latq) * TH["paper_qwen_calls"] / 4 / 3600) if latq else None}
    return bk


# ------------------------------------------------------------------------------------------ main analysis
def analyze(logs_dir, out_path, stim=None):
    stim = stim or load_stim()
    logs_dir = Path(logs_dir)
    all_jev, jev_recs, bad_j, files_j = _read_logs(logs_dir, "jev")
    all_qwen_raw, qwen_recs, bad_q, files_q = _read_logs(logs_dir, "qwen_local")
    server_recs = [r for r in all_qwen_raw if r.get("key") == "__server__"]
    all_qwen = [r for r in all_qwen_raw if r.get("key") != "__server__"]
    qwen_recs = {k: v for k, v in qwen_recs.items() if k != "__server__"}
    V, parse_bk, parsed = build_tables(stim, jev_recs, qwen_recs)

    res = {"idea": "02_probability_axioms", "n_stimuli": len(stim),
           "n_by_domain": dict(Counter(it["domain"] for it in stim)), "bootstrap": {"n": NBOOT, "seed": SEED,
           "stratified_by": "domain"}, "thresholds": TH}
    res.update(bookkeeping(stim, all_jev, all_qwen, jev_recs, qwen_recs, server_recs, V, parsed,
                           {"jev": bad_j, "qwen": bad_q}))
    res["parse"] = parse_bk
    res["log_files"] = {"jev": files_j, "qwen": files_q}

    # ---- PRIMARY: negation complementarity, Jev (r0) vs Qwen (rounded), common complete pairs
    cmp_main, per_main = compare(V, "jev", "qwen", stim)
    prim = cmp_main["neg_mad"]
    if prim.get("n_items"):
        dvec = np.array(per_main["neg_mad"][0]) - np.array(per_main["neg_mad"][1])
        prim["perm_p_two_sided"] = stats.paired_perm_test(dvec, n=10000, seed=SEED)
        prim["n_pairs"] = sum(1 for it in stim for k in range(3)
                              if all(q in V["jev"][it["id"]] and q in V["qwen"][it["id"]] for q in (f"S{k}", f"N{k}")))
    res["primary"] = prim
    res["secondary_jev_vs_qwen"] = {m: v for m, v in cmp_main.items() if m != "neg_mad"}
    res["by_domain"] = {dom: compare(V, "jev", "qwen", stim, metrics=["neg_mad", "neg_signed", "neg_contra",
                                                                        "exh_signed", "mono_mag", "equiv_signed",
                                                                        "xtype_abs", "s_acc", "ch_acc"], domain=dom)[0]
                        for dom in sorted(set(it["domain"] for it in stim))}
    # ---- R2-M2: per (domain, label k) breakdown; equiv_signed = d_k - n_k diagnoses a misread negation
    #      (D_k's disjunction wording contains no "not")
    lab_metrics = ["neg_mad", "neg_signed", "neg_logit", "equiv_signed"]
    res["by_domain_label"] = {}
    for dom in sorted(set(it["domain"] for it in stim)):
        sd = [it for it in stim if it["domain"] == dom]
        for k, lab in enumerate(sd[0]["classes"]):
            Vk = keep_pairs(V, sd, lambda it, kk, k=k: kk == k)
            res["by_domain_label"][f"{dom}:{lab}"] = dict(
                compare(Vk, "jev", "qwen", sd, metrics=lab_metrics)[0], neg_polarity=neg_polarity(sd[0], k))
    # ---- R2-M2 gate P: everything except the negative-polarity pairs (PQA 'negative', NLI 'contradiction')
    Vnp = keep_pairs(V, stim, lambda it, k: not neg_polarity(it, k))
    res["polarity_robustness"] = compare(Vnp, "jev", "qwen", stim, metrics=["neg_mad", "neg_signed"])[0]
    res["polarity_robustness"]["excluded_labels"] = TH["P_neg_polarity_labels"]
    res["polarity_robustness"]["n_pairs"] = sum(
        1 for it in stim for k in range(3) if not neg_polarity(it, k)
        and all(q in V["jev"][it["id"]] and q in V["qwen"][it["id"]] for q in (f"S{k}", f"N{k}")))
    # ---- R2-S3 (report only): PQA items inside / outside the public Jevals PubMedQA suite
    res["jevals_suite_subsets"] = {
        name: compare(V, "jev", "qwen", [it for it in stim if it["domain"] == "pqa" and it["in_jevals_suite"] == flag],
                      metrics=["neg_mad", "neg_signed", "s_acc"])[0]
        for name, flag in (("pqa_in_suite", True), ("pqa_out_of_suite", False))}
    # ---- noise floors
    nfj = noise_floor(V, "jev_r0", "jev_r1", stim)
    if nfj.get("n_items"):
        # S10c: if the repeats are (near-)identical (e.g. a response cache), the measured NF is not informative ->
        # use max(measured, Jevals prior 0.007) for F3 and S-a
        fb = nfj["frac_identical"] >= TH["nf_identical_frac"]
        nfj["nf_fallback_applied"] = bool(fb)
        nfj["nf_used"] = max(nfj["nf_neg"], TH["nf_prior"]) if fb else nfj["nf_neg"]
    res["noise_floor"] = {"jev": nfj,
                          # expected ~0: temperature-0 logprob readout; this only checks server determinism (S2)
                          "qwen": noise_floor(V, "qwen", "qwen_r1", stim)}
    # ---- robustness of the primary
    rob = {}
    for a, b in (("jev_avg", "qwen"), ("jev", "qwen_raw"), ("jev", "qwen_mass05"), ("jev", "qwen_mass08"),
                 ("jev_r1", "qwen")):
        c = compare(V, a, b, stim, metrics=["neg_mad", "neg_signed", "neg_contra", "neg_logit"])[0]
        rob[f"{a}__vs__{b}"] = c
    res["robustness"] = rob
    # ---- R2-S1: verbal readout of Qwen on ALL items (gating D3: same direction as Delta)
    res["verbal_readout_check"] = {
        "jev__vs__qwen_verbal": compare(V, "jev", "qwen_verbal", stim,
                                        metrics=["neg_mad", "neg_signed", "neg_logit"])[0],
        "qwen_verbal__vs__qwen_logprob": compare(V, "qwen_verbal", "qwen", stim,
                                                 metrics=["neg_mad", "neg_signed", "neg_logit"])[0]}
    # ---- Jev bundle (all 10 questions in one request) vs single-question calls
    res["jev_bundle"] = {"vs_single_r0": compare(V, "jev_bundle", "jev_r0", stim,
                                                 metrics=["neg_mad", "neg_signed", "exh_signed", "mono_mag",
                                                          "equiv_abs", "xtype_abs"])[0],
                         "answer_shift": bundle_vs_single(V, stim)}
    # ---- pair level: confidence bins, mid-range subset, agreement
    res["pair_level"] = pair_level(V, stim)
    # ---- human ambiguity (NLI): does incoherence track human label entropy?
    amb = {}
    for v in ("jev", "qwen"):
        xs, ys = [], []
        for it in stim:
            if it["domain"] != "nli":
                continue
            pa, pb = restrict(V["jev"][it["id"]], V["qwen"][it["id"]])
            m = item_metrics(pa if v == "jev" else pb, it)["neg_mad"]
            if not math.isnan(m):
                xs.append(m)
                ys.append(it["reference"]["human_entropy"])
        if len(xs) >= 5:
            r, lo, hi = stats.corr_ci(xs, ys, seed=SEED)
            amb[v] = {"spearman_negmad_vs_entropy": r, "ci": [lo, hi], "n": len(xs)}
    res["human_ambiguity_nli"] = amb

    # ---- GO / NO-GO (PREREG section 6). R2-S5: every statistic is rounded with R() before it is compared.
    c, nfj = res["calls"], res["noise_floor"]["jev"]
    mass_med = res["qwen_readout_mass"]["all"].get("median")
    ps = res["paper_scale"]

    def _le(x, t):
        return x is not None and R(x) <= t

    F = {"F1_success_rate": R(c["jev_success_rate"]) >= TH["success_rate"]
         and R(c["qwen_success_rate"]) >= TH["success_rate"],
         "F2_qwen_mass": mass_med is not None and R(mass_med) >= TH["qwen_mass_median"],
         "F3_jev_noise_floor": _le(nfj.get("nf_used"), TH["jev_noise_floor_max"]),
         "F4_paper_scale": (_le(ps["jev_cost_usd"], TH["paper_cost_usd_max"])
                            and _le(ps["jev_hours_at_conc4"], TH["paper_hours_max"])
                            and _le(ps["qwen_hours_at_conc4_server"], TH["paper_hours_max"]))}
    Sa = Sb = D = D1 = D1_avg = D2 = D3 = P = False
    D2_level, D2_used, D3_used, P_detail, flags = None, None, None, None, {}
    coh_claim, coh_claim_np, story = None, None, None
    if prim.get("n_items"):
        nf = nfj.get("nf_used", float("inf"))
        nfq = res["noise_floor"]["qwen"].get("nf_neg", float("inf"))
        sacc = cmp_main["s_acc"].get("jev")
        sacc_soft = cmp_main["s_acc_soft"].get("jev")

        def s_a(m):          # m: a compare() entry for neg_mad
            return (R(m["jev"]) >= TH["Sa_point"] and R(m["jev_ci"][0]) >= TH["Sa_lo"]
                    and R(m["jev_ci"][0]) >= R(TH["Sa_noise_mult"] * nf))

        def s_b(m):          # control meaningfully incoherent, Jev less so (monotone; see module docstring)
            return (R(m["jev"]) < R(m["qwen"])
                    and R(m["qwen"]) >= TH["Sb_qwen_point"] and R(m["qwen_ci"][0]) >= TH["Sb_qwen_lo"]
                    and R(m["qwen_ci"][0]) >= R(TH["Sb_qwen_noise_mult"] * nfq)
                    and sacc_soft is not None and R(sacc_soft) >= TH["Sb_jev_acc_soft"])

        Sa, Sb = s_a(prim), s_b(prim)
        mj, mq = R(prim["jev"]), R(prim["qwen"])
        dlo, dhi = R(prim["diff_ci"][0]), R(prim["diff_ci"][1])
        delta = R(prim["diff"])
        ratio = R(max(mj, mq) / max(min(mj, mq), TH["D_ratio_floor"]))
        D1 = abs(delta) >= TH["D_abs_diff"] and ratio >= TH["D_ratio"] and (dlo > 0 or dhi < 0)
        # D1_avg (S1): the gap must survive averaging Jev's two identical repeats
        ra = rob["jev_avg__vs__qwen"]["neg_mad"]
        D1_avg = bool(ra.get("n_items")) and R(ra["diff"]) * delta > 0 and abs(R(ra["diff"])) >= TH["D1_avg_abs_diff"]
        # D2 confidence-confound guard (R2-M1). Level 1: confidence-MATCHED pairs (|s_J - s_Q| <= 0.10), >= 30 items:
        #   Delta_match must have the sign of Delta and |Delta_match| >= 0.03. Level 2 (fewer than 30 items): logit
        #   scale, neg_logit diff with the sign of Delta AND its 95% CI excluding 0 on that side.
        #   "Not evaluable" never passes by itself.
        cm = res["pair_level"].get("conf_matched_s") or {}
        if cm.get("n_items", 0) >= TH["D2_match_min_items"]:
            D2_level = "matched_s_0.10"
            D2_used = {"source": "pair_level.conf_matched_s", "n_items": cm["n_items"], "n_pairs": cm["n_pairs"],
                       "diff": cm["diff"], "diff_ci": cm["diff_ci"]}
            D2 = R(cm["diff"]) * delta > 0 and abs(R(cm["diff"])) >= TH["D2_match_abs_diff"]
        else:
            D2_level = "logit"
            nl = cmp_main["neg_logit"]
            if nl.get("n_items"):
                D2_used = {"source": "secondary_jev_vs_qwen.neg_logit", "n_items": nl["n_items"], "diff": nl["diff"],
                           "diff_ci": nl["diff_ci"], "matched_items": cm.get("n_items", 0)}
                D2 = R(nl["diff"]) * delta > 0 and (R(nl["diff_ci"][0]) > 0 if delta > 0 else R(nl["diff_ci"][1]) < 0)
        # D3 (R2-S1): with Qwen's VERBAL readout (all items) the neg_mad gap must point the same way as Delta
        vm = res["verbal_readout_check"]["jev__vs__qwen_verbal"]["neg_mad"]
        if vm.get("n_items", 0) >= TH["D3_verbal_min_items"]:
            D3_used = {"source": "verbal_readout_check.jev__vs__qwen_verbal.neg_mad", "n_items": vm["n_items"],
                       "diff": vm["diff"], "diff_ci": vm["diff_ci"]}
            D3 = R(vm["diff"]) * delta > 0
        else:
            D3_used = {"n_items": vm.get("n_items", 0), "not_evaluable": True}
        D = D1 and D1_avg and D2 and D3
        # P (R2-M2): without the negative-polarity pairs, the S branch that passed must still hold and Delta must keep
        # its sign with |Delta| >= 0.03
        pr = res["polarity_robustness"]["neg_mad"]
        if pr.get("n_items"):
            S_np = (Sa and s_a(pr)) or (Sb and s_b(pr))
            D_np = R(pr["diff"]) * delta > 0 and abs(R(pr["diff"])) >= TH["P_abs_diff"]
            P = S_np and D_np
            P_detail = {"branch_holds": bool(S_np), "delta_holds": bool(D_np), "n_items": pr["n_items"],
                        "jev": pr["jev"], "jev_ci": pr["jev_ci"], "qwen": pr["qwen"], "qwen_ci": pr["qwen_ci"],
                        "diff": pr["diff"], "diff_ci": pr["diff_ci"]}
        prim["ratio_larger_over_smaller"] = ratio
        prim["jev_s_argmax_acc"] = sacc
        prim["jev_s_argmax_acc_soft"] = sacc_soft
        # REPORT-ONLY (R2-S2): the report may call Jev "coherent" only if its MAD point estimate AND 95% CI upper
        # bound are < 0.05 (fewer than 5 quantisation steps); also shown on the non-polarity subset.
        def _coh(m):
            return bool(m.get("n_items")) and R(m["jev"]) < TH["coherent_claim_below"]                 and R(m["jev_ci"][1]) < TH["coherent_claim_below"]
        coh_claim, coh_claim_np = _coh(prim), _coh(pr)
        story = ("jev_less_coherent_than_llm" if delta > 0 else
                 ("jev_coherent_llm_not" if coh_claim else "jev_more_coherent_both_violate") if delta < 0 else
                 "no_difference")
        # pre-declared NON-gating report flags: the report must discuss any flag that is true
        bd = res["by_domain"]
        dn, dp = (bd.get(x, {}).get("neg_mad", {}).get("diff") for x in ("nli", "pqa"))
        flags = {"domain_sign_disagree": (dn is not None and dp is not None and R(dn) * R(dp) < 0),
                 "D2_fell_back_to_logit": D2_level == "logit",
                 "verbal_diff_ci_includes_0": bool(D3_used and "diff_ci" in D3_used
                                                   and R(D3_used["diff_ci"][0]) <= 0 <= R(D3_used["diff_ci"][1]))}
    Fall = all(F.values())
    S = bool(Sa or Sb)
    GO = bool(Fall and S and D and P)
    res["decision"] = {"F": F, "F_all": Fall, "S_a_jev_incoherent": bool(Sa), "S_b_jev_more_coherent": bool(Sb),
                       "S": S, "D1_gap": bool(D1), "D1_avg_robust": bool(D1_avg),
                       "D2_confound_guard": bool(D2), "D2_level": D2_level, "D2_detail": D2_used,
                       "D3_verbal_direction": bool(D3), "D3_detail": D3_used, "D": bool(D),
                       "P_polarity_robust": bool(P), "P_detail": P_detail,
                       "jev_coherent_claim": coh_claim, "jev_coherent_claim_non_polarity": coh_claim_np,
                       "story": story,
                       "GO": GO, "verdict": "GO" if GO else "NO-GO", "report_flags": flags}

    res = _clean(res)
    Path(out_path).write_text(json.dumps(res, indent=1, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return res


def _clean(o):
    """numpy scalars -> python, NaN/inf -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


# ------------------------------------------------------------------------------------------ selftest
SCENARIOS = (("jev_offset", "GO"), ("coherent", "NO-GO"), ("null_branch", "GO"), ("confound", "NO-GO"),
             ("confident_negation_failure", "GO"), ("low_mass", "NO-GO"), ("sa_noise", "NO-GO"), ("d1_ci", "NO-GO"),
             ("d2_logit_confound", "NO-GO"), ("d2_logit_pass", "GO"), ("polarity_driven", "NO-GO"),
             ("verbal_flip", "NO-GO"), ("gray_zone_better", "GO"))
OWN_QWEN_S = ("confound", "confident_negation_failure", "d2_logit_confound", "d2_logit_pass")


def _fake_truth(stim, scenario, r):
    """Per item: (s_jev, dev_jev, s_qwen, dev_qwen), arrays over k; N_k answer = 1 - s_k + dev_k, D_k = 1 - s_k.
    jev_offset / low_mass / verbal_flip: Jev dev 0.10 everywhere, Qwen coherent (low_mass: Qwen mass 0.6 -> F2
      fails; verbal_flip: Qwen's VERBAL answers carry dev 0.20 -> D3 fails).
    coherent: both coherent. null_branch: Jev coherent (+repeat noise), Qwen dev 0.08 -> S-b.
    confound (R2-M1, true differential confidence): the SAME law for both systems (|dev| 0.10 on mid-range pairs,
      0 on confident pairs), Jev confident (0.95 / 0.03) where Qwen is mid-range, both 0.50 on one label ->
      S-b, D1 pass, but the confidence-matched gap is 0 -> D2 (level 1) fails.
    confident_negation_failure (R2-M1 reviewer scenario): identical confidence; Jev coherent with per-call noise
      sd 0.01; Qwen s = p + N(0, 0.02) and, when p_k < 0.10, N_k ~ U(0.20, 0.45) -> GO (S-b); the old mid-range
      rule would have vetoed it.
    sa_noise: 1/3 of items Jev dev 0.16 (MAD 0.054, between-item variance) + Jev repeat noise {-.03,0,0,.03} ->
      CI lower bound >= 0.03 but < 3*NF -> S-a fails.
    d1_ci: heavy tails (34 Jev items / 22 Qwen items with dev 0.45) -> |Delta| >= 0.03, ratio >= 1.5 but CI incl. 0.
    d2_logit_confound: Jev confident (0.97/0.02) and nearly coherent (dev -0.02), Qwen mid-range with dev +0.08 ->
      no confidence-matched pairs -> logit level: Qwen better on the logit scale -> D2 fails -> NO-GO.
    d2_logit_pass: Jev 0.97/0.02 coherent, Qwen 0.80/0.15 with |dev| 0.08 -> no matched pairs -> logit level
      agrees -> GO (S-b).
    polarity_driven (R2-M2): Jev dev 0.15 ONLY on the negative-polarity pair (MAD exactly 0.05 -> S-a only after
      rounding, R2-S5), Qwen coherent -> F, S, D pass but P fails -> NO-GO.
    gray_zone_better (monotonicity of S): Jev dev 0.14 on 1/3 of the items (MAD 0.047, 95% CI upper bound > 0.05),
      Qwen dev 0.15 everywhere -> S-a fails (point < 0.05) and the OLD S-b (Jev CI upper < 0.05) would fail too,
      although Jev is 3x more coherent than a clearly incoherent control -> new S-b passes -> GO, with
      jev_coherent_claim = False (the report may not call Jev "coherent")."""
    T = {}
    perm = r.permutation(len(stim))
    heavy_j, heavy_q = set(perm[:34].tolist()), set(perm[34:56].tolist())
    for idx, it in enumerate(stim):
        ref = it["classes"].index(it["reference"]["label"])
        s = np.round(r.uniform(0.15, 0.85, 3), 2)
        dj, dq, sq = np.zeros(3), np.zeros(3), None
        if scenario in ("jev_offset", "low_mass", "verbal_flip"):
            dj[:] = 0.10
        elif scenario == "null_branch":                    # informative S: reference label clearly highest
            s = np.round(r.uniform(0.15, 0.40, 3), 2)
            s[ref] = 0.80
            dq[:] = 0.08
        elif scenario == "confound":
            mid = (ref + 1) % 3
            s = np.array([0.03, 0.03, 0.03])
            s[ref], s[mid] = 0.95, 0.50
            dj = np.where(np.arange(3) == mid, 0.10, 0.0)
            sq = s.copy()
            sq[ref] = round(float(r.uniform(0.55, 0.75)), 2)
            sq[(ref + 2) % 3] = round(float(r.uniform(0.20, 0.35)), 2)
            dq[:] = 0.10
        elif scenario == "confident_negation_failure":
            if "human_dist" in it["reference"]:
                h = np.array([it["reference"]["human_dist"][c] for c in it["classes"]]) ** 2
                p = h / h.sum()
            else:
                p = np.full(3, 0.15)
                p[ref] = 0.70
            s = np.round(p, 2)
            sq = np.round(np.clip(p + r.normal(0, 0.02, 3), 0.0, 1.0), 2)
            for k in range(3):
                if p[k] < 0.10:
                    dq[k] = round(float(r.uniform(0.20, 0.45)), 2) - (1 - sq[k])
        elif scenario == "sa_noise":
            s = np.round(r.uniform(0.2, 0.8, 3), 2)
            if idx % 3 == 0:
                dj[:] = 0.16
        elif scenario == "d1_ci":
            if idx in heavy_j or idx in heavy_q:
                s = np.array([0.5, 0.5, 0.5])
            if idx in heavy_j:
                dj[:] = 0.45
            if idx in heavy_q:
                dq[:] = 0.45
        elif scenario in ("d2_logit_confound", "d2_logit_pass"):
            s = np.array([0.02, 0.02, 0.02])
            s[ref] = 0.97
            if scenario == "d2_logit_confound":
                dj[:] = -0.02
                sq = np.round(r.uniform(0.3, 0.7, 3), 2)
                dq[:] = 0.08
            else:
                sq = np.array([0.15, 0.15, 0.15])
                sq[ref] = 0.80
                dq = np.where(sq > 0.5, 0.08, -0.08)
        elif scenario == "polarity_driven":
            dj = np.array([0.15 if neg_polarity(it, k) else 0.0 for k in range(3)])
        elif scenario == "gray_zone_better":               # informative S, as in null_branch
            s = np.round(r.uniform(0.15, 0.40, 3), 2)
            s[ref] = 0.80
            if idx % 3 == 0:
                dj[:] = 0.14
            dq[:] = 0.15
        T[it["id"]] = (s, dj, s if sq is None else sq, dq)
    return T


def _fake_logs(d, stim, scenario, seed=123):
    """Fabricated logs with KNOWN answers (see _fake_truth). Every scenario also contains: a failed-then-retried
    key, an unparseable Jev answer without retry (-> r1 fallback, F1 counts it as missing), an unparseable Jev
    answer WITH a parseable '|retry1' record, a Jev record without usage.cost, 4 missing Qwen keys, Qwen mass 0.7
    on the N questions of the 20 repeat items, one unparseable verbal record, bundle answers under the opaque ids
    and server-side timings on the Qwen decide records. 'coherent' also cuts jev.jsonl with a truncated line after
    half of its records and continues in jev_c1.jsonl (run.py's continuation file)."""
    r = np.random.default_rng(seed)
    pj, pq, pv = planned_keys(stim)
    T = _fake_truth(stim, scenario, r)
    byid = {it["id"]: it for it in stim}
    jev, qwen = [], [{"key": "__server__", "ok": True, "server": {"build_info": "fake"}, "at": "2000-01-01T00:00:00.000Z"}]
    noise, levels = {}, {"null_branch": [-0.01, 0.0, 0.0, 0.01], "sa_noise": [-0.03, 0.0, 0.0, 0.03]}

    def jev_val(it, qid, rep):
        s, dj = T[it["id"]][0], T[it["id"]][1]
        if qid == "CH":
            z = s / s.sum()
            return {"type": "choice", "choice": it["classes"][int(np.argmax(z))],
                    "probabilities": {c: round(float(z[i]), 2) for i, c in enumerate(it["classes"])}}
        k = int(qid[1])
        v = s[k] if qid[0] == "S" else (1 - s[k] + dj[k] if qid[0] == "N" else 1 - s[k])
        if scenario == "confident_negation_failure":          # per-call noise, same value for bundle and r0
            if (it["id"], qid, rep) not in noise:
                noise[(it["id"], qid, rep)] = float(r.normal(0, 0.01))
            v += noise[(it["id"], qid, rep)]
        elif rep == 1 and scenario in levels:
            if (it["id"], qid) not in noise:
                noise[(it["id"], qid)] = float(r.choice(levels[scenario]))
            v += noise[(it["id"], qid)]
        return {"type": "noul", "noul": round(float(min(1.0, max(0.0, v))), 2)}

    def qwen_p(it, qid, verbal=False):
        _, _, s, dq = T[it["id"]]
        k = int(qid[1])
        if qid[0] == "S":
            return float(s[k])
        if qid[0] == "D":
            return float(1 - s[k])
        extra = 0.20 if (verbal and scenario == "verbal_flip") else float(dq[k])
        return float(min(1.0, max(0.0, 1 - s[k] + extra)))

    ks = sorted(pj)
    for t, key in enumerate(ks, 1):
        iid, qid, rep = key.split("|")
        it = byid[iid]
        ts = f"2000-01-01T00:{(t // 60) % 60:02d}:{t % 60:02d}.000Z"
        answers = ({oq: jev_val(it, q, 0) for oq, q in it["bundle_qids"].items()} if qid == "bundle"
                   else {"q": jev_val(it, qid, int(rep[1]))})
        rec = {"key": key, "ok": True, "system": "jev", "meta": {}, "request": {}, "response":
               {"model": "typesafe/jev-1.13-20260917", "answers": answers, "usage": {"cost": 3e-5}},
               "http_status": 200, "error": None, "attempts": 1, "latency_s": 0.5, "started_at": ts, "finished_at": ts}
        if key == ks[0]:
            jev.append({**rec, "ok": False, "response": {"error": "x"}, "http_status": 500, "error": "http 500"})
        if key in (ks[4], ks[8]):                                   # "<first item>|D1|r0" and "|N0|r0"
            good = rec["response"]["answers"]
            rec["response"] = {**rec["response"], "answers": {"q": {"type": "noul"}}}    # format anomaly
            if key == ks[8]:                                        # ... re-asked once by run.py, parses
                jev.append(rec)
                rec = {**rec, "key": key + RETRY, "response": {**rec["response"], "answers": good}}
        if key == ks[12]:
            rec["response"] = {k: v for k, v in rec["response"].items() if k != "usage"}  # missing usage.cost
        jev.append(rec)
    drop = set(sorted(pq)[:4])                                      # 4 missing Qwen keys
    for key in sorted(pq) + sorted(pv):
        if key in drop:
            continue
        iid, qid, rep = key.split("|")
        it = byid[iid]
        rec = {"key": key, "system": "qwen3.8-27b-local", "meta": {"domain": it["domain"]}, "ok": True,
               "mode": "verbal" if rep == "verbal" else "logprob", "kind": "choice" if qid == "CH" else "noul",
               "latency_s": 30.0, "started_at": "2000-01-01T01:00:00.000Z", "finished_at": "2000-01-01T01:00:30.000Z"}
        if rep != "verbal":
            rec["mass"] = 0.6 if scenario == "low_mass" else (0.7 if (qid[0] == "N" and it["qwen_repeat"]) else 0.95)
            rec["timings"] = {"prompt_ms": 250.0, "predicted_ms": 0.001}
        if qid == "CH":
            s = T[iid][2]
            z = s / s.sum()
            rec["probs"] = {c: float(z[i]) for i, c in enumerate(it["classes"])}
        elif rep == "verbal" and key == sorted(pv)[0]:
            rec["probs"] = None                                     # unparseable verbal answer
        else:
            v = qwen_p(it, qid, verbal=(rep == "verbal"))
            rec["p_yes"] = v
            rec["probs"] = {"yes": v, "no": 1 - v}
        qwen.append(rec)
    for name, recs in (("jev.jsonl", jev), ("qwen_local.jsonl", qwen)):
        cut = len(recs) // 2 if (name == "jev.jsonl" and scenario == "coherent") else len(recs)
        with open(Path(d) / name, "w", encoding="utf-8") as f:
            for x in recs[:cut]:
                f.write(json.dumps(x) + "\n")
            if cut < len(recs):
                f.write('{"key": "truncated|S0|r1", "ok": tr')           # killed mid-write
        if cut < len(recs):                                              # run.py then continues in jev_c1.jsonl
            with open(Path(d) / "jev_c1.jsonl", "w", encoding="utf-8") as f:
                for x in recs[cut:]:
                    f.write(json.dumps(x) + "\n")


def selftest():
    stim = load_stim()
    tmp = Path(tempfile.mkdtemp(prefix="axioms_selftest_"))
    # exact expected MAD_neg (None = random by design, not checked)
    exp_mad = {"jev_offset": 0.10, "low_mass": 0.10, "verbal_flip": 0.10, "confound": 0.10 / 3,
               "sa_noise": 54 * 0.16 / 160, "d1_ci": 34 * 0.45 / 160, "d2_logit_confound": 0.02,
               "polarity_driven": 0.05, "confident_negation_failure": None, "gray_zone_better": 54 * 0.14 / 160}
    exp_qmad = {"null_branch": 0.08, "confound": 0.10, "d1_ci": 22 * 0.45 / 160, "d2_logit_confound": 0.08,
                "d2_logit_pass": 0.08, "confident_negation_failure": None, "gray_zone_better": 0.15}
    n_suite = sum(it["domain"] == "pqa" and it["in_jevals_suite"] for it in stim)
    n_pqa = sum(it["domain"] == "pqa" for it in stim)
    try:
        ok = True
        for scen, expect in SCENARIOS:
            d = tmp / scen
            d.mkdir()
            _fake_logs(d, stim, scen)
            r1 = analyze(d, d / "results.json", stim)
            txt1 = (d / "results.json").read_text(encoding="utf-8")
            analyze(d, d / "results2.json", stim)
            same = txt1 == (d / "results2.json").read_text(encoding="utf-8")
            p, dec, c = r1["primary"], r1["decision"], r1["calls"]
            nfj = r1["noise_floor"]["jev"]
            nf = nfj["nf_used"]
            pl = r1["pair_level"]
            mid = pl["midrange_both"]
            print(f"[selftest:{scen}] MAD jev={p['jev']:.4f} {[round(x, 4) for x in p['jev_ci']]} qwen={p['qwen']:.4f} "
                  f"diff={p['diff']:.4f} {[round(x, 4) for x in p['diff_ci']]} ratio={p['ratio_larger_over_smaller']:.3f} "
                  f"nf={nfj['nf_neg']:.4f} nf_used={nf:.4f} matched={pl['conf_matched_s']['n_items']} "
                  f"mid10={mid['n_items']}/{mid.get('diff', float('nan')):.4f}")
            print(f"[selftest:{scen}] verdict={dec['verdict']} S_a={dec['S_a_jev_incoherent']} S_b={dec['S_b_jev_more_coherent']} "
                  f"D1={dec['D1_gap']} D1_avg={dec['D1_avg_robust']} D2={dec['D2_confound_guard']}({dec['D2_level']}) "
                  f"D3={dec['D3_verbal_direction']} P={dec['P_polarity_robust']} "
                  f"F={[k for k, v in dec['F'].items() if not v] or 'all'} flags={dec['report_flags']}")
            base = dec["D1_gap"] and dec["D1_avg_robust"]
            matched = dec["D2_level"] == "matched_s_0.10"
            gates = {
                "jev_offset": dec["S_a_jev_incoherent"] and base and dec["D2_confound_guard"] and matched
                and dec["D3_verbal_direction"] and dec["P_polarity_robust"] and nfj["nf_fallback_applied"]
                and abs(nf - TH["nf_prior"]) < 1e-12 and dec["story"] == "jev_less_coherent_than_llm",
                "verbal_flip": dec["S_a_jev_incoherent"] and base and dec["D2_confound_guard"]
                and not dec["D3_verbal_direction"] and dec["P_polarity_robust"],
                "coherent": not dec["S"] and not dec["D1_gap"] and c["log_unparseable_lines"]["jev"] == 1
                and r1["log_files"]["jev"] == ["jev.jsonl", "jev_c1.jsonl"],
                "null_branch": dec["S_b_jev_more_coherent"] and base and dec["D2_confound_guard"] and matched
                and dec["D3_verbal_direction"] and dec["P_polarity_robust"] and 0 < nf < 0.02
                and not nfj["nf_fallback_applied"] and dec["jev_coherent_claim"] is True
                and dec["story"] == "jev_coherent_llm_not",
                "confound": dec["S_b_jev_more_coherent"] and base and matched and not dec["D2_confound_guard"]
                and dec["D3_verbal_direction"] and dec["D2_detail"]["n_items"] >= TH["D2_match_min_items"]
                and abs(dec["D2_detail"]["diff"]) < 1e-9,
                # GO, and the old level-1 mid-range rule (same sign, |gap| >= 0.03) would have vetoed it
                "confident_negation_failure": dec["S_b_jev_more_coherent"] and base and dec["D2_confound_guard"]
                and matched and dec["D3_verbal_direction"] and dec["P_polarity_robust"]
                and not nfj["nf_fallback_applied"] and mid["n_items"] >= 30
                and not (R(mid["diff"]) * R(p["diff"]) > 0 and abs(R(mid["diff"])) >= 0.03),
                "low_mass": dec["S"] and dec["D"] and dec["P_polarity_robust"] and not dec["F"]["F2_qwen_mass"],
                "sa_noise": (not dec["S_a_jev_incoherent"] and p["jev"] >= TH["Sa_point"] and p["jev_ci"][0] >= TH["Sa_lo"]
                             and p["jev_ci"][0] < TH["Sa_noise_mult"] * nf and not nfj["nf_fallback_applied"]),
                "d1_ci": (dec["S_a_jev_incoherent"] and not dec["D1_gap"] and abs(p["diff"]) >= TH["D_abs_diff"]
                          and p["ratio_larger_over_smaller"] >= TH["D_ratio"] and p["diff_ci"][0] < 0 < p["diff_ci"][1]),
                "d2_logit_confound": (dec["S_b_jev_more_coherent"] and base and pl["conf_matched_s"]["n_items"] < 30
                                      and dec["D2_level"] == "logit" and not dec["D2_confound_guard"]),
                "d2_logit_pass": (dec["S_b_jev_more_coherent"] and base and dec["D2_level"] == "logit"
                                  and dec["D2_confound_guard"] and dec["D3_verbal_direction"] and dec["P_polarity_robust"]),
                "polarity_driven": (dec["S_a_jev_incoherent"] and base and dec["D2_confound_guard"]
                                    and dec["D3_verbal_direction"] and not dec["P_polarity_robust"]
                                    and not dec["P_detail"]["branch_holds"] and R(p["jev"]) == 0.05
                                    and all(abs(v["neg_mad"]["jev"] - (0.15 if v["neg_polarity"] else 0.0)) < 1e-6
                                            for v in r1["by_domain_label"].values())),
                # the old S-b (Jev CI upper < 0.05) would have failed: S was non-monotone in Jev's MAD
                "gray_zone_better": (dec["S_b_jev_more_coherent"] and not dec["S_a_jev_incoherent"] and base
                                     and dec["D2_confound_guard"] and matched and dec["D3_verbal_direction"]
                                     and dec["P_polarity_robust"] and dec["jev_coherent_claim"] is False
                                     and R(p["jev"]) < 0.05 <= R(p["jev_ci"][1])
                                     and dec["story"] == "jev_more_coherent_both_violate"),
            }[scen]
            mass = r1["qwen_readout_mass"]
            ps = r1["paper_scale"]
            em, eq = exp_mad.get(scen, 0.0), exp_qmad.get(scen, 0.0)
            checks = {
                "deterministic": same,
                "verdict": dec["verdict"] == expect,
                "gates": bool(gates),
                "F_all": dec["F_all"] == (scen != "low_mass"),
                "jev_mad": em is None or abs(p["jev"] - em) < 1e-6,
                "qwen_mad": eq is None or abs(p["qwen"] - eq) <= 0.0101,
                "bundle_identical": abs(r1["jev_bundle"]["answer_shift"]["bundle_vs_r0"]) < 1e-12,
                "missing_qwen_excluded": c["qwen_keys_parsed"] == c["qwen_planned_keys"] - 4,
                "failed_then_ok": c["jev_fail"] == 1 and c["jev_http_ok_rate"] == 1.0,
                "F1_parseable": (c["jev_keys_parsed"] == c["jev_planned_keys"] - 1 and c["jev_retry_keys_in_log"] == 1
                                 and c["jev_unplanned_keys_in_log"] == 0),
                "anomaly_counted": (r1["parse"]["format_anomalies"].get("single") == 2
                                    and r1["parse"]["format_anomalies"].get("qwen_verbal") == 1
                                    and not r1["parse"]["format_anomalies"].get("bundle")),
                "retry_used": r1["parse"]["jev_retry_used"].get("single") == 1,
                "fallback_used": r1["parse"]["jev_r0_missing_used_r1"] == 1,
                "cost_missing": c["jev_ok_records_missing_cost"] == 1,
                "mass_by_form": (scen == "low_mass") or (
                    mass["form_N"]["min"] == 0.7 and mass["form_S"]["min"] == 0.95
                    and r1["robustness"]["jev__vs__qwen_mass08"]["neg_mad"]["n_items"] == 140),
                "verbal": (c["qwen_verbal_keys_parsed"] == c["qwen_verbal_planned_keys"] - 1
                           and c["qwen_verbal_planned_keys"] == 6 * len(stim)
                           and r1["verbal_readout_check"]["jev__vs__qwen_verbal"]["neg_mad"]["n_items"] == len(stim)),
                # F4 (R2-S4): server-side 0.250001 s gates; the 30 s wall-clock latency (queueing) would have failed
                "F4_server": (abs(ps["qwen_hours_at_conc4_server"] - 0.250001 * TH["paper_qwen_calls"] / 4 / 3600) < 1e-9
                              and ps["qwen_hours_at_conc4_wallclock"] > TH["paper_hours_max"]),
                "by_label": len(r1["by_domain_label"]) == 6 and sum(v["neg_polarity"] for v in r1["by_domain_label"].values()) == 2,
                "suite_subsets": (r1["jevals_suite_subsets"]["pqa_in_suite"]["neg_mad"]["n_items"] == n_suite
                                  and r1["jevals_suite_subsets"]["pqa_out_of_suite"]["neg_mad"]["n_items"] == n_pqa - n_suite),
                "exh_signed": scen in OWN_QWEN_S or abs(r1["secondary_jev_vs_qwen"]["exh_signed"]["jev"]
                                                        - r1["secondary_jev_vs_qwen"]["exh_signed"]["qwen"]) < 0.02,
            }
            bad = [k for k, v in checks.items() if not v]
            print(f"[selftest:{scen}] {'ok' if not bad else 'FAILED: ' + str(bad)}")
            ok &= not bad
        print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=str(HERE / "logs"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    res = analyze(a.logs_dir, a.out)
    p = res["primary"]
    print(json.dumps({"primary": p, "noise_floor": res["noise_floor"], "decision": res["decision"],
                      "calls": res["calls"], "qwen_mass": res["qwen_readout_mass"]["all"]}, indent=1, ensure_ascii=False))
