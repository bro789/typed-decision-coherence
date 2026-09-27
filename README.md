# Beyond Calibration: Do a Typed-Decision Model's Probabilities Obey the Probability Axioms?

Data and analysis code for the paper by Keyi Li, Yihao He, and Quanyi Li (Software College, Northeastern University,
Shenyang, China).

The paper tests whether the probabilities that a typed-decision model (TypeSafe's Jev) and an open-weight LLM
(Qwen3.8-27B) give to logically related questions obey the probability axioms. Each of 160 items (100 from
ChaosNLI-MNLI, 60 from PubMedQA) has three mutually exclusive labels, and both systems answer ten questions about it:
whether the label is X_k (S_k), whether it is not X_k (N_k), whether it is one of the other two labels (D_k), and which
label applies (CH). This repository contains the items, every request and response, and the scripts that compute every
number, table, and figure in the paper.

## Layout

```
_common/                                   helpers imported by the analysis (JSONL logs, bootstrap statistics)
02_probability_axioms/
  stimuli/items.jsonl                      the 160 items and their ten questions
  stimuli/manifest.json                    sampling seeds; PubMedQA items that belong to the public Jevals suite
  official_qwen/logs/jev.jsonl             3,360 Jev requests and responses
  official_qwen/logs/qwen_local.jsonl      2,680 Qwen3.8-27B calls (1,720 first-token, 960 verbalized)
  official_qwen/results.json               output of analyze.py: every statistic in the paper
  official_qwen/results_posthoc.json       exploratory quantities (Appendix D of the paper)
  analyze.py                               analysis from the logs
  ccfa-workfiles/figures/paper-displays/source/make_displays.py
                                           figures, tables, and exploratory quantities
  manuscript/figures/, manuscript/tables/  the paper's figures and tables as written by make_displays.py
```

The directory names are those of the working copy in which the analysis ran; the scripts locate their inputs relative
to their own location, so they run unchanged from a clone of this repository.

## Reproducing the results

Python 3.12 with the packages in `requirements.txt`. From the repository root:

```
pip install -r requirements.txt
python -X utf8 02_probability_axioms/analyze.py --logs-dir 02_probability_axioms/official_qwen/logs --out 02_probability_axioms/official_qwen/results.json
python -X utf8 02_probability_axioms/ccfa-workfiles/figures/paper-displays/source/make_displays.py
```

The first command rebuilds `results.json` from the logs. The second writes `results_posthoc.json` and the figures and
tables under `02_probability_axioms/manuscript/`. Both are deterministic (bootstrap and permutation seed 0) and
reproduce `results.json`, `results_posthoc.json`, and the tables in this repository byte for byte; the figures are
redrawn from the same values. `python -X utf8 02_probability_axioms/analyze.py --selftest` runs the analysis on
synthetic logs with known answers.

## Data format

- `items.jsonl`: one JSON object per item with `id`, `domain` (`nli` or `pqa`), `source`, `state` (the text both
  systems see), `classes` (the three labels), `questions` (`S0`-`S2`, `N0`-`N2`, `D0`-`D2`, `CH`: type and question
  text, and for `CH` the options with their definitions), `reference` (reference label; for NLI also the distribution
  and entropy of the 100 human labels), `in_jevals_suite`, `qwen_repeat` (the 20 items whose S and N questions were
  asked twice), and `bundle_qids` (the opaque question ids used when Jev received all ten questions in one request).
- `jev.jsonl`: one record per call, keyed `<item>|<question>|r<repeat>` or `<item>|bundle|r0`, with the request
  (model, state, questions), the response (model version, answers, usage and cost), and timestamps.
- `qwen_local.jsonl`: the first record describes the server (vLLM version, model, context length). The other records
  are keyed `<item>|<question>|r<repeat>` for the first-token readout (messages, templated prompt, the 40 most likely of
  the 100 scored next tokens with their log-probabilities, label probabilities, and label mass) or
  `<item>|<question>|verbal` for the verbalized readout (request, reply text, parsed probabilities).
- `results.json`: every statistic reported in the paper, in sections named as in `analyze.py` (`primary`,
  `secondary_jev_vs_qwen`, `robustness`, `pair_level`, ...). It also carries bookkeeping fields used while developing
  the analysis, such as thresholds and decision flags, which the paper does not use.

## Systems

- Jev: `typesafe/jev-1.13` through OpenRouter's decisions endpoint; every response reports version
  `jev-1.13-20260917`.
- Qwen3.8-27B: official BF16 weights served with vLLM 0.19.1 on two RTX A6000 GPUs, thinking disabled. The prompts and
  the two readouts are described in Section 4 and Appendix A of the paper.

## Data sources and licenses

The NLI items come from ChaosNLI (Nie et al., 2020; CC BY-NC 4.0), which is built on MNLI (Williams et al., 2018). The
PubMedQA items come from the expert-labelled PubMedQA set (Jin et al., 2019; MIT license). Reuse of the item texts is
subject to those licenses.
