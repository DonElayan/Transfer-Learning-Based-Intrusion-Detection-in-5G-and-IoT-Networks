# BoT-IoT → ToN-IoT transfer learning (`bot_ton_tl`)

A complete, leakage-aware pipeline for answering one question:

> **Does knowledge learned on BoT-IoT (source, Argus flows) let us detect attacks on
> ToN-IoT (target, Zeek flows) with far fewer ToN-IoT labels than training from scratch?**

It turns the audit work (Phase 1–2.5) into explicit decisions and then runs the
experiment: three stages, each a command, each writing a report you can quote in a
methods section.

```
raw CSVs ──prepare──► bot.parquet / ton.parquet ──diagnose──► domain-shift report
                                          └────────experiment──► learning curves, tables
```

---

## 1. How the problem is set up

| | |
|---|---|
| **Source** | BoT-IoT 5 % release (`UNSW_2018_IoT_Botnet_Full5pc_1..4.csv`, 3,668,522 flows; 477 normal) — all labels used |
| **Target** | ToN-IoT network (`Network_dataset_1..23.csv`, 22,339,021 flows; 796,380 normal) — **only k labelled flows per class** |
| **Tasks** | `binary` (normal vs attack — defined identically in both datasets) and `multiclass` (all ToN-IoT classes, via a new output head) |
| **Evaluation** | learning curves: macro-F1 on a fixed ToN-IoT test set vs. k ∈ {10, 50, 100, 500, 1000} labels per class, mean ± std over seeds |
| **TL counts as working when** | `finetune` / `dann` beat **both** no-transfer baselines (`target_only`, `rf`) at small k. If they fall below them, that is negative transfer, and it is a reportable result too. |

### Methods compared

| method | ToN labels | what it is |
|---|---|---|
| `source_only` | 0 | BoT-IoT model applied as is (binary: P(attack) = 1 − P(normal)) |
| `dann_zero` | 0 | DANN model (aligned with *unlabelled* ToN flows) applied as is |
| `target_only` | k/class | same network, random init, trained on the k-shot budget |
| `rf` | k/class | random forest on the same budget, the strong tabular baseline reviewers will ask for |
| `frozen` | k/class | BoT-IoT encoder frozen + new dense head (the recipe of this repo's `TL - CNN-TL.ipynb`) |
| `finetune` | k/class | BoT-IoT encoder, **LP-FT**: train new head, then unfreeze all with 10× smaller encoder LR |
| `dann` | k/class | encoder pre-trained with DANN (BoT labels + unlabelled ToN flows), then LP-FT |
| `finetune_full` | k/class | BoT-IoT encoder, then trained **exactly like `target_only`** (same LR, steps, early stopping). The only difference is the initialisation, so `finetune_full − target_only` is the clean transfer effect |
| `dann_full` | k/class | same, from the DANN encoder |

The encoder is a ~30k-parameter MLP (`--arch mlp`, default) or a 1-D CNN mirroring the
repository's CNN-B (`--arch cnn1d`). Both are small enough for constrained IoT/edge devices.

---

## 2. Decisions taken (and the evidence)

The Phase 2.x audits left every item "UNVERIFIED". These are the choices this pipeline
makes. Each one is a flag, so you can run any of them as an ablation.

| # | Issue | Decision | Evidence / reason |
|---|---|---|---|
| 1 | Which fields are comparable? | proto, src/dst port, duration, src/dst bytes, src/dst packets (+ state as an ablation) | Only these exist in both Argus and Zeek outputs. BoT-IoT's window aggregates (`TnBPSrcIP`, `N_IN_Conn_P_*`, …), `*_number` encodings, IPs, timestamps, ids, `seq` and all label columns are **never** features. |
| 2 | **`sbytes` ↔ `src_bytes`? No: ↔ `src_ip_bytes`** | ToN `src_ip_bytes`/`dst_ip_bytes` (`--ton-bytes ip`) | Zeek `src_bytes` (= `orig_bytes`) counts **payload only**, so it is 0 for SYN scans and floods. Argus `sbytes` counts every packet byte. In this repo's real `datasets/BoT-IoT.csv` (5.83 M rows): 0 rows have `spkts>0 & sbytes==0`. The prepare report prints the same check for ToN. |
| 3 | Argus counts the Ethernet header | subtract 14 bytes/packet from BoT bytes (`--bot-l2-bytes 14`) | Real BoT-IoT: minimum bytes/packet = **42 = 14 Eth + 20 IP + 8 UDP**, and a lone TCP RST is 60 bytes (padded minimum frame). Zeek `*_ip_bytes` start at the IP header. |
| 4 | Hex ports, `-1`, blanks | hex decoded; `-1`, blank, out of range and junk (`xinetd`, `nut`, `login` seen in real data) → missing; port features only for TCP/UDP, as coarse ranges + a few service flags | ICMP "ports" are type/code and ARP rows have none. Raw port numbers act as dataset shortcuts, so compare `--features volume shape proto` against the default. |
| 5 | ToN `src_bytes='0.0.0.0'` (869 rows, files 1/22/23) | row dropped and counted | A byte counter holding an IP address marks a corrupt record. That is 0.004 % of ToN. |
| 6 | Zeek `-` | `-` in duration / byte columns → 0 (counted) | Zeek leaves duration/payload unset for single-packet connections. |
| 7 | `state` vs `conn_state` | **off by default**. `--features ... state` uses a coarse map (established / no_reply / reset / other) | The taxonomies differ, so the coarse map is a stated hypothesis (`schema.STATE_COARSE`), not an equivalence. |
| 8 | **DoS vs DDoS** | kept separate; run `--merge dos+ddos` as well | Real BoT-IoT: 24,145 of the 24,178 feature vectors that carry two labels are `dos`/`ddos`. A single flow cannot show how many hosts attack. |
| 9 | Label mapping | binary is the primary task. Multiclass fine-tuning learns all 10 ToN classes with a new head, so no mapping is needed. `source_only` multiclass is scored only on normal / dos / ddos / reconnaissance≈scanning (`multiclass_shared`) | ToN classes such as xss, injection, password, ransomware, backdoor and mitm have no BoT-IoT counterpart. Theft is never mapped. |
| 10 | Duplicates and leakage | exact de-duplication on (feature vector, class); **train/val/test assigned per feature-vector hash** | Identical vectors can never sit on both sides of a split. Label-conflicting twins are kept but stay in one split. Counts are in the prepare report. |
| 11 | BoT-IoT has almost no normal traffic | class-balanced mini-batches; macro metrics as headline | 477 normal flows vs 3.67 M attacks. Accuracy would hide every normal-traffic failure. |
| 12 | Scaling | one scaler fitted on BoT train + **unlabelled** ToN pool (`--norm shared`); only continuous columns are z-scored, indicators stay 0/1 | The features are in the same physical units (log bytes, packets, seconds). `--norm per_domain` (AdaBN-style) is offered as an ablation. |
| 13 | Few-shot model selection | early stopping uses 20 % **of the k-shot budget only**; the ToN validation split is never loaded | Using thousands of extra labelled ToN flows for early stopping would quietly break the "scarce labels" premise. |
| 14 | Test set | fixed across seeds (hash split, 20 %), ≤ 50k rows/class. `--test-files N ...` holds out whole ToN files as a scenario split | Class/file concentration in ToN means a file split cannot test every class, so report it as a separate evaluation. |

---

## 3. Running it (Windows, PyCharm terminal)

**Easiest: one file.** [`bot_ton_tl_all_in_one.py`](../bot_ton_tl_all_in_one.py) (repo root)
contains the whole pipeline. Edit the SETTINGS block at its top (`DATA_ROOT`,
`QUICK_TRIAL`, ...) and press Run in PyCharm. `--self-test` checks the installation on
synthetic data. The commands below are the same pipeline as separate modules.

```bat
cd path\to\Transfer-Learning-Based-Intrusion-Detection-in-5G-and-IoT-Networks
python -m pip install -r requirements-tl.txt
:: GPU (optional): install the CUDA build of torch from pytorch.org first

:: 0) dry run on synthetic CSVs with the real headers (2 minutes, no data needed)
python -m bot_ton_tl.synthetic --out synthetic_raw
python -m bot_ton_tl.prepare --root synthetic_raw --out prepared_synth
python -m bot_ton_tl.experiment --prepared prepared_synth --out results\synth --quick

:: 1) harmonise + de-duplicate the 27 real CSVs (all 25.9 M rows; a few minutes per dataset)
python -m bot_ton_tl.prepare --root "D:\Abdallah\Pycharm folder\pythonProject\ToN-IoT FULL" --out prepared
::    -> read prepared\prepare_report.txt first (drops, duplicates, label conflicts, byte evidence)

:: 2) how far apart are the datasets?
python -m bot_ton_tl.diagnose --prepared prepared --out results\diagnose

:: 3) main experiment: 3 seeds x 5 budgets x 2 tasks x 7 methods
python -m bot_ton_tl.experiment --prepared prepared --out results\main --include-full
```

Every stage prints `--help`. Useful switches:

* `prepare --max-rows-per-file 200000` runs a quick trial on the head of each file.
* `prepare --bot-files datasets\BoT-IoT.csv` uses another BoT-IoT release, such as the
  15-column file in this repo (DDoS and DoS 2M rows each; 9,543 normal).
* `experiment --device cuda`, `--arch cnn1d`, `--seeds 0 1 2 3 4` and `--shots 5 10 20 50`.

Measured here on CPU: preparing the repo's 5.83 M-row BoT-IoT CSV takes 51 s. Pre-training
the source model (157k balanced rows) takes 14 s and reaches in-domain macro-F1 0.853 /
balanced accuracy 0.948; most of the remaining error is DoS↔DDoS (see decision 8).

### Outputs

| file | content |
|---|---|
| `prepared/prepare_report.txt` / `.json` | rows read/kept per file, drops by reason, invalid tokens, hex/sentinel port counts, duplicates per class, label-conflict vectors, byte-semantics evidence |
| `results/diagnose/diagnose_report.txt` | domain-classifier AUC / proxy A-distance overall and per matched class, most-shifted features (KS) |
| `results/<run>/summary.txt` | macro-F1 (and ROC-AUC for binary) tables, method × labels-per-class, mean ± std |
| `results/<run>/results.csv` | one row per task/method/budget/seed with all metrics (accuracy, balanced acc, precision, DR, FAR, F1, ROC-AUC, PR-AUC, params, train time) |
| `results/<run>/per_class.csv` | per-class detection rate (binary) / recall (multiclass) |
| `results/<run>/learning_curve_*.png` | the figure for the paper |
| `results/<run>/models/` | pre-trained source & DANN encoders, scaler parameters |

---

## 4. Recommended experiment plan

1. **Main result:** the default run above, which gives binary + multiclass learning curves.
2. **Zero-day / unseen attacks** (the original project goal): hold out ToN classes that
   BoT-IoT never saw and that are never labelled. Then read their binary detection rate in
   `per_class.csv`:
   `--unseen-classes ransomware mitm backdoor`
3. **Feature ablations:** `--features volume shape proto` (no ports) and
   `--features volume shape proto ports state`.
4. **Label ambiguity:** `--merge dos+ddos`.
5. **Byte semantics:** re-run `prepare` with `--ton-bytes payload --bot-l2-bytes 0`. This is
   the mapping the earlier audits assumed. Compare `source_only` and `finetune`.
6. **Normalisation:** `--norm per_domain` and `--norm source`.
7. **Scenario split:** `--test-files ...` after checking per-file class support in the
   Phase 2.4 report. Report it separately from the hash split.

`summary.txt` ends with a **transfer-gain table**. It gives each transfer method minus
`target_only` and minus `rf`, paired by seed, as the mean difference and the number of seeds
where transfer wins. Claim a gain only where it wins on every seed.

Feature group `timing` (duration, packet rate, byte rate) is separate. It depends on how
Argus and Zeek cut flows, and it was the most shifted group in the diagnostics. Compare it
with `--features volume shape proto ports`.

Report mean ± std over ≥ 3 seeds (5 for the paper). State the test-set class counts
(printed in `summary.txt`). Do not tune anything on the ToN test set.

---

## 5. What was reused from the ChatGPT audit scripts, and what was changed

**Reused, and correct:** expected file names and row counts (3,668,522 / 22,339,021).
Header normalisation. ToN `Network_dataset_6.csv`'s extra `uid` column is harmless here,
because columns are selected by name. Hex port decoding, `-1` sentinels, the `0.0.0.0`
records, the `bytes = sbytes + dbytes` identity (so aggregates are not used as features),
the duplicate and label-conflict concern, and keeping labels, IPs and timestamps out of the
predictors.

**Changed:**
* the `sbytes ↔ src_bytes` pairing → `sbytes ↔ src_ip_bytes` with an L2 correction
  (decisions 2–3)
* audit-only phases with every item "UNVERIFIED" → explicit, flag-controlled decisions plus
  an actual model
* chunk-stratified samples for duplicate statistics → exact de-duplication over all rows
* row-random splits → feature-hash group splits
* none of those scripts trained a model → baselines + three TL methods + learning curves

---

## 6. A strong alternative worth knowing

Sarhan et al. regenerated both datasets from the original pcaps with nProbe, as
**NF-BoT-IoT(-v2)** and **NF-ToN-IoT(-v2)**. Both releases share an *identical* NetFlow
feature set (43 features in v2). That removes the Argus-vs-Zeek mismatch at the source.
If a reviewer challenges the feature mapping, running the same `experiment` design on the
NF-* versions is the cleanest robustness check. It would need a small `prepare` mapping
for the NetFlow columns.

---

## 7. References

* Koroniotis et al., *Towards the development of realistic botnet dataset in the IoT for network forensic analytics: Bot-IoT dataset*, FGCS 100, 2019.
* Moustafa, *A new distributed architecture for evaluating AI-based security systems at the edge: Network TON_IoT datasets*, SCS 72, 2021.
* Sarhan, Layeghy, Moustafa, Portmann, *NetFlow Datasets for Machine Learning-based Network Intrusion Detection Systems*, 2020 (arXiv:2011.09144). Original BoT-IoT and ToN-IoT feature sets are incompatible, and NF-* versions share one.
* Guerra-Manzanares & Huang, *Exploring Robust Intrusion Detection: A Benchmark Study of Feature Transferability in IoT Botnet Attack Detection*, 2026 (arXiv:2602.23874). Zero-adaptation cross-dataset transfer degrades sharply even when the same extractor (Argus, Zeek or CICFlowMeter) is used on both sides.
* Rodríguez, Valls, Otero et al., *Transfer-Learning-Based Intrusion Detection Framework in IoT Networks*, Sensors 22(15):5621, 2022. The CNN-B / CNN-TL work this repository implements.
* Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 17, 2016 (DANN).
* Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022 (LP-FT).
* Ben-David et al., *A theory of learning from different domains*, Machine Learning 79, 2010 (proxy A-distance).
* Zeek `conn.log` field reference (`orig_bytes` = payload bytes, `orig_ip_bytes` = IP-level bytes); Argus `ra(1)` field reference (`sbytes`, `dur`, `state`).
