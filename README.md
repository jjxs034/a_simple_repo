# PowerBrain: code, configuration and reproduction materials

Anonymous release for the review of the paper "PowerBrain: A Detector-Grounded Multimodal Large Language Model for
Efficient Power Line Inspection on Edge Devices". It contains the code of every experiment, the training and evaluation
configuration, the Chinese prompts, the question rules (implemented in `powerbrain/bench.py`), the InsPLAD question sets,
candidate lists, per-question outputs and adapters, and de-identified per-question records of the ED and TD runs. The ED
and TD images and annotations are provided under an agreement with the company and cannot be released (see "Not
included").

## Contents

| Path | What it is |
|---|---|
| `powerbrain/` | Pipeline, training, evaluation and report scripts (Python) |
| `edge/` | Conversion for the RK3588 board and the board-side measurement scripts |
| `scripts/` | `final_round.sh` (all ED/TD runs of the paper), `run_eval3.sh` (its evaluation step), `freeze_final.sh` (all tables from the per-question outputs), `pub_insplad.sh` (the InsPLAD replication) |
| `configs/` | `training.json`, `candidate_list.json`, `degradations.json`: every setting, with the file and line it comes from |
| `prompts/prompts_zh.md` | The Chinese prompts and three complete examples |
| `data/insplad/bench/` | InsPLAD question sets, cross-fitting fold groups and candidate lists |
| `data/insplad/results/` | InsPLAD per-question outputs of the five systems and their summary |
| `data/edtd/` | De-identified per-question records of the ED and TD runs |
| `adapters/insplad/` | The four InsPLAD adapters, each split into two parts |
| `requirements.txt` | Python packages of the experiment environment |
| `LICENSE`, `DATA_LICENSE.md`, `LICENSES/` | Licences (see the last section) |

## Setup

- Python 3.12, CUDA 12.8, the packages of `requirements.txt` (PyTorch 2.10, Transformers 4.57.6, PEFT 0.21.2,
  Accelerate 1.12.0, Ultralytics 8.4.174).
- Base models: `Qwen/Qwen3-VL-4B-Instruct` and `Qwen/Qwen3-VL-2B-Instruct`.
- InsPLAD-det images and annotations: Vieira-e-Silva et al., Mendeley Data, DOI 10.17632/5n3fjgvfyz.1. Extract it so
  that one folder holds `annotations/`, `train/` and `val/`.
- Run the Python scripts from `powerbrain/` (they import each other by module name), or set `PB_CODE` to that folder.

Paths are given by environment variables:

| Variable | Meaning |
|---|---|
| `PB_PL_ROOT` | the extracted InsPLAD-det folder (used when questions or detector data are built) |
| `PB_PATH_MAP` | `old prefix=new prefix;...`, maps image paths stored in the question files to this machine (`core.local_path`) |
| `PB_FOLD_GROUPS` | `data/insplad/bench/fold_groups.json` for InsPLAD: images of one tower share a cross-fitting fold |
| `PB_SCENARIOS` | `pl` for InsPLAD |
| `PB_W`, `PB_E`, `PB_M4`, `PB_M8` | work folder, data and output folder, 4B and 8B base models of `scripts/final_round.sh` |
| `PB_PUB`, `PB_ENV` | work folder and Python environment of `scripts/pub_insplad.sh` |
| `PB_GRID_DATA`, `PB_ED_DIR`, `PB_TD_DIR` | location of the ED and TD source data (not released) |
| `PB_EDGE` | the board-side folder (default `~/pb_edge`) |

The InsPLAD question files and `fold_groups.json` store image paths with the placeholder prefix `INSPLAD_ROOT/`
(for example `INSPLAD_ROOT/val/292-1_DJI_0130.jpg`). Map it to your copy with
`export PB_PATH_MAP="INSPLAD_ROOT=/path/to/InsPLAD-det"`. Splits and degradations depend only on file names
(`bench.is_val`, `corrupt._rng`). The fold of an image is that of its tower, which `head.fold_of` looks up in
`fold_groups.json` by the same path string as in the question files, so both must carry the same prefix (step 3
rewrites them together).

## InsPLAD materials

- `pl_test.jsonl`: 2,200 test questions on 1,572 images of the official validation partition (600 which-class,
  1,000 presence, 600 count questions). `pl_fit.jsonl` (9,907) and `pl_val.jsonl` (1,093) split the official
  training partition; `pl_train.jsonl` holds both.
- `fold_groups.json`: every training image to its tower (167 towers); the two cross-fitting folds are drawn by tower,
  so no tower is split between them.
- `f_evid_pl_eval_m576.jsonl`: candidate lists of the full detector for the test and validation questions;
  `f_evid_pl_fit_m576.jsonl`: out-of-fold lists for training (the method); `_ins`: in-sample lists; `_gt`: lists
  from the annotated boxes (the two controls of Table IV).
- `results/<run>/shard0.jsonl`: one record per test question with the option log-probabilities, the predicted and the
  reference answer, correctness, input lengths and time. `results/pl_detector.jsonl`: the detector with rules.
- `results/p30_summary.json`: accuracies, paired differences and 95% intervals (2,000 resamples of test images).

| Run | System | Accuracy (%) | Where in the paper |
|---|---|---|---|
| `pl_one576` | PowerBrain, out-of-fold lists + 576-token thumbnail | 94.2 | Table IV, InsPLAD column |
| `pl_abl_ins` | trained on in-sample lists | 91.9 | Table IV |
| `pl_abl_gt` | trained on lists from annotated boxes | 90.8 | Table IV |
| `pl_w576` | 576-token whole image | 92.5 | Table IV note; Section III-B (+1.6, 0.6 to 2.6) |
| `pl_detector` | detector with rules | 92.5 | summary only |

## InsPLAD adapters

All adapters are LoRA adapters of the language model (`configs/training.json`), stored in float32 as trained. Each
archive holds `adapter_model.safetensors`, `adapter_config.json`, `train_args.json` (paths shown relative to the work
folder `$PUB` of `scripts/pub_insplad.sh`) and the PEFT model card; only the path fields were changed. The archives
are split into parts of at most 90 MiB. To rebuild and check them:

```
cd adapters/insplad
for a in pl_one576 pl_abl_ins pl_abl_gt pl_w576; do cat insplad_adapter_$a.tar.part-* > insplad_adapter_$a.tar; done && sha256sum -c SHA256SUMS
```

| Archive | Bytes | sha256 | Produces |
|---|---|---|---|
| `insplad_adapter_pl_one576.tar` | 132,208,640 | `dcdf7a91325df7ca8dee804c61d28a4a98c4428a9369ff5a9acd9968500c07da` | `results/pl_one576` |
| `insplad_adapter_pl_abl_ins.tar` | 132,208,640 | `7464af426cbcd8a39046271a2982e4c7da92219a0b124b3e94988c1792f9c3e7` | `results/pl_abl_ins` |
| `insplad_adapter_pl_abl_gt.tar` | 132,208,640 | `b0e983353b5540533839d5eb05632252ff25246002c164d99e4b03dd2bfffa84` | `results/pl_abl_gt` |
| `insplad_adapter_pl_w576.tar` | 132,208,640 | `3e3ef18f968a1bbf4203b693684dce7a00889fe323d0cde284c7298081ca571a` | `results/pl_w576` |

## ED and TD: de-identified per-question records

`data/edtd/` holds one record per question and run of the ED and TD experiments, with no image, question text,
option, reference answer, model answer, score, box or file name. The records are licensed under CC BY-NC 4.0; the
underlying ED and TD data are not released (see "Licences").

| File | Columns |
|---|---|
| `questions.csv.gz` | `file`, `order`, `qh`, `ih`, `dataset`, `form`, `frame_size_bin`, `evidence_area_bin`, `near_duplicate` |
| `outputs.csv.gz` | `run`, `system`, `seed`, `condition`, `policy`, `qh`, `correct` |
| `outputs_rules.csv.gz` | the two rule rows of Table III, which the report scripts compute from detections |
| `runs.json` | per run: system, seed, condition, files, records and input policies |

- `qh` and `ih` are keyed hashes (HMAC-SHA256, first 16 hex digits) of the question and of the image; the key is not
  released. Questions are listed in the order of the original question files, so the images enter the bootstrap in
  the same order as in the paper and the intervals come out identical. In the `file` column, `ed_test_notTaDiao` is
  the ED test set without the questions about the tower crane, used for the held-out hazard class experiment.
- `condition` is `clean` or the degradation and its strength (for example `fog:2`). `run` is the name used by
  `scripts/final_round.sh` and `scripts/freeze_final.sh`.
- The three derived fields are coarse class labels: `frame_size_bin` (megapixels of the source image: <=1.5, 1.5-4,
  4-6, >6), `evidence_area_bin` (share of the frame covered by the boxes of the asked class: <0.2%, 0.2-0.5%,
  0.5-1%, 1-5%, >=5%), and `near_duplicate` for test images (0: no training or validation image within 26 bits of a
  256-bit difference hash; 1: closest at 8 to 26 bits; 2: closest at 7 bits or fewer).

To recompute the ED and TD numbers of Tables II-IV, Fig. 4 and the size, resolution, form and near-duplicate strata:

```
cd powerbrain
python recompute_edtd.py --data ../data/edtd --out edtd_tables.json
```

On the authors' machine this reproduces all 1,689 compared entries of the frozen reports (accuracies, seed means and
standard deviations, 95% intervals and paired differences), none different. The rows of Table V measured on the
RK3588 board come from the aggregates of the board logs and are not in the per-question records.

## Reproduction

### 1. InsPLAD numbers from the released outputs (seconds, standard library only)

```
cd powerbrain
D=../data/insplad
python p30_report.py --items $D/bench/pl_test.jsonl \
  --system ours=$D/results/pl_one576/shard0.jsonl ins=$D/results/pl_abl_ins/shard0.jsonl \
           gt=$D/results/pl_abl_gt/shard0.jsonl w576=$D/results/pl_w576/shard0.jsonl rules=$D/results/pl_detector.jsonl \
  --pair ours-w576 ours-rules ours-ins ours-gt ins-gt --out p30_summary.json
```

The output is identical to `data/insplad/results/p30_summary.json`.

### 2. Evaluating a released adapter on InsPLAD (one GPU)

```
cd powerbrain
export PB_PATH_MAP="INSPLAD_ROOT=/path/to/InsPLAD-det" PB_SCENARIOS=pl
python evaluate.py --items ../data/insplad/bench/pl_test.jsonl --out-dir out/pl_one576 \
  --model Qwen/Qwen3-VL-4B-Instruct --lora /path/to/insplad_pl_one576 --policies g:e \
  --regions e=../data/insplad/bench/f_evid_pl_eval_m576.jsonl --shard 0 --num-shards 1
```

Use `--policies u576` without `--regions` for `pl_w576`.

### 3. The InsPLAD replication from scratch (one node with four GPUs, about 1.5 hours)

`scripts/pub_insplad.sh` runs every step: detector data, the full detector and the two fold detectors (YOLO11m,
longer side 1280, 30 epochs), detections, candidates and lists, the four adapters, the detector with rules, and the
evaluation. Before it, prepare `$PB_PUB`:

```
mkdir -p $PB_PUB/bench $PB_PUB/weights
ln -s /abs/path/to/InsPLAD-det $PB_PUB/data           # the job reads PB_PL_ROOT=$PB_PUB/data
for f in pl_train.jsonl pl_fit.jsonl pl_val.jsonl pl_test.jsonl fold_groups.json; do
  sed "s#INSPLAD_ROOT/#$PB_PUB/data/#g" data/insplad/bench/$f > $PB_PUB/bench/$f
done
cp -r powerbrain $PB_PUB/code                         # plus yolo11n.pt in $PB_PUB/code (Ultralytics AMP check)
# $PB_PUB/weights: Qwen3-VL-4B-Instruct/ and yolo11m.pt
```

The question files are the ones of the paper; `python bench.py --scenarios pl --out-dir <dir>` rebuilds them, and
`fold_groups.json` maps every training image to its tower (`bench.tower_of`). A step whose output exists is
skipped.

### 4. How the ED and TD runs were produced

These runs need the ED and TD data. The scripts show every step; `scripts/final_round.sh` expects the code in
`$PB_W/pb`, `run_eval3.sh` in `$PB_W` and a Python environment in `$PB_W/.venv`:

- `scripts/final_round.sh` defines every run of the paper: candidate lists, the degradation plan of the training
  images, all adapters (three seeds for the main rows), evaluation on the clean and degraded test images, held-out
  classes, held-out question forms, open-form questions, and the single-GPU cost measurement.
- `scripts/freeze_final.sh` writes every table from the stored per-question outputs: `audit_final.py` (consistency
  of settings and outputs), `maintable.py` (Table II and the rows of Table IV), `strata.py` and `dupeffect.py`
  (strata and near-duplicate check), `robustreport.py` and `robust_ci.py` (Fig. 4 and the degraded columns of
  Table IV), `presence.py`, `xreport.py` and `openformreport.py` (Table III), `locoreport.py` and `tier2_report.py`
  (held-out classes and Tier 2), then `mdtables.py` and `texnumbers.py` (the numbers in the paper).
- Cost (Table II time and memory, Fig. 2): `cost_pack.py`; training cost: `train_cost.py`; problem (12) on the
  validation images: `select_demo.py`.
- Table V (RK3588): `edge/merge_lora.py`, `edge/dump_prompts.py` and `edge/convert_rkllm.py` (language model),
  `edge/convert_detector.py`, `edge/check_detector.py` and `edge/compare_detections.py` (detector), then on the board
  `edge/bench_setup_rock5bp.sh` (fixed clocks), `edge/board_p29.sh` with `edge/board_watch.py`, and
  `edge/p29_summary.py`, `edge/p29_compare.py`, `edge/p29_costfit.py` and `edge/boardnumbers.py` for the numbers. The
  operating-point sweep uses `edge/board_p31*.sh` and `edge/p31*_analysis.py`.

Comments in the code refer to pre-registered protocols by number (P16 to P32) and to the authors' run records under
`docs/results/`; these notes are not part of the release.

## Question rules

Every answer is derived by fixed rules from the boxes that come with each dataset; the rules are implemented in
`powerbrain/bench.py` (base forms), `powerbrain/bench_x.py` (further forms) and `powerbrain/openform.py` (forms without
options), and the code is authoritative. A which-class question is asked only about an image whose boxes all belong to
one class; its options are the five classes of ED or TD, or, among the 17 classes of InsPLAD, that class and four
others drawn at random. A presence question asks whether a class is present, with yes and no in equal shares. A count
question asks how many instances of a countable class are present, with the options 1, 2, 3 and 4 or more. One
training image in ten is held out for validation by a hash of its file name. No separate written annotation protocol
exists.

## Not included

- ED and TD images, annotations and boxes, question texts, reference and model answers, scores and file names: provided
  under an agreement with the company and cannot be released.
- The ED and TD adapters: trained on the company's data, and of no use without the ED and TD images.
- Detector weights.
- Raw board outputs.

## Licences

- Code: GNU Affero General Public License v3.0 (`LICENSE`), because `powerbrain/detector.py`,
  `powerbrain/branchstate.py`, `powerbrain/cost_pack.py` and `edge/convert_detector.py` use Ultralytics, which is
  AGPL-3.0.
- InsPLAD-derived files (`data/insplad/`) and the four InsPLAD adapters (`adapters/insplad/`): Creative Commons
  Attribution Non Commercial 3.0 Unported, the licence of InsPLAD-det; see `DATA_LICENSE.md`.
- ED and TD per-question records (`data/edtd/`): Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0); attribution to the authors of the paper (names withheld during review). The licence covers only the
  de-identified records; the underlying ED and TD data are not released and not covered (see `DATA_LICENSE.md`).
- Base models: Qwen3-VL-4B-Instruct and Qwen3-VL-2B-Instruct (Qwen Team) are licensed under the Apache License 2.0
  (`LICENSES/Apache-2.0.txt`). The adapters are LoRA weights to be applied to Qwen3-VL-4B-Instruct; this
  repository contains no base-model weights.
