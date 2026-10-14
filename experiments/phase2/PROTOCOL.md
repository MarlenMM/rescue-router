# Phase 2 protocol (pre-registered 14 Oct 2026, before any Phase 2 GPU run)

Written before the runs so that no choice depends on their results. Changes after a run starts are logged at the
end with the reason. Budget: Kaggle 2×T4 sessions (a session is billed once for its wall time; 30 h/week).

## Questions

| id | question | how |
|---|---|---|
| A3 | How much does the 50-shot result move with the training seed and with the choice of the 50 photos? | 3 seeds × 3 support draws (the submitted model is seed 0, draw 0): 8 new runs, eval100 |
| A4 | Learning curve and RescueNet with **one** recipe (the published curve mixes recipes) | 1/5/10/20/100 shots + RescueNet 20/50, recipe below, tag `p2` |
| A5 | Same-label-budget baseline | DeepLabV3-ResNet50 on the same photos (`src/baseline_seg.py`; on the Mac, no Kaggle quota) |
| F1 | What do the 30 tuning labels buy? | every run decoded with label-free thresholds (`src/prior_decode.py` matching) next to the val30-tuned ones |
| F2 | Is val30 over-fitted? | the submitted model on `floodnet_val_confirm100` (100 val photos, disjoint from val30) |

## Fixed recipe (= the submitted model, 8 Oct)

Training: `finetune.py --shot k --n_steps 1000 --amp fp16 --grad_ckpt --freeze_rest --defer_val --val_every 250
--snap_from 750 --chunk 1 --channel_chunk 1 --gbs 2` (lr 1e-3 cosine), snapshots at 750 and 1000, their weight average
(`avg_ckpt.py`) is the run's checkpoint. Prediction: fp16, 384 tiles, stride 192. Decode (CPU, `run_decode.py`): guided
filter r 2 ε 1e-3 → per-class τ tuned on val30 (coordinate ascent, drivable-precision floor .567 as for the submitted
model) → SAM grow. RescueNet: its published decode rule (`decode_tune.py`: τ + building vote), no SAM.
100 shots: trained on the first 100 photos of the support order, predicted with the first 50 (`--eval_shot 50`; 100
supports do not fit a T4 at prediction).

## Runs and cost (measured T4 costs: 2.6 s/step; 17 s/photo at 50 supports; `kaggle/phase2/make_plan.py cost`)

| session | runs | quota |
|---|---|---|
| `var` | 50 shots, (seed, draw) ∈ {0,1,2}² minus (0,0): train + val30 + eval100 | ~5.6 h |
| `curve` | FloodNet 1/5/10/20 (`p2`), 100 (`p2`, eval_shot 50), RescueNet 20/50 (`p2`); submitted model on confirm100 | ~4.2 h |

Order: `curve` first (it also carries F2, and A4 matters most for the write-up), then `var`. Before each push:
`python kaggle/phase2/make_plan.py preflight <session>` must end with PREFLIGHT OK (toy chain on the Mac), the code
is committed, and `kaggle/b13/make_kernel.py` builds the kernel from that commit. After each session: download
(`kaggle/fetch_output.py`), `run_decode.py --sam` per FloodNet run, `phase2_report.py --routing`, refresh the off-site
backup (`src/backup.py pack` + `release/backup/upload.sh`).

## Evaluation

- **val30**: only for choosing τ (as before). **eval100**: every run's report split (paired across runs: same photos).
  **confirm100**: F2 only. **Full test**: not used in Phase 2 (headline unchanged unless a later adoption passes the
  full-test rule).
- Per run: mIoU (τ .5 / label-free / tuned / tuned + SAM), per-class IoU, drivable precision, stranded F1 and
  false-safe rate (router v3, eval100).
- A3: mean ± sd over the 9 runs, and a two-way split of the variance into support draw, training seed and rest
  (`phase2_report.py`). The submitted model's place in that distribution is reported as is (it was not chosen among
  seeds, so there is no selection bias to correct).
- A4: one point per shot (single run; A3 gives the noise level at 50 shots).
- Every number is reported, including runs that fail or look worse; a failed run is re-run once if the cause is
  infrastructure (OOM, session loss), otherwise reported as failed.

## Decisions this protocol allows (and no others)

1. **Write-up**: the A3 sd replaces "single run" caveats; the A4 curve replaces the mixed-recipe curve; A5 is the
   like-for-like comparison; F1 states the label budget with a number.
2. **Headline model**: unchanged by Phase 2 (no Phase 2 run is a candidate: the seeds/draws are replicates, not
   variants). A later model change (roadmap B) needs its own pre-registration and the full-test paired rule.
3. **F2**: if the submitted model's confirm100 mIoU is more than .03 below its val30 (.516), every val30-based claim in
   the write-up is marked as possibly optimistic.

## Router parameters (added 14 Oct, before any confirm100 prediction exists)

val30 cannot choose router parameters: it holds 52 stranded houses and every off-road gate from 0.3 up gives the same
stranded F1 (.980). Once the curve session has predicted confirm100 with the submitted model, the off-road gate is
re-chosen on val30 ∪ confirm100 (130 photos, pooled) from {none, .5, .3, .2, .15, .1} by max stranded F1 (a tie that
includes 0.3 keeps 0.3; otherwise the tied gate with the lowest false-safe rate), and replaces 0.3 only if its
false-safe rate on the same photos is not higher; it is then reported on eval100 and the full test with the
paired bootstrap (`src/router_sweep.py --splits val30,confirm100,eval100`). The confidence tier's cautious gate is
re-chosen the same way with its published rule (the largest gate whose "confident truck" houses are ≥ 99 % truly
truck-reachable), now on val30 ∪ confirm100 (`src/tiers.py`). F2 (segmentation) is unaffected: confirm100
plays no part in choosing the decode. Not candidates (measured 14 Oct, `results/routing/router_sweep/`): calibrated
wetness for the router (no change on eval100) and a limit on off-road distance (false safe .119 → .010 but stranded F1
.819 → .658: a precision/recall trade-off that the confident tier already offers).

## Deviation log

(none yet)
