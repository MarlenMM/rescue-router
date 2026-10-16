# Phase 2 protocol (pre-registered 14 Oct 2026, before any Phase 2 GPU run)

Written before the runs so that no choice depends on their results. Changes after a run starts are logged at the
end with the reason. Budget: Kaggle 2×T4 sessions (a session is billed once for its wall time; 30 h/week).

**Reading guide:** the sections were added in order, all before any Phase 2 run; where they differ, the later one
holds: "Router parameters" and "Model choice" (14 Oct) and **Amendment 1** (14 Oct night: curve on the full test,
RescueNet 5/10, lean variance design, baseline candidates, decode floor, label errors). Superseded sentences are
marked "→ amendment 1". **Amendment 2** (15 Oct) is procedural: pinned image, environment check, baseline draw seeds.
**Amendment 3** (15 Oct) adds two product-model candidates: the baselines' seed ensemble, without and with flips.
**Amendment 4** (16 Oct) moves RescueNet 5-20 shots to its full test set (+41 min of quota; Marlen keeps or drops it).
**Amendment 5** (16 Oct) adds two support draws of RescueNet 20 shots to `var` (+91 min of quota; Marlen keeps or drops
it), replaces RescueNet's draws 1-2 by clean draws 3-4 (the old ones overlap val30), and runs RescueNet first in each lane.

## Questions

| id | question | how |
|---|---|---|
| A3 | How much does the 50-shot result move with the training seed and with the choice of the 50 photos? | 3 seeds × 3 support draws (the submitted model is seed 0, draw 0): 8 new runs, eval100 → amendment 1: 4 runs (draws 1, 2 at seed 0; seeds 1, 2 at draw 0) |
| A4 | Learning curve and RescueNet with **one** recipe (the published curve mixes recipes) | 1/5/10/20/100 shots + RescueNet 20/50, recipe below, tag `p2` → amendment 1: + RescueNet 5/10; 1-20 shots on the full test |
| A5 | Same-label-budget baseline | DeepLabV3-ResNet50 on the same photos (`src/baseline_seg.py`; on the Mac, no Kaggle quota) |
| F1 | What do the 30 tuning labels buy? | every run decoded with label-free thresholds (`src/prior_decode.py` matching) next to the val30-tuned ones |
| F2 | Is val30 over-fitted? | the submitted model on `floodnet_val_confirm100` (100 val photos, disjoint from val30) |

## Fixed recipe (= the submitted model, 8 Oct)

Training: `finetune.py --shot k --n_steps 1000 --amp fp16 --grad_ckpt --freeze_rest --defer_val --val_every 250
--snap_from 750 --chunk 1 --channel_chunk 1 --gbs 2` (lr 1e-3 cosine), snapshots at 750 and 1000, their weight average
(`avg_ckpt.py`) is the run's checkpoint. Prediction: fp16, 384 tiles, stride 192. Decode (CPU, `run_decode.py`): guided
filter r 2 ε 1e-3 → per-class τ tuned on val30 (coordinate ascent, drivable-precision floor .567 as for the submitted
model; → amendment 1 item 5: min(.567, the run's own τ .5 value)) → SAM grow. RescueNet: its published decode rule (`decode_tune.py`: τ + building vote), no SAM.
100 shots: trained on the first 100 photos of the support order, predicted with the first 50 (`--eval_shot 50`; 100
supports do not fit a T4 at prediction).

## Runs and cost (measured T4 costs: 2.6 s/step; 17 s/photo at 50 supports; `kaggle/phase2/make_plan.py cost`)

**Changed by Amendment 1 (below): curve 1-20 shots on the full test + RescueNet 5/10; var = 4 runs (L design).**

| session | runs | quota |
|---|---|---|
| `var` | 50 shots, (seed, draw) ∈ {0,1,2}² minus (0,0): train + val30 + eval100 | ~5.6 h |
| `curve` | FloodNet 1/5/10/20 (`p2`), 100 (`p2`, eval_shot 50), RescueNet 20/50 (`p2`); submitted model on confirm100 | ~4.2 h |

Order: `curve` first (it also carries F2, and A4 matters most for the write-up), then `var`. Before each push:
`python kaggle/phase2/make_plan.py preflight <session>` must end with PREFLIGHT OK (toy chain on the Mac), the code
is committed, and `kaggle/b13/make_kernel.py` builds the kernel from that commit. After each session: download
(`kaggle/fetch_output.py`), `run_decode.py --sam` per FloodNet run, `phase2_report.py --routing`, refresh the off-site
backup (`src/backup.py pack` + `release/backup/upload.sh`). → amendment 1: `kaggle/phase2/launch_check.py <session>`
before the push, `kaggle/phase2/post.py <session>` runs every CPU step after the download.

## Evaluation

- **val30**: only for choosing τ (as before). **eval100**: every run's report split (paired across runs: same photos).
  **confirm100**: F2 only (→ also the router, tier and model choices, sections below). **Full test**: not used in
  Phase 2 (headline unchanged unless a later adoption passes the full-test rule) → amendment 1: the 1-20-shot curve
  runs are scored on it, paired with the baseline; the product-model choice is reported on it.
- Per run: mIoU (τ .5 / label-free / tuned / tuned + SAM), per-class IoU, drivable precision, stranded F1 and
  false-safe rate (router v3, eval100).
- A3: mean ± sd over the 9 runs, and a two-way split of the variance into support draw, training seed and rest
  (`phase2_report.py`) → amendment 1: mean ± sd over the 5 runs, sd over draws (seed 0) and over seeds (draw 0). The submitted model's place in that distribution is reported as is (it was not chosen among
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

## Model choice (added 14 Oct, before any Phase 2 run; after the A5 baseline result)

The same-budget baseline (DeepLabV3-ResNet50, the same 50 photos, no tuning labels) beats the submitted Chameleon model
on the full test: mIoU .581 vs .507 (paired +.074 [+.045, +.108]; +.068 without the 14 test photos that overlap a
training photo), stranded F1 .813 vs .752 (+.060 [+.015, +.115]), false safe .081 vs .111 (−.030 [−.053, −.012]).
val30 ranks them the other way (.470 vs .516; confirm100: baseline .546): val30 cannot choose between models either.
So, once the curve session has predicted confirm100 with the submitted Chameleon model, the product model (Chameleon,
baseline, or their probability average w ∈ {0, .25, .5, .75, 1}, `src/ensemble.py`) is chosen on val30 ∪ confirm100
by mIoU under the drivable-precision floor, and reported on the full test with the paired bootstrap. The baseline
curve (5/10/20/100/all 1,445 photos) and its two other support draws run on the Mac (no Kaggle quota) and answer
where, if anywhere, Chameleon's few-shot adaptation wins.

## Amendment 1 (14 Oct 2026, ~21:30 KST; before any Phase 2 GPU run and before any result of the runs it adds)

After A5 (the same-budget baseline beats Chameleon at 50 shots) the questions that matter are where, if anywhere,
Chameleon wins (fewer labels, a second disaster) and whether the baseline's lead survives other support draws.
Changes (`kaggle/phase2/make_plan.py cost`: curve + var 8.6 h of quota instead of 9.8 h; the two GPU lanes are now
split exactly and the submitted model's confirm100 prediction runs first):

1. **curve:** the FloodNet 1/5/10/20-shot `p2` runs predict the whole test set (448 photos) instead of eval100, so
   each point is paired photo by photo with the baseline curve on the full test; their eval100 caches are cut from
   it on the CPU (`src/subset_cache.py`; bit-exact: on the submitted model's T4 caches the eval100 photos of the
   full-test cache equal the eval100 cache byte for byte). RescueNet **5 and 10** shots (`p2`) are added: the
   low-label end on the second disaster. Cost: +3.3 h T4 (+1.7 h of quota on 2×T4).
2. **var:** the lean "L" design replaces the 3×3 grid: support draws 1 and 2 with training seed 0, and training seeds 1
   and 2 with draw 0 (4 runs, 2.9 h of quota instead of 5.6 h). Reported: mean ± sd over the three draws (seed 0) and
   over the three seeds (draw 0), each with the submitted model as the (0, 0) run; the baseline's draws 0-2 (Mac) are
   paired with Chameleon's by draw. The full grid stays available as session `var_full` (not planned).
3. **Baseline side (Mac, no quota):** RescueNet 5/10/20/50 shots, FloodNet 1 shot, training seeds 1 and 2 on draw 0 at
   50 shots, with the same fixed recipe (`src/baseline_seg.py`), so every Chameleon point has a same-photos partner.
4. **Product-model candidates** added to the "Model choice" rule (same rule: highest mIoU of the confusion summed over
   val30 + confirm100, `src/ensemble.py --choose_on val30,confirm100`; then reported on the full test with the
   paired bootstrap):
   - **self-trained baseline:** teacher = the 50-shot draw-0 baseline; pseudo-labels for the 1,395 unlabelled FloodNet
     train photos (`src/baseline_pseudo.py`: the four flips averaged, a pixel kept when its class probability is
     ≥ .9, else ignored); student = the same recipe from COCO weights, 3,000 steps of 8 labelled + 8 pseudo-labelled
     crops (`--bs 8 --accum 2 --pseudo_frac 0.5`). Uses no labels beyond the 50.
   - **test-time flips** (four views averaged) of the plain and the self-trained baseline.
   - **small models** for CPU / in-browser use (DeepLabV3-MobileNetV3, LR-ASPP-MobileNetV3), trained like the
     self-trained student (50 labels + the same pseudo-labels): candidates only by the same rule; otherwise reported
     as the speed / accuracy trade-off.
   None of these settings is tuned on any labelled photo; they are fixed here before the runs.
5. **Decode floor for runs other than the submitted model** (a fix found before any result): the "Fixed recipe"
   decode keeps val30 drivable precision ≥ .567 (the first 50-shot model's τ .5 value). Few-shot runs cannot reach
   that (published τ .5 val30 drivable precision: 1 shot .06, 5 shots .56, 10 .51, 20 .42), so every threshold move
   would be rejected and their "tuned" decode would stay at τ .5, understating Chameleon exactly where it may win. The
   floor is therefore min(.567, the run's own τ .5 val30 drivable precision) (`src/run_decode.py`): .567 for the
   submitted model (unchanged, its thresholds reproduce) and the published curve's own rule (decode_tune.py: never
   below the run's τ .5 decode) for the others.
6. **Label errors in FloodNet** (found 14 Oct by the false-safe analysis): 11 test photos and 2 validation photos
   (val30 7769, confirm100 7762) label large dry lawns or banks as water (`splits/floodnet_label_errors.tsv`,
   `src/label_audit.py`: visual check of photo and label only, never of a prediction). Every full-test and eval100
   result is reported on all photos (primary, as published) and without these photos (secondary); choices use all
   photos as registered.
7. **Confidence tier when no gate reaches 99 %** (added 14 Oct 23:00, before computing it; seen so far: the baseline's
   gate-only sweep, where no gate reaches 99 % on val30 ∪ confirm100): the candidates are extended with an off-road
   reach limit (the truck may cross dry yards only within R m of a predicted road), R ∈ {40, 20, 10} at every candidate
   gate; among all candidates whose confident-truck houses are ≥ 99 % truly truck-reachable on val30 ∪ confirm100, the
   one with the most confident houses is chosen (for gates alone this is the published rule: the largest gate). It
   changes only which houses are "confident"; statuses and the stranded F1 keep the tuned gate. `src/tiers.py --reach`.

## Amendment 2 (15 Oct 2026, ~21:30 KST; procedural only, before any Phase 2 GPU run)

1. **Environment.** Both kernels pin the Kaggle image of every 5–8 Oct run (`kaggle kernels pull -m`:
   `gcr.io/kaggle-private-byod/python@sha256:2757e0c7…`) and, before the lanes start, the submitted model predicts the
   first 3 photos of `floodnet_test_full_part1` as the published run did; `src/env_check.py` compares 4×4 block means
   with the published T4 cache and **stops the session** if max |diff| > 8/255 or mean > 1/255 (cross-device repeats
   stay within 2/255). Both sessions therefore mount `marlenmelis/rr-final-ckpt`. A stopped session is an
   infrastructure failure (re-run once after the cause is found), not a result.
2. **Baseline draws** (amendment 1 item 3, Mac): the 50-shot support draws 1 and 2 run with training seed 0
   (`--seed 0 --support_seed d`, runs `…_50shot_s0_sup<d>`), mirroring Chameleon's L design (seed 0, draw d), instead of
   seed = draw as first queued on 14 Oct (that run was killed part-way by a reboot of the Mac at ~03:00 on 15 Oct;
   nothing of it was evaluated). Every Mac job is in `experiments/phase2/mac_queue.txt` (`src/job_queue.py`, resumes
   after a crash).
3. Nothing else changes: runs, recipes, decodes, splits and choice rules are as registered.

## Amendment 3 (15 Oct 2026, ~24:00 KST; before any result of the candidates it adds, before any Phase 2 GPU run)

**Product-model candidates added** to the "Model choice" rule (same rule, same splits: highest mIoU of the confusion
summed over val30 + confirm100; ties: fewer labels, then the simpler model; then reported on the full test with the
paired bootstrap):
- **seed ensemble** `baseline_deeplabv3_r50_50shot_s0_ens3`: the mean softmax of the three 50-shot draw-0 baselines
  with training seeds 0, 1 and 2 (the runs of amendment 1 item 3), decoded by argmax (`src/baseline_avg.py`);
- the same with **test-time flips** (`…_ens3_tta`: each member's four-view average, then the mean).
Reason: averaging independently trained models is the standard way to reduce variance and usually improves both
accuracy and calibration; it uses no labels beyond the 50 and no tuning, at 3× the CPU inference of one model
(~1.4 s per photo). Nothing about these candidates is tuned on any labelled photo. In the "simpler" tie-break an
ensemble counts as more complex than any single model; the full order (`kaggle/phase2/post.py` `candidate_order`, until
now alphabetical) is: the plain ResNet-50 baseline, its flips, self-trained, self-trained + flips, the small models,
then the seed ensembles (ties at 6 decimals are unlikely; the order only decides those). The Mac jobs are in `experiments/phase2/mac_queue.txt`
(`seed1tta`, `seed2tta`, `ens3`, `ens3tta`); `kaggle/phase2/post.py` passes them to `src/ensemble.py`.

## Amendment 4 (16 Oct 2026, ~00:30 KST; before any Phase 2 GPU run; for Marlen to keep or drop at the launch)

**RescueNet 5, 10 and 20 shots (`curve`) predict RescueNet's whole test set (450 photos) instead of eval100**, as
amendment 1 did for FloodNet 1-20 shots: eval100 is cut from it on the CPU (`kaggle/phase2/post.py` step 1,
`src/subset_cache.py --dataset rescuenet`), and each point is paired photo by photo with the baseline on the full
RescueNet test (the Mac predicts it for every RescueNet baseline: `experiments/phase2/mac_queue.txt`). Reason: the
second disaster at few labels is where Chameleon may still win, and 100 photos cannot resolve differences of a few
points (FloodNet eval100 paired intervals are ±.04-.05 wide). RescueNet 50 shots stays on eval100 (17 s per photo).
Cost: +80 T4 minutes, +41 min of quota (`curve` 385 min instead of 344; curve + var 9.4 h). If dropped: in
`kaggle/phase2/make_plan.py` give the three runs their default splits again (one line), and nothing else changes.
On the Mac (no quota) the baseline side gets the realistic cross-disaster competitor at every low-label point:
DeepLabV3 started from the FloodNet 50-shot baseline and fine-tuned on the same 5 / 10 / 20 RescueNet photos (`ft50`;
it has also seen 50 FloodNet labels, reported as such, never instead of the plain baseline).
Equal total labels (16 Oct, after the first RescueNet point showed Chameleon's lead at 20 photos comes from its 30
tuning photos, the baseline gaining nothing from tuning): baseline partners at k + 30 = 35, 40, 50 and 80 RescueNet
photos (nested supports, val30 excluded), reported next to Chameleon k (+ 30 tuning) as a second view
(`curve_figure.py` curve_total.png); no choice depends on it.
The comparison is reported on the full RescueNet test where both have it (`baseline_report.py` pairs on the largest
common photo set), on eval100 otherwise; both models' label-free (k labels) decodes are reported next to the tuned
ones for RescueNet too (16 Oct: the RescueNet chain has no guided filter, so its thresholds are matched on the raw
probabilities, `baseline_report.label_free_tau`).

## Amendment 5 (16 Oct 2026, ~02:25 KST; before any Phase 2 GPU run; item 1 for Marlen to keep or drop at the launch)

1. **RescueNet 20 shots, two more support draws** (`var`): the only point where Chameleon leads so far (RescueNet 20
   photos with its 30 tuning photos: baseline −.061 [−.098, −.027] on eval100, stranded F1 .605 vs .818) rests on one
   support draw and one training run, and the bootstrap intervals cover test photos only. Draws 3 and 4 with training
   seed 0 and the one recipe (`p2`), predicted on RescueNet's full test like amendment 4
   (`rescuenet_20shot_s0_stratified_sup{3,4}_p2`); with the `curve` session's draw-0 run they give mean ± sd over
   three draws (`phase2_report.py`, lean spread). The baseline gets the same draws on the Mac (`rn20d3`, `rn20d4`,
   seed 0, full test) and `baseline_report.py` pairs them by draw. Reported: Chameleon − baseline per draw (paired
   bootstrap on the full RescueNet test) and the three draws' mean ± sd of both models, tuned and label-free. No
   choice depends on it; it says whether "Chameleon wins on RescueNet at 20 photos" holds beyond one draw.
   Cost: +182 T4 minutes, **+91 min of quota** (`var` 269 min instead of 178; curve + var 10.9 h of 30 h/week).
   If dropped: `VAR_RN_DRAWS = ()` in `kaggle/phase2/make_plan.py` (one line), and the two Mac jobs are reported
   as baseline draws only.
2. **RescueNet support draws 3 and 4 instead of 1 and 2** (a fix found before any use of them): RescueNet's supports and
   its val30 come from the same val pool, and val30 was cut outside the draw-0 supports only (5 Oct), so the
   published draws 1 and 2 contain 8 and 5 val30 photos (3 and 1 among their first 20). A run on them would tune its
   thresholds on photos it trained on. Draws 3 and 4 use the same rule (`prep_data.support_order`, seeds 3 and 4) on
   the pool without val30 (`src/splits_local.py rn_draws`; hashes appended to `splits/SHA256SUMS`; the published
   files are unchanged). `finetune.py` and `baseline_seg.py` now stop when a support photo is in the dataset's val30 or
   confirm100 (`rr.runs.check_supports`). FloodNet's draws are train photos and are not affected; no published or
   queued run used RescueNet draws 1-2.
3. **Lane order** (procedural): within each GPU lane the prediction-only group still runs first, then the RescueNet
   runs, then the rest longest first (`make_plan.lanes`). The 2-GPU split and each lane's length are unchanged; if a
   session is cut short, the runs of the open question exist first.

## Launch decisions (Marlen, 16 Oct 2026, ~08:20 KST; before any Phase 2 GPU run)

- **Amendment 4 kept** (RescueNet 5-20 shots on the full RescueNet test, +41 min of quota) and **amendment 5 kept**
  (RescueNet 20-shot draws 3 and 4 in `var`, +91 min). Sessions as built: `curve` 10 job groups, ~6.4 h of quota;
  `var` 6 job groups, ~4.5 h; both may run at the same time (the account runs two GPU sessions).
- The private dataset `marlenmelis/rr-final-ckpt` (the submitted checkpoint, 4,999,531 bytes) was created on 16 Oct and
  is ready; both sessions mount it. The launch itself (kernel pushes) is done in a new chat from the hand-off
  `PHASE2_HANDOFF.md` (project folder).

## Deviation log

Every entry is procedural: no registered run, recipe, split, decode or choice rule was changed, and no result
was looked at before the step it concerns (16-17 Oct 2026).

1. **Launch timing.** `curve` was pushed on 16 Oct 08:41 KST and passed its environment check after 2.9 min
   (`ENVCHECK PASS: max |diff| 0.50/255, mean 0.106/255`). `var` was meant to follow at once but was pushed at 16:53:
   the watcher read `kaggle kernels logs` (no `-f`), which prints nothing until a script kernel ends, so the check's
   line was seen only when `curve` had finished. No effect on any run (same image, same code `132a463`); `var` passed
   the same check (identical values). The two sessions ran one after the other instead of side by side.
2. **Session lengths.** `curve` 8.13 h (planned ~6.4 h), `var` 5.63 h (~4.5 h): fine-tunes took 47-63 min and full-test
   predictions 29-65 min on the T4s. Every job group started before its start limit (curve 7.1 h, var 5.2 h); none
   failed or was skipped (`curve` 10/10, `var` 6/6).
3. **Mac restart during post-processing** (16 Oct ~19:30-20:40): `router_sweep.py`'s 8 parallel `route_eval` workers
   (~4.6 GB each) next to the Mac queue's 7.5 GB MPS training exhausted the 24 GB of memory and 33 GB of swap; the Mac
   was restarted, killing every local job. Post-processing resumed where it stopped (each step skips finished
   outputs; all 146 router-sweep files checked as valid JSON). Since then the CPU pools size themselves to the free
   memory (`src/rr/mem.py`) and `src/memguard.py` pauses the youngest heavy job of this repository when memory runs
   out (PLAN §16.8). A second worker for the remaining router-sweep configurations ran the same jobs in reverse order
   (identical commands; finished files skipped).
4. **The last baseline jobs ran on a Kaggle 2×T4 (CUDA) instead of the Mac (MPS)** (Marlen, 16 Oct ~21:00: "we can
   run anything you want in kaggle"): `rn35`, `rn40`, `rn80` (amendment 4, equal total labels) and `rn20d3`,
   `rn20d4` (amendment 5), session `rr-p2-baselines` (`kaggle/phase2/baselines_kernel.py`; same script at `132a463`,
   recipe, seeds and photos; `rn35` was stopped on the Mac at step 600 of 3,000 and rerun from scratch on Kaggle). Device
   control in the same session: the RescueNet 20-photo baseline (draw 0), already trained on the Mac, trained again on
   CUDA: full RescueNet test .387 vs .393, CUDA − MPS **−.005 [−.010, −.002]** (eval100 −.008 [−.018, +.001];
   `results/baselines/report/device_control.md`, `experiments/device_control.py`), the size of the spread between
   support draws (sd .008); the moved baselines may sit up to that much lower than Mac runs would have.
5. **The queue's `reports` step was run in its Phase 2 form.** `rebuild_all.py --only same_budget` called
   `baseline_report.py` and `curve_figure.py` without `--phase2`, which would have written the pre-Phase-2 pairing
   over the Phase 2 report; they were run with `--phase2 results/phase2/rr` (and `rebuild_all.py` now passes it when
   the Phase 2 runs exist). `choice_bl` ran unchanged.
6. **Reporting additions (no choice depends on them):** (a) the registered product-model choice on the full test with
   the paired bootstrap the protocol names, which `ensemble.py` did not compute (`ensemble.py --paired`: vs the
   submitted Chameleon and the plain baseline, all photos and without the label-error photos; plus routing of the
   chosen mix, the router reading the mix's probabilities); (b) `baseline_report.py` pairs each A3 replicate with the
   Chameleon run of the same training seed and support draw (before: of the same draw, so the baselines' seeds 1-2
   were compared with the submitted seed-0 model) and reports the replicates of both models on one photo set per
   group (FloodNet 50: eval100; RescueNet 20: the full test), tuned and with equal labels, with mean ± sd
   (amendment 5's registered report).
7. **RescueNet's cautious tier** has no confirm100 split: it stays at its published val30 choice (0.15); the
   registered re-choice on val30 ∪ confirm100 concerns FloodNet.
