# Rescue Router

Few-shot flood mapping from drone photos, and rescue routes on the predicted map. KAIST AIC200 mini-project
(Chameleon, Track A + B).

**Status (15 Oct 2026).** Submitted on 8 Oct (tag `submitted-2026-10-08`; the sections below describe that model).
Since then: an ordinary segmenter (DeepLabV3-ResNet50) fine-tuned on the same photos beats the submitted Chameleon
model at 50 labels and, with the same labels for both, at every label count measured so far (5–50); the one-recipe
Chameleon curve, its training variance and the second disaster are the next GPU round (Phase 2, prepared, not
started). Details: "Since submission" below, `PLAN.md` §16.

**Track A (segmentation).** [Chameleon](https://github.com/GitGyun/chameleon) (Kim et al., ECCV 2024), meta-trained
by the course, is adapted with **50 labelled FloodNet photos** to five classes: clear road, flooded road, flooded
building, dry building, water. **Label budget:** 30 more labelled validation photos (val30) chose the checkpoint, the
decode thresholds, the router settings and every Batch 13 idea, so the system has seen 80 labelled photos in all.
With thresholds set from the 50 training labels alone (class shares matched on unlabelled photos,
`src/prior_decode.py`) it scores **.468**; with no thresholds at all (plain 0.5), .278. On the full 448-photo FloodNet test set the final model reaches **mIoU .507 [.474, .538]**
(bootstrap 95 % CI) with its decode tuned on 30 validation photos; our first model reached .472, and a round of 36
improvement ideas tested on the validation photos only (Batch 13: a 1,000-step schedule, a weight average of two
snapshots, a guided-filter decode and SAM to redraw building outlines; `writeup/batch13_improvements.md`) added
+.035 [+.013, +.057] (paired). PSPNet trained on 1,445 labelled photos scores .848 on the same classes, so this is far
below full supervision; flooded roads are the gap (IoU .31 vs .82). **A standard supervised model trained on the same
50 photos does better** (14 Oct, `src/baseline_seg.py`, `results/baselines/report/`): DeepLabV3-ResNet50 (COCO
weights) reaches **.581** with no tuning labels (paired +.074 [+.045, +.107]), routes better (stranded F1 .813 vs .752)
and more safely (false safe 8 % vs 11 %), and runs in 0.5 s per photo on a CPU (Chameleon: 17 s on a T4 GPU). Learning curve and the second dataset (RescueNet, Hurricane Michael, 20 labels: .421 on eval100, a fixed
100-photo subset of the test set) are in `results/kaggle_v2/report/`.

**Track B (routing).** A router on 4×4-pixel cells sends a truck along dry roads, then a boat across water, then a
walk to each house (never back into a vehicle), with vehicle widths, parked cars, tree canopy and a corner rule. On
the predicted maps it finds the houses that cannot be reached by road with **F1 .752** on the full FloodNet test
(.819 on eval100; RescueNet .818; the first model's .759 / .835 are within noise of these). The dangerous error is
the **false-safe rate**: 11 % [5 %, 18 %] of the houses it shows as truck-reachable are not (full test; RescueNet 3 %);
7 % once 11 test photos are set aside whose FloodNet labels mark dry lawns and river banks as water (42 of the 105
false-safe houses are in two of them, where the model is right; `splits/floodnet_label_errors.tsv`). In most of the
rest the truck route crosses water or flooded road that the model left unlabelled and the router took for dry yard
(`results/routing/false_safe/`), which is why each truck-reachable house gets a confidence tier: all 208 "confident" houses on FloodNet eval100 were truly truck-reachable; the others say
"check first" (79 %). The "labelled map" these numbers
compare with is the same router run on the human labels: the datasets label pixels, not reachability.

**Try it:** <https://marlenmm.github.io/rescue-router-demo/> (public copy of `docs/`). The web viewer (`docs/`, no server needed) shows all 200 eval100 photos: prediction vs labelled map,
house statuses and tiers, animated truck → boat → walk routes, minutes and metres, a water-probability layer and
shareable links. The Gradio app (`src/gradio_app.py`) does the same in Python and, on a GPU, runs the model on an
uploaded photo.

## Repository

| path | what |
|---|---|
| `src/` | all code: data prep, fine-tuning and inference (GPU), metrics, decode tuning, router, evaluation, reports, viewer export, Gradio |
| `src/rr/` | data loading, tiled inference, decode rules and metrics |
| `src/tests/` | Python tests (router, metrics/decode, scene packing, Phase 2 tooling, baseline resume, job queue) |
| `app/` | web viewer source (plain HTML/JS + Leaflet 1.9.4, vendored); `app/js/router.js` is the JS port of `src/router.py` |
| `app/test/` | JS router parity test against the Python reference (bit-exact on all 200 scenes); Korean coverage of the interface |
| `docs/` | the built viewer (GitHub Pages folder), made by `src/build_viewer.py`; published at <https://marlenmm.github.io/rescue-router-demo/> by `src/publish_demo.sh` |
| `notebook/` | the final notebook: generator `build_notebook.py`, executed copy `rescue_router_executed.ipynb` (the submitted one) |
| `writeup/`, `slides/` | write-up (LaTeX, figures, PDF) and slides (PPTX, sources, GIF backup, recording storyboard) |
| `release/` | submission builder and the app zip's README |
| `kaggle/` | the Kaggle GPU notebooks (training queue, checkpoint selection, test-time flips, live checks, Gradio share link) |
| `splits/`, `supports/` | fixed evaluation splits and the support (training) photo lists, with SHA-256 sums |
| `demo/uploads/` | three FloodNet test photos (not in eval100) for the live-upload demo |
| `experiments/phase2/` | the pre-registered protocol of the next GPU round and the Mac's baseline job list |
| `kaggle/phase2/` | its session plans, Kaggle kernel builder and the local pre-flight |
| `release/backup/` | off-site backup manifest (SHA-256 of every backed-up file) and upload script |
| `MODEL_CARD.md`, `DATA.md` | what the model is, how far to trust it; data sources, licences and splits |
| `PLAN.md` | the project plan and the full log of decisions and results (§16 first since 14 Oct) |

## Since submission (14–15 Oct 2026)

The submitted state is tag `submitted-2026-10-08`; the public demo stays frozen until grading (`release/DEMO_FROZEN`).
Work since then (PLAN §16; §16.5 for the latest review):

- **Same-budget baseline** (`src/baseline_seg.py`, `src/baseline_report.py`): DeepLabV3-ResNet50 fine-tuned on the
  same 50 photos beats Chameleon on the full test (mIoU .581 vs .507, stranded F1 .813 vs .752, false safe .081 vs
  .111; paired intervals exclude 0) and runs on a CPU. **Learning curve** (eval100, same photos, baseline − Chameleon;
  `results/baselines/report/baselines.md`, `curve/curve.png`): 5 photos −.013 [−.053, +.025], 10 photos −.014
  [−.061, +.032] (level, although Chameleon's decode also uses the 30 tuning labels), 20 photos +.047 [+.004, +.090],
  50 photos +.074 on the full test; the baseline reaches .609 with 100 photos and .728 with all 1,445 (full test).
  **With equal labels** (Chameleon decoded with thresholds from its k training labels only) the baseline is ahead at
  every point: 5 +.021 [−.012, +.053], 10 +.035 [+.004, +.066], 20 +.096 [+.054, +.138], 50 +.114 [+.083, +.148].
  1 photo, RescueNet 5–50, other support draws and training seeds run on the Mac
  (`experiments/phase2/mac_queue.txt`, restartable: `src/job_queue.py`); Chameleon's one-recipe curve is Phase 2.
- **FloodNet label errors** (`src/label_audit.py`, `splits/floodnet_label_errors.tsv`): 11 test and 2 validation
  photos label large dry areas as water; results are given with and without them.
- **Phase 2** (GPU, not started): pre-registered in `experiments/phase2/PROTOCOL.md` (amendment 1: one-recipe curve
  paired with the baseline on the full test, lean variance design, RescueNet 5–50 shots, product-model candidates);
  `kaggle/phase2/make_plan.py` builds the sessions (pinned to the 8 Oct Kaggle image; each session first checks that
  the submitted model reproduces its published predictions, `src/env_check.py`, and stops otherwise),
  `kaggle/phase2/launch_check.py` checks the launch list, `kaggle/phase2/post.py` runs every CPU step afterwards.
- Engineering: off-site backup (`src/backup.py`), configurable paths (`RR_DATA`), `pyproject.toml` + `uv.lock`, CI,
  the Chameleon scripts on Apple MPS / CPU for pre-flight checks (`RR_DEVICE`), calibration, label-free decode,
  false-safe analysis, CSV house list, truck stop points and a Korean interface in the viewer (local `docs/`;
  links now open exactly the view they name, Esc closes a route, the About text states the false-safe rate and the
  baseline result). Baseline training resumes after a crash or reboot (`--save_every`).
  Model and data: `MODEL_CARD.md`, `DATA.md`.

## Setup (CPU)

```bash
uv sync --locked          # CPU environment from uv.lock (or the pip line below); .venv may be a symlink out of iCloud
```

```bash
python3.12 -m venv ~/.venvs/rr && ~/.venvs/rr/bin/pip install -r requirements.txt
```

Data: the preprocessed 768×576 photos and labels in `~/rr_data/{floodnet_r768,rescuenet_r768}` (from
`src/prep_data.py`; FloodNet 1024-px photos for the viewer in `~/rr_data/floodnet_r1024`). Prediction caches,
checkpoints and decodes in `results/kaggle_v2/rr/` (the data zip holds them packed; restore with
`python src/cache_pack.py unpack --src <unzipped folder>`). Node.js ≥ 20 for the JS test.

## Reproduce

GPU (not needed to rebuild the numbers): fine-tuning and the prediction caches ran on Kaggle T4s with
`kaggle/queue/run_queue_kaggle.py` (queues `core`, `rerun50`, `select50`, `extra`, `flips`); see PLAN §15.

CPU, everything from the caches in one command (~30 min on 8 cores; logs in `results/rebuild_logs/`):

```bash
~/.venvs/rr/bin/python src/rebuild_all.py
```

It rebuilds the result tables, decode tuning, routing reports (eval100, full test, gate sweeps, physical-rule
ablation, tiers, sensitivity), bootstrap CIs, near-duplicate check, failure analysis, figures, the test-time-flip
report, the 200 viewer scenes and `docs/`, runs the tests and checks the headline numbers against the published ones
(`--check` alone does only that; `--list` shows the steps).

## Run

```bash
python3 -m http.server 8766 --directory docs
```

then open <http://localhost:8766/#s=rescuenet_12195&h=8&m=gt> (canal row: truck, boat across the canal, then on foot).

```bash
~/.venvs/rr/bin/python src/gradio_app.py
```

serves the Gradio app from the caches on <http://127.0.0.1:7860>; `src/gradio_viewer.py` (same arguments) puts the web
viewer inside Gradio, so an uploaded photo gets every viewer visual (animated route, swipe, layers), and keeps the
classic view as a second tab. On a plain CPU, `--live_baseline results/baselines/ckpt/<run>.pt` serves uploads with
the same-budget baseline (~0.5 s per photo; a comma list of checkpoints = their mean, `--live_tta` adds flips).
Live inference with Chameleon needs a GPU:
`kaggle/gradio_live/colab_gradio_live.py` (one Colab cell, T4) or `kaggle/gradio_live/run_gradio_live.py` (Kaggle T4)
starts it with a public share link.

## Submission

`bash release/build_release.sh` assembles the KLMS zip `20260876_MarlenMelis_project.zip` (notebook, write-up,
slides, data zip, app zip with `release/README_app.md`) and checks that it holds exactly one `.ipynb`.

## Licences and credits

- Our code and documentation: MIT (`LICENSE`). Third-party models, data and vendored code keep their own licences,
  listed below; the datasets are not covered by the MIT licence.
- Model: Chameleon, Kim et al., *Chameleon: A Data-Efficient Generalist for Dense Visual Prediction in the Wild*,
  ECCV 2024 (<https://github.com/GitGyun/chameleon>); course starter
  <https://github.com/GitGyun/AIC200-Project-StructuredPrediction>. Second pretrained model (Batch 13): SAM ViT-B,
  Kirillov et al., *Segment Anything*, ICCV 2023 (<https://github.com/facebookresearch/segment-anything>, Apache-2.0),
  used only to redraw the outlines of the buildings Chameleon predicts.
- Data: FloodNet (Rahnemoonfar et al., 2021), CDLA-Permissive-1.0
  (<https://github.com/BinaLab/FloodNet-Supervised_v1.0>); RescueNet (Rahnemoonfar et al., 2023), figshare CC0
  (GitHub release CC BY-NC-ND 4.0). Viewer photos and labels are derived from these datasets.
- Viewer: Leaflet 1.9.4 (BSD-2-Clause, `app/vendor/leaflet/LICENSE`).
