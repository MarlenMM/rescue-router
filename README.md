# Rescue Router

Few-shot flood mapping from drone photos, and rescue routes on the predicted map. KAIST AIC200 mini-project
(Chameleon, Track A + B).

**Track A (segmentation).** [Chameleon](https://github.com/GitGyun/chameleon) (Kim et al., ECCV 2024), meta-trained
by the course, is adapted with **50 labelled FloodNet photos** to five classes: clear road, flooded road, flooded
building, dry building, water. On the full 448-photo FloodNet test set it reaches **mIoU .472 [.437, .506]** with a
decode tuned on 30 validation photos (.420 [.386, .455] at τ 0.5; bootstrap 95 % CIs). PSPNet trained on 1,445 labelled
photos scores .848 on the same classes, so this is far below full supervision; flooded roads are the gap (IoU .29 vs
.82). Learning curve and the second dataset (RescueNet, Hurricane Michael, 20 labels: .421 on eval100) are in
`results/kaggle_v2/report/`.

**Track B (routing).** A router on 4×4-pixel cells sends a truck along dry roads, then a boat across water, then a
walk to each house (never back into a vehicle), with vehicle widths, parked cars, tree canopy and a corner rule. On
the predicted maps it finds the houses that cannot be reached by road with **F1 .759** on the full FloodNet test
(.835 on eval100; RescueNet .818). Each truck-reachable house gets a confidence tier: 301 "confident" houses on
FloodNet eval100 were 98.7 % truly truck-reachable; the others say "check first". The "labelled map" these numbers
compare with is the same router run on the human labels: the datasets label pixels, not reachability.

**Try it.** The web viewer (`docs/`, no server needed) shows all 200 eval100 photos: prediction vs labelled map,
house statuses and tiers, animated truck → boat → walk routes, minutes and metres, a water-probability layer and
shareable links. The Gradio app (`src/gradio_app.py`) does the same in Python and, on a GPU, runs the model on an
uploaded photo.

## Repository

| path | what |
|---|---|
| `src/` | all code: data prep, fine-tuning and inference (GPU), metrics, decode tuning, router, evaluation, reports, viewer export, Gradio |
| `src/rr/` | data loading, tiled inference, decode rules and metrics |
| `src/tests/` | Python tests (router, metrics/decode, scene packing) |
| `app/` | web viewer source (plain HTML/JS + Leaflet 1.9.4, vendored); `app/js/router.js` is the JS port of `src/router.py` |
| `app/test/` | JS router parity test against the Python reference (bit-exact on all 200 scenes) |
| `docs/` | the built viewer (GitHub Pages folder), made by `src/build_viewer.py` |
| `kaggle/` | the Kaggle GPU notebooks (training queue, checkpoint selection, test-time flips, live checks, Gradio share link) |
| `splits/`, `supports/` | fixed evaluation splits and the support (training) photo lists, with SHA-256 sums |
| `demo/uploads/` | three FloodNet test photos (not in eval100) for the live-upload demo |
| `PLAN.md` | the project plan and the full log of decisions and results (§15 "Quick summary" first) |

## Setup (CPU)

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

serves the Gradio app from the caches on <http://127.0.0.1:7860>. Live inference on uploads needs a GPU:
`kaggle/gradio_live/run_gradio_live.py` starts it on a Kaggle T4 with a public share link.

## Licences and credits

- Model: Chameleon, Kim et al., *Chameleon: A Data-Efficient Generalist for Dense Visual Prediction in the Wild*,
  ECCV 2024 (<https://github.com/GitGyun/chameleon>); course starter
  <https://github.com/GitGyun/AIC200-Project-StructuredPrediction>.
- Data: FloodNet (Rahnemoonfar et al., 2021), CDLA-Permissive-1.0
  (<https://github.com/BinaLab/FloodNet-Supervised_v1.0>); RescueNet (Rahnemoonfar et al., 2023), figshare CC0
  (GitHub release CC BY-NC-ND 4.0). Viewer photos and labels are derived from these datasets.
- Viewer: Leaflet 1.9.4 (BSD-2-Clause, `app/vendor/leaflet/LICENSE`).
