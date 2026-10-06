# Rescue Router: app (Track B)

Marlen Melis, 20260876. KAIST AIC200 mini-project, Track A + B.

A drone photo after a flood is mapped by Chameleon (adapted from 50 labelled FloodNet photos) into clear road, flooded
road, flooded/dry building and water. A router on that map works out how a rescue team reaches each house: truck on
dry roads, then a boat across water, then the last stretch on foot (never back into a vehicle). The same pipeline,
from the model to the routes, runs end to end in the notebook (`20260876_MarlenMelis_code.ipynb`, sections 8–9).
This zip holds the parts that run outside it.

**Live web viewer:** <https://marlenmm.github.io/rescue-router-demo/>
(best start: <https://marlenmm.github.io/rescue-router-demo/#s=rescuenet_12195&h=8&m=gt>, a canal row reached by
truck, then boat, then on foot).

## Contents

| path | what |
|---|---|
| `viewer/` | the web viewer, built (plain HTML/JS + Leaflet 1.9.4, vendored; no server code, no CDN, no storage). The same files as the live site. All 200 eval100 test photos (100 FloodNet, 100 RescueNet), 8 showcase scenes first. |
| `src/` | all Python code (identical to the `src/` in the data zip). The Gradio app is `src/gradio_app.py`; the router `src/router.py` (the viewer's `viewer/js/router.js` is its JS port, bit-exact on all 200 scenes). |
| `requirements.txt` | Python packages (CPU) |
| `kaggle/gradio_live/` | scripts that serve the Gradio app with live inference behind a public share link: `colab_gradio_live.py` (one Colab cell, T4) and `run_gradio_live.py` (Kaggle T4) |
| `demo/uploads/` | three FloodNet test photos the model never saw, for the live-upload demo (expected results in its README) |
| `rr_data/rescuenet_r768/test/images/` | the 98 RescueNet eval100 photos the data zip leaves out (it only needs their labels); the Gradio app shows them |
| `demo_route.gif` | recording backup: RescueNet 12195, house 8, truck → boat → on foot |

## Run the web viewer locally

Any static file server works (opening `index.html` directly from disk is blocked by browsers' file:// rules):

```bash
python3 -m http.server 8766 --directory viewer
```

then open <http://localhost:8766/#s=rescuenet_12195&h=8&m=gt>. Click a house for its route; switch between the
prediction, the labelled map (the same router run on the human labels) and a swipe between them; choose where the team
enters; the water-probability layer and `&cb=1` (colour-blind palette) are in the side panel. Every view has its own
link ("Copy link to this view").

## Run the Gradio app (CPU, from the prediction caches)

It needs the prediction caches from the data zip (`20260876_MarlenMelis_data.zip`):

```bash
unzip 20260876_MarlenMelis_data.zip -d rr && cd rr
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python src/cache_pack.py unpack --src results/kaggle_v2/rr_packed     # restores the caches bit for bit
cp -R <this app folder>/rr_data/. rr_data/                                     # optional: all 100 RescueNet photos
.venv/bin/python src/gradio_app.py --data rr_data
```

and open <http://127.0.0.1:7860>. Pick a scene or type a photo id (e.g. `7301`, `12195`), click a house for its route.
Without the optional copy the app shows all FloodNet eval100 photos and the two RescueNet showcase scenes.

## Live inference on an uploaded photo (GPU)

Uploads run the FloodNet 50-shot model live (~15 s per photo on a T4). It needs the course's meta-trained checkpoint
`generalist.ckpt` (course Drive) and the starter repository (Chameleon code):

```bash
python src/gradio_app.py --live --ckpt results/kaggle_v2/rr/ckpt/floodnet_50shot_s0_stratified.ckpt \
    --mt_ckpt <generalist.ckpt> --chameleon <AIC200-Project-StructuredPrediction>/chameleon \
    --data rr_data --support_file supports/floodnet_stratified_s0.txt --amp fp16 --share
```

The notebook runs exactly this model on `demo/uploads/floodnet_7307.jpg` (section 9). `kaggle/gradio_live/colab_gradio_live.py`
does the same in one Colab cell (T4; data zip in My Drive) and prints a public `gradio.live` link, used for the
presentation; `run_gradio_live.py` is the Kaggle version.

## Data and licences

FloodNet (CDLA-Permissive-1.0, <https://github.com/BinaLab/FloodNet-Supervised_v1.0>); RescueNet (figshare release
CC0, <https://springernature.figshare.com/collections/6647354>; the GitHub release states CC BY-NC-ND 4.0). Viewer
photos and labels are downsampled copies of these datasets. Model: Chameleon (Kim et al., ECCV 2024,
<https://github.com/GitGyun/chameleon>). Map: Leaflet 1.9.4 (BSD-2-Clause, `viewer/vendor/leaflet/LICENSE`).
