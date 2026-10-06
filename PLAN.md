# Rescue Router: Project Plan

**Few-shot flood mapping and rescue routing with Chameleon.** KAIST AIC.20000 mini-project, **Track A + Track B**.
Student: 20260876. Presentation: Thu 8 Oct 2026, in class (16:00). KLMS deadline: 8 Oct 23:59 KST. Target submission: **8 Oct 12:30**.

This file is the single source of truth for the build. Claude Code executes it. When reality differs from the plan, update this file.

---

## PART A: Plain-language overview

### What we're building
After a flood, rescue teams fly drones and get hundreds of photos, but someone still has to decide which streets a truck can use and which houses are cut off. Our AI colours every pixel of a drone photo:

- green: clear road
- blue: flooded road
- red: flooded house
- grey: dry house
- dark blue: open water

A route planner reads that map. A coordinator clicks a stranded house and the app draws the fastest way there: by truck on clear roads, then by boat across the flooded stretch. It also counts every house no truck can reach.

### Why it matters (Korea hook)
- **Seoul, 8 Aug 2022:** 141.5 mm of rain in one hour, the highest since records began in 1907. Gangnam's streets became rivers. 11 dead and 8 missing by 11 Aug (Central Disaster and Safety Countermeasures HQ).
- **Osong, 15 Jul 2023:** an embankment collapse flooded the Gungpyeong No. 2 underpass. 14 dead, 17 vehicles submerged. The entry-closure rule was tightened from 15 cm to 5 cm of water. The life-or-death decision is "is this road passable right now?"

### The story (headline)
**"A new disaster, mapped from 50 labelled photos in hours, not weeks."**

- Standard AI needs a whole labelled dataset per disaster. FloodNet's training split alone is 1,445 images, and labelling takes about **1 hour per image** according to both dataset papers.
- Chameleon adapts from a handful of examples, so we prove it with a curve: 1 → 5 → 10 → 20 → 50 photos.
- We then repeat the adaptation on a different hurricane (RescueNet).

### Wow moment
On the live web map, click a red house. A truck icon drives along green streets, stops at the water's edge, and a boat continues to the house. The counter reads "N houses unreachable by road". A toggle shows the answer the true map gives.

### What "done" looks like
- One notebook that reproduces everything.
- A public interactive map (GitHub Pages).
- A Gradio app.
- A 2–3 page write-up.
- Title slide + 6 slides.
- A data zip, an app zip and a screen recording.

---

## PART B: Technical execution plan

## 0. Compute and workflow (Google Colab, driven by Claude Code)

**Setup:** Marlen has Google Colab connected to Claude both via **MCP** and via **CLI**. Per the TA, **any GPU is allowed**.

### 0.1 GPU choice
- **Development and all experiments:** the strongest available Colab GPU, in order of preference **A100 80 GB → A100 40 GB → L4 → T4**. Check with `nvidia-smi` at the start of every session and log it.
- **Precision:** fp32 by default, matching the starter, so results are reproducible on the TA's T4. Allow a `--bf16` flag on A100 for speed in the large run queue only if fp32 throughput is too slow. If used, report it and verify that one checkpoint gives the same metrics (±0.5 IoU) when inference runs in fp32.
- **Compute units:** check the balance before launching the overnight queue. An A100 consumes units much faster than a T4, so estimate cost from the timed 50-step run (§4.4).

### 0.2 Budget rule: run only what answers a question
Marlen has usage limits (Colab compute units and Claude usage). **No run is repeated just for the sake of repeating.** Every GPU job must produce a number or figure that goes into the write-up, slides or demo. Otherwise it isn't run.

- **One run per curve point (seed 0).** Extra seeds only if a result looks suspicious and must be confirmed (optional block R6).
- **No learning-rate sweep by default.** Use the starter's lr. Sweep only if G1 fails.
- **Resolution ablation only if needed:** run 1024×768 only if G1 shows thin roads being missed at 768×576.
- **Curve points are evaluated on eval100.** The full 448-image test is run once, for the final model only.
- **The GPU is used only for fine-tuning and inference.** All coding, data prep (CPU), routing, metrics from caches, figures, the web viewer, the write-up and slides run with **no GPU runtime attached**. Disconnect the GPU runtime as soon as a job finishes, because an idle runtime still burns units.
- **GPU tier:** use the cheapest GPU that fits the job. T4/L4 for short tests and inference; A100 only for the batched fine-tuning queue, and only if the timed test (§4.4) shows T4/L4 would be too slow.
- **Big-GPU benefit kept:** eval_shot = 50 at inference, with no memory compromises.
- **Claude usage:** while a queue runs, Claude Code does **not** poll the log in a loop. It checks once at each planned milestone (G1/G2/G3 and the morning review) or when Marlen asks.

### 0.3 Division of work between machines
- **Mac (local project folder, iCloud):** source of truth for code (`src/`, notebook, `PLAN.md`, write-up, slides, `app/`). Claude Code edits here.
- **Git:** a private GitHub repo `rescue-router` mirrors the code. Colab pulls it with `git pull`. The public web viewer lives in the same repo (`docs/` → GitHub Pages) or a separate public repo if the main one stays private.
- **Google Drive** (`MyDrive/AIC200-my_work/rescue_router/`): persistent storage for checkpoints, logs, prediction caches, split files and preprocessed data. **Everything that took GPU time is written to Drive immediately.**
- **Colab local disk** (`/content/`): working copy of the datasets for fast I/O. Download datasets **directly on Colab**, not on the Mac: faster network, and it keeps the iCloud folder small.
- **Course assets:** the shared course folder `AIC200-StructuredPrediction` (meta-checkpoint ~2.8 GB, task checkpoints) is a shortcut in MyDrive. Copy the meta-checkpoint to `/content/` once per session for speed.

### 0.4 Surviving Colab disconnects (critical for overnight runs)
- Run the queue as a **background process**, not inside a notebook cell: `nohup python src/run_all.py --queue queue.yaml > /content/drive/MyDrive/.../logs/queue.log 2>&1 &`.
- **Resumable queue:** each run checks for `ckpt/<run_id>.pt` plus `metrics/<run_id>.json` on Drive and skips if both exist. Checkpoint every 500 steps, and resume from the last checkpoint if interrupted.
- **Per-run JSON logs** (config, git commit hash, GPU, timings, loss every 50 steps, val30 mIoU every 250 steps).
- Claude Code checks `queue.log` only at milestones (see §0.2), not continuously. If the runtime died, it reconnects, re-mounts Drive, restarts the queue, and completed runs are skipped automatically.
- The queue script ends by writing `DONE` to the log. Once it's done, the GPU runtime is released.
- Keep the Colab tab or session alive during the overnight queue (Colab Pro background execution if available). Colab's maximum session length is finite, so the queue must tolerate a restart.

### 0.5 Final notebook must still run on a plain T4
The TA's reference runtime is a Colab T4.

- The final notebook auto-detects GPU memory.
- On less than 20 GB it uses eval_shot = 20 (the nested first 20 of the 50 supports) and `chunk_size=1`, `channel_chunk_size=1`, and **prints which setting it used**.
- Headline numbers come from our A100 runs at eval_shot = 50 (cached predictions produced by our own code, shipped in the data zip). The notebook recomputes the headline model live and verifies it against the cache.
- **G3 test (Wed 22:00) runs on a fresh T4 runtime.**

---

## 1. Verified facts sheet

| Item | Value | Consequence |
|---|---|---|
| Chameleon | Kim et al., ECCV 2024 oral, arXiv 2404.18459. BEiT-v2-L image encoder, ViT-L label encoder, DPT decoder. Fine-tunes biases + matching temperature + one LayerNorm. Paper tasks use ≤50 labels. Cellpose used 224 tiles with 50% overlap and Gaussian blending. ISIC/DAVIS used 384 crops. | 384 tiles/crops with Gaussian blend; 50 shots = top of the documented range |
| FloodNet | 2,343 UAV images, Hurricane Harvey (Fort Bend County TX, 2017), 4000×3000. Split 1,445 / 450 / 448. Classes: 0 Background, 1 Building-flooded, 2 Building-non-flooded, 3 Road-flooded, 4 Road-non-flooded, 5 Water, 6 Tree, 7 Vehicle, 8 Pool, 9 Grass. CDLA-Permissive-1.0. | Supports from train, tuning on val, evaluation on test |
| FloodNet quirk | ~~Some images are 4592×3072 while their labels are 4000×3000~~ **Checked 5 Oct: in the Dropbox release every label matches its image size** (352 pairs are 4592×3072 / 4592×3072; 0 mismatches). **Real quirk: 57 images carry EXIF orientation 3 (180°) and their labels match the rotated view** (edge score 1.5–4.2 rotated vs 0.7–1.0 raw, 57/57). Class presence (>0 px, all 2,343): road-flooded 264, road-non-flooded 1,175, bldg-flooded 245, water 1,059. Train images with flooded road/bldg ≥0.1 %: 165 of 1,445; 1-shot-eligible: 6 | Export applies the EXIF rotation; 4592×3072 resized anisotropically to 768×576 |
| FloodNet baselines | Paper Table 4, test, 713×713 | §5.2 |
| RescueNet | 4,494 UAV images, Hurricane Michael (FL, Oct 2018). Split 3,595 / 449 / 450. Classes: 0 Background, 1 Water, 2 Bldg-No-Damage, 3 Minor, 4 Major, 5 Total-Destruction, 6 Vehicle, 7 Road-Clear, 8 Road-Blocked, 9 Tree, 10 Pool. About 1 h annotation per image. | Separate task definition (§3.4); supports from val, evaluation on test |
| RescueNet licence | Conflict: GitHub says CC BY-NC-ND 4.0; figshare/HF say CC0 | Redistribute only a minimal attributed subset |

**Hour-1 task:** read the real signatures and defaults of `su.build_config`, `register_custom`, `CustomDenseDataset`, `derive_stage2_overrides`, `predict_queries`, `predict_image`, plus the default image size and checkpoint filenames, from `chameleon/starter_utils.py` and `chameleon/downstream/custom/`. Record them in §13 of this file. Every value marked **[start]** below gets adjusted to match.

---

## 2. Data

### 2.1 Downloads (run on Colab)
```bash
# FloodNet (official Dropbox; dl=1 gives a zip)
wget -O floodnet.zip "https://www.dropbox.com/scl/fo/k33qdif15ns2qv2jdxvhx/ANGaa8iPRhvlrvcKXjnmNRc?rlkey=ao2493wzl1cltonowjdbrnp7f&dl=1"
# RescueNet (figshare collection 6647354): list articles, download val + test only
curl -s https://api.figshare.com/v2/collections/6647354/articles | python -m json.tool
```

Fallbacks:
- **FloodNet:** download the split folders one by one, or use the Kaggle mirror `aletbm/aerial-imagery-dataset-floodnet-challenge` or HF `torchgeo/floodnet`.
- **RescueNet:** HF `isp-uv-es/rescuenet-taco` or the BinaLab GitHub Dropbox link.
- We **never download the RescueNet train archive**: supports come from val (449) and evaluation uses test (450).

### 2.2 Preprocessing (`src/prep_data.py`, CPU, once, output to Drive)
1. **Alignment.** If the image size ≠ label size, resize the image to the label size (the label is authoritative) and log the pair in `size_mismatch.csv`. Overlay label edges on 5 mismatched pairs. If the label is a crop rather than a resize, switch to a centre crop and document it.
2. **Downsample** to the working resolution (768×576; and 1024×768 for the ablation set). Use area interpolation for images and nearest-neighbour for labels.
3. **Map classes** to harmonized channel ids (§3) and save a uint8 PNG (channel id, 254 = other, 255 = ignore).
4. **Statistics.** Write `stats.csv` with per-image class pixel fractions.

**Done 5 Oct (CPU Colab, `src/prep_data.py`, sha256 `6617cb54…`).** Output in `MyDrive/AIC200-my_work/rescue_router/`:
- `data/floodnet_r768.tar` (535 MB, all 2,343), `data/floodnet_r1024_subset.tar` (287 MB, 738 = test_full + val30 + supports s0–s2), `data/rescuenet_r768.tar` (253 MB, val 449 + test 450). Layout inside: `<ds>_<res>/<split>/{images/<id>.jpg (q95), labels/<id>.png (channel ids), labels_raw/<id>.png (original class ids)}`.
- `meta/`: `stats_{floodnet,rescuenet}.csv` (sizes, EXIF, per-class fractions at full res), `alignment_*.csv` + `alignment/*.jpg` (5 EXIF pairs, raw vs rotated), `size_mismatch.csv` (empty), `raw_manifest.json` (zip sha256 + sources), `prep_log.json`. `code/prep_data.py`.
- "Present" = ≥ 0.1 % of an image's pixels. Images: DCT 2× decode + INTER_AREA; labels: INTER_NEAREST from full res. No 255 (ignore) pixels exist in either source.
- RescueNet val/test zips are CC0 on figshare; their 4056×3040 images (52) also have matching labels.

### 2.3 Fixed splits (`splits/`, created before any support selection, SHA-256 printed in the notebook)
- Stratification flag "target" = flooded road or flooded building present (RescueNet: road-blocked or major/total-destruction building). eval100 = 50 target + 50 non-target (seed 0); val_tune30 = 15 + 15 (seed 0). Hashes in `splits/SHA256SUMS`.
- `floodnet_test_full.txt` (448)
- `floodnet_test_eval100.txt` (seed 0; 50% contain a flooded road or building)
- `floodnet_val_tune30.txt` (30 val images)
- `floodnet_support_pool.txt` (official train, 1,445)
- `rescuenet_test_full.txt` (450), `rescuenet_test_eval100.txt`, `rescuenet_support_pool.txt` (official val, 449)

### 2.4 Support sets: nested, stratified, ordered
- The custom loader takes the **first `shot` files**, so supports are written with priority prefixes (`000_<id>.jpg`, …). This gives 1 ⊂ 5 ⊂ 10 ⊂ 20 ⊂ 50 ⊂ 100 automatically.
- **Stratified random (main):** seeds {0, 1, 2}.
  - Every prefix of length k ∈ {5, 10, 20, 50, 100} has ≥⌈k/2⌉ images with a flooded road or flooded building, and ≥1 with water.
  - The 1-shot image must contain road-clear, road-flooded and a building.
  - Flooded classes are rare (road-flooded appears in 335 images vs road-non-flooded in 1,725), so pure random fails at small shots.
- **Done 5 Oct:** `supports/{floodnet,rescuenet}_stratified_s{0,1,2}.txt`, 100 ordered ids each (line 1 = 1-shot image). Order alternates target / non-target, so every prefix k has exactly ⌈k/2⌉ targets; the water rule is asserted for k ∈ {5…100}. FloodNet has only 6 1-shot-eligible train images, so s0 and s2 share the 1-shot image (7614). The `000_<id>` run folders are built from these lists at GPU time.
- **Diverse (selection experiment, 20 shots):** (needs GPU features, not done yet) greedy k-centre on mean-pooled frozen Chameleon BEiT features, with the same coverage constraint.

### 2.5 Run folder layout (on Colab disk; regenerated from splits)
```
runs_data/floodnet_s{seed}[_{sel}]/train/{images,labels}/000_<id>.jpg|png …
runs_data/floodnet_s{seed}/test/{images,labels}/<id>.jpg|png
```

### 2.6 Data zip (`20260876_<NAME>_data.zip`, target < 300 MB)
- Splits + hashes, `stats.csv`, `size_mismatch.csv`.
- Preprocessed supports (all seeds, deduplicated) + FloodNet eval100 images and labels.
- RescueNet 50 supports + eval100, with `LICENSES.md` quoting both licences and attribution.
- `README_data.md` (counts, links, class mapping).
- Final checkpoints (~20 MB each: FloodNet best, RescueNet best, FloodNet 20-shot s0).
- Prediction caches `preds/*.npz` (uint8 probabilities) for every curve point.

---

## 3. Task formulation

### 3.1 FloodNet channels (C = 5)
| ch | Name | Source class | Routing role |
|---|---|---|---|
| 0 | road_clear | 4 Road-non-flooded | truck-drivable |
| 1 | road_flooded | 3 Road-flooded | boat-navigable; truck forbidden |
| 2 | bldg_flooded | 1 Building-flooded | target, high priority |
| 3 | bldg_dry | 2 Building-non-flooded | target; obstacle |
| 4 | water | 5 Water | boat-navigable |

- Background, tree, grass, pool and vehicle are negative in all channels (M = 1, decoded as "other").
- Vehicles on roads are handled by a 5-px morphological closing of the road channels at decode time, not by changing labels.
- **Label format:** Y = one-hot (5, H, W) in {0, 1}. M = 1 except at ignore pixels (255).
- **Loss:** `bce` (one sigmoid per channel).
- **Decode:** argmax if max prob ≥ τ (tuned on val30, start 0.5), else "other".

**Rejected alternatives (for the write-up):**
- (a) All 10 classes: more cost, dilutes 50 shots; kept only as a 20-shot ablation.
- (b) A single "passable" channel: no targets, no boat leg.
- (c) A cost-regression map: no labels exist for it.
- (d) Tree as an obstacle: canopy covers real roads in nadir views.
- (e) Pool merged into water: pools aren't routes.

**Optional ch5 `flood_any`** (union of the wet classes): add only if G1 shows road_flooded/water confusion.

### 3.2 Resolution and tiling
- **Main working resolution:** 768×576. 1024×768 is the ablation, promoted to main if it wins (see §0.2).
- **Training:** random 384×384 crops (scale 0.5–1.0), h/v flips, 90° rotations, brightness/contrast jitter 0.2, **no hue jitter** (water colour matters).
- **Inference:** 384 tiles, stride 192 (6 tiles at 768×576; 15 at 1024×768), Gaussian-weighted blending (σ = tile/4).
- **Supports at inference:** 384 crops. Check how `derive_stage2_overrides` sizes supports; pass crops if it resizes whole images.
- **Evaluation:** at working resolution against nearest-downsampled labels.

### 3.3 Label conversion
```python
FLOOD2CH = {4:0, 3:1, 1:2, 2:3, 5:4}            # others -> 254
RESC2CH  = {7:0, 8:1, 2:2, 3:2, 4:3, 5:3, 1:4}  # RescueNet, see 3.4
def to_onehot(lbl, C):
    Y = np.stack([(lbl == c) for c in range(C)]).astype(np.float32)
    M = np.repeat((lbl != 255)[None], C, 0).astype(np.float32)
    return Y, M
```

### 3.4 RescueNet task (same 5 slots, different meaning)
| ch | Name | Source |
|---|---|---|
| 0 | road_clear | 7 Road-Clear |
| 1 | road_blocked | 8 Road-Blocked (water, sand or debris) |
| 2 | bldg_intact | 2 No-Damage + 3 Minor |
| 3 | bldg_damaged | 4 Major + 5 Total-Destruction |
| 4 | water | 1 Water |

Report three things:
1. **Zero-label transfer** of the FloodNet 50-shot model, on the shared super-classes only (road-any, building-any, water).
2. **Re-adaptation:** 10/20/50 RescueNet labels.
3. **Supervised references,** as context only (reported range 57–98% mIoU, protocols differ).

**RescueNet routing:** road_blocked is impassable to both truck and boat. Buildings with no route get the status "ISOLATED: needs foot/air team".

---

## 4. Fine-tuning

### 4.1 Hyperparameters
| Param | Value |
|---|---|
| dataset | `custom` via `register_custom(FloodDataset, Learner, n_tasks=5)` |
| loss_type | `bce` |
| shot | 1 / 5 / 10 / 20 / 50 / 100 |
| global_batch_size | starter default **[start]**; ×2 on A100 if memory allows |
| n_steps | 1,000 (shot ≤ 10), 2,000 (20/50/100) **[start]**; convergence check on 20-shot s0 (val30 every 250 steps) |
| lr | starter default; sweep {0.3×, 1×, 3×} on 20-shot s0 (val30); fixed afterwards |
| channel_sampling | all 5 channels |
| precision | fp32 (bf16 only under the §0.1 rule) |
| trainable params | starter set only (biases, temperature, LayerNorm) |
| seeds | torch, numpy, random; cudnn deterministic where possible |

### 4.2 Run matrix (queue file `queue.yaml`, processed in this order). Lean by design (§0.2)
| Block | Runs | Purpose | Status |
|---|---|---|---|
| R1 | FloodNet {1, 5, 10, 20} s0 = 4 | data-efficiency curve | core |
| R2 | FloodNet 50-shot s0 = 1 | main model + showcase | core |
| R3 | FloodNet 20-shot "diverse" s0 = 1 | support selection (vs R1 20-shot) | core |
| R4 | RescueNet {20, 50} s0 = 2 | new disaster | core |
| R5 | FloodNet 100-shot s0 = 1 | does adding more stop helping? | optional, only if time and units allow at G2 |
| R6 | FloodNet 5-shot s1, s2 = 2 | luck check at small shots | optional, only if the curve looks noisy |
| — | lr sweep, 1024×768 run, C=4/flood_any variant | fixes | **only if G1 fails** |

Core = **8 fine-tunes**. Repeats of the same configuration are not run. Checkpoints: `ckpt/{dataset}_{shot}shot_s{seed}_{sel}.pt` on Drive.

**Reuse instead of re-running:**
- Each checkpoint's predictions are cached once (`preds/*.npz`). Every metric, figure, routing evaluation and failure analysis is computed from caches on CPU.
- The final notebook recomputes live only the headline 50-shot model; everything else loads from caches.

### 4.3 Stage-2 inference
- `eval_shot = shot` on A100.
- On OOM: `chunk_size` → `channel_chunk_size` → eval_shot = 20 (nested).
- Write predictions to `preds/<run_id>_<split>.npz` as uint8 probabilities, so all metrics and figures are computed from caches.

### 4.4 First-hour timing (decides everything)
Time 50 fine-tune steps and tiled inference on 10 images at 768×576, at 20 and at 50 shots. From that, compute total queue hours and compute-unit cost.

- Curve points are always evaluated on eval100. The full 448-image test runs once, for the final 50-shot model only.
- From the timing, pick the cheapest GPU tier on which the 8 core runs finish overnight.
- This timing test is the only exploratory GPU use. It reuses the G1 20-shot run (§10) rather than being a separate job.

---

## 5. Evaluation

### 5.1 Segmentation
- Per-channel IoU, mIoU (5 channels), pixel accuracy. IoU is computed from the confusion matrix accumulated over the dataset.
- One run per shot (seed 0). Report it honestly as single runs; add the 5-shot spread only if R6 was needed.
- **Plot:** mIoU and road_flooded IoU vs shot (log x), with a PSPNet dashed line.
- **Binary routing metrics:** "drivable" (ch0) precision/recall/IoU and "wet" (ch1 ∪ ch4 ∪ ch2) IoU. **Drivable precision matters most**: a false "clear" sends a truck into water.

### 5.2 Published baselines (FloodNet paper Table 4, test, 713×713)
| Method | Bldg flooded | Bldg non-flooded | Road flooded | Road non-flooded | Water | mIoU (9 classes) |
|---|---|---|---|---|---|---|
| ENet | 6.94 | 47.35 | 12.49 | 48.43 | 48.95 | 42.61 |
| DeepLabV3+ | 32.7 | 72.8 | 52.00 | 70.2 | 75.2 | 61.53 |
| PSPNet | 68.93 | 89.75 | 82.16 | 91.18 | 92.00 | 79.69 |

Rules:
- Compare **per-class IoU on the 5 matching classes only**. Our 5-class decode may inflate IoU slightly; say so.
- Framing: "Chameleon, 50 labels vs PSPNet, 1,445 labels (3.5%)".
- RescueNet supervised numbers are context only.

### 5.3 Failure analysis
- Mine the 12 worst images by road IoU, plus false-"clear" clusters (predicted ch0, ground truth wet).
- Tag each failure:
  1. shadow/dark asphalt seen as water;
  2. muddy/shallow water seen as clear (dangerous);
  3. canopy breaking a road;
  4. narrow roads/driveways missed;
  5. pool/water/flooded-yard confusion;
  6. flooded vs dry building ambiguity;
  7. tile-border artefacts.
- Report counts, a 2×3 gallery, and the routing consequence of each.

### 5.4 Routing evaluation
- **Grid:** 4×4 px cells (192×144 at 768×576), majority vote per cell.
- **Base:** the clear-road cell of the largest ground-truth road component touching the image border; the same base for predicted and ground-truth maps.
- **Buildings:** ground-truth building components, opened with a 3×3 kernel, area ≥ 150 px. Target = 2-cell dilation ring.
- **Status:** TRUCK / BOAT / ISOLATED.
- **Metrics:**
  1. **Stranded detection:** precision/recall/F1 of "not TRUCK", plus a 3×3 status confusion matrix.
  2. **Route safety violation (RSV):** the fraction of truck-leg cells that are wet or building in ground truth (mean). **Unsafe-route rate:** the share of truck legs crossing ≥3 consecutive ground-truth wet cells.
  3. **Route cost ratio:** C_G(route_P) / C_G(route_G), with median and % infeasible.
  4. **Reachability agreement:** % of buildings with the same status.
- **Ablation:** risk-aware vs plain argmax truck costs.

**Implemented 6 Oct (`src/router.py: route_metrics, summarize_routing`; `src/route_eval.py`).** Definitions as built:
- Infeasible: the predicted route drives a truck onto GT water, flooded or blocked road, or enters a GT building. A boat over GT dry ground is only charged boat cost.
- RSV: the share of truck-leg cells that are GT hazard (road_flooded/blocked, water) or building, averaged over routes with a truck leg.
- Cost ratio: C_G(route_P) / C_G(route_G), median over feasible predicted routes to GT-reachable buildings whose GT route costs > 0. About 27 % of houses touch the image-border entry cells and cost 0; an earlier version counted those as ratio 0 and gave a spurious median of 1.0. `infeasible_rate` = share of GT-reachable buildings with no predicted route or an infeasible one.

**GT maps, eval100, default router** (`results/routing/gt_*_eval100_off*.json`, not in git). GT-vs-GT agreement is 1.0 (sanity check):

| dataset | router | images with base | TRUCK | BOAT | ISOLATED |
|---|---|---|---|---|---|
| FloodNet | off-road ×4 | 100 (all via road or land) | 471 | 144 | 125 |
| FloodNet | roads only | 92 (39 road, 53 water) | 19 | 165 | 555 |
| RescueNet | off-road ×4 | 100 | 267 | 5 | 20 |
| RescueNet | roads only | 76 | 40 | 4 | 196 |

- FloodNet, off-road, by GT building class: flooded buildings split 348 TRUCK / 134 BOAT / 102 ISOLATED, dry buildings 123 / 10 / 23.
- So a "flooded" label often means water touched the house, while a dry path still reaches its yard.
- All 125 isolated FloodNet buildings have land in their ring, but that land is cut off by water or buildings.

---

## 6. Track B system

### 6.1 Router (`src/router.py`, NumPy + heapq; ported to JS)
- **State and moves:** (cell, mode ∈ {TRUCK, BOAT}), 8-connected.
- **TRUCK:**
  - allowed on decoded road_clear;
  - cost = len × (1 + λ·(p1 + p4)), λ = 5.
- **BOAT:**
  - allowed on decoded road_flooded or water, or where p1 + p4 ≥ 0.5;
  - cost = len × 3.
- **Switching:** TRUCK → BOAT is allowed next to a boat cell, with a launch penalty of 40. There is no switch back.
- **Goal:** the ring around the building; building cells are impassable.
- **Search:** Dijkstra (~55k states, milliseconds). Statuses for all buildings come from 2 multi-source runs.
- **Visuals:** the road skeleton (skimage) is drawn as an extra layer only.
- **Tests:** unit tests on synthetic grids with known answers.

**Implemented 6 Oct (`src/router.py`, tests `src/tests/test_router.py`, 14 pass; GT evaluation `src/route_eval.py`).** Differences from the text above:
- **Off-road truck (new default, `offroad=4`).** The truck may also cross dry "other" land (lawn, trees, background, pools, vehicles) at 4× the road cost. Water, flooded road (RescueNet: blocked road) and buildings still stop it. Why: with roads only (the rule above), the GT maps of FloodNet eval100 leave **555 of 739 buildings ISOLATED** (19 TRUCK, 165 BOAT), because at 768×576 a 4-px cell is ~0.3 m and almost every house sits behind its own yard. Then "stranded" would measure yards, not floods. A wider goal ring does not fix it (ring 12 cells still leaves 345 isolated). `offroad=None` restores the strict rule.
- **Search:** one multi-source Dijkstra (`scipy.sparse.csgraph`, 2 layers × 27,648 cells) gives every building's status. TRUCK = a ring cell is reachable in the truck layer, BOAT = only in the boat layer.
- **Base fallbacks:** if no GT clear-road component touches the border, use the largest dry-land component touching the border (truck enters over land), then water (boat).
- **Grid majority-vote ties** go to the less drivable class (road_clear loses every tie).
- **Speed:** ~40 ms per image (both maps), ~4 s for eval100.

### 6.2 Notebook pipeline
`load_chameleon` → fine-tuned checkpoint → `tiled_predict` → `decode` → `router.status_all` / `router.route(click)` → overlays → metrics.

### 6.3 Gradio app (in the notebook, `share=True`)
- **Inputs:** upload, or one of 8 showcase scenes. Controls: risk-aware toggle, boat-speed slider.
- **Outputs:**
  - colour overlay;
  - building status overlay (green truck / orange boat / red isolated);
  - click → route (truck leg white, boat leg cyan) plus a route summary;
  - "Unreachable by road: N / M".
- The model is loaded once. Inference uses eval_shot from the §0.5 auto-detect.

### 6.4 Web viewer (GitHub Pages, `app/` → `docs/`)
- **Stack:** plain HTML/JS + Leaflet 1.9 (`L.CRS.Simple`, `L.imageOverlay`) + a custom swipe slider (`clip-path`), modelled on Theme Worlds. No build step.
- **Per scene** (8: 6 FloodNet, 2 RescueNet):
  - `photo.webp` (2048 wide)
  - `pred_*.png` and `gt_*.png` (RGBA)
  - `grid.json` (base64 Uint8)
  - `buildings.json` (id, polygon, centroid, status_pred, status_gt)
  - `meta.json`
- **Features:**
  - layer toggles, opacity, prediction/GT swap;
  - click a building → JS Dijkstra → animated truck then boat icons;
  - unreachable counter;
  - "disagreements with GT" toggle;
  - scene picker, legend.
- **Assets** are exported from the notebook (`export_web_assets` cell).
- **Test** on phone and laptop, and in incognito.
- **Screen recording** (60–90 s, 1080p): open scene → toggles → swipe → click 3 houses (truck, truck + boat, isolated) → GT toggle.

---

## 7. Notebook (`20260876_<NAME>_code.ipynb`, all outputs saved)
0. Title, "Track A + B", contents, runtime table, GPU used.
1. **Setup:** installs, clone, Drive mount, flags `RUN_TRAINING=False`, `RUN_FULL_TEST=False`, GPU auto-detect (§0.5), seeds, versions.
2. **Data:** fetch the data zip (default) or the full datasets; mapping table; mismatch summary; split hashes; samples.
3. **Formulation:** channels, Y/M illustration, tiling diagram.
4. **Fine-tuning:** dataset class + `register_custom` + `build_config`. Retrain the full matrix if `RUN_TRAINING`; otherwise a 200-step demo run plus loading checkpoints. Loss plots from logs.
5. **Results:** metrics per shot. The headline 50-shot and 20-shot s0 models are recomputed live; other points come from caches, which is stated. Curve, baselines, selection, qualitative grid.
6. **Failure analysis.**
7. **Cross-disaster** (RescueNet).
8. **Routing:** code, status maps, metrics, ablation.
9. **App:** Gradio launch; web-asset export.
10. **Reproducibility:** pretrained-weights list with links, runtime log, git commit hash.

Runtime target with `RUN_TRAINING=False`: ≤ 40 min on a T4.

---

## 8. Write-up (2–3 pages, one section per graded point)
1. **Problem and motivation:** Osong/Seoul, the first-hours mapping gap, 1 h/image labelling cost, the goal.
2. **Formulation:** channel table, Y/M/BCE/decoding, tiling, rejected alternatives, router as the consumer.
3. **Data and resources:**
   - FloodNet (link, CDLA-Permissive-1.0, counts) and RescueNet (figshare DOI 10.6084/m9.figshare.c.6647354, licence note, counts).
   - **Weights:** (1) Chameleon meta-trained checkpoint (course Drive; github.com/GitGyun/chameleon), which contains (2) the BEiT-v2-L image encoder and (3) the ViT-L label encoder initialisations. No other models.
   - **Compute:** Google Colab A100/L4 (state which), plus a T4 reproducibility check.
4. **Method and setup:** hyperparameter table, augmentations, run matrix, timings, inference settings, router costs.
5. **Results and analysis:** curve with seeds, baseline table and caveats, selection, resolution, cross-disaster, routing metrics, failure categories, demo screenshot, limitations:
   - single frames, no georeferencing;
   - US hurricanes rather than Korean urban floods;
   - temporal change.

---

## 9. Slides (title + 6) and the 5-minute script
| # | Content | Time |
|---|---|---|
| Title | "Rescue Router: mapping a new flood from 50 photos", name, 20260876, **Track A + B**, hero split image | 0:10 |
| 1 | Problem + best result: Osong/Seoul facts; hero numbers (road IoU at 50 shots vs PSPNet with 1,445; RSV; stranded F1); route screenshot | 0:50 |
| 2 | Formulation: photo → 5 channels → cost grid → truck/boat route; why not 10 classes | 0:40 |
| 3 | Data: FloodNet + RescueNet, counts, "1 h per image", nested 1 ⊂ … ⊂ 50, fixed test set | 0:35 |
| 4 | What we tried: curve with seed bands + PSPNet line; diverse vs random; resolution | 0:50 |
| 5 | Live demo (browser, ≤60 s); GIF backup on the slide | 1:00 |
| 6 | What we learned: failure gallery → routing risk; RescueNet re-adaptation; limitations; "Few-shot dense prediction makes the map; the map makes the decision." | 0:45 |

Rehearse 3× on Thursday morning; stay under 4:50.

---

## 10. Schedule (KST)

### Mon 5 Oct
- **18:30–19:30:**
  - Colab: pick the strongest GPU, mount Drive, clone the starter, copy the meta-checkpoint, run the starter demo cells.
  - **Read and record the real API (§13).**
  - Time 50 steps + inference (§4.4).
  - Create the private repo `rescue-router`.
- **18:30–21:00 (parallel, on Colab):** FloodNet download; RescueNet val + test download. Unzip to `/content`, copy the originals' manifest to Drive.
- **21:00–22:00:** `prep_data.py`, alignment check, splits, stratified supports s0–s2, written to Drive.
- **22:00–23:30:** 20-shot s0 short run; tiled inference on val30.
- **G1 (23:30):**
  - Pass: val30 road_clear IoU ≥ 0.5 and road_flooded ≥ 0.25. Then launch the overnight background queue (R0 → R1 → R2 → R3), resumable.
  - Fail: try 1024×768 / lr ×3 / C=4 or flood_any; rerun once; launch the best.
- **Overnight:** Claude Code checks the queue log periodically and restarts if the runtime dropped.

### Tue 6 Oct
- **08:00–09:00:** review logs, fix failures, lock lr; queue anything missing.
- **09:00–13:00 (CPU/Mac):** `router.py` + unit tests; routing metrics on GT; τ tuning on val30.
- **G2 (13:00):**
  - Freeze the main FloodNet model (50-shot s0; resolution decided).
  - Queue R4 (RescueNet) and R5; run on a second runtime if possible.
- **13:00–17:00:** prediction caches + metrics for all checkpoints; Gradio app.
- **17:00–22:00:** web viewer v1, `export_web_assets`, Pages deploy.
- **22:00:** queue the full-448 test inference for the final models overnight.

### Wed 7 Oct
- **08:00–11:00:** routing evaluation, ablation, failure mining and tagging, all figures.
- **11:00–15:00:** write-up draft (tables filled from metric JSONs).
- **15:00–19:00:** slides draft, screen recording, viewer polish.
- **19:00–22:00:** notebook cleanup into the §7 structure; READMEs; data/app zips; size check.
- **G3 (22:00):** fresh **T4** runtime, notebook top to bottom with `RUN_TRAINING=False`, outputs saved. Fix or fall back to cached results (stated).
- **22:00–00:00:** final numbers into the write-up and slides; PDF export.

### Thu 8 Oct
- **08:00–10:30:** rehearse ×3; test the demo on a phone hotspot.
- **10:30–12:30:** assemble and verify the zip (§12); **submit by 12:30**.
- **13:00–16:00:** chemistry lab.
- **16:00:** present.

### Fallback ladder (in order)
1. **Low performance:** 1024×768 → lr ×3 → a 3-channel "drivable / wet / building" task.
2. **Memory:** chunk sizes → eval_shot = 20 → batch 2.
3. **Time:** R1 down to 2 seeds at shots 10/20; R3 to 1 seed; drop R5.
4. **RescueNet fails:** HF TACO repack; else drop it and use a "held-out neighbourhoods" story on FloodNet val.
5. **Pages fails:** ship the viewer in the app zip and demo locally.
6. **Colab runtime lost repeatedly:** smaller queue chunks (one run per process), lower-tier GPU with a longer queue, or a second account/runtime.

---

## 11. Risks
| Risk | Mitigation |
|---|---|
| Starter API differs from assumptions | Hour-1 API read; our `tiled_predict` wrapper |
| Colab disconnect kills the overnight queue | Background `nohup` queue, resumable runs, everything on Drive, periodic checks |
| Compute units run out | Cost estimate from the timed run; bf16 / fewer seeds / L4 fallback |
| Inference too slow | Prediction caches; eval100 fallback |
| Rare flooded classes missing | Stratified supports; per-class IoU |
| Image/label misalignment | Resize-to-label + visual check + log |
| Flooded road vs water vs mud confusion | Wet-union metric, risk-aware router, failure analysis |
| Final notebook fails on the TA's T4 | §0.5 auto-detect, G3 on a fresh T4, `RUN_TRAINING=False` default, self-contained data zip |
| RescueNet licence | Minimal subset, both licences quoted, regeneration script |
| Overclaiming against baselines | Per-class comparison with caveats |
| Live demo fails | GIF backup, offline local copy |

---

## 12. Submission checklist
```
20260876_<NAME>_project.zip
├── 20260876_<NAME>_code.ipynb      # single notebook, all outputs, RUN_TRAINING=False
├── 20260876_<NAME>_writeup.pdf     # 2–3 pages + references
├── 20260876_<NAME>_slides.pdf      # title (Track A + B) + 6 slides
├── 20260876_<NAME>_data.zip
└── 20260876_<NAME>_app.zip         # app/, README.md (live URL + local run), recording.mp4
```
- [ ] Only one .ipynb anywhere in the zip
- [ ] Notebook ran top to bottom on a fresh **T4**
- [ ] Every number in the slides and write-up appears in a notebook output
- [ ] Weights list starts with Chameleon, with links
- [ ] Data counts and links for both datasets
- [ ] "Track A + B" on the title slide and in the write-up
- [ ] 1 title + ≤6 slides; talk ≤ 5:00
- [ ] Pages URL works in incognito
- [ ] Zip uploads to KLMS (test early)
- [ ] `<NAME>` filled in exactly as on KLMS

---

## 13. Recorded starter API (filled in 5 Oct from source; read from `chameleon/starter_utils.py`, `chameleon/downstream/custom/`, `downstream/base_learner.py`, `train/trainer.py`, `train/train_utils.py`, `args.py`, `chameleon.ipynb`)

**`su.build_config(stage, dataset, result_dir=None, flags=(), **overrides)`** (`starter_utils.py:208`)
- Wraps `args.parse_args`; always adds `--precision fp32 --single_gpu`; stage 1 always adds `--no_eval` (train loader sized `n_steps × global_batch_size`, no evaluator, no val support set).
- `overrides` → `--key value` (lists → `--key v1 v2`, bools → `--key True`). **No value may contain a space.** `None` values are skipped.
- Sets `result_dir = ckpt_dir = /tmp/aic200/<dataset>_stage<stage>`, `num_workers = min(cfg, 2)`.
- Base YAML: `downstream/custom/configs/train_config.yaml` (stage 1) / `test_config.yaml` (stage 2). `data_paths.yaml` is looked up in the CWD first, then the `chameleon/` package root → write it with `su.write_data_paths(DEMO_DATA_ROOT, extra={"custom": ROOT}, chameleon_root=CHAMELEON_DIR)`.
- **Custom-task defaults (train_config.yaml):** seed 0, shot 20, img_size 224, base_size 256, support_idx 0, randomflip True, randomjitter True (**ColorJitter hue = 0.2**, applied with p = 0.5), randomrotate False (ignored by `CustomDenseDataset`), global_batch_size 2, n_steps 200, eval_shot 20, eval_batch_size 4, eval_size 20, chunk_size 1, channel_chunk_size 1, channel_sampling −1 (all channels), attn_dropout 0.5, optimizer adam, **lr 1e-3**, lr_pretrained 2e-4, lr_schedule constant, no warmup, weight_decay 0, loss_type bce, mask_value −1, autocrop False, head/label_decoder/relpos/embed tuning all False. `n_tasks` is not in the YAML: pass `n_tasks=` or rely on `register_custom`.
- `n_schedule_steps` defaults to `n_steps` in stage 1. ISIC (closest paper task) used img 384 / base 448, gbs 5, n_steps 10000, eval_shot 50, randomjitter/rotate/blur True, flip False.

**Default image size / crop size**
- Custom default `img_size = 224` (model input / training crop), `base_size = 256` (images resized to this first). The YAML comment "must stay 224 unless you also change the encoder name" is outdated: `train_utils.load_model` (`train_utils.py:547`) renames `beitv2_large_patch16_224 → _384` and `vit_large_patch16_224 → _384` automatically for any square `img_size` and resizes pos-embeds / rel-pos tables. **So `img_size=384` works** (as ISIC does). `base_size` may be a pair `[H, W]` → we use `base_size=[576, 768]` (H, W order).

**`register_custom(dataset_cls=None, learner_cls=None, n_tasks=None)`** (`downstream/custom/registry.py`): call before `build_config`/`load_chameleon`; `None` keeps the previous value. `n_tasks` resolution: `config.n_tasks` first, then the registry.

**`CustomDenseDataset(config, split, base_size, crop_size, eval_mode=False, resize=False, dset_size=-1)`** (`downstream/custom/dataset.py`) — hooks:
- Layout: `ROOT/{train,test}/{images,labels}/<stem>.<ext>`; `split='valid'` reuses `train/`. Stems are **sorted by name**; split `train` takes `stems[support_idx*shot : +shot]` (wraps around) → our `000_<id>` prefix scheme works.
- `load_image(stem)` → `(3, H, W)` in [0, 1]. `load_label(stem)` → `(Y, M)` `(C, H, W)` — **override** (one-hot of our uint8 channel PNG, M = 0 at 255).
- `__getitem__`: `X = Resize(base_size)(X)` (bilinear, antialias); Y/M resized nearest only if shapes differ; then `postprocess`.
- `postprocess(X, Y, M)` — **the place for crops and augmentations; override it**:
  - train (`eval_mode=False, resize=False`): jitter (hue 0.2!) p = 0.5, hflip p = 0.5, then **random `img_size` crop** of the base image (no scale jitter, no v-flip/rot90). → Our override: random-scale crop 0.5–1.0, h/v flip, rot90, brightness/contrast 0.2, **hue 0**.
  - support (`eval_mode=True, resize=True`, built by `BaseLearner.get_support_data`): **the whole base image is squashed to `img_size`** (`Resize(img_size)`), i.e. 768×576 → 384×384 = 2× downscale + aspect distortion. → Our override: deterministic 384 crops for supports (per §3.2).
  - query/test (`eval_mode=True, resize=False`, `crop_size=base_size`): centre crop of `base_size` = the whole image.
- `eval_mode=True` returns `(X, Y, M, aux=zeros(1))`.

**`CustomLearner`** (`downstream/custom/learner.py`): bce loss = `(M * BCEWithLogits).mean()`; `postprocess_logits` = sigmoid; `postprocess_final` = `> 0.5` per channel (no argmax); `compute_metric` = per-image mean IoU over all channels (not a dataset confusion matrix) → we compute our own metrics from cached probabilities.

**Support set sizing / `get_support_data`** (`base_learner.py:73`): `dset_size = eval_shot if eval_shot > 0 and not autocrop else shot`; loaded in **one batch** (`batch_size=len(data)`), then encoded in `chunk_size` chunks.

**`derive_stage2_overrides(ft_ckpt_path, verbose=True)`** (`starter_utils.py:168`): copies from the checkpoint's saved config `class_name, shot, eval_shot, img_size, base_size, support_idx, chunk_size, channel_chunk_size, autocrop, autocrop_minoverlap, head/label_decoder/relpos/input_embed/output_embed tuning, separate_alpha`; drops `eval_shot ≤ 0`. It does **not** resize anything itself: supports are sized by the dataset (`resize=True` → squashed to `img_size`, see above). `load_model` merges meta → fine-tuned → our config (ours wins), so `eval_shot`, `chunk_size` etc. can be overridden at stage 2. `n_tasks`/`loss_type` are not inherited by the helper but come back through the ft-config merge.

**Inference** (`train/trainer.py:inference`, `preprocess_query`):
- `autocrop=False` (default): a query of size ≠ `img_size` is **downsampled whole to `img_size`**, predicted, then upsampled to `base_size` (`T.Resize`). No tiling.
- `autocrop=True`: `dataset.utils.AutoCrop(img_size, autocrop_minoverlap)` tiles with stride `(1−overlap)·img`, blends with a Gaussian of **σ = crop/2** (not /4); forces `chunk_size=1` and makes supports use `shot` (ignores `eval_shot`). Caveat: for 576×768 with 384 tiles / stride 192 its offset rule produces duplicate tiles (rows 0,192,192; cols 0,192,384,384 → 12 tiles, 6 unique). → **We write our own `tiled_predict`** (6 unique tiles, σ = tile/4) calling `model.inference` on 384×384 tiles (input = `img_size` → no resize path).
- Channel chunks via `channel_chunk_size`, instance chunks via `chunk_size`.

**`predict_queries(model, n_query, support_verbose=False)`** → list of dicts `X, Y, M, aux, raw` (raw = sigmoid probs at `base_size`, before threshold); iterates `learner.get_test_loader()` (`eval_batch_size`, first `eval_size` test files); prints peak GPU memory. **`predict_image(model, image)`** → `(raw, pred)` at the image's own resolution; resizes the PIL image to `base_size` (bilinear) and calls `model.inference` — tiling only if `autocrop=True`. Neither does tiling on its own.

**Other helpers**: `load_chameleon(config, mt_ckpt, ft_ckpt=None, verbose=True)` → `(model, config)` (fp32, `.cuda()`, `setup()`); `save_finetuned(model, path)` stores only fine-tuned tensors + config (~20 MB); `finetuned_parameter_names`, `trainable_parameter_report`, `free_gpu`, `T4_BUDGET_GB = 14.7`. Training loop is plain PyTorch in the notebook (`get_optimizer(cfg, model.model)`, `model.learner.get_train_loader()`, `Y_pred, Y_q, M_q, _ = model(batch)`, `learner.compute_loss`).

**Checkpoint filenames in the shared Drive.** Verified 5 Oct from the course folder `1sw2E-ocZmLo5ph9ZdvgHzk7uIhYDndb6` (owner kdgyun425, shared "anyone with link: reader"):
- `checkpoints/metatrained/generalist.ckpt`: 2,967,114,775 bytes, file id `1_UrVMFy18kEcrxwD1b_ohxuTfJIsOWsG`. ← the one we use.
- `checkpoints/finetuned/` uses the **dataset folder names**: `ap10k/{Antelope,Cat,Elephant,Giraffe,Hippo,Horse,Mouse,Pig}.ckpt`, `davis2017/` (30 videos incl. `judo`), `isic2018/0.ckpt`, `fsc147/best.ckpt`, `cellpose/best.ckpt`. `demo_data/` sits next to `checkpoints/`.
- **Our copy:** Marlen's MyDrive had no shortcut, so the meta-checkpoint was copied server-side (Drive connector) to `MyDrive/AIC200-StructuredPrediction/checkpoints/metatrained/generalist.ckpt` (file id `1AwUEMbTfNX17ZUTv73GDKKY2RLv6UnQD`). `su.find_meta_trained(CHECKPOINT_ROOT)` works unchanged with the starter's paths. Fine-tuned task checkpoints and demo data were not copied (not needed). `MyDrive/AIC200-Project-StructuredPrediction` is just a code copy.
- Demo data: `.../AIC200-StructuredPrediction/demo_data/<dataset>/{images,labels}`. Our writable dir: `MyDrive/AIC200-my_work/`.

**Measured 5 Oct on a Colab T4** (`logs/probe_t4.json`; each config in a fresh process, 384 px, fp32, gbs 2):
- Training, all 5 channels: **OOM**. With 3 or 2 channels per step: OOM. With gradient checkpointing of the 24 BEiT blocks (`--grad_ckpt`): 5 channels OOM; **3 channels fit, 11.8 GB peak, 4.3 s/step** (2000 steps ≈ 2.4 h). → The T4 cannot run the planned recipe. Training moves to an A100 (preferred) or L4, per §0.1/§0.2.
- Inference, 20-shot (384 support crops), `chunk_size=1, channel_chunk_size=1`: support encoding 25 s at 5.7 GB peak; tiled prediction **11.2 s/image** (6 tiles × 5 channels) at 6.2 GB peak. Batched (`chunk 6 / channel_chunk 5`): OOM. → The §0.5 T4 notebook path works for inference only: eval100 at 20 shots ≈ 19 min.
- 50-shot inference and the bigger-GPU numbers: to be measured (`src/probe.py`).

**Measured 5 Oct on a Colab T4 with fp16** (`logs/probe_t4_fp16.json`, code `8f1cb58`, one fresh process per config, 384 px, gbs 2):
- Plain fp16 autocast gave **NaN loss**. Two overflows (found with forward hooks): (1) the ViT label encoder computes `(q @ kᵀ) * scale`, scaling after the fp16 matmul, so it is NaN from block 0; (2) label-encoder features reach ~1.5e4 and the matching module's `fc_v` overflows (6.5e4 → inf). Fix in `rr/compat.py`: attention logits in fp32 when q/k are fp16, whole matching module in fp32 under fp16 autocast. fp32 runs take the original code. Remaining fp16 max is ~1.5e4 (label encoder), so 4× headroom.
- Grad ckpt on the image encoder only did not save memory under fp16 (3 ch: 11.7 GB, same as fp32). The label encoder's 24 blocks were also storing activations. `--grad_ckpt` now checkpoints **both encoders (48 blocks)**.
- **Training, fp16 + gc, all 5 channels: fits, 10.15 GB peak, 3.15 s/step** (probe median; 3.57 s/step in the real run incl. data loading). GradScaler warms up from 65536 to ~1024 (6 skipped steps in the first 50).
- **Inference fp16, 20-shot:** chunk 6 / channel_chunk 5 OOM. chunk 1/1: 8.1 s/image (fp32 was 11.2), 6.9 GB peak. **chunk 6 / channel_chunk 1: 5.7 s/image, 11.0 GB peak** (support encoding 10 s).
- **Inference fp16, 50-shot, chunk 1/1: 17.2 s/image, 12.0 GB peak**, support encoding 29 s. → 50-shot runs and caches must use `--chunk 1 --channel_chunk 1`; eval100 at 50 shots ≈ 29 min.

**Starter issues found on the GPU (worked around in `src/`, the vendored starter is untouched):**
1. `model/transformers/helpers.py::resize_rel_pos_bias` calls `scipy.interpolate.interp2d`, which was removed in SciPy 1.14 (Colab has 1.16), so every img_size ≠ 224 crashed. → `rr/compat.py` shims it with `RectBivariateSpline` (same grid spline; max difference to interp2d's grid path 3.6e-14).
2. `register_custom(dataset_cls=…)` is never read; the loaders use `CustomLearner.BaseDataset` (= `CustomDenseDataset`). → `rr.data.FloodLearner` sets `BaseDataset = FloodDataset`.
3. `trainer.inference` tests `X.shape[-2:] != config.img_size` (torch.Size vs list, always unequal), so even 384 inputs are upsampled to `base_size`. → `tiled_predict` stores `img_size` as a tuple.

**Implemented 5 Oct (commit `df79806`, private repo github.com/MarlenMM/rescue-router; code reaches Colab as `rescue_router/rr_code_<hash>.tar.gz` on Drive because a private repo cannot be cloned without a token):** `src/rr/data.py` (`FloodDataset`: training crop side 320–480 px → 384, i.e. scale 0.8–1.2 around the inference scale, h/v flip, rot90, brightness/contrast 0.2, no hue; support = best 384 window by channel coverage), `src/rr/infer.py` (`encode_support`, `tiled_predict`: 6 tiles at 768×576, 15 at 1024×768, σ = 96), `src/rr/metrics.py`, `src/finetune.py` (resumable run, val30 every 250 steps, `_last.pt` every 500, logs/metrics/preds on Drive), `src/predict.py` (stage-2 caches). CPU smoke test on synthetic data passed against the real starter code (config, crops, support crops, loader, tiles, metrics).

**Consequences for §2–§4:** use `img_size=384`, `base_size=[576,768]`, our own `FloodDataset.postprocess` (augmentations + support crops), our own `tiled_predict`, own metrics from caches.

## 15. Status / handoff (updated Tue 6 Oct, 17:45 KST)

**Done**
- §13 starter API recorded, including 3 starter bugs and their workarounds.
- §2.1–2.4 data on Drive: preprocessed tars, splits + SHA256SUMS, stratified supports s0–s2, meta/stats. EXIF-180° fix applied to 57 FloodNet images. The s0 lists, val/eval100 splits and the RescueNet pool are mirrored in the repo (`splits/`, `supports/`; SHA256 verified against Drive).
- fp16 recipe working on T4 (commits `54c01ee`, `8f1cb58`; §13 "Measured … with fp16"). fp16 vs fp32 eval agrees within 0.3 IoU points (§0.1 check passed).
- G1 run (old recipe, Colab) done: narrow fail, see below.
- **Track B router (6 Oct 04:10):** `src/router.py` + unit tests (15) + `src/route_eval.py` + `src/route_report.py`. GT routing statistics are in §5.4/§6.1. The default is the off-road truck (×4); see §6.1 for why. **Kept:** at 12:00 Marlen left this and the ablation to Claude's judgement ("do what is needed, don't skip things").
- **Decision: dry-building ablation skipped.** The model has no flooded/dry signal (per-building AUC ~0.5). Loss up-weighting would mostly shift the dry threshold, which τ tuning already explored without gain. The wet-context decode fixes dry buildings at zero GPU cost.
- **Mac CPU environment:** `~/.venvs/rr` (Python 3.12: numpy, scipy, scikit-image, matplotlib, pillow, pytest). Tests: `~/.venvs/rr/bin/python -m pytest src/tests -q`. The preprocessed data (both r768 tars, extracted) is in `~/rr_data/`, outside iCloud. `kaggle/fetch_output.py` downloads notebook outputs with resume, because the Kaggle CLI's single GET breaks on large files with IncompleteRead. Run it with `~/.venvs/kaggle/bin/python`.
- CPU scripts ready for when the queue finishes: `src/collect_results.py` (tables + curve), `src/decode_tune.py` (per-channel τ + building vote on val30 → eval100). `rr.metrics.decode` now takes a per-channel τ (candidate if p_c ≥ τ_c, winner = max p_c/τ_c; a scalar τ gives the old rule) and `bldg_t` (one flooded/dry label per predicted building).
- Queue code (commit `136c2fb`): best-val30 checkpoint + cosine lr in `finetune.py`; `predict.py` skips existing caches; `src/run_queue.py`; `rescuenet_val_tune30` (30 pool images outside the s0 supports, 15 target / 15 non-target, seed 0); Kaggle notebooks in `kaggle/`.

**Compute (6 Oct):** Marlen has **no Colab subscription** (free tier; about 3 h of GPU use left on 6 Oct). All GPU work moved to **Kaggle** (account `marlenmelis`, phone-verified, 30 GPU-h/week, 0 used before this queue). Colab is still usable for short CPU work.

**Recipe v2 (used for ALL queue runs; supersedes the "fp32" rows of §0.1/§4.1):** 384 px, all 5 channels, gbs 2, lr 1e-3 with a **cosine schedule**, `--amp fp16 --grad_ckpt`, **the checkpoint with the best val30 mIoU is kept** (val every 250 steps; every 500 at 50 shots). Steps: 1000 for ≤10 shots, 2000 for 20/50. Inference fp16; `--chunk 6 --channel_chunk 1` at ≤20 shots, `--chunk 1 --channel_chunk 1` at 50. G1 (recipe v1: constant lr, final checkpoint) is superseded; its 20-shot point is rerun in the queue.

**Kaggle setup (all from the Mac, no browser needed):**
- CLI: `~/.venvs/kaggle/bin/kaggle` (venv; auth via `~/.kaggle/access_token`, created by Marlen). Kernel pushes need short paths: `cd` into the folder and use `-p .` (long iCloud/scratchpad paths fail with "File name too long").
- Notebook output `marlenmelis/rr-fetch-data` (`kaggle/fetch/fetch.py`, CPU, done): `floodnet_r768.tar`, `rescuenet_r768.tar`, `generalist.ckpt` (sizes verified) and `rescuenet_val_tune30.txt`. The Drive tars were link-shared only for this fetch and are **private again**.
- Private dataset `marlenmelis/rr-code`: `rr_code.tar.gz` = `src/ splits/ supports/ VERSION` at `136c2fb`. After a code change: rebuild the tarball (`git archive HEAD src splits supports` + `VERSION`) and run `kaggle datasets version -p . -m "<msg>"` from its folder.
- GPU notebook `marlenmelis/rr-gpu-queue` (`kaggle/queue/run_queue_kaggle.py`, machine `NvidiaTeslaT4` = 2× T4, internet on): setup (~8 min), then one `run_queue.py` per GPU (own starter copy each, because the starter writes `data_paths.yaml` into its folder). Prints progress every 10 min. Live log: Chrome → `https://www.kaggle.com/code/marlenmelis/rr-gpu-queue/log`; the rows can be read with JS (`[...document.querySelectorAll('table tr')].map(r => r.innerText)`). Status: `kaggle kernels status marlenmelis/rr-gpu-queue`.

**Core queue result (notebook `rr-gpu-queue`; finished about 08:33 KST, 5.7 h; outputs in `results/kaggle_v2/rr/`):** 5 of 7 runs OK (FloodNet 1/5/10/20, RescueNet 20), each with its eval100 cache.
- **Both 50-shot runs FAILED:** CUDA OOM in the first in-process val30 evaluation (step 500). 50-shot inference needs ~12 GB on its own and does not fit next to the training state on a 14.6 GB T4. No `_last.pt` was written (val runs before the step-500 resume save), so their eval100 jobs failed too (no checkpoint).
- **Fix and relaunch** (pre-authorized by Marlen for clear, fixable failures; commit `b8e2daf`): `finetune.py --defer_val` saves a snapshot `ckpt/<run>_step<N>.ckpt` at every val step instead of validating. `src/select_ckpt.py` then validates each snapshot on val30 at 50 shots in a fresh process (fp16, chunk 1/1, 12 GB) and keeps the best: the same rule as in-process.
- **Notebook `marlenmelis/rr-gpu-queue50`** (`kaggle/queue/run_queue_kaggle.py` with `QUEUE = 'rerun50'`, metadata in `kaggle/queue50/`), **launched 09:21 KST, ETA ~12:45 KST.** GPU 0: FloodNet 50 → select → eval100. GPU 1: RescueNet 50 → select → eval100. It is a separate slug, so the first queue's outputs stay downloadable.
- `rr-code` dataset rebuilt with `COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata`, because Kaggle unpacks the tarball and macOS had added `._*` files (`tar t` hides them).
- The progress watcher in the first session died on a DNS error while the Mac slept. The new one retries.
- **rr-gpu-queue50 result (finished ~11:25 KST):** both 50-shot trainings OK (2.0 h each; snapshots at steps 500/1000/1500/2000 in `results/kaggle_v2/q50/rr/ckpt/`). `select_ckpt.py` failed at once: argparse abbreviation matching read the shared `--data /content/rr_data` as `--dataset`. Fixed with `allow_abbrev=False` (commit after `b8e2daf`).
- **Relaunched as `marlenmelis/rr-gpu-select50`** (`QUEUE = 'select50'`, metadata `kaggle/select50/`, input = the queue50 output), pushed 11:57 KST. It copies the 8 snapshots and 2 training logs, then per GPU runs `select_ckpt` (4 val30 evaluations at 50 shots) and eval100. **Done:** FloodNet 50 in 40 + 30 min, RescueNet 50 in 44 + 33 min. Outputs are in `results/kaggle_v2/s50/` and merged into `results/kaggle_v2/rr/` (the 50-shot files only). Kaggle GPU use on 6 Oct: about 5.9 + 2.1 + 1.3 ≈ 9.3 session-hours of 30.
- Speed (T4, fp16): training 3.6–4.1 s/step. eval100 inference: 2.7 / 3.7 / 4.1 / 5.9 s per image at 1 / 5 / 10 / 20 shots (FloodNet), 6.5 at RescueNet 20.

**Results, all 7 runs (best-val30 checkpoint, τ = 0.5; tables from `src/collect_results.py`, curve `results/kaggle_v2/report/curve_floodnet.png`):**

#### floodnet val30 (τ = 0.5)

| shot | best step | road_clear | road_flooded | bldg_flooded | bldg_dry | water | mIoU | drv P | drv R | wet IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 500 | 0.056 | 0.298 | 0.227 | 0.212 | 0.004 | 0.160 | 0.057 | 0.867 | 0.459 |
| 5 | 500 | 0.476 | 0.280 | 0.638 | 0.000 | 0.177 | 0.314 | 0.556 | 0.769 | 0.546 |
| 10 | 1000 | 0.346 | 0.342 | 0.487 | 0.027 | 0.283 | 0.297 | 0.509 | 0.519 | 0.590 |
| 20 | 1000 | 0.317 | 0.369 | 0.673 | 0.017 | 0.398 | 0.355 | 0.424 | 0.556 | 0.643 |
| 50 | 1000 | 0.299 | 0.331 | 0.727 | 0.271 | 0.553 | 0.436 | 0.567 | 0.387 | 0.655 |

#### floodnet eval100 (τ = 0.5)

| shot | n img | road_clear | road_flooded | bldg_flooded | bldg_dry | water | mIoU | drv P | drv R | wet IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100 | 0.101 | 0.292 | 0.242 | 0.241 | 0.005 | 0.176 | 0.102 | 0.863 | 0.467 |
| 5 | 100 | 0.454 | 0.324 | 0.673 | 0.015 | 0.136 | 0.320 | 0.539 | 0.741 | 0.599 |
| 10 | 100 | 0.497 | 0.317 | 0.509 | 0.098 | 0.221 | 0.328 | 0.671 | 0.658 | 0.612 |
| 20 | 100 | 0.412 | 0.362 | 0.674 | 0.002 | 0.335 | 0.357 | 0.540 | 0.635 | 0.634 |
| 50 | 100 | 0.511 | 0.398 | 0.721 | 0.211 | 0.446 | 0.457 | 0.805 | 0.583 | 0.618 |

#### rescuenet val30 (τ = 0.5)

| shot | best step | road_clear | road_blocked | bldg_intact | bldg_damaged | water | mIoU | drv P | drv R | wet IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 1500 | 0.635 | 0.091 | 0.472 | 0.410 | 0.855 | 0.493 | 0.735 | 0.824 | 0.775 |
| 50 | 2000 | 0.663 | 0.103 | 0.627 | 0.356 | 0.696 | 0.489 | 0.812 | 0.783 | 0.625 |

#### rescuenet eval100 (τ = 0.5)

| shot | n img | road_clear | road_blocked | bldg_intact | bldg_damaged | water | mIoU | drv P | drv R | wet IoU |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 100 | 0.613 | 0.049 | 0.354 | 0.271 | 0.645 | 0.386 | 0.731 | 0.791 | 0.472 |
| 50 | 100 | 0.621 | 0.052 | 0.592 | 0.364 | 0.401 | 0.406 | 0.763 | 0.770 | 0.308 |

- **Val30 history (mIoU per eval):**
  - FloodNet 50-shot (snapshots, `select_ckpt.py`): .267 / **.436** / .411 / .425 (steps 500–2000).
  - RescueNet 50-shot: .270 / .401 / .478 / **.489**.
  - 1-shot: .151 / **.160** / .135 / .142
  - 5-shot: .166 / **.314** / .285 / .281
  - 10-shot: .212 / .283 / .294 / **.297**
  - 20-shot: .196 / .290 / .288 / **.355** / .322 / .347 / .345 / .347
  - RescueNet 20-shot: best .493 at step 1500.
- With cosine lr, the late evaluations are stable (20-shot: .32–.35 after step 1000). The v1 swings (water .48 ↔ .06) are gone.
- Recipe v2 at 20 shots (val30 .355) is not better than G1's best eval (.389 at step 1750, recipe v1). It is about the same as G1's final checkpoint (.336).
- **The curve is flat from 5 to 20 shots** (eval100 mIoU .320 → .328 → .357), then **jumps at 50 shots** (.457 at τ 0.5, .492 tuned). At 50 shots: drivable precision .805, road_flooded .398, dry buildings .211 (no fix needed).
- **RescueNet 50-shot is not better than 20** on eval100 (.406 vs .386 at τ 0.5; .402 vs .421 tuned). Buildings improve (intact .35 → .59) but water drops (.65 → .40), and its tuned decode did not transfer. road_flooded sits at .29–.36 and is far from PSPNet's .82 with 1,445 labels.
- **Dry buildings collapse** (IoU ≤ .10 from 5 shots on). At 1 shot, both building classes sit at ~.24: the 1-shot image has dry and flooded buildings in similar amounts.
- Water rises steadily with shots (.005 → .34).

**Decode tuning (6 Oct 09:35; `src/decode_tune.py`, results `results/kaggle_v2/rr/decode/<run>.json`, table `results/kaggle_v2/report/decode.md`).** Everything is tuned on val30 only and applied unchanged to eval100.
- **Per-channel τ** (`rr.metrics.decode`: channel c is a candidate if p_c ≥ τ_c; the winner is the largest p_c/τ_c). Coordinate ascent on val30 mIoU, with the constraint that **val30 drivable precision may not drop** below the τ = 0.5 decode. Without that constraint, the 10-shot tuning cut eval100 drivable precision from .671 to .534.
- **Dry-building diagnosis.** From 5 shots on, the model hardly predicts dry buildings (20-shot: 8 k predicted dry px against 1.3 M GT dry px). GT dry pixels go to flooded (69 %) or other (30 %). Per GT building, the share p_dry/(p_dry + p_flooded) does **not** separate dry from flooded buildings (AUC .52 / .48 / .53 on eval100 at 5 / 10 / 20 shots). So no threshold, bias or per-building vote on the building channels can fix it, and the vote (`bldg_t`) gains nothing on FloodNet.
- **The fix that works: the wet-context rule** (`ctx = (d, f)`): a predicted building is flooded iff ≥ f of the pixels within d px of it (Voronoi-split between buildings) are decoded wet (road_flooded or water). This follows FloodNet's own definition (water around the house), and the model's wet channels are decent. The decoded-wet ring alone gives AUC .63–.75 (d 32–48) for flooded vs dry; the GT wet ring gives .66–.73, so the GT flooded/dry label is itself partly ambiguous.
- **Chosen per run** by val30 mIoU: FloodNet 5/10/20 → τ + wet context (d 96–128 px ≈ 7–10 m, f 0.01–0.02, i.e. "any water nearby"). FloodNet 1 → τ only. RescueNet 20 → τ + vote (`bldg_t` .4; intact/damaged is a per-building property).
- d and f sit at the edge of a wide grid, so the rule is close to "any wet pixel within ~10 m". With the narrower grid (d ≤ 64) the 20-shot eval100 was slightly better (.429 vs .422). That is a small val30 overfit; report the chosen values.

| run | split | decode | ch0 | ch1 | ch2 | ch3 | ch4 | mIoU | drv P | drv R | wet IoU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| floodnet_1shot | val30 | τ=0.5 | 0.056 | 0.298 | 0.228 | 0.212 | 0.004 | **0.160** | 0.057 | 0.868 | 0.459 |
| floodnet_1shot | val30 | tuned τ | 0.149 | 0.340 | 0.490 | 0.255 | 0.039 | **0.255** | 0.163 | 0.625 | 0.302 |
| floodnet_1shot | eval100 | τ=0.5 | 0.101 | 0.292 | 0.243 | 0.241 | 0.005 | **0.176** | 0.102 | 0.863 | 0.467 |
| floodnet_1shot | eval100 | tuned τ | 0.220 | 0.278 | 0.518 | 0.264 | 0.039 | **0.264** | 0.265 | 0.564 | 0.313 |
| floodnet_5shot | val30 | τ=0.5 | 0.476 | 0.280 | 0.638 | 0.000 | 0.177 | **0.314** | 0.555 | 0.769 | 0.546 |
| floodnet_5shot | val30 | tuned τ + wet ctx | 0.512 | 0.339 | 0.716 | 0.221 | 0.312 | **0.420** | 0.680 | 0.674 | 0.658 |
| floodnet_5shot | eval100 | τ=0.5 | 0.454 | 0.324 | 0.673 | 0.015 | 0.136 | **0.320** | 0.539 | 0.741 | 0.599 |
| floodnet_5shot | eval100 | tuned τ + wet ctx | 0.427 | 0.360 | 0.656 | 0.088 | 0.235 | **0.353** | 0.654 | 0.551 | 0.629 |
| floodnet_10shot | val30 | τ=0.5 | 0.346 | 0.342 | 0.488 | 0.027 | 0.283 | **0.297** | 0.510 | 0.519 | 0.590 |
| floodnet_10shot | val30 | tuned τ + wet ctx | 0.346 | 0.336 | 0.665 | 0.320 | 0.323 | **0.398** | 0.510 | 0.519 | 0.624 |
| floodnet_10shot | eval100 | τ=0.5 | 0.497 | 0.317 | 0.510 | 0.098 | 0.221 | **0.329** | 0.671 | 0.658 | 0.612 |
| floodnet_10shot | eval100 | tuned τ + wet ctx | 0.495 | 0.313 | 0.647 | 0.221 | 0.238 | **0.383** | 0.668 | 0.657 | 0.640 |
| floodnet_20shot | val30 | τ=0.5 | 0.317 | 0.368 | 0.673 | 0.017 | 0.398 | **0.355** | 0.424 | 0.556 | 0.643 |
| floodnet_20shot | val30 | tuned τ + wet ctx | 0.327 | 0.376 | 0.671 | 0.333 | 0.571 | **0.456** | 0.442 | 0.556 | 0.716 |
| floodnet_20shot | eval100 | τ=0.5 | 0.412 | 0.362 | 0.674 | 0.002 | 0.334 | **0.357** | 0.540 | 0.635 | 0.634 |
| floodnet_20shot | eval100 | tuned τ + wet ctx | 0.422 | 0.394 | 0.653 | 0.167 | 0.477 | **0.422** | 0.561 | 0.631 | 0.661 |
| floodnet_50shot | val30 | τ=0.5 | 0.299 | 0.331 | 0.727 | 0.270 | 0.553 | **0.436** | 0.567 | 0.387 | 0.655 |
| floodnet_50shot | val30 | tuned τ | 0.363 | 0.415 | 0.733 | 0.373 | 0.534 | **0.484** | 0.569 | 0.500 | 0.696 |
| floodnet_50shot | eval100 | τ=0.5 | 0.511 | 0.398 | 0.721 | 0.211 | 0.446 | **0.457** | 0.805 | 0.583 | 0.618 |
| floodnet_50shot | eval100 | tuned τ | 0.526 | 0.430 | 0.720 | 0.341 | 0.441 | **0.492** | 0.759 | 0.632 | 0.656 |
| rescuenet_20shot | val30 | τ=0.5 | 0.635 | 0.090 | 0.473 | 0.410 | 0.855 | **0.493** | 0.735 | 0.824 | 0.775 |
| rescuenet_20shot | val30 | tuned τ + vote | 0.676 | 0.141 | 0.596 | 0.499 | 0.877 | **0.558** | 0.898 | 0.732 | 0.810 |
| rescuenet_20shot | eval100 | τ=0.5 | 0.613 | 0.049 | 0.354 | 0.271 | 0.645 | **0.386** | 0.731 | 0.791 | 0.472 |
| rescuenet_20shot | eval100 | tuned τ + vote | 0.607 | 0.059 | 0.484 | 0.309 | 0.644 | **0.421** | 0.810 | 0.707 | 0.499 |
| rescuenet_50shot | val30 | τ=0.5 | 0.663 | 0.103 | 0.627 | 0.356 | 0.696 | **0.489** | 0.812 | 0.784 | 0.625 |
| rescuenet_50shot | val30 | tuned τ + vote | 0.691 | 0.118 | 0.662 | 0.537 | 0.750 | **0.552** | 0.915 | 0.738 | 0.689 |
| rescuenet_50shot | eval100 | τ=0.5 | 0.621 | 0.052 | 0.592 | 0.364 | 0.401 | **0.406** | 0.763 | 0.770 | 0.308 |
| rescuenet_50shot | eval100 | tuned τ + vote | 0.621 | 0.055 | 0.526 | 0.389 | 0.418 | **0.402** | 0.786 | 0.747 | 0.340 |

- floodnet_1shot_s0_stratified: τ = [0.8, 0.3, 0.1, 0.8, 0.1], vote = None, wet ctx (d px, f) = None, chosen = tau
- floodnet_5shot_s0_stratified: τ = [0.7, 0.3, 0.3, 0.45, 0.15], vote = 0.2, wet ctx (d px, f) = [96, 0.02], chosen = tau_ctx
- floodnet_10shot_s0_stratified: τ = [0.5, 0.6, 0.2, 0.8, 0.5], vote = 0.2, wet ctx (d px, f) = [96, 0.02], chosen = tau_ctx
- floodnet_20shot_s0_stratified: τ = [0.5, 0.4, 0.5, 0.45, 0.25], vote = None, wet ctx (d px, f) = [128, 0.01], chosen = tau_ctx
- floodnet_50shot_s0_stratified: τ = [0.4, 0.35, 0.45, 0.35, 0.45], vote = None, wet ctx (d px, f) = None, chosen = tau
- rescuenet_20shot_s0_stratified: τ = [0.8, 0.6, 0.3, 0.3, 0.25], vote = 0.4, wet ctx (d px, f) = None, chosen = tau_bldg
- rescuenet_50shot_s0_stratified: τ = [0.6, 0.55, 0.5, 0.25, 0.25], vote = 0.7, wet ctx (d px, f) = None, chosen = tau_bldg

- **eval100 mIoU gains:** +.09 (1-shot), +.03 (5), +.05 (10), +.07 (20), **+.035 (50)**, +.04 (RescueNet 20), −.004 (RescueNet 50).
- **At 50 shots the model learns flooded vs dry itself:** per-building p_dry-share AUC .763 (val30) / .733 (eval100), against ~.5 at 5–20 shots. Tuning keeps plain per-channel τ (the wet-context rule no longer helps), and dry IoU on eval100 rises .211 → .341.
- FloodNet 50 tuned: eval100 drivable precision falls .805 → .759, although the val30 constraint held (.567 → .569). Report both decodes for the headline model.
- **Dry buildings on eval100:** .015 → .088 / .098 → .221 / .002 → .167 at 5 / 10 / 20 shots.
- Drivable precision never drops on val30 by construction. On eval100 it rises at 1 / 5 / 20 shots and RescueNet, and is flat at 10 (.671 → .668).
- **Write-up framing:** the dry-building gain comes from a decode rule built on the model's wet predictions, not from the model's building channels. The model itself does not learn flooded vs dry buildings from these supports, which hold ~4× more flooded than dry building pixels.

**Routing evaluation on predictions (eval100, 6 Oct 12:00, table regenerated 17:30 with the off-road gate g = .3 on predicted maps; `src/route_report.py` → `results/routing/routing.md`).** Same GT base and GT buildings for both maps. The decode is τ = 0.5 or the tuned decode of each run:

| run | decode | truck cost | stranded P | stranded R | stranded F1 | status agreement | RSV | unsafe routes | cost ratio (median) | infeasible |
|---|---|---|---|---|---|---|---|---|---|---|
| floodnet_1shot | τ=0.5 | risk-aware λ=5 | 0.878 | 0.799 | 0.837 | 0.788 | 0.137 | 0.149 | 1.160 | 0.377 |
| floodnet_1shot | τ=0.5 | plain λ=0 | 0.878 | 0.799 | 0.837 | 0.788 | 0.105 | 0.159 | 1.137 | 0.372 |
| floodnet_1shot | tau | risk-aware λ=5 | 0.485 | 1.000 | 0.653 | 0.445 | 0.003 | 0.000 | 1.043 | 0.180 |
| floodnet_1shot | tau | plain λ=0 | 0.485 | 1.000 | 0.653 | 0.445 | 0.003 | 0.000 | 1.049 | 0.180 |
| floodnet_5shot | τ=0.5 | risk-aware λ=5 | 0.935 | 0.587 | 0.721 | 0.764 | 0.105 | 0.237 | 1.029 | 0.348 |
| floodnet_5shot | τ=0.5 | plain λ=0 | 0.935 | 0.587 | 0.721 | 0.764 | 0.142 | 0.264 | 1.000 | 0.407 |
| floodnet_5shot | tau_ctx | risk-aware λ=5 | 0.935 | 0.647 | 0.765 | 0.793 | 0.090 | 0.209 | 1.030 | 0.301 |
| floodnet_5shot | tau_ctx | plain λ=0 | 0.935 | 0.647 | 0.765 | 0.793 | 0.123 | 0.244 | 1.000 | 0.364 |
| floodnet_10shot | τ=0.5 | risk-aware λ=5 | 0.865 | 0.810 | 0.837 | 0.778 | 0.067 | 0.084 | 1.054 | 0.289 |
| floodnet_10shot | τ=0.5 | plain λ=0 | 0.865 | 0.810 | 0.837 | 0.778 | 0.102 | 0.126 | 1.001 | 0.358 |
| floodnet_10shot | tau_ctx | risk-aware λ=5 | 0.858 | 0.810 | 0.834 | 0.774 | 0.049 | 0.096 | 1.071 | 0.250 |
| floodnet_10shot | tau_ctx | plain λ=0 | 0.858 | 0.810 | 0.834 | 0.774 | 0.074 | 0.126 | 1.014 | 0.299 |
| floodnet_20shot | τ=0.5 | risk-aware λ=5 | 0.909 | 0.595 | 0.719 | 0.766 | 0.094 | 0.213 | 1.043 | 0.332 |
| floodnet_20shot | τ=0.5 | plain λ=0 | 0.909 | 0.595 | 0.719 | 0.766 | 0.126 | 0.236 | 1.000 | 0.361 |
| floodnet_20shot | tau_ctx | risk-aware λ=5 | 0.911 | 0.606 | 0.728 | 0.774 | 0.091 | 0.205 | 1.038 | 0.320 |
| floodnet_20shot | tau_ctx | plain λ=0 | 0.911 | 0.606 | 0.728 | 0.774 | 0.125 | 0.232 | 1.000 | 0.354 |
| floodnet_50shot | τ=0.5 | risk-aware λ=5 | 0.899 | 0.725 | 0.802 | 0.784 | 0.070 | 0.165 | 1.045 | 0.328 |
| floodnet_50shot | τ=0.5 | plain λ=0 | 0.899 | 0.725 | 0.802 | 0.784 | 0.092 | 0.175 | 1.000 | 0.382 |
| floodnet_50shot | tau | risk-aware λ=5 | 0.899 | 0.725 | 0.802 | 0.781 | 0.068 | 0.162 | 1.035 | 0.325 |
| floodnet_50shot | tau | plain λ=0 | 0.899 | 0.725 | 0.802 | 0.781 | 0.092 | 0.185 | 1.000 | 0.380 |
| rescuenet_20shot | τ=0.5 | risk-aware λ=5 | 0.900 | 0.360 | 0.514 | 0.932 | 0.060 | 0.104 | 1.055 | 0.140 |
| rescuenet_20shot | τ=0.5 | plain λ=0 | 0.900 | 0.360 | 0.514 | 0.932 | 0.076 | 0.094 | 1.034 | 0.147 |
| rescuenet_20shot | tau_bldg | risk-aware λ=5 | 0.900 | 0.360 | 0.514 | 0.938 | 0.044 | 0.077 | 1.050 | 0.088 |
| rescuenet_20shot | tau_bldg | plain λ=0 | 0.900 | 0.360 | 0.514 | 0.938 | 0.058 | 0.087 | 1.029 | 0.121 |
| rescuenet_50shot | τ=0.5 | risk-aware λ=5 | 0.636 | 0.280 | 0.389 | 0.914 | 0.067 | 0.092 | 1.071 | 0.162 |
| rescuenet_50shot | τ=0.5 | plain λ=0 | 0.636 | 0.280 | 0.389 | 0.914 | 0.067 | 0.095 | 1.044 | 0.151 |
| rescuenet_50shot | tau_bldg | risk-aware λ=5 | 0.500 | 0.280 | 0.359 | 0.908 | 0.064 | 0.088 | 1.080 | 0.151 |
| rescuenet_50shot | tau_bldg | plain λ=0 | 0.500 | 0.280 | 0.359 | 0.908 | 0.062 | 0.091 | 1.051 | 0.147 |

- *The bullets below were written for the gate-free table (`results/routing/routing_nogate.md`); with the gate, recall is higher and precision lower (see "Off-road gate").*
- **Risk-aware truck costs help every run.** RSV falls by ~25–35 % (FloodNet 10-shot tuned: .121 → .087) and infeasible routes by 4–10 points, at a median cost premium of 3–5 %.
- **FloodNet 50-shot routing:** tuned decode, risk-aware: stranded F1 .667 (P .993, R .502), status agreement .758, RSV .103, unsafe routes .251, infeasible .315. RescueNet 50 is weak at stranded detection (only 25 GT non-truck houses; F1 .36 tuned).
- **Stranded detection: high precision (.98–1.0), low recall (.30–.64).** Predicted maps over-connect: they miss water barriers, so cut-off houses look truck-reachable. That is the dangerous direction; feed it into the failure analysis.
- **The tuned decode raises stranded recall at 5 and 20 shots** (.275 → .636, .316 → .461), mostly through more water (τ_water .15–.25). At 10 shots it lowers recall (.59 → .52).
- 1-shot tuned is degenerate (τ_road_clear .8 → almost no road; every house "stranded", recall 1.0, precision .49). Don't headline it.

**Track A completion (6 Oct 15:45–17:45).** GPU: notebook `rr-gpu-extra` (`QUEUE = 'extra'`, metadata `kaggle/extra/`), 1.7 h on 2× T4. It ran the zero-label transfer and the full 448-image test of FloodNet 50-shot, split into `splits/floodnet_test_full_part{1,2}.txt`. `splits/floodnet_test_full.txt` was regenerated from the 448 test ids, sorted numerically; it matches the SHA-256 in `splits/SHA256SUMS`. `predict.py --query_dataset` keeps the checkpoint's own supports and takes queries from the other dataset. Outputs are in `results/kaggle_v2/extra/`, with caches merged into `results/kaggle_v2/rr/preds/`. Kaggle GPU use on 6 Oct: ≈ 11 of 30 session-hours.

*§5.2 PSPNet comparison* (`src/baseline_table.py` → `results/kaggle_v2/report/baselines.md`; full-test metrics in `full_test_floodnet50.json`):

| method | labels | bldg flooded | bldg dry (non-flooded) | road flooded | road clear (non-flooded) | water | mean of these 5 | paper mIoU (9 cl.) |
|---|---|---|---|---|---|---|---|---|
| ENet (paper) | 1,445 | 6.9 | 47.4 | 12.5 | 48.4 | 49.0 | 32.8 | 42.6 |
| DeepLabV3+ (paper) | 1,445 | 32.7 | 72.8 | 52.0 | 70.2 | 75.2 | 60.6 | 61.5 |
| PSPNet (paper) | 1,445 | 68.9 | 89.8 | 82.2 | 91.2 | 92.0 | 84.8 | 79.7 |
| Chameleon 50-shot, eval100, τ 0.5 | 50 | 72.1 | 21.1 | 39.8 | 51.1 | 44.6 | 45.7 | – |
| Chameleon 50-shot, eval100, tuned | 50 | 72.0 | 34.1 | 43.0 | 52.6 | 44.1 | 49.2 | – |
| Chameleon 50-shot, full test (448 img), τ 0.5 | 50 | 55.8 | 29.8 | 29.3 | 50.1 | 45.0 | 42.0 | – |
| Chameleon 50-shot, full test (448 img), tuned | 50 | 57.7 | 50.0 | 29.1 | 54.4 | 44.8 | 47.2 | – |

Caveats: the paper trains on 10 classes at 713×713 and reports the full 448-image test; we train a 5-channel task (other classes decoded as "other") at 768×576 with 50 labels (3.5 % of 1,445). The 5-class decode removes confusions with tree/grass/vehicle/pool, which can inflate our IoU a little. eval100 is a stratified subset (50 % with flooded road/building), not the paper split.

- **Full test, FloodNet 50-shot:** mIoU .420 (τ 0.5) / .472 (tuned). Drivable precision .857 / .830, recall .547 / .612, wet IoU .548 / .584.
- The full test is lower than eval100 on flooded buildings (.56 vs .72), because eval100 is stratified 50 % flooded. It is higher on dry buildings (.50 tuned).
- **Headline framing:** with 50 labels (3.5 %), Chameleon beats ENet on 4 of 5 classes (water 45 vs 49) and DeepLabV3+ on flooded buildings (58 vs 33). It is far below PSPNet on every class, worst on flooded roads (29 vs 82).

*§3.4 cross-disaster* (`src/transfer_eval.py` → `results/kaggle_v2/report/transfer.md`): RescueNet eval100 on the shared super-classes road (0+1), building (2+3) and water (4):

| model (RescueNet eval100) | road IoU | building IoU | water IoU | mean |
|---|---|---|---|---|
| zero-label transfer: FloodNet 50-shot (0 RescueNet labels), τ 0.5 | 0.353 | 0.263 | 0.225 | 0.280 |
| zero-label transfer: FloodNet 50-shot, FloodNet-tuned decode | 0.375 | 0.366 | 0.223 | 0.321 |
| re-adapted: RescueNet 20-shot, τ 0.5 | 0.625 | 0.548 | 0.645 | 0.606 |
| re-adapted: RescueNet 20-shot, tuned decode | 0.600 | 0.640 | 0.644 | 0.628 |
| re-adapted: RescueNet 50-shot, τ 0.5 | 0.649 | 0.603 | 0.401 | 0.551 |
| re-adapted: RescueNet 50-shot, tuned decode | 0.645 | 0.632 | 0.418 | 0.565 |

- **Zero-label transfer reaches about half of what 20 RescueNet labels give** (mean .32 vs .63). A new disaster needs its own few labels, which is the story's point.

*§5.3 failure analysis* (`src/failures.py` → `results/kaggle_v2/report/failures/`, with `per_image.json`, `worst.json` and contact sheets `sheet_1..4.png`). FloodNet 50-shot, tuned decode, eval100. The 12 worst images by road score (mean IoU of the road channels covering ≥ 1 % of the image), plus the 4 largest false-"clear" clusters, all tagged by eye:

| tag | failure | images (of the 16) | routing consequence |
|---|---|---|---|
| 1 | dark surface confused (here: a dark water channel decoded as clear road) | 6377 | false "clear" (no houses in the scene) |
| 2 | **shallow / muddy / sandy water decoded as dry land or clear road** | 7284, 7355, 7370, 7486, 7308 | **the dangerous one.** Without the gate: 25 stranded houses shown as truck-reachable in these 5 images. With the off-road gate: 6 (7355: 3, 7308: 3) |
| 3 | canopy breaking a road | – | – |
| 4 | road / driveway missed (pale concrete decoded as other) | 6420, 6468 | small: the truck still crosses off-road at ×4 cost |
| 5 | water ↔ flooded-road (or pond) confusion | 6468, 6775, 6785, 6786, 7268, 7935 | benign: both are boat cells |
| 6 | flooded ↔ dry building | 6695, 6468 | status priority only |
| 7 | tile-border artefact (class switch along the tile row y ≈ 192) | 6775, 6785, 6786 | benign (inside water) |
| 8 | **GT label ambiguity**: bridges over the flood labelled "flooded road" (7291, 6695); houses in a dry street labelled "flooded" (6412, qualitative figure) | 7291, 6695 (+ 6412) | counted as "unsafe" / "missed", though the prediction is arguably right (7291: 2 houses) |
| 9 | thin structures (power lines over water) decoded as clear road | 7268, 7759 | could open a false truck "bridge"; none here (no houses) |

- Gallery: `results/kaggle_v2/report/figures/failure_gallery.png` (2×3: 7370, 7486, 7308, 7291, 6775, 6420; GT | prediction, with tag and routing consequence).
- Qualitative grid: `figures/qualitative.png` (7320, 7415 flooded; 6412, 9107 dry; columns photo, GT, 1/5/20/50-shot).
- Overlay colours (`src/viz.py`): green clear road, blue flooded road, red flooded building, **amber dry building** (grey vanished on grey roofs), navy water. RescueNet uses magenta for blocked roads.

*Off-road gate (router, from the failure analysis).* 92 % of the GT-water cells on predicted truck routes were predicted "other" land, crossed off-road (median p_wet .21). New router option `offroad_maxwet = g`: off-road only where p1 + p4 < g. **Prediction grids only; GT grids are unchanged.** Tuned on FloodNet 50 val30 (stranded F1 .548 → .971 at g = .3), then applied to eval100 unchanged. **It is the default in `route_eval.py` (`PRED_OFFROAD_MAXWET = 0.3`); the web viewer's JS router must use the same value.** Sweep (`results/routing/gate/gate.md`, tuned decode, λ = 5):

| split | gate g | stranded P | R | F1 | agreement | RSV | unsafe | cost ratio | infeasible | GT stranded | pred stranded |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val30 | none | 1.000 | 0.377 | 0.548 | 0.783 | 0.089 | 0.225 | 1.045 | 0.288 | 53 | 20 |
| val30 | 0.5 | 1.000 | 0.377 | 0.548 | 0.783 | 0.089 | 0.225 | 1.045 | 0.288 | 53 | 20 |
| val30 | 0.3 | 1.000 | 0.943 | 0.971 | 0.894 | 0.064 | 0.097 | 1.060 | 0.318 | 53 | 50 |
| val30 | 0.2 | 0.773 | 0.962 | 0.857 | 0.818 | 0.048 | 0.058 | 1.084 | 0.359 | 53 | 66 |
| val30 | 0.15 | 0.699 | 0.962 | 0.810 | 0.783 | 0.043 | 0.053 | 1.150 | 0.388 | 53 | 73 |
| val30 | 0.1 | 0.520 | 0.981 | 0.680 | 0.646 | 0.023 | 0.009 | 1.234 | 0.459 | 53 | 100 |
| val30 | 0.05 | 0.452 | 0.981 | 0.619 | 0.571 | 0.026 | 0.011 | 1.576 | 0.529 | 53 | 115 |
| val30 | 0 | 0.366 | 1.000 | 0.535 | 0.429 | 0.000 | 0.000 | 2.347 | 0.618 | 53 | 145 |
| eval100 | none | 0.993 | 0.502 | 0.667 | 0.758 | 0.103 | 0.251 | 1.035 | 0.315 | 269 | 136 |
| eval100 | 0.5 | 0.993 | 0.546 | 0.705 | 0.774 | 0.100 | 0.236 | 1.035 | 0.315 | 269 | 148 |
| eval100 | 0.3 | 0.899 | 0.725 | 0.802 | 0.781 | 0.068 | 0.162 | 1.035 | 0.325 | 269 | 217 |
| eval100 | 0.2 | 0.811 | 0.892 | 0.850 | 0.746 | 0.038 | 0.085 | 1.094 | 0.328 | 269 | 296 |
| eval100 | 0.15 | 0.659 | 0.955 | 0.780 | 0.647 | 0.028 | 0.051 | 1.103 | 0.423 | 269 | 390 |
| eval100 | 0.1 | 0.588 | 0.989 | 0.738 | 0.572 | 0.007 | 0.009 | 1.184 | 0.478 | 269 | 452 |
| eval100 | 0.05 | 0.518 | 1.000 | 0.683 | 0.480 | 0.001 | 0.004 | 1.345 | 0.559 | 269 | 519 |
| eval100 | 0 | 0.470 | 1.000 | 0.640 | 0.408 | 0.000 | 0.000 | 1.527 | 0.631 | 269 | 572 |

- **eval100 at g = .3:** stranded F1 .667 → .802, unsafe routes .251 → .162, RSV .103 → .068, at precision .993 → .899. Missed stranded houses fall from 134 to 74.
- The routing table above, "Routing evaluation on predictions", is now **with the gate** (regenerated); the gate-free version is `results/routing/routing_nogate.md`.
- With the gate, FloodNet routing on eval100 (risk-aware, tuned decode): stranded F1 .653 / .765 / .834 / .728 / .802 at 1 / 5 / 10 / 20 / 50 shots.

**Still open for Track A** (all need the Track B code or come later): the final notebook (§7) with `RUN_TRAINING=False`, which recomputes FloodNet 50 live and loads the rest from caches; the G3 check on a fresh T4; the write-up and slides. Dropped on purpose: R3 diverse supports, R5 100-shot, R6 seeds, the 1024-px ablation.

**Earlier plan (kept for reference): the core queue on Kaggle, launched Tue 6 Oct 02:40 KST, ETA 11:30–12:00 KST** (Kaggle session limit 12 h; no new job starts after 9.5 h):
- GPU 0: FloodNet 1-shot → eval100 cache → 50-shot → cache → 20-shot → cache → 10-shot → cache.
- GPU 1: RescueNet 20-shot → cache → FloodNet 5-shot → cache → RescueNet 50-shot → cache.
- Check at 03:30 (0.5 h in): both GPUs training at 3.7–4.1 s/step (10–20 % slower than Colab: the two jobs share 4 CPU cores), peak 10–14 GB. The best-checkpoint code works (FloodNet 1-shot val mIoU 0.151 at step 250). The RescueNet val split works (20-shot step 250: road_clear 0.32, road_blocked 0.00, bldg_intact 0.56, bldg_damaged 0.05, water 0.64, mIoU 0.315).
- Dropped from the plan: R3 (diverse supports), R5, R6, the full-448 test (eval100 only, unless time is left).

**Collecting results (when `kaggle kernels status` says COMPLETE):** `kaggle kernels output marlenmelis/rr-gpu-queue -p results/kaggle_v2` (into the project folder; `results/` is not tracked by git). Contents: `rr/ckpt/*.ckpt` (best-val checkpoints), `rr/metrics/*.json` (best + final val, and eval100), `rr/preds/*_val_tune30.npz` + `*_test_eval100_fp16.npz` (uint8 probs + confusion), `rr/logs/` (per-run JSON, queue logs). Check `queue_gpu*.log` for FAILED jobs, and any leftover `*_last.pt` (unfinished run → push the notebook again; finished runs are not inside the new session, so either rerun only the missing jobs or attach the previous output).

**G1 result (FloodNet 20-shot s0, 2000 steps, fp16 recipe; finished 6 Oct 00:28 KST, 2 h 20 min on the T4).** Outputs on Drive: `ckpt/floodnet_20shot_s0_stratified.ckpt` (5 MB), `metrics/…json`, `preds/…_floodnet_val_tune30.npz` (fp16 probs + confusion), `logs/…json|.out`, `logs/…_val30_fp32check.out`. val30, τ = 0.5:

| step | road_clear | road_flooded | bldg_flooded | bldg_dry | water | mIoU | drivable P / R | wet IoU |
|---|---|---|---|---|---|---|---|---|
| 250 | 0.198 | 0.078 | 0.669 | 0.000 | 0.012 | 0.191 | 0.46 / 0.26 | 0.26 |
| 500 | 0.171 | 0.117 | 0.656 | 0.000 | 0.397 | 0.268 | 0.51 / 0.21 | 0.47 |
| 750 | 0.179 | 0.227 | 0.675 | 0.000 | 0.488 | 0.314 | 0.41 / 0.24 | 0.63 |
| 1000 | 0.270 | 0.339 | 0.669 | 0.016 | 0.445 | 0.348 | 0.52 / 0.36 | 0.65 |
| 1250 | 0.421 | 0.356 | 0.665 | 0.000 | 0.128 | 0.314 | 0.46 / 0.82 | 0.54 |
| 1500 | 0.382 | 0.363 | 0.658 | 0.019 | 0.063 | 0.297 | 0.59 / 0.52 | 0.47 |
| 1750 | **0.520** | **0.301** | 0.647 | 0.000 | 0.477 | **0.389** | 0.64 / 0.73 | 0.68 |
| 2000 (final) | 0.480 | 0.362 | 0.666 | 0.051 | 0.120 | 0.336 | 0.53 / 0.84 | 0.49 |

- **Gate:** formally a narrow **fail** at the final step (road_clear 0.480 < 0.5; road_flooded 0.362 ≥ 0.25 passes). Step 1750 passed both. Val30 swings a lot between evals (water 0.48 ↔ 0.06–0.13), consistent with a constant lr 1e-3 at gbs 2 and only the final checkpoint being kept.
- **Final confusion (rows GT, cols pred):** water GT is mostly decoded "other" (1.11 M of 1.82 M px) or road_flooded (0.45 M). Dry buildings go to flooded (136 k of 256 k) or other (102 k); only 16 k are correct. The supports have 4× more flooded- than dry-building pixels (621 k vs 148 k; val30 872 k vs 256 k). 119 k road_flooded GT px are decoded road_clear (unsafe for routing).
- **τ sweep on the cached probs (CPU):** τ 0.4 gives the best mIoU 0.379 (water 0.36) but road_clear 0.455. No τ lifts road_clear to 0.5.
- **§0.1 fp16 check: passed.** The same checkpoint evaluated in fp32 (chunk 1/1) agrees within 0.3 IoU points on every channel (road_clear 0.477 vs 0.480, mIoU 0.335 vs 0.336). fp32 inference costs 12.3 s/image vs 5.6.
- Colab tab disconnects did not stop the runtime: the background process kept running through two "reconnects".

**Next (in order):**
1. ~~Track B router + unit tests + routing metrics on GT maps~~ (done 04:10).
2. ~~Download, check logs, tables + curve~~ (done; all 7 runs as of 15:30). The 50-shot outputs were fetched into `results/kaggle_v2/{q50,s50}/` with `kaggle/fetch_output.py`; their `*50shot*` files were copied into `results/kaggle_v2/rr/`. Then `src/collect_results.py`, `src/decode_tune.py --runs …` and `src/route_report.py` were rerun. Originally this step said: fill the data-efficiency curve (val30 + eval100) and the per-class table vs PSPNet (§5.2).
3. ~~Decode tuning on val30 from the caches~~ (done, all 7 runs). ~~Routing evaluation on predictions~~ (done, all 7 runs, `src/route_report.py`). See "Decode tuning" below.
4. ~~Optional dry-building ablation~~: **skipped** (see the decision above).
5. **Next phase (new chat, from 6 Oct ~16:00): Track B product.** Marlen wants to see the final product first; the write-up and slides wait until it exists. In order:
   1. **Showcase scenes + export (CPU, Mac).** Pick 8 eval100 scenes: 6 FloodNet from the 50-shot model, 2 RescueNet. Cover truck-only, truck + boat, isolated, and one honest failure (a missed water barrier). Use the per-run tuned decode (`results/kaggle_v2/rr/decode/<run>.json`, key `chosen`). For RescueNet, use 20-shot (eval100 tuned .421) rather than 50 (.402), unless the scenes look better with 50; state which. Write `export_web_assets` (in `src/`) to produce, per scene: photo, pred/GT RGBA overlays, cell-grid JSON (classes + wet probability), buildings JSON (id, polygon, centroid, status pred/GT), meta. Local images are 768×576 (`~/rr_data`); §6.4's 2048-px photo would need the original FloodNet/RescueNet files, so use 768 (or 1024 from the r1024 subset tar on Drive) unless the originals are fetched for the 8 scenes only.
   2. **Web viewer** (`app/` source → built into `docs/`): plain HTML/JS + Leaflet 1.9 (`L.CRS.Simple`), no build step. Layer toggles + opacity, prediction/GT swap, swipe slider, building status layer, click a house → **JS port of `src/router.py`** (same costs: λ = 5, boat ×3, launch 40, off-road ×4 **gated at p_wet < .3 on predicted maps** (none on GT maps), no switch back; must reproduce the Python statuses and costs exactly on the exported grids; test that). Overlay colours from `src/viz.py` → animated truck then boat, "Unreachable by road: N / M" counter, "disagreements with GT" toggle, scene picker, legend, attribution (FloodNet CDLA-Permissive; RescueNet licence note). Check it on phone width and in incognito.
   3. **Gradio app** (§6.3): showcase scenes from the caches (CPU), upload → live inference (GPU; test on Kaggle later, or leave it as a notebook cell). Risk-aware toggle, boat-speed slider, click → route + summary.
   4. **Deploy:** a separate **public** repo `MarlenMM/rescue-router-demo` holding only the built `docs/` viewer (decision 6 Oct 15:40: keep `rescue-router` private until grading, then fold everything into it: clean PLAN/history, make it public, Pages from `docs/`, retire the demo repo with a redirect). `gh` CLI is logged in as MarlenMM. **Creating the repo and publishing need Marlen's explicit OK at that moment.** Script the export (private `docs/` → demo repo) as one command.
   5. Screen recording (60–90 s) later, by Marlen.
6. ~~Failure analysis, PSPNet table, cross-disaster transfer, full-448 test, figures~~ (done 17:45; see "Track A completion"). After the product: write-up, slides, final notebook (§7), G3 T4 check.

## 14. Sources
- FloodNet: https://github.com/BinaLab/FloodNet-Supervised_v1.0 · paper https://arxiv.org/abs/2012.02951
- RescueNet: https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation · https://www.nature.com/articles/s41597-023-02799-4 · https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1 · https://huggingface.co/datasets/isp-uv-es/rescuenet-taco
- Chameleon: https://arxiv.org/abs/2404.18459 · https://github.com/GitGyun/chameleon · starter https://github.com/GitGyun/AIC200-Project-StructuredPrediction
- Seoul 2022: https://www.cnn.com/2022/08/09/asia/seoul-south-korea-rain-flooding-intl-hnk/index.html
- Osong 2023: https://www.koreaherald.com/view.php?ud=20230717000749
