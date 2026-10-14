# Rescue Router: Project Plan

**Few-shot flood mapping and rescue routing with Chameleon.** KAIST AIC.20000 mini-project, **Track A + Track B**.
Student: 20260876. Presentation: Thu 8 Oct 2026, in class (16:00). KLMS deadline: 8 Oct 23:59 KST. Target submission: **8 Oct 12:30**.

This file is the single source of truth for the build. Claude Code executes it. When reality differs from the plan, update this file.

**Since 14 Oct (after submission): §16 "After submission" is the live roadmap; read it first.** **How to read it (cleaned up 7 Oct):** **§15 "Status / handoff" is the submission-time state**: its Quick summary first, then the
improvement batches (the roadmap), then the chronological log. Parts A and B (§0–§14) are the plan as written on
5 Oct; they are kept for the record, and where reality went another way the section says so in a *[Superseded]* note
(main changes: all GPU work on Kaggle T4s with an fp16 recipe instead of Colab A100s; the router grew walking, vehicle
widths and confidence tiers; the schedule). The repository overview for readers is `README.md`.

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

*[Superseded 6 Oct: Marlen has no Colab subscription, so all GPU work ran on **Kaggle** (2× T4 notebooks driven from
the Mac with the Kaggle CLI) with recipe v2 (fp16 + gradient checkpointing, cosine lr, best-val30 checkpoint); see §15
"Compute", "Recipe v2" and "Kaggle setup". Colab was used only for the CPU data preparation. The budget rule §0.2
still holds.]*

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

### 2.6 Data zip (`20260876_<NAME>_data.zip`, target < 300 MB): contents spec (revised 7 Oct, Batch 8)
*[Superseded the 5 Oct list: no seeds s1–s2 were trained, and the caches are packed losslessly.]* Built in Batch 12.
```
data/
├── README_data.md        counts, sources/links, class mapping (FloodNet 10 -> 5 slots, RescueNet 11 -> 5 slots),
│                         how each file was made, how to restore the caches
├── LICENSES.md           FloodNet CDLA-Permissive-1.0, RescueNet (figshare CC0; GitHub release CC BY-NC-ND 4.0), attribution
├── splits/  supports/    the repo's split and support lists + SHA256SUMS (also in the code)
├── images/               the 768x576 photos + labels (channel ids) of every photo a cache covers: FloodNet test
│                         (448, incl. eval100) + val30 + the 50 supports; RescueNet eval100 + val30 + the 50 supports
│                         (prep_data.py output, JPEG q95 / PNG)
├── ckpt/                 the 7 fine-tuned checkpoints (FloodNet 1/5/10/20/50, RescueNet 20/50; ~5 MB each: adapted
│                         parameters only, the meta-checkpoint is the course's generalist.ckpt)
├── preds/*.ptar          prediction caches packed by `src/cache_pack.py` (lossless PNG; `unpack` restores the npz
│                         bit for bit and checks SHA-256): eval100 + val30 of all 7 runs, FloodNet 50 full test,
│                         FloodNet 50 -> RescueNet eval100 (zero-label transfer)
├── decode/               the tuned decodes (rr/decode/*.json) and the logs/metrics JSONs of every run
└── report/               results/ JSON + Markdown tables (no PNG duplicates of the write-up figures)
```
Not included (stated in README_data): step snapshots of the 50-shot runs, the full-test halves (merged), the
test-time-flip caches (not adopted; reproducible with `kaggle/flips`), the 1024-px photos (viewer only; they are in
the app zip's scenes). **Measured 7 Oct (Batch 8):** caches 497 MB as npz → **264 MB packed** (lossless WebP, all 16 verified bit-exact
after unpacking); checkpoints 35 MB; labels (all covered photos) ~12 MB; photos of eval100 + val30 (FloodNet + RescueNet)
~60 MB (the other FloodNet test photos would add ~70 MB; not needed: the scripts read labels and caches, only the viewer
export and figures need photos, and those cover eval100); report JSON/MD ~10 MB. **Total ≈ 380 MB, above the 300 MB
target.** Open for Marlen (Batch 12): the KLMS upload limit decides: (a) ship ≈ 380 MB if KLMS allows it; (b) under 300:
leave the zero-label-transfer cache (31 MB) and the 1/5/10-shot val30 caches (21 MB) out and drop the RescueNet 50-shot
caches (19 MB), stating that those numbers are reproducible from the checkpoints; (c) host the caches as a public
Kaggle dataset and put only a link + the 50-shot caches in the zip.

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

| dataset | router | images with base | TRUCK | BOAT | FOOT | ISOLATED |
|---|---|---|---|---|---|---|
| FloodNet | off-road ×4 + walking + physical rules (v3, default) | 100 (all via road or land) | 470 | 148 | 117 | 5 |
| FloodNet | roads only + walking + physical rules | 92 | 25 | 199 | 477 | 38 |
| RescueNet | off-road ×4 + walking + physical rules (v3, default) | 100 | 267 | 3 | 18 | 4 |
| RescueNet | roads only + walking + physical rules | 76 | 58 | 4 | 167 | 11 |

*Router v2 (walking, point vehicles): FloodNet off-road 471 / 144 / 125 / 0, roads only 19 / 165 / 516 / 39; RescueNet off-road 267 / 5 / 16 / 4, roads only 40 / 4 / 185 / 11.* *Router v1 (truck/boat only, 6 Oct): FloodNet off-road 471 / 144 / – / 125 isolated, roads only 19 / 165 / – / 555; RescueNet off-road 267 / 5 / – / 20, roads only 40 / 4 / – / 196. With walking, every FloodNet house that v1 called isolated is reachable by boat, then on foot.* **These statuses are the router run on the labelled maps; the datasets label pixels, not reachability** (viewer and app call this the "labelled map", not "ground truth").

- FloodNet, off-road, by GT building class: flooded buildings split 348 TRUCK / 134 BOAT / 102 ISOLATED, dry buildings 123 / 10 / 23.
- So a "flooded" label often means water touched the house, while a dry path still reaches its yard.
- All 125 v1-isolated FloodNet buildings have land in their ring, cut off by water or buildings: with walking (v2) all 125 are BOAT + FOOT.

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
- **Off-road truck (new default, `offroad=4`).** The truck may also cross dry "other" land (lawn, trees, background, pools, vehicles) at 4× the road cost. Water, flooded road (RescueNet: blocked road) and buildings still stop it. Why: with roads only (the rule above), the GT maps of FloodNet eval100 leave **555 of 739 buildings ISOLATED** (19 TRUCK, 165 BOAT), because at 768×576 a 4-px cell is small (0.3–0.7 m, see "Router v3" below) and almost every house sits behind its own yard. Then "stranded" would measure yards, not floods. A wider goal ring does not fix it (ring 12 cells still leaves 345 isolated). `offroad=None` restores the strict rule.
- **Search:** one multi-source Dijkstra (`scipy.sparse.csgraph`, 3 layers × 27,648 cells) gives every building's status. TRUCK = a ring cell is reachable in the truck layer, BOAT = only in the boat layer, FOOT = only on foot.
- **Router v2: walking (decided by Marlen 6 Oct ~18:45).** A third mode FOOT: modes only go forward, truck → boat → foot (or truck → foot, or boat → foot from a water entry), never back into a vehicle. Foot moves on dry land (road_clear, 'other' with the same off-road gate on predicted maps) and, RescueNet only, on blocked road (debris: walkable, not drivable; Marlen's choice). Cost 8 per cell (slowest mode), +10 to get out of a truck or boat. Statuses TRUCK / BOAT / FOOT / ISOLATED (= no route: air team). `foot_cost=None` gives the v1 truck/boat router. **`route(b)` now ends in the layer of the building's status** (a TRUCK house gets its road route, a BOAT house its boat route); v1 took the cheapest over both layers, which sometimes showed a boat route to a truck-reachable house. Tests: 19 (`src/tests/test_router.py`).
- **Router v3: physical rules (6 Oct 20:19, Batch 1 of the improvement list).** Ground scale measured from the labelled cars (median car 1.8 m wide): **RescueNet 7.7 cm/px (cell 0.31 m); FloodNet flies at two heights, ids 7000–7999 at 16.6 cm/px (cell 0.66 m, most flooded scenes), most others ~8.5 cm/px (0.34 m).** The camera estimate for FloodNet's stated 200 ft is ~10 cm/px; the cars disagree, so the cars are used. One scale per dataset (`PROFILES[...]['cell_m']`): FloodNet uses the larger cell (0.66 m), so scale errors can only under-block, never close a real gap. Rules, sizes in metres: (1) **vehicle width:** a cell is open to the truck (half-width 1.25 m) / boat (0.8 m) only if its centre is ≥ half-width + half a cell from every wall cell; a vehicle reaches a house when its edge is within the ring (goal = building dilated by ceil(clearance) + ring cells; foot keeps the plain ring); (2) **parked cars:** road gaps ≤ 2 m closed (PLAN §3.1, never implemented before), dry gated land only; (3) **canopy:** a boat may cross non-building, non-road land strips ≤ 3 m between water at 2× boat cost (tree crowns over floodwater decode as 'other'); (4) **corners:** no diagonal step between two obstacle cells touching at a corner. `router.POINT_VEHICLES` = the v2 router; `route_eval.py --point_vehicles` reproduces v2 exactly (checked), `--router '<json>'` overrides any parameter. JS: Python exports per-cell passability flags (`Grid.flags`) and per-mode goal rings, the JS does the search; all 474 routes of the 8 scenes bit-exact. Tests: 24. **Gate re-checked on val30: 0.3 is still best (F1 .980).**

  Ablation (eval100, tuned decode, λ 5; `results/routing/physical/ablation.md`):

  | dataset | router | labelled map: truck / boat / foot / none | stranded P | R | F1 | RSV | unsafe | infeasible |
  |---|---|---|---|---|---|---|---|---|
  | FloodNet 50 | point vehicles (v2) | 471 / 144 / 125 / 0 | .899 | .725 | .802 | .067 | .161 | .365 |
  | FloodNet 50 | + truck width only | 470 / 143 / 127 / 0 | .868 | .804 | .835 | .052 | .121 | .369 |
  | FloodNet 50 | + boat width only | 471 / 148 / 116 / 5 | .899 | .725 | .802 | .067 | .161 | .354 |
  | FloodNet 50 | + car closing only | 471 / 144 / 125 / 0 | .899 | .725 | .802 | .067 | .161 | .365 |
  | FloodNet 50 | + canopy only | 471 / 146 / 123 / 0 | .899 | .725 | .802 | .066 | .160 | .358 |
  | FloodNet 50 | + corner rule only | 471 / 144 / 125 / 0 | .899 | .725 | .802 | .067 | .161 | .365 |
  | FloodNet 50 | **all four (v3 default)** | 470 / 148 / 117 / 5 | .868 | .804 | **.835** | **.052** | **.121** | .348 |
  | RescueNet 20 | point vehicles (v2) | 267 / 5 / 16 / 4 | .900 | .360 | .514 | .045 | .076 | .139 |
  | RescueNet 20 | **all four (v3 default)** | 267 / 3 / 18 / 4 | 1.000 | .400 | **.571** | .043 | .073 | .201 |

  - **Truck width carries almost all of the gain** (predicted maps open 1-cell squeezes between houses that are often misread water or yards). Boat width makes 5 FloodNet houses (all in scene 7241) unreachable: flood pockets between houses narrower than a boat. **Car closing and the corner rule change nothing measurable** (the off-road truck already passes parked cars; widths already keep vehicles off corners); kept as stated rules.
  - Truck statuses changed, so stranded P/R/F1 changed with v3 (unlike v2); all tables below are v3, v2 results kept in `results/routing_v2/`.
- (v2 note) Truck statuses are unchanged by walking, so **stranded P/R/F1 (= not TRUCK) and the gate tuning are identical to v1**. Status agreement is over 4 classes now; infeasible rises because formerly isolated houses now have a GT route that predicted walks can fail (walking onto GT water).
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

*[Superseded: what actually happened is in §15. Remaining, from Wed 7 Oct (one batch per prompt, §15 "Improvement
batches"): Batch 9 final notebook + G3 on a fresh T4 (Wed morning, needs OK), Batch 10 write-up (Wed), Batch 11 slides
+ recording storyboard (Wed evening; Marlen records), Batch 12 packaging + Pages release (Thu morning, publishing needs
OK), submit by Thu 12:30, live Gradio link pushed ~15:15 for the 16:00 presentation (§15 0l).]*

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
*[7 Oct: the app zip = the built viewer `docs/` (open `index.html` through any static server, or the Pages URL), the
Gradio app (`src/gradio_app.py` + caches it needs, or the share-link notebook), README with both URLs and the local
commands, and Marlen's recording; the data zip per §2.6. Every reported number is reproducible with
`src/rebuild_all.py` (CPU) from the data zip.]*
- [x] Only one .ipynb anywhere in the zip (checked recursively by `release/build_release.sh`)
- [x] Notebook ran top to bottom on a fresh **T4** (Kaggle, G3, 25.8 min; the Colab path is untested)
- [x] Every number in the slides and write-up appears in a notebook output (slides checked 7 Oct 05:45; PSPNet shows as 84.8)
- [x] Weights list starts with Chameleon, with links (write-up §3)
- [x] Data counts and links for both datasets
- [x] "Track A + B" on the title slide and in the write-up
- [x] 1 title + ≤6 slides (PPTX, 7 slides with notes); talk ≤ 5:00 (notes ≈ 4:40; Marlen rehearses)
- [x] Pages URL works in a fresh browser profile (browser pane, desktop + 375 px phone width, no console errors)
- [ ] Zip uploads to KLMS (Marlen: limit 3 GB; the zip is 458 MB)
- [x] `<NAME>` filled in exactly as on KLMS (`20260876_MarlenMelis`, Marlen 7 Oct)

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

## 16. After submission: review and roadmap (14 Oct 2026; read this first now)

**Context (Marlen, 14 Oct):** the KLMS submission is in (8 Oct, tag `submitted-2026-10-08` = commit 0556993). From now
on this is our own project, without the course rules (one notebook, 50-label story, Chameleon-only, deadline). **One
constraint stays until the grade is out: the public demo (MarlenMM/rescue-router-demo, Pages) does not change.**
`src/publish_demo.sh` now refuses to run while `release/DEMO_FROZEN` exists (override `RR_UNFREEZE_DEMO=1` only after
the grade). The private code repo, `docs/` in it and everything else may change; new viewer builds are previewed
locally (`python3 -m http.server 8766 --directory docs`).

### 16.1 What the earlier improvement rounds left open (from §15 and `writeup/batch13_improvements.md`)
- **Batch 13 ideas not run (GPU budget / deadline):** I9 LoRA (coded + tested), I11 gradient accumulation, I14 Dice/Lovász,
  I21 test-time adaptation, I25 two-scale, I26 seeds, I35 saved support encodings. **I24 denser tiles passed the +.010
  bar** (+.0103 val30, 2× prediction) and was left out only for the quota.
- **Bundles never split:** I12+I13 and I15+I18+I19 lost as bundles (−.04, −.10); a part may still help alone (low value).
- **Two recipes in one story:** the 1/5/10/20 curve and RescueNet use the first recipe; only FloodNet 50 has the Batch 13
  model + decode chain.
- **Better pixels did not give better routes:** full-test mIoU +.035, but stranded F1 .759 → .752 and building-detection
  F1 .853 → .806. Batch 13 chose on mIoU only.
- **Calibration:** at τ .5 the final model scores .278 (full test); its tuned thresholds are part of the model.
- **`--freeze_rest` memory saving** (5.7 vs 10.1 GB) seen in a 30-step probe, not in the 1000-step runs (11.4 GB peak).
- Rejected only because of the course rules (now open): more labels, other segmenters, 100-shot and larger points,
  1024-px/native-resolution model, uncertainty-driven labelling, mosaics.

### 16.2 Fixes (things that are wrong or risky now)
- **F1 Label budget wording (honesty).** "50 labelled photos" is the training set; checkpoint, decode τ, the SAM step,
  the router gates and all 36 Batch 13 decisions used 30 more labelled photos (val30). Say "50 training + 30 tuning
  labels (80)" in README/write-up/viewer About, and measure a 50-only variant (τ and checkpoint chosen by
  cross-validation on the 50 supports).
- **F2 val30 over-use.** One 30-photo set chose checkpoint, decode, gates and 36 ideas → winner's curse on every val30
  number (the full-test paired check protects only the final switch). Use a second, untouched FloodNet val split
  (420 unused val photos) to confirm choices; report choices made on val30 as such.
- **F3 No variance from training.** Every point is one seed and one support draw; the bootstrap CIs cover test-photo
  sampling only. → R-A3.
- **F4 Data and results are not backed up.** `results/` (4.0 GB, incl. `b13/` 1.3 GB, checkpoints, caches) is untracked
  and sits inside iCloud Drive (can be evicted by "Optimise Mac storage"); `~/rr_data` (2.4 GB) is on this Mac only;
  Time Machine failed to mount its destination (checked 14 Oct). → private Kaggle/HF dataset with a SHA-256 manifest.
- **F5 Hard-coded paths.** 48 references to `~/rr_data`/`~/.venvs` in `src/` → env vars (`RR_DATA`, `RR_RESULTS`, `RR_PY`)
  with today's values as defaults.
- **F6 No LICENSE** in the code repo (**done 14 Oct: MIT, Marlen's choice**); RescueNet's two licences (figshare CC0 vs GitHub CC BY-NC-ND 4.0) → state that
  we use the figshare (CC0) release everywhere, before anything goes public.
- **F7 Small code issues:** `finetune.py:238` ignores the `load_state_dict` result on resume (assert no unexpected keys);
  ruff (default rules + bugbear) reports 459 findings, almost all style; the B023 hits in `bootstrap_ci.py` /
  `sam_refine.py` / `b13_final_check.py` are false positives (lambdas called inside the loop), 2 unused variables.

### 16.3 Roadmap (P1 = first; GPU-h = Kaggle T4 hours, 30 h/week)
**A. Evaluation and science**
- **A1 (P1, CPU)** F1 + F2: label-budget wording; 50-only variant; second validation split.
- **A2 (P1, CPU)** Routing safety metrics as headline next to F1: false-safe rate (house shown truck-reachable that
  is not), route-length/ETA error per house, an APLS-style path score.
- **A3 (P1, ~10 GPU-h)** Variance: FloodNet 50 with the final recipe × 3 seeds × 3 support draws (I26); report
  mean ± sd next to the bootstrap CI.
- **A4 (P1, ~8 GPU-h)** One recipe everywhere: curve 1/5/10/20/50 (+100, +200) and RescueNet 20/50 with the final recipe
  and decode chain.
- **A5 (P1, ~4 GPU-h)** Same-budget baselines (the missing comparison): a standard segmenter (e.g. SegFormer-B2 or a
  U-Net with an ImageNet backbone) trained on the same 50 photos, and on 100/200/1,445 → where does Chameleon win, and
  where does full supervision overtake it (PSPNet .848 is at 1,445 labels).
**B. Model (target: flooded roads .31 vs PSPNet .82, the dangerous error)**
- **B1 (P2, ~6 GPU-h)** Finish the open ideas: I24 (adopt for offline evaluation; keep 6 tiles for the live app),
  I25 two-scale, I9 LoRA, I14 Dice/Lovász, I11; I21 last.
- **B2 (P2)** Routing-aware model choice: add val routing F1 and false-safe rate to the adoption rule.
- **B3 (P2, CPU)** Per-class temperature scaling on val so τ .5 and `p_wet` mean something; gates then carry over
  between models.
- **B4 (P2, GPU)** More context for wet roads vs water: larger/downscaled whole-photo pass; road-connectivity loss
  (clDice); native-resolution tiles (originals 4000×3000, we use 768×576).
- **B5 (P3, GPU)** Active learning: add 10–20 photos chosen by wet/dry uncertainty vs 10–20 random; gain per label.
**C. Routing (Track B logic)**
- **C1 (P2, CPU)** Dispatch plan per photo: boat launch points, houses grouped per launch point, multi-house boat tours,
  priority order → "N boats, M trips, ≈ T minutes".
- **C2 (P2, CPU)** Probability-aware routing (expected cost / chance constraint) instead of hard gates; route
  confidence = joint passability; compare with the tiers.
- **C3 (P3, CPU)** Road graph from the drivable cells (skeleton) → turn-by-turn text, APLS, faster solves.
- **C4 (P3, CPU)** Per-photo ground scale from detected cars (now one constant per dataset).
- **C5 (P3, research)** Stitch overlapping frames (feature matching + homography) into area maps; FloodNet has no
  geotags.
**D. Product**
- **D1 (P2, ~3 GPU-h)** Small student model (e.g. SegFormer-B0) distilled from the final model on the 1,445 FloodNet
  train photos *without their labels* + the 50 supports; measure vs the teacher; export to ONNX.
- **D2 (P2)** Upload in the static viewer with the student in the browser (ONNX Runtime Web): no GPU server needed.
- **D3 (P2)** Permanent live demo: Hugging Face Space (student on CPU, or ZeroGPU for Chameleon) instead of 3-h Colab
  links.
- **D4 (P2)** Viewer: export a team briefing (PNG/PDF + GeoJSON of routes), batch triage list (photos ranked by stranded
  houses / "check first"), Korean UI, offline PWA (service worker), screen-reader labels; `docs/` 56 MB → AVIF/WebP and
  lazy loading.
- **D5 (P3)** Gradio: concurrency limit, EXIF orientation, size checks, clear error states (or fold it into D2/D3).
**E. Engineering**
- **E1 (P1)** F4 backups; F5 config; `pyproject.toml` + `uv.lock` (CPU env) and a pinned GPU env (Kaggle image digest
  or Dockerfile).
- **E2 (P1)** CI (GitHub Actions): pytest, the JS parity test on a 10-scene fixture, ruff on changed files; pre-commit.
- **E3 (P2)** Tidy: one-off scripts (`b13_*`, `probe`, `postproc_sweep`, `decode_ctl_check`, slide patchers) →
  `experiments/`; one CLI (`rr predict | decode | route | viewer`); CPU smoke tests for finetune/infer with a tiny fake
  model; a test for the `final_decode` chain.
- **E4 (P3)** Repo size: `.git` 132 MB (viewer builds, PPTX, MP4) → large media to GitHub Releases / LFS, or a fresh
  history for the public repo.
**F. Docs and open-sourcing (after the grade)**
- **F-1** LICENSE, CITATION.cff, model card (label budget, US hurricanes ≠ Korea, not for operational use), data card.
- **F-2** Split PLAN.md (197 KB): `ROADMAP.md` (live) + `notes/log_2026-10.md` (history) + methods; README without
  jargon (val30/eval100/Batch 13 defined).
- **F-3** Technical report / blog post with A3–A5 results; make the code repo public; then unfreeze and republish the
  demo (README link to the code).

**Suggested order:** Phase 1 (CPU, ~1–2 days): E1, E2, A1, A2, F7 fixes, B3. Phase 2 (GPU week 1, ~22 GPU-h): A3, A4,
A5. Phase 3 (GPU week 2): B1, B2, B4, D1. Phase 4: C1, C2, D2–D4. Phase 5 (after the grade): F-1…F-3, demo republish.

### 16.4 Progress log
- **14 Oct, Phase 1 (CPU), done:**
  - **F4 backups:** `src/backup.py` packs results/ (4.3 GB), the 3 data tars, the submitted zip and the untracked
    project files (patched starter, course PDFs, slides PPTX, prompts) into 6 parts (6.3 GB) with a committed
    SHA-256 manifest (`release/backup/`, 1,139 files); `release/backup/upload.sh` sends them to the private Kaggle
    dataset `marlenmelis/rr-backup` and the private release `backup-2026-10-14` of MarlenMM/rescue-router. Verified:
    local restore bit-exact; GitHub's asset digests = our sums; one Kaggle part downloaded and hashed. SAM weights
    are public (URL + hash in `EXTERNAL.tsv`). **Refresh after each GPU batch** (`pack`, then `upload.sh`).
  - **F5 config:** `src/rr/paths.py` (`RR_DATA`, `RR_SAM_WEIGHTS`). **E1:** `pyproject.toml` + `uv.lock` (CPU torch
    wheels on Linux; `.venv` is a symlink to `~/.venvs/rr-uv`, outside iCloud), `requirements-gpu.txt` (torch
    2.11.0 image + the pip line the GPU scripts run). **E2:** GitHub Actions (ruff, pytest, JS parity on 200 scenes),
    green. **F7:** unused imports, resume assert in finetune.
  - **Splits:** 6 files listed in `splits/SHA256SUMS` were never committed (Drive only): rebuilt from the local labels by
    `src/splits_local.py regen`, each matching its published hash (incl. **support orders s1/s2 for A3**).
    **F2:** `splits/floodnet_val_confirm100.txt` (100 val photos disjoint from val30, 39 with a flooded class; same
    rule, which reproduces val30 exactly); confirmation only. Needs a GPU pass (~30 min) per model.
  - **A1 wording:** README, viewer About, Gradio headers, app README: 50 training + 30 tuning labels. The 50-only
    variant needs GPU (cross-validated thresholds on the supports) → Phase 2.
  - **A2:** `false_safe_rate` (houses shown truck-reachable that are not) in `router.summarize_routing`, routing.md and
    the CIs: FloodNet 50 full test **.111 [.054, .176]**, eval100 .119 (first model .108), RescueNet 20 .026. Also
    worth stating: 25 % of the final model's routes are infeasible on the labelled map (eval100, λ 5).
  - **B3 calibration** (`src/calibrate.py`, `report/calibration/`, rebuild step `calibration`): per-class Platt on
    val30 halves the ECE (full test .033 → .019; water .079 → .025, its raw p is too low). Plain 0.5 on calibrated
    probabilities: full test .430 (raw .279, tuned .503, both without SAM): about 2/3 of the tuning gain is
    calibration; the rest is IoU-optimal thresholds (dry buildings .17 → .47). Not adopted for the decode (tuned is
    better); candidate for the viewer's water-probability layer and the router's risk term (C2).
  - All published numbers unchanged (`rebuild_all.py --check`).
  - **Phase 1 left:** pre-commit hook (optional), F6 LICENSE (Marlen's choice of licence), write-up source wording
    (the submitted PDF stays as submitted; the tech report F-3 replaces it).

- **14 Oct, pre-Phase-2 improvement cycles (CPU only; Marlen: "iterate until perfect before Phase 2"):**
  - **Cycle 1, Phase 2 readiness.** `generalist.ckpt` is on this Mac (Drive desktop folder → `~/rr_data/weights/`).
    `rr/device.py` makes every Chameleon script run on CUDA (unchanged), Apple MPS or CPU (`RR_DEVICE`): MPS reproduces
    the T4 cache within 2/255 (5-shot), training works, but MPS is 35–100× slower (50-shot inference swaps at 24 GB),
    so the Mac pre-flights GPU jobs and runs small models only. `finetune --support_seed` (draw ≠ seed, `_sup<k>`,
    stored in the checkpoint) and `--eval_shot`. `run_decode.py` (= the submitted chain for any run; reproduces
    τ [.25,.25,.45,.10,.15], val30 .5163, eval100 .5158 / .5185 with SAM), `phase2_report.py` (per run + mean ± sd +
    seed/draw variance split + routing; reproduces F1 .819 / false safe .119), `kaggle/phase2/make_plan.py`
    (sessions `curve` ~4.2 h, `var` ~5.6 h of quota; `kernel` builds the Kaggle folder; `preflight` runs the job
    chains at toy size on the Mac: `var` OK), `experiments/phase2/PROTOCOL.md` (pre-registered). Label-free decode
    (`prior_decode.py`, all 6 shares matched): full test **.468** (val30-tuned .503, plain .278). Same-budget
    baseline `baseline_seg.py` (DeepLabV3-R50, MPS). Routing: `false_safe.py`: 83–87 % of false-safe houses cross
    water/flooded road left unlabelled ("other") as dry yard; `router_sweep.py` (predicted map only): calibrated
    wetness no help, an off-road reach limit trades F1 (.819 → .658) for safety (.119 → .010); val30 cannot pick
    router parameters (52 stranded houses, F1 .980 for every gate ≥ .3) → gate re-chosen on val30 ∪ confirm100
    (pre-registered). Viewer: CSV house list. MODEL_CARD.md, DATA.md, README "Since submission", pre-commit.
  - **Cycle 2.** `baseline_report.py` (full test with CIs, paired Chameleon − baseline, val30-tuned variant, routing)
    + argmax decode; baseline curve 5/10/20/100/all queued on the Mac (`~/rr_local/baseline_queue.sh`); regression
    test of `run_decode`; `decode_tune.py` finds fp16 val30 caches; kernel builder mounts a dataset
    (`marlenmelis/rr-final-ckpt`, staged in `~/rr_push/rr-final-ckpt/`, not uploaded); `var` kernel builds.
  - **Cycles 3–4.** Pre-flights: `curve` 28/29 steps OK (the last, 50-shot prediction of the submitted model, swaps
    on the Mac and was stopped: same code path as passed steps), `--eval_shot` < `--shot` verified (trained on 2,
    predicted with 1). Router sweep: pooled val30 ∪ confirm100 choice coded (ties keep 0.3); tier gate likewise
    (`tiers.py --choose_on`, default byte-identical). Viewer: truck stop points (boat launch / walk-on, grouped
    within 20 m: counter, map, CSV), Korean interface (`&lang=ko`, English text unchanged), "+30 for tuning".
    Kaggle connector unauthenticated: check the quota in the browser before launching.
  - **A5 result (the big one).** DeepLabV3-ResNet50 (COCO weights) fine-tuned on the **same 50 photos** (3,000 steps,
    384 crops, no validation labels; 70 min on the Mac's MPS): full test **.581 [.545, .617]** vs Chameleon .507:
    paired **+.074 [+.045, +.108]**, +.068 [+.039, +.097] without the 14 overlapping test photos; flooded road .451 vs
    .307; eval100 .558. Routing, full test, paired: stranded F1 **.813 vs .752 (+.060 [+.015, +.115])**, false safe
    **.081 vs .111 (−.030 [−.053, −.012])**. CPU inference 0.46 s/photo (42 M parameters). val30 ranks the other way
    (.466 / tuned .470 vs .516); confirm100 .546. Ensemble (`ensemble.py`): val30 picks Chameleon alone (w 1), but on
    the test every w < 1 is better and the plain baseline best. → protocol: model choice on val30 ∪ confirm100;
    baseline curve (5/10/20/100/all) + draws 1–2 queued on the Mac (`~/rr_local/baseline_queue*.sh`). README and
    MODEL_CARD say so.
  - **Phase 2 launch checklist:** commit → preflight `curve` and `var` OK → check Kaggle quota → upload
    `rr-final-ckpt` → `make_plan.py kernel curve` → push → after: fetch, `run_decode.py --sam`, `decode_tune.py`
    (RescueNet), `phase2_report.py --routing`, `router_sweep.py --splits val30,confirm100,eval100`, backup refresh.

## 15. Status / handoff (updated Wed 7 Oct, 18:15 KST)

### Quick summary (read this first; details below)
**Deadline:** submit Thu 8 Oct 12:30 KST, present 16:00. **Done:** all GPU training/inference (7 runs), all Track A analysis, the Track B product (scenes, web viewer, Gradio on CPU), improvement Batches 1–9 + the RescueNet gate, **final notebook + G3 passed on a fresh T4** (`notebook/rescue_router_executed.ipynb`, 25.8 min, live recompute = cache exactly). **Write-up done** (`writeup/20260876_MarlenMelis_writeup.pdf`, 3 pages incl. references). **Slides done** (Claude Slides artifact https://claude.ai/artifact/YYgLzsuLXTeRiQmWUtpyJw, title + 6, talk in the speaker notes; Marlen downloads the PDF for KLMS). **Batch 12 done (05:46, see 0r): submission built and demo public.** KLMS zip `~/rr_push/submission/20260876_MarlenMelis_project.zip` (458 MB: code.ipynb 8.4, writeup.pdf 1.0, slides.pptx 1.6, data.zip 363, app.zip 84 MB; one .ipynb), rebuilt by `bash release/build_release.sh`; **public viewer https://marlenmm.github.io/rescue-router-demo/** (repo MarlenMM/rescue-router-demo). Marlen uploads the zip to KLMS (limit 3 GB). **Since then (0s):** `src/gradio_viewer.py` = the web viewer inside Gradio, so a live upload gets every viewer visual (animated truck → boat → walk); classic view as 2nd tab; in the app zip and the Colab live cell; checked in CPU test mode only. **Not done:** recording (Marlen; the app zip has the GIF), the Colab path of the notebook and the Colab live-link rehearsal (both need GPU OK and the data zip in My Drive). **File names: `20260876_MarlenMelis_*`** (Marlen, 7 Oct; the course example is `20261234_JohnDoe`). **Rebuild everything on CPU:** `python src/rebuild_all.py` (15 min, ends with a headline check).

**Track A (segmentation), headline = FloodNet 50-shot on the full 448-image test:** mIoU .420 [.386, .455] (τ 0.5) / **.472 [.437, .506]** (tuned decode), bootstrap 95 % CI. Per class (tuned): road clear .544, road flooded .291, bldg flooded .577, bldg dry .500, water .448. PSPNet (1,445 labels): 84.8 on the same 5 classes; we beat ENet on 4/5 classes and DeepLabV3+ on flooded buildings; flooded roads are the gap (29 vs 82). Curve (eval100, stratified, τ 0.5 / tuned): 1 → 5 → 10 → 20 → 50 shots = .176/.264, .320/.353, .329/.383, .357/.422, .457/.492; paired bootstrap: 1→5 and 20→50 real, 5→10 noise, 10→20 borderline. RescueNet: 20-shot eval100 .386/.421, 50-shot .406/.402; zero-label transfer of FloodNet 50 reaches about half of 20 RescueNet labels (super-class mean .32 vs .63). Near-duplicate train/test frames: 14/448 FloodNet (3/100 RescueNet), removing them changes mIoU ≤ .004. Failure analysis: shallow/muddy water read as dry land is the dangerous error; GT ambiguity at bridges.

**Track B (routing), router v3** (`src/router.py`, JS port `app/js/router.js`, bit-exact): cell = 4×4 px; truck → boat → foot, never back into a vehicle (v2, walking; RescueNet debris walkable on foot); vehicle widths (truck 1.25 m / boat 0.8 m half-width), parked-car gaps closed, canopy strips for boats, corner rule (v3; ground scale from labelled cars: FloodNet 0.66 m/cell (two flight heights, larger used), RescueNet 0.31 m); off-road truck ×4 on dry land, gated on predicted maps: **FloodNet 0.3, RescueNet 0.2** (each tuned on its own val30); statuses TRUCK / BOAT / FOOT / ISOLATED; the "labelled map" (not "ground truth": the datasets label pixels, not reachability) is the same router on the human labels. **Confidence tiers** (cautious gate FloodNet 0.1, RescueNet 0.15): FloodNet 50 eval100: 301 houses "confident truck" 98.7 % truly truck-reachable, 189 "check first" 74 %; RescueNet's errors are not wetness-related, so its tiers add little (confident 97.4 %, val30 cannot reach 99 %). **Routing, tuned decode, risk-aware:** FloodNet 50 stranded F1 **.835 [.752, .899]** (eval100), **.759 [.638, .856]** (full test); RescueNet 20 **.818 [.524, .966]** (eval100). Statuses never depend on cost values (verified: 14 cost changes, 0 status changes); sensitivity to gate/ring/cell size/widths is small (FloodNet F1 .78–.86). Building detection on predicted maps: FloodNet F1 .85, RescueNet .63.

**Product:** web viewer `app/` → `docs/` (`src/build_viewer.py`; Leaflet 1.9.4 vendored in `app/vendor/leaflet/`, no CDN or storage: works offline and in private windows; **all 200 eval100 photos** (100 FloodNet + 100 RescueNet, 1,032 houses), the 8 showcase scenes first: FloodNet 9107/7320/7301/7423/7611/7241, RescueNet 15833/12195; `docs/` 55.7 MB): prediction / labelled map / swipe, status + tier badges, disagreements, user-chosen entry point, risk toggle, animated truck → boat → walk, ≈ minutes/metres, counter; since Batch 4: browse/sort all test photos, water-probability layer, dashed labelled-map route on disagreement, view in the URL (`#s=rescuenet_12195&h=8&m=gt`), 1024-px FloodNet photos, packed `scene.rrb` per scene; since Batch 5: keyboard house list, colour-blind palette toggle (`&cb=1`). Gradio `src/gradio_app.py` (any eval100 photo from the caches; photo-id box; upload warning; `--live` checked on a Kaggle T4 in Batch 7: live = cache bit for bit, ~15 s per photo, public share link + upload end to end 20.6 s; `--fake_live` = CPU test mode). Live hosting for the presentation: `kaggle/gradio_live/` (share link, prepared, not run; see 0l), demo upload photos in `demo/uploads/`. Best handover demo: RescueNet 12195, labelled map, house 8 (truck → boat → foot). Honest failure demo: FloodNet 7241 (all 6 wrongly truck-reachable houses are "check first").

**Key files:** results `results/kaggle_v2/rr/` (preds incl. merged full-test cache, ckpt, decode), routing `results/routing/` (`routing.md`, `gate/`, `gate_rescuenet/`, `physical/ablation.md`, `tiers/`, `sensitivity/`, `full/`; older routers in `results/routing_v1`, `_v2`), report `results/kaggle_v2/report/` (`ci/ci.md`, `dupes/`, `failures/`, `figures/` incl. `calibration_pwet.png`), scenes `app/scenes/`, contact sheets `results/web/`. Data `~/rr_data`, env `~/.venvs/rr` (Gradio 6.29 inside), Node for the JS tests. Scripts added 6 Oct: `export_web_assets.py`, `build_viewer.py`, `gradio_app.py`, `publish_demo.sh`, `tiers.py`, `bootstrap_ci.py`, `dupes.py`, `sensitivity.py`; `route_eval.py --router '<json>' --point_vehicles`. Added 7 Oct: `tta.py`, `rebuild_all.py`, `gate_sweep.py`, `physical_ablation.py`, `cache_pack.py`, `kaggle/build_rr_code.sh`, `kaggle/gradio_live/`.

**Waiting for Marlen's explicit OK:** the live Gradio run on presentation day (~3 h, now on **Colab**: `kaggle/gradio_live/colab_gradio_live.py`, one cell; Kaggle `run_gradio_live.py` is the fallback) and, before it, a short Colab T4 rehearsal (~40 min: the notebook's Colab path + the live link); both need `20260876_MarlenMelis_data.zip` (and the app zip) dragged into My Drive by Marlen. **Next (7 Oct 17:25):** Batch 13 moved before the deadline: improve the FloodNet 50-shot model (36 ideas I1–I37, batches 13.1–13.6, adoption rule, GPU budget) in the "Batch 13" block below; full switch only if the final model passes (talk material by ~15:00 Thu, the rest by 23:59), else the current zip (16:01) is submitted. Marlen uploads a zip to KLMS in any case. Test-time flips (Batch 7) were tried and **not adopted** (0m). New chat prompt: `NEW_CHAT_PROMPT.md`. **Batch 13 (8 Oct 05:10): DONE through 13.6; the submission now uses the Batch 13 model** (0x, 0y): full test mIoU **.507** [.474, .538] (first model .472; paired Δ +.035 [+.013, +.057]; Chameleon alone .503); routing stranded F1 **.752** full / .819 eval100 (first .759 / .835; differences within noise); confident tier 208 houses 100 % (cautious gate 0.05); building detection F1 .806. Model: published recipe + 1,000-step schedule + `--freeze_rest`, weight average of steps 750+1,000; decode: guided filter → tuned τ → SAM grow (the clean-up was removed: it deleted small houses). Published files: `results/published_7oct.tar` + run `..._pub`; old zip: `~/rr_push/submission_fallback_7oct/`. Every idea and its measured change: `writeup/batch13_improvements.md`. Data bundle 554 MB (`rr-bundle-b13`). **Demo video done (8 Oct 11:30, 0za): `slides/rescue_router_demo.mp4`** (90 s, 1080p30, 16.2 MB; recorder `slides/demo_video/record_demo.py`).


### Improvement batches (agreed with Marlen 6 Oct ~20:00; one prompt each, in this order)
1. ~~Router passability rules~~ (done, router v3): vehicle widths, parked-car gaps, canopy strips, corner rule.
2. ~~Confidence tiers + readable outputs~~ (done, see 0g): p_wet calibration check; cautious gate chosen on val30; tiers per house (confident truck / probably truck / needs boat or foot; eval100 probe with v2: 288 houses 99 % correct, 235 at 70 %, 217 at 90 %); tier table + tests; tiers in viewer and Gradio; costs → rough minutes / metres (scale per dataset, §6.1 v3).
3. ~~Honest evaluation package~~ (done, see 0h; follow-up: RescueNet gate tuned on its own val30, see 0i): full 448-test headline; bootstrap 95 % CIs per curve point + paired 20 vs 50; routing on the full test (FN50); near-duplicate train/test frames on FloodNet (7085↔7081, 8462↔8463, …) and RescueNet, results with/without; statement that stranded metrics do not depend on cost values + status sensitivity to gate, cell size, ring, boat threshold; building detection on predicted maps; caveats (val30 reused for checkpoint/decode/gate; 50-shot validated every 500 steps).
4. ~~Viewer features~~ (done, see 0j): browse all 200 eval100 photos (8 showcase first; 55.7 MB with packing); dashed ghost of the other map's route on disagreement; water-probability layer; scene + house + entry + view in the URL; 1024-px FloodNet photos.
5. ~~Viewer robustness~~ (done, see 0k): Leaflet vendored; binary grid packing (Batch 4); keyboard house list; colour-blind palette toggle; `clip` → `clip-path`; phone + private-window retest.
6. ~~Gradio~~ (done, see 0l): upload warning (+ 4:3 crop instead of stretch); photo-id box + Enter; live hosting plan (Kaggle share-link notebook prepared, recording as backup, fallbacks); demo upload photos.
7. ~~GPU before the deadline~~ (done, see 0m): rebuild `rr-code` (`kaggle/build_rr_code.sh`); live end-to-end check (kaggle/live, ~15 min) + a short `rr-gradio-live` run (share link + upload, Marlen records it); optional test-time flips for FN50 (~1–2 GPU-h eval100, ~4 full test).
8. ~~Engineering~~ (done, see 0n): one-command CPU rebuild (`src/rebuild_all.py`, verified to reproduce every result file); tests for metrics/decode; own README; data zip contents spec (§2.6, ≈ 380 MB measured); PLAN cleanup.
9. ~~Final notebook §7 + G3 on a fresh T4~~ (done, see 0o): live recompute of 10 eval100 photos at 50 and 20 shots = cache exactly; 25.8 min.
10. ~~Write-up~~ (done, see 0p): fix "hours, not weeks" (50 labels ≈ 50 labeller-hours, 3.5 % of 1,445); full-test + CIs lead, far below PSPNet (flooded roads 29 vs 82); labelled-map statuses = our router on human labels + router assumptions; limitations (US vs Korea, single frames, near-duplicates, caveats); one section per graded point; confidence tiers; figures.
11. ~~Slides~~ (done, see 0q): (title + 6), GIF backup, storyboard/script for Marlen's recording.
12. ~~Packaging + release~~ (done, see 0r): data zip (the G3 bundle, 363 MB: KLMS allows 3 GB) + app zip, one .ipynb; demo public on GitHub Pages; checked at desktop + phone width.
13. **Moved before the deadline (7 Oct 17:20, see "Batch 13" below; was: after the deadline):** multi-crop supports (now one 384 crop ≈ 33 % of a support image), 100-shot point, seeds, class-balanced supports, 1024-px model, uncertainty-driven choice of photos to label, georeferenced mosaics.
Ruled out by CPU probes (6 Oct): averaging the 20- and 50-shot models (eval100 mIoU .492 → .408); smoothing the probabilities before decoding (.492 → .491): the errors are semantic (shallow water as land), not noisy edges.

### Batch 13: model improvements before the deadline (agreed with Marlen 7 Oct ~17:20; supersedes item 13 below)
**Scope (Marlen):** try **all** ideas I1–I37 below except I26 (3 training seeds: too slow) and the rejected list; **FloodNet 50-shot only** (the 1/5/10/20 curve and RescueNet stay as they are, labelled "baseline recipe"). Adopt every change that improves **or does not hurt** (a change with no extra running cost is kept when it is not worse; a change that costs prediction time or another model must win clearly). Always apply the efficiency items (I33–I37) while doing the rest. **Full switch by ~11:00 Thu if the final model passes**, else the current submission (`~/rr_push/submission/20260876_MarlenMelis_project.zip`, built 7 Oct 16:01) stays and the experiments are reported as tried. **GPU budget:** Kaggle ~12–13 h this week (check `get_accelerator_quota` first), Colab ~1 h today (free tier). **Colab through the official CLI** (`colab new -s <name> --gpu T4`, `colab exec/upload/download/stop`; logged in, no browser, no clicks; tested 7 Oct 17:38). No Drive mounts (consent per session): upload from the Mac; generalist.ckpt via gdown from the course's public folder (file id `1_UrVMFy18kEcrxwD1b_ohxuTfJIsOWsG`). **Timeline (Marlen, 7 Oct 17:30):** only the PPTX is needed for the 16:00 talk; everything else (notebook, write-up, zips) is submitted by **23:59 Thu**. Model work may run until **15:00 Thu**. **No GPU permission needed any more** (stay within the budget); still ask before anything public and before replacing the submission zip. 13.1 → 13.5 run back to back without waiting for a new prompt.

**Reference numbers (FloodNet 50-shot, current):** val30 tuned .484 (τ .5 .436); eval100 .492 / .457; full test .472 [.437, .506] / .420; dry buildings eval100 .21 (τ .5) / .34 tuned; flooded roads full test .29 (PSPNet .82). Costs on a T4: training ≈ 3.5 s/step (50-shot peak 10.1 GB, chunk 1); prediction ≈ 17 s/photo at 6 tiles (val30 ≈ 9 min, eval100 ≈ 30 min, full test ≈ 2.1 h; support encoding 11.3 GB, inference peak 12.0 GB of 14.6). Best checkpoint so far: step 1000 of 2000. Local: step-500/1000/1500/2000 checkpoints (`results/kaggle_v2/rr/ckpt/*_step*.ckpt`) and their val30 caches; all 1,445 train photos in `~/rr_data`.

**Adoption rule (fixed before any result):** every variant is scored on val30 with its own decode re-tuned on val30 (`decode_tune.py` rule) against .484. (a) **No running cost** (training-side change, decode/post-processing on CPU): kept if val30 tuned Δ ≥ 0 and val30 drivable precision not lower. (b) **Running cost** (slower prediction, extra model, extra prediction passes): kept only if Δ ≥ +.01. Winners are combined; the combination must again beat .484 on val30. **Headline switch only if** the final model's paired full-test difference vs .472 (bootstrap, `bootstrap_ci.py`) is positive with the 95 % CI lower bound > 0. Every variant tried is logged with its val30 numbers (table for the write-up: "tried, kept, dropped"). Never choose on eval100 or the full test.

**The ideas (I-numbers as discussed with Marlen; "bundle" = tested together in one run to fit the time):**
1. Support set: **I1** support-view ensemble (2–3 different 384 crops per labelled photo, one view per pass, probabilities averaged; memory stays, prediction ×views); **I2** support crop weights favouring dry buildings (`rr/data.py` CROP_WEIGHTS, now 1,3,3,1,1); **I3** self-support (add each test photo's own most confident crop as an extra support; memory risk at 51); **I4** class-balanced choice of the 50 photos (new support list; needs a training run).
2. What is trained (starter flags, `starter_utils.py:296`): **I5** `label_decoder_tuning`; **I6** `relpos_tuning`; **I7** `head_tuning`; **I8** `separate_alpha` (per-channel matching temperature); **I9** LoRA in attention (new code).
3. How it is trained: **I10** authors' episodes: train with shot 20 drawn from the 50, `eval_shot` 50 (ISIC config); **I11** global batch 5–8 (grad accumulation; now 2); **I12** rare-class loss weights (flooded road, dry building; inverse-frequency, as in FloodNet work); **I13** training crops biased to rare classes; **I14** Dice/Lovász term added to BCE; **I15** rotation + blur augmentation (ISIC config); **I16** lr 0.5× / 2× (Cellpose uses 3e-3) on a 1000-step cosine schedule; **I17** EMA of the tuned weights; **I18** matching-attention dropout 0.3 / 0.7 (now 0.5); **I19** rare-class copy-paste augmentation.
4. Unlabelled data: **I20** pseudo-labels on ~200 unused train photos (confident pixels only), retrain; **I21** per-photo test-time adaptation (entropy/confident-pixel bias update; unstable, slow).
5. Prediction: **I22** average the step-1000/1500/2000 checkpoints' predictions (val30 caches exist: CPU); **I23** average the checkpoints' weights (CPU, then one val30 pass); **I24** denser tiles, stride 128 (12 tiles, 2×); **I25** two-scale (+1.25× = 18 tiles, 3×; both I24+I25 = 36 tiles, 6×: never stack with I1 if the final run does not fit). (**I26** dropped.)
6. Probabilities → classes (CPU): **I27** per-class bias offsets; **I28** two-step decode (wet vs dry, then class); **I29** clean-up (remove specks, close road/water gaps); **I30** edge-aware refinement (dense CRF / guided filter); **I31** SAM/SAM2 refinement of building outlines (second pretrained model, list it in the write-up with link, report Chameleon-only and +SAM separately; Chameleon still decides every class); **I32** calibration of p_wet for the router's gates (re-tune gates on val30 if the model changes).
7. Efficiency and cost (apply throughout): **I33** one process per GPU runs all inference variants (the 3 GB generalist loads once); **I34** 1000-step schedule (best step was 1000); **I35** save encoded supports to disk for the app (start ~30 s faster); **I36** `torch.compile` / channels-last for prediction (keep only if outputs agree); **I37** Colab detection fix in the notebook (`ON_KAGGLE` true on Colab because of an empty `/kaggle/input`): `ON_KAGGLE = os.path.isdir('/kaggle/input') and not IN_COLAB`.
Rejected (unchanged): more labels (val30 / RescueNet as training data), non-Chameleon segmenters, 100-shot, 1024-px model, uncertainty-driven labelling, mosaics, extra-photo seeds.

**Unpacking (revised 17:30, more time):** the binding limit is now Kaggle GPU hours, not the clock: fully unpacked the training ideas are ~18 runs × ~1.3 GPU-h ≈ 23 GPU-h (+ ~5 h final) > budget. So: CPU ideas and prediction-only ideas each alone (cheap; one model load per GPU, I33); **I17 EMA scored inside every training run** (free single test); **single runs** for I5, I6, I7, I8, I9, I10, I10+I11 (authors' full recipe), I4, I14, I16 (2× lr); **small bundles only for same-mechanism or tiny items**: I12+I13 (rare classes), I15+I18+I19 (regularisers); I20 after round 1 on the best recipe. = 12 training runs in round 1 (3 waves on 4 T4s). A winning bundle is split into singles in round 2 if budget remains. Measure how Kaggle counts a 2×T4 session first; if it counts per GPU, cut the single runs in the order I9, I14, I16, I4 (lowest expected value first). The 4-lane bundle plan below is superseded where it differs.

**Batches (13.1–13.5 back to back; "∥" = runs in parallel):**
- ~~**13.1 (CPU, ~2 h, Wed evening): infrastructure + free experiments.**~~ (done 18:15, see 0u) Check Kaggle quota. Add options without changing defaults (defaults must reproduce today's caches bit for bit; tests): `finetune.py` flags for I5–I8, I10–I19 (incl. I34 steps, I17 EMA, I11 accumulation, I12 weights, I13 sampling, I14 loss, I15 aug, I18 dropout); `predict.py`/`rr/infer.py`/`rr/data.py` for I1 views, I2 weights, I23 weight-averaged ckpt, I24 stride, I25 scales, I36 compile, and **I33 `src/infer_sweep.py`** (one model load, loops variants, writes one cache per variant). CPU now: I22, I27, I28, I29, I30, I32 on the existing val30 caches → log. Build the Kaggle round-1 kernels (rr-code rebuilt by `kaggle/build_rr_code.sh`). I37 in `build_notebook.py`. Commit.
- ~~**13.2 (GPU round 1, ~2 h wall, Wed ~19:30): 5 lanes ∥.**~~ (done 22:30, see 0v) Lane A (Colab 1 h or Kaggle): I33 sweep on val30 with the current checkpoint: I1 (views 1, 2), I2, I23, I24, I25, I36 (+ I3 if ready). Kaggle 2 sessions × 2 T4 = lanes B–E, each a 1000-step 50-shot run + val30 per snapshot: **B** authors' recipe I10+I11+I15+I17 (+I34); **C** more trained parts I5+I6+I8 (+I7 via I5); **D** rare classes I12+I13; **E** I14+I18+I19 (or I4 class-balanced supports if its list is ready).
- ~~**13.3 (CPU, ~30 min): decide.**~~ (done 23:15, see 0v) Score every lane on val30 (own decode), apply the rule, log, design the combined recipe.
- ~~**13.4 (GPU round 2, ~2–3 h, Wed night): combine + the slow ideas.**~~ (done 01:00, see 0w) Combined recipe run (winners of B–E; I4 if it won or was untested) ∥ I20 pseudo-labels (predict ~200 unlabelled photos with the best model, then a run on supports + pseudo-labels) ∥ I9 LoRA run (if coded) ∥ I21 and I31 tests on val30. Decide again on val30.
- ~~**13.5 (GPU round 3, overnight, ~2.5–6 h): final predictions**~~ (done 02:55, PASSED, see 0x) with the winning setup: val30, eval100, full test (two halves ∥), FloodNet-50 → RescueNet eval100 transfer cache. Then the paired full-test check (headline rule).
- ~~**13.6 (Thu 07:00–15:00 for the talk, the rest after 16:30 until 23:59): full switch or keep.**~~ (done 8 Oct 05:40, see 0y) By ~14:30: results, viewer, live app, new slide numbers (Marlen re-exports the PPTX by 15:00). After the talk: notebook re-run, write-up, zips, submit by 23:59. Details: If it passed: `rebuild_all.py` (all reports, gates re-checked I32, viewer scenes, tests), Gradio `LiveModel` with the same prediction setup + I35, notebook regenerated (I37 included) and re-run on a T4 (G3; time × the prediction-cost factor), data bundle rebuilt (`make_bundle.py` at the new commit; Drive copy for Colab), app zip, write-up and slides numbers (Marlen re-exports the PPTX), project zip, Pages republish (needs OK). If it did not pass: keep the current zip; add the "tried" table to the write-up/notebook only if time allows.
- **After the deadline:** whatever did not fit (single-idea ablations of the bundles, I26 seeds, the rejected list).
**Honest risk:** all 36 ideas in one night is ambitious; bundles trade attribution for time (a winning bundle is kept whole). Expected total gain +.02–.05 (I5/I6/I10/I11 the most likely), not guaranteed.

### Batch 12 hand-off (written 7 Oct 05:28 KST for a new chat)
*Done 7 Oct 05:46 (see 0r). Kept for the record; the Thursday steps at the end still apply.*
**Goal:** the KLMS submission `20260876_MarlenMelis_project.zip` by **Thu 8 Oct 12:30 KST** (hard deadline 23:59) and the public demo. Course rules (README_starter.md, local only): ONE `.ipynb` in the whole zip, saved with all outputs; every result from our own code; every pretrained weight listed; data disclosed; Track B app source + README as `_app.zip`.

**Ask Marlen first (one AskUserQuestion):**
1. Data zip size: the bundle is **363 MB** (> our 300 MB target; KLMS limit unknown). Options from §2.6: (a) ship 363 MB if KLMS allows; (b) under 300 MB: drop the zero-label-transfer cache (31 MB), the 1/5/10-shot val30 caches (21 MB) and the RescueNet 50-shot caches (19 MB), stating they are reproducible from the checkpoints (then the notebook's curve cell must skip missing caches, and the headline check stays); (c) caches as a public Kaggle dataset + link (publishing: needs OK).
2. **OK to create and publish** the public repo `MarlenMM/rescue-router-demo` (GitHub Pages from the built viewer): `bash src/publish_demo.sh --create`.
3. The slides PDF: Marlen downloads it from the deck https://claude.ai/artifact/YYgLzsuLXTeRiQmWUtpyJw → `slides/20260876_MarlenMelis_slides.pdf` (Claude cannot export it). The recording `recording.mp4` (storyboard `slides/recording_script.md`) is Marlen's; it goes into the app zip if ready, else the GIF `slides/demo_route.gif`.

**Then build (all CPU; commit first, the bundle packs git HEAD):**
- `python src/make_bundle.py --out ~/rr_bundle --zip <out>/20260876_MarlenMelis_data.zip` (3 min; the notebook finds `20260876_*_data.zip`). Check: no `.ipynb` inside, size.
- `20260876_MarlenMelis_code.ipynb` = `notebook/rescue_router_executed.ipynb` (executed on a fresh T4, 0 errors; generator `notebook/build_notebook.py`; do NOT regenerate without re-running G3).
- `20260876_MarlenMelis_writeup.pdf` = `writeup/20260876_MarlenMelis_writeup.pdf` (rebuild: `writeup/build.sh`).
- `20260876_MarlenMelis_app.zip`: `docs/` (the built viewer, runs from any static server), `src/gradio_app.py` + the `src/` modules it imports (router, export_web_assets, viz, rr/) or simply `src/`, `kaggle/gradio_live/`, `demo/uploads/`, a short `README_app.md` (Pages URL, `python3 -m http.server 8766 --directory docs`, the Gradio commands, the share-link notebook), `recording.mp4` or `demo_route.gif`. No `.ipynb`.
- `20260876_MarlenMelis_project.zip` = code.ipynb + writeup.pdf + slides.pdf + data.zip + app.zip. Verify: one `.ipynb` (`unzip -l` recursively into the inner zips), names exactly as the course example, sizes.
- Publish (after OK): `bash src/publish_demo.sh --create` → https://marlenmm.github.io/rescue-router-demo/ ; test in the browser pane at desktop + phone width and in a fresh profile (incognito); put the URL into `README_app.md`, `README.md`, the slides' notes if wanted. §12 checklist.
- Untested: the Colab path of the notebook (Drive popup needs Marlen). Worth doing now that Colab is the default for short GPU work: it is the TA's environment (a G3 re-run ≈ 30 min on a Colab T4).
- **GPU platform rule (Marlen, 7 Oct 05:34): Google Colab for GPU jobs under ~5 h, Kaggle only for day-long or > 5 h continuous runs** (Kaggle's weekly quota, ≈ 15.5 of 30 h used, is kept for long jobs). Colab is driven through the colab-mcp tools (`open_colab_browser_connection`); Marlen switches the runtime to T4 and approves the Drive popup when asked (memory: colab-drive-operations); free tier = limited daily GPU, idle sessions drop.
- **Thursday ~15:15 KST:** live Gradio share link for the talk (~3 h → **Colab** under the rule above; needs OK): in a Colab T4 runtime run the steps of `kaggle/gradio_live/run_gradio_live.py` adapted to Colab paths (bundle zip from Drive or `/content`, `generalist.ckpt` from the course Drive folder, `python src/gradio_app.py --live --share ...`), keep the session alive, send Marlen the `gradio.live` link. Prepare and test this cell before Thursday (a short run). Kaggle `rr-gradio-live` (`cd kaggle/gradio_live && ~/.venvs/kaggle/bin/kaggle kernels push -p .`) stays the fallback.

### Log (chronological; later entries supersede earlier ones)

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
| floodnet_1shot | τ=0.5 | risk-aware λ=5 | 0.864 | 0.778 | 0.819 | 0.758 | 0.139 | 0.157 | 1.087 | 0.414 |
| floodnet_1shot | τ=0.5 | plain λ=0 | 0.864 | 0.778 | 0.819 | 0.758 | 0.096 | 0.151 | 1.080 | 0.396 |
| floodnet_1shot | tau | risk-aware λ=5 | 0.509 | 0.996 | 0.674 | 0.484 | 0.009 | 0.003 | 1.000 | 0.249 |
| floodnet_1shot | tau | plain λ=0 | 0.509 | 0.996 | 0.674 | 0.484 | 0.009 | 0.003 | 1.000 | 0.253 |
| floodnet_5shot | τ=0.5 | risk-aware λ=5 | 0.905 | 0.670 | 0.770 | 0.766 | 0.092 | 0.169 | 1.024 | 0.411 |
| floodnet_5shot | τ=0.5 | plain λ=0 | 0.905 | 0.670 | 0.770 | 0.766 | 0.131 | 0.202 | 1.000 | 0.463 |
| floodnet_5shot | tau_ctx | risk-aware λ=5 | 0.893 | 0.711 | 0.792 | 0.778 | 0.080 | 0.147 | 1.021 | 0.339 |
| floodnet_5shot | tau_ctx | plain λ=0 | 0.893 | 0.711 | 0.792 | 0.778 | 0.101 | 0.176 | 1.003 | 0.376 |
| floodnet_10shot | τ=0.5 | risk-aware λ=5 | 0.871 | 0.852 | 0.861 | 0.777 | 0.055 | 0.083 | 1.021 | 0.329 |
| floodnet_10shot | τ=0.5 | plain λ=0 | 0.871 | 0.852 | 0.861 | 0.777 | 0.085 | 0.124 | 1.004 | 0.395 |
| floodnet_10shot | tau_ctx | risk-aware λ=5 | 0.859 | 0.856 | 0.857 | 0.762 | 0.043 | 0.084 | 1.026 | 0.268 |
| floodnet_10shot | tau_ctx | plain λ=0 | 0.859 | 0.856 | 0.857 | 0.762 | 0.060 | 0.102 | 1.011 | 0.316 |
| floodnet_20shot | τ=0.5 | risk-aware λ=5 | 0.863 | 0.656 | 0.745 | 0.750 | 0.075 | 0.170 | 1.038 | 0.346 |
| floodnet_20shot | τ=0.5 | plain λ=0 | 0.863 | 0.656 | 0.745 | 0.750 | 0.104 | 0.196 | 1.016 | 0.374 |
| floodnet_20shot | tau_ctx | risk-aware λ=5 | 0.865 | 0.667 | 0.753 | 0.751 | 0.076 | 0.174 | 1.035 | 0.335 |
| floodnet_20shot | tau_ctx | plain λ=0 | 0.865 | 0.667 | 0.753 | 0.751 | 0.104 | 0.200 | 1.014 | 0.366 |
| floodnet_50shot | τ=0.5 | risk-aware λ=5 | 0.868 | 0.778 | 0.820 | 0.755 | 0.054 | 0.119 | 1.042 | 0.359 |
| floodnet_50shot | τ=0.5 | plain λ=0 | 0.868 | 0.778 | 0.820 | 0.755 | 0.067 | 0.140 | 1.008 | 0.408 |
| floodnet_50shot | tau | risk-aware λ=5 | 0.868 | 0.804 | 0.835 | 0.758 | 0.052 | 0.121 | 1.038 | 0.348 |
| floodnet_50shot | tau | plain λ=0 | 0.868 | 0.804 | 0.835 | 0.758 | 0.070 | 0.164 | 1.010 | 0.386 |
| rescuenet_20shot | τ=0.5 | risk-aware λ=5 | 1.000 | 0.720 | 0.837 | 0.973 | 0.060 | 0.094 | 1.061 | 0.219 |
| rescuenet_20shot | τ=0.5 | plain λ=0 | 1.000 | 0.720 | 0.837 | 0.973 | 0.071 | 0.087 | 1.039 | 0.229 |
| rescuenet_20shot | tau_bldg | risk-aware λ=5 | 0.947 | 0.720 | 0.818 | 0.966 | 0.037 | 0.045 | 1.051 | 0.191 |
| rescuenet_20shot | tau_bldg | plain λ=0 | 0.947 | 0.720 | 0.818 | 0.966 | 0.047 | 0.059 | 1.037 | 0.184 |
| rescuenet_50shot | τ=0.5 | risk-aware λ=5 | 0.684 | 0.520 | 0.591 | 0.908 | 0.058 | 0.075 | 1.087 | 0.243 |
| rescuenet_50shot | τ=0.5 | plain λ=0 | 0.684 | 0.520 | 0.591 | 0.908 | 0.060 | 0.082 | 1.037 | 0.247 |
| rescuenet_50shot | tau_bldg | risk-aware λ=5 | 0.609 | 0.560 | 0.583 | 0.911 | 0.061 | 0.087 | 1.101 | 0.229 |
| rescuenet_50shot | tau_bldg | plain λ=0 | 0.609 | 0.560 | 0.583 | 0.911 | 0.062 | 0.100 | 1.045 | 0.233 |

*Router v3 (walking + physical rules; RescueNet gate 0.2 since 0i), regenerated 6 Oct 20:19.*
*Earlier: router v2 (walking, point vehicles) in `results/routing_v2/routing.md`; the v1 truck/boat table is in `results/routing_v1/routing.md` (same stranded P/R/F1, agreement .78 → .77 and infeasible +.04 for FloodNet 50 tuned).*

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
| 2 | **shallow / muddy / sandy water decoded as dry land or clear road** | 7284, 7355, 7370, 7486, 7308 | **the dangerous one.** Without the gate: 25 stranded houses shown as truck-reachable in these 5 images (v1). With the off-road gate: 6 (7355: 3, 7308: 3); router v3: 7 (7355: 3, 7308: 3, 7486: 1) |
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
| val30 | none | 1.000 | 0.385 | 0.556 | 0.783 | 0.110 | 0.237 | 1.026 | 0.414 | 52 | 20 |
| val30 | 0.5 | 1.000 | 0.404 | 0.575 | 0.788 | 0.110 | 0.232 | 1.026 | 0.409 | 52 | 21 |
| val30 | 0.3 | 1.000 | 0.962 | 0.980 | 0.899 | 0.055 | 0.078 | 1.031 | 0.343 | 52 | 50 |
| val30 | 0.2 | 0.769 | 0.962 | 0.855 | 0.798 | 0.054 | 0.062 | 1.120 | 0.333 | 52 | 65 |
| val30 | 0.15 | 0.676 | 0.962 | 0.794 | 0.732 | 0.036 | 0.059 | 1.261 | 0.338 | 52 | 74 |
| val30 | 0.1 | 0.543 | 0.981 | 0.699 | 0.556 | 0.018 | 0.008 | 1.511 | 0.455 | 52 | 94 |
| val30 | 0.05 | 0.490 | 0.981 | 0.654 | 0.490 | 0.018 | 0.010 | 1.519 | 0.551 | 52 | 104 |
| val30 | 0 | 0.371 | 1.000 | 0.542 | 0.313 | 0.000 | 0.000 | 2.286 | 0.641 | 52 | 140 |
| eval100 | none | 0.994 | 0.585 | 0.737 | 0.769 | 0.091 | 0.206 | 1.020 | 0.389 | 270 | 159 |
| eval100 | 0.5 | 0.994 | 0.637 | 0.777 | 0.786 | 0.088 | 0.185 | 1.020 | 0.378 | 270 | 173 |
| eval100 | 0.3 | 0.868 | 0.804 | 0.835 | 0.758 | 0.052 | 0.121 | 1.038 | 0.348 | 270 | 250 |
| eval100 | 0.2 | 0.769 | 0.900 | 0.829 | 0.662 | 0.029 | 0.074 | 1.069 | 0.333 | 270 | 316 |
| eval100 | 0.15 | 0.673 | 0.952 | 0.788 | 0.573 | 0.021 | 0.034 | 1.092 | 0.381 | 270 | 382 |
| eval100 | 0.1 | 0.606 | 0.985 | 0.750 | 0.466 | 0.005 | 0.007 | 1.146 | 0.471 | 270 | 439 |
| eval100 | 0.05 | 0.543 | 1.000 | 0.704 | 0.378 | 0.000 | 0.000 | 1.139 | 0.558 | 270 | 497 |
| eval100 | 0 | 0.485 | 1.000 | 0.653 | 0.296 | 0.000 | 0.000 | 1.344 | 0.608 | 270 | 557 |

*(router v3: walking + physical rules; v2 sweep in `results/routing_v2/gate/gate.md`)*

- **eval100 at g = .3 (v3):** stranded F1 .737 → .835, unsafe routes .206 → .121, RSV .091 → .052, at precision .994 → .868. Missed stranded houses fall from 112 to 53 (of 270). (v2: F1 .667 → .802, missed 134 → 74; v1 likewise.)
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
   0. **Progress (6 Oct 18:15).** Step 1 done: Marlen picked the 8 scenes from the contact sheets (`results/web/contact_sheet_{proposed,alternates}.jpg`): FloodNet 50-shot 9107 (dry, all truck), 7320 (truck + isolated), 7301 (truck + boat + isolated), 7423 (boat rescues), 7611 (mostly isolated), 7241 (honest failure: 14 boat houses predicted truck-reachable); RescueNet **20-shot** 15833 (clear road, intact/damaged), 12195 (honest failure: canal houses, GT 2 boat + 4 isolated, predicted all truck). `src/export_web_assets.py` writes `app/scenes/<ds>_<id>/` (photo 768×576, pred/gt RGBA, grid.json with float32 risk/p_boat, buildings.json with Python statuses/costs at λ 5 and 0, meta.json) and `--sheet` contact sheets. JS router `app/js/router.js` + parity test `node --test app/test/router.test.mjs`: all statuses and all 364 route costs match Python bit-for-bit (float32 truck cost as in NumPy); dropping the gate changes 15 statuses, so the test catches it.
   0b. **Step 2 done (6 Oct 18:22): web viewer** `app/` (index.html, css/style.css, js/app.js, js/router.js; Leaflet 1.9.4 from unpkg with SRI) → `docs/` via `python src/build_viewer.py` (copy without tests, + `.nojekyll`; 7.4 MB). Local: `.claude/launch.json` configs `viewer` (app/, :8765) and `viewer-docs` (docs/, :8766). Features: scene picker (failures tagged), Prediction / Ground truth / Swipe (panes clipped at the divider, statuses per side), flood-map toggle + opacity, house status badges, disagreements toggle (dashed magenta), entry points, risk-aware toggle (λ 5 ↔ 0, rerouted live), click/tap a house → route card + animated truck then boat, "Unreachable by road N / M" counter (both maps in swipe), legend, attribution. Phone: bottom sheet + dock (route card + counter), checked at 375×812. **Viewer rule:** a TRUCK house is drawn with its cheapest road-only route (`route(ring, true)`); router.py's `route()` may return a cheaper truck + boat route to a truck-reachable house, which read as "truck can reach it" but showed a boat. Parity test also covers the road-only routes. Truck → boat handovers with a real truck leg: GT 7301 (4 houses), pred 7423 (3-cell truck leg), 7241 (GT, pred), 12195; on most predicted maps boats launch at the image edge.
   0c. **Step 3 (CPU part) done (6 Oct 18:27): Gradio app** `src/gradio_app.py` (Gradio 6.29 in `~/.venvs/rr`; launch config `gradio`, :7860). Any eval100 photo from the caches (8 showcase scenes first), Prediction / Ground truth, risk-aware toggle, boat cost slider (1–6 per cell), click a house → route drawn (truck white, boat cyan) + summary with GT agreement, "unreachable by road" counts. Shares `build_scene` / `make_grids` with the export (refactor verified: exported assets byte-identical). Upload path: `--live` loads the checkpoint once (`LiveModel`, same recipe as predict.py) and routes an uploaded photo with base + buildings taken from the prediction (no GT). Headless smoke test: 484 house clicks over all 8 scenes × both maps × two cost settings; upload path checked on CPU with a cached photo standing in for the model. **Not yet run on a GPU:** `kaggle/live/live_check.py` (notebook `marlenmelis/rr-gpu-live`, 1 T4, ~15 min) loads LiveModel, compares 3 live predictions with the eval100 cache, runs a 4000×3000 upload and saves the rendered views. Needs Marlen's OK first, plus an rr-code rebuild that also ships `results_decode/floodnet_50shot_s0_stratified.json` (results/ is not in git).
   0d. **Step 4 prepared, not run:** `bash src/publish_demo.sh --create` (runs the JS parity test, builds docs/, creates the public repo `MarlenMM/rescue-router-demo`, pushes docs/ + a README with attribution, enables Pages from main /). Later updates: without `--create`. The repo does not exist yet; waiting for Marlen's explicit OK.
   0e. **Router v2 + viewer changes (6 Oct 19:19, Marlen's three requests):** (1) walking mode (see §6.1 "Router v2"), Python + JS (parity: all 484 routes of the 8 scenes bit-exact; JS statuses/costs = Python), Gradio; all routing reports, the gate sweep, failures and figures regenerated from the caches (v1 results kept in `results/routing_v1/`). (2) **Entry point chosen by the user** in the viewer ("Choose on map" / "Auto") and Gradio ("Clicking the photo will… set the entry point"): a 3×3-cell staging area, truck on dry ground, boat on water (decided on the labelled map, as the auto entry); evaluation keeps the automatic rule (reproducible, not hand-picked). (3) "Ground truth" → **"Labelled map"** in the viewer and Gradio, with the note that it is the router on the human labels. Captions of 7320 / 7301 / 7611 updated (no more "isolated"). Best handover demo: RescueNet 12195 labelled map, house 8: truck → boat across the canal → on foot.
   0f. **Batch 1 done (6 Oct 20:19): router physical rules** (see §6.1 "Router v3"): vehicle widths, parked-car gap closing, canopy strips for boats, corner rule; ground scale measured from labelled cars; all routing reports, the gate sweep (0.3 still best on val30), failures, figures and the 8 scenes regenerated; viewer About text mentions widths; viewer bug fixed (the grid `flags` were dropped on load). Scene changes: 7241 labelled map now 0 truck / 6 boat / 3 foot / 5 no route (boat too wide between houses), prediction 6 truck / 8 foot; 12195 labelled map 8 truck / 4 foot (no boat house any more), the truck → boat → foot handover is still house 8 (46 + 81 + 24 cells). Next: Batch 4 of the improvement list below.
   0g. **Batch 2 done (6 Oct 20:30): confidence tiers + readable outputs.** `src/tiers.py` → `results/routing/tiers/{tiers.md,tiers.json}`, `figures/calibration_pwet.png`. The predicted map is routed twice: tuned gate 0.3 and a **cautious gate 0.1** (`route_eval.PRED_CAUTIOUS_MAXWET`; chosen on FloodNet 50 val30 as the largest candidate whose confident-truck houses are ≥ 99 % truck-reachable; the val30 candidates 0.05–0.25 differ by about one house (98.4–99.0 %), so the choice is noisy). Tiers (`router.tier`): CONFIDENT_TRUCK (truck even on the cautious map) / PROBABLE_TRUCK (only with the tuned gate: check first) / BOAT / FOOT / ISOLATED. Router v3, labelled map as reference:

     | run | split | confident truck: houses, truly truck-reachable | probably truck (check first) | not truck (boat/foot/none): truly not truck |
     |---|---|---|---|---|
     | FloodNet 50 | val30 | 104, 99.0 % | 44, 97.7 % | 50, 100 % |
     | FloodNet 50 | **eval100** | **301, 98.7 %** | **189, 74.1 %** | 250, 87.2 % |
     | RescueNet 20 | val30 | 72, 90.3 % | 7, 85.7 % | 5, 80 % |
     | RescueNet 20 | eval100 | 270, 97.4 % | 12, 33.3 % | 10, 100 % |

     - **The tiers catch the dangerous errors in the showcase failures:** 7241, all 6 houses wrongly predicted truck-reachable are "check first" (0 confident); 12195, the 4 canal houses are "check first".
     - Calibration of p_wet (cell mean of p1 + p4 vs labelled wet share): ECE FloodNet .069 / .074 (val30 / eval100; cells decoded 'other', where the gate acts: .043 / .046), RescueNet .016–.023. FloodNet p_wet is slightly too high below 0.3 (at 0.25 the labelled wet share is ~0.12) and too low above 0.5; RescueNet's 'other' cells are over-predicted wet above 0.3 (blocked-road channel). No recalibration applied (the gates are tuned on the raw values).
     - Readable outputs: `router.eta` = metres per mode + minutes for a route (display only; assumed flood-time speeds truck 20 km/h on road, 5 off-road (= the ×4 cost), boat 6 (3 under canopy), on foot 3; +5 min to launch a boat, +1 min to get out). Distances use a per-scene scale from the photo's labelled cars (`meta.m_per_cell`; dataset value if < 2 cars). Viewer and Gradio: yellow "check first" badges on the prediction, counter line "N more look truck-reachable: check first", route card with tier sentence (dataset's eval100 shares), "≈ X min · truck … → boat … → on foot …", and confident houses drawn with their cautious-map route (avoids land that may be wet). JS parity extended to the cautious map, tiers and ETA (bit-exact). Tests: 26 Python, 8 JS scene tests.
   0h. **Batch 3 done (6 Oct 20:47): honest evaluation package.** Scripts: `src/bootstrap_ci.py` → `results/kaggle_v2/report/ci/ci.{md,json}`; `src/dupes.py` → `report/dupes/dupes.json`; `src/sensitivity.py` → `results/routing/sensitivity/sensitivity.{md,json}`; full-test routing → `results/routing/full/` (the two full-test caches merged into `preds/floodnet_50shot_s0_stratified_floodnet_test_full_fp16.npz`).
     - **Headline = the full 448-image test (FloodNet 50-shot):** mIoU **.420 [.386, .455]** (τ 0.5) / **.472 [.437, .506]** (tuned); drivable precision .857 [.819, .890] / .830 [.792, .863]. Per class, tuned: road clear .544 [.495, .594], road flooded .291 [.224, .362], bldg flooded .577 [.487, .654], bldg dry .500 [.437, .561], water .448 [.392, .503]. eval100 (stratified 50 % flooded) is reported as the curve split only: .457 [.406, .510] / .492 [.435, .549].
     - **Bootstrap 95 % CIs over images (B = 2000)** for every curve point (`ci.md`). **Paired differences (eval100):** 1→5 shots +.14 [+.10, +.18] (τ 0.5); **5→10 not distinguishable from 0** (+.008 [−.030, +.049]); 10→20 borderline (+.028 [−.003, +.060]; tuned +.040 [+.007, +.071]); **20→50 clearly positive** (+.100 [+.059, +.142]; tuned +.069 [+.024, +.114]). So: "flat between 5 and 20, real jump at 50" holds with uncertainty.
     - **Near-duplicates (`dupes.py`):** ORB + RANSAC homography on the 3 most similar supports per test photo (≥ 25 inliers and ≥ 20 % cover; 5 pairs at 10–24 inliers checked by eye and recorded in `MANUAL`). Test photos overlapping a support photo: **FloodNet 14 / 448 with the 50-shot supports** (9 with the 20-shot), all neighbouring frames or the same spot from another flight (e.g. 7085/7081, 8462/8463, 8942/8206); **RescueNet 3 / 100** (20-shot). A lower bound (only the top 3 candidates verified). **Effect: none that matters** — mIoU without them changes by ≤ .004 (full test tuned .472 → .473), stranded F1 .759 → .753 (full test), .835 → .832 (eval100).
     - **Routing on the full test (FloodNet 50, router v3, tuned, λ 5):** 448 photos, 1,232 houses (labelled map 893 truck / 182 boat / 150 foot / 7 none); stranded P .847, R .687, **F1 .759 [.638, .856]**, agreement .803, RSV .062, unsafe .122, cost ratio 1.069, infeasible .331 (τ 0.5: F1 .748). eval100: F1 .835 [.752, .899]. RescueNet 20 eval100: .571 [.222, .857] (only 25 cut-off houses: very uncertain).
     - **Cost independence verified:** changing any of 14 cost values (λ 0/20, boat 1.5/6, launch 0/200, foot 4/16, alight 0/50, off-road ×2/×8, canopy ×1/×4) changes **0** statuses on both maps, both datasets. Statuses (and so stranded P/R/F1) depend only on passability; costs only choose among routes (RSV, cost ratio).
     - **Sensitivity (FloodNet 50 eval100, stranded F1, default .835):** gate 0.2 / 0.5: .829 / .777; ring 1 / 3 cells: .834 / .843; boat threshold 0.3 / 0.7: .835 / .835; cell 8 px (same ring in metres): .831; vehicle half-widths ×0.5 / ×1.5: .808 / .863. Robust; the gate is the main lever. **RescueNet 20:** default .571, gate 0.2: **.818**, half-widths ×1.5: .636; the 0.3 gate was tuned on FloodNet only — a RescueNet gate tuned on its own val30 is a consistent, untried option.
     - **Building detection on predicted maps** (same opening/area rule; IoU ≥ 0.5): FloodNet 50 P .836, R .870, **F1 .853**, flooded/dry agreement of matched buildings .857 (IoU ≥ 0.25: F1 .891); RescueNet 20 P .557, R .733, F1 .633 (many false buildings), intact/damaged agreement .780.
     - **Caveats for the write-up:** val30 (30 images) is reused for checkpoint selection, decode tuning, the off-road gate and the cautious gate (eval100 and the full test are untouched); the 50-shot runs were validated every 500 steps on snapshots (`select_ckpt.py`), the others every 250 steps in-process — a small confound in the curve.
   0i. **RescueNet off-road gate, option A (Marlen, 6 Oct ~20:50; done 21:09).** The 0.3 gate was tuned on FloodNet only; RescueNet's own val30 sweep (`results/routing/gate_rescuenet/gate.md`, RescueNet 20 tuned decode) picks **0.2** (stranded F1 .471 → .556 on val30); eval100 .571 → **.818** (P .947, R .720). Gates are now per dataset (`route_eval.PRED_OFFROAD_MAXWET = {'floodnet': 0.3, 'rescuenet': 0.2}`, cautious `PRED_CAUTIOUS_MAXWET = {'floodnet': 0.1, 'rescuenet': 0.15}`, the RescueNet cautious gate chosen on its val30 with the same rule; no candidate reaches 99 % there, best 90.8 %). Regenerated: routing report, gate-free report, physical ablation, tiers, bootstrap CIs, sensitivity, scenes, viewer (About text) — FloodNet numbers unchanged. RescueNet now: stranded F1 .818 [.524, .966] (CI), physical ablation point vehicles .791 → v3 .818, tiers confident 273 houses 97.4 % / no "check first" on eval100; scene 12195 prediction now equals the labelled map (8 truck / 4 foot), caption changed from "honest failure" to "canal row reachable only on foot". The Batch 1 ablation table in §6.1 and the RescueNet numbers in 0h were computed with the old 0.3 gate.
   0j. **Batch 4 done (6 Oct 22:08): viewer features.** Marlen's choices (6 Oct ~21:20): download the 1024-px photos; browse both datasets, packed.
     - **1024-px photos:** `floodnet_r1024_subset.tar` (286,525,440 bytes, size checked against Drive) downloaded through Marlen's logged-in Chrome (Drive "download anyway" page; no sharing change) into `~/rr_data/floodnet_r1024/`. Every FloodNet scene shows the 1024×768 photo (q86) under the unchanged 768-grid overlays; `export_web_assets.photo_hi` checks each against the 768 image (mean |diff| ≤ 6 grey levels after downscaling, catches the EXIF-180 fix): 0 of 100 failed. RescueNet stays 768 (no 1024 subset).
     - **Packed format (Batch 5's "binary grid packing", done here):** one `scene.rrb` per scene = gzip of [uint32 header length][header JSON: grid meta, buildings with polygons + goal rings, array layout][cls u8, risk f32, flags u8 of both maps + cautious flags]; `p_boat` dropped (the JS router never read it; boat passability is in the flags). Read by `router.js parseScene` (DecompressionStream; also accepts an already-decompressed payload). The Python reference per house moved to `ref.json` (parity test only; `build_viewer.py` does not copy it). Per scene ~117 KB instead of ~920 KB (grid.json + buildings.json). Python round-trip test `src/tests/test_export.py` (bit-exact arrays, deterministic gzip).
     - **Browse all eval100:** `export_web_assets.py --out app/scenes` now exports the 8 showcase scenes + every eval100 photo of both datasets (200 scenes, 8 workers, ~1.5 min; `--showcase` for the 8 only); `scenes.json` carries counts, disagreements, "check first" count and per-photo mIoU. Panel: showcase row, then "All test photos" (FloodNet / RescueNet, sort by id / most cut off / most disagreements / lowest mIoU), thumbnails with "cut-off/houses" badge and a pink dot for disagreements, ‹ › buttons and N / P keys; auto caption for non-showcase photos; "No houses in this photo" for the 58 empty ones.
     - **Ghost route:** when the selected house's status differs between the maps, the other map's route is drawn in dashed magenta under the main route, with a card line ("Labelled map route: truck → boat").
     - **Water-probability layer** (Classes | Water probability, prediction only): p_wet = p1 + p4 per routing cell; clear below the cautious gate, faint yellow between the gates, **solid yellow where a truck may cross only on the normal map, not the cautious one** (exactly the cells behind "check first"), blue ramp at/above the normal gate; building cells left clear; legend bar with the per-dataset gates. On 7241 it shows the shallow-water yards the model put at 0.1–0.3.
     - **View in the URL:** `#s=<scene>&h=<house id>[gt]&e=<x>,<y>&m=gt|swipe&l=pwet&r=0` (old `#floodnet_7241` links still work); scene changes push history (back/forward work), other changes replace it; "Copy link to this view" button.
     - Checks: Python 27 tests; JS parity **200/200 scenes bit-exact** (13 s); viewer and built `docs/` checked in the browser pane at desktop and phone width (no console errors; link restore with house + entry + swipe + layer; back button; empty scenes). `docs/` 1,206 files, 55.7 MB. No router change, so routing reports unchanged.
   0k. **Batch 5 done (6 Oct 22:14): viewer robustness.** Marlen's choices (6 Oct ~22:09): vendor Leaflet; colour-blind palette as a toggle, off by default.
     - **Leaflet vendored:** `app/vendor/leaflet/` = leaflet@1.9.4 `leaflet.js`, `leaflet.css`, `images/` (5 PNGs), `LICENSE` (BSD-2), 188 KB, from unpkg; both files match the published SRI hashes byte for byte (the `integrity` attributes stay). The viewer now makes no external request and uses no browser storage, so it works offline and in a private window exactly as in a normal one ("incognito" retest = fresh browser profile in the pane + this audit).
     - **Binary grid packing:** done in Batch 4 (0j).
     - **Keyboard house list** (panel section "Houses", after Map): every house with its status/tier badge, word ("truck (confident)", "truck, check first", "boat", …), sub-class and "≠ <other map's status>" on disagreement; follows the map source (prediction / labelled map; swipe = prediction). `role="listbox"`, roving tabindex: ↑/↓/Home/End move, Enter/Space or click sends the team (same as clicking the house on the map); the selection is marked (`aria-selected`).
     - **Colour-blind palette** (checkbox "Colour-blind palette", URL `&cb=1`): Okabe-Ito. Status: truck blue #0072B2, boat orange #E69F00, on foot pink #CC79A7, check first yellow #F0E442, no route black (+ white X). Classes by meaning: clear road sky blue, flooded road dark blue, blocked road (RescueNet) bluish green #009E73 (not pink, which is the on-foot status), flooded/damaged building vermillion, dry/intact building yellow, water navy. The class overlays are recoloured in the browser (canvas, exact colour lookup; the PNGs are alpha 0/255 with exact class colours), so no extra files.
     - **CSS `clip` replaced** by `clip-path: polygon(...)` for the swipe (works on Leaflet's zero-size panes; checked at desktop and phone width).
     - Checks: Python 27, JS parity 200/200; `app/` and built `docs/` in the browser pane at desktop and phone width (swipe, colour-blind toggle + link restore, keyboard list), no console errors. Note: the local `python -m http.server` once dropped 1 of ~100 parallel thumbnail requests (dev server only). `docs/` 1,214 files, 55.9 MB.
   0l. **Batch 6 done (6 Oct 22:24): Gradio.**
     - **Upload warning:** a note under the upload box and, after an upload, in the header + a toast: no labels, so the houses and the entry point come from the prediction (missed houses get no route, nothing to compare with); FloodNet 50-shot model, use a nadir drone photo from a similar height; minutes/metres assume FloodNet's scale. `prep_upload` now **centre-crops other aspect ratios to 4:3 instead of stretching** (stretching would distort vehicle widths and the scale) and reports rotation / crop / upscaling in the warning. "No houses found" / "No houses in this photo" messages instead of "0 / 0".
     - **Scene choice by typing:** typing in the dropdown + Enter already worked for exact ids, but the list did not visibly narrow and an unknown id stayed in the box with no feedback. Added a **"type a photo id, then Enter" box** (`7241`, `12195`, `fn 7241`, `rn 12195`; ids are unique across datasets: FloodNet 6362–9112, RescueNet 10794–20428) with a message for unknown ids; dropdown entries now say "FloodNet 6362 · 5 houses, 2 not reachable by truck" / "· no houses" (from `app/scenes/scenes.json`).
     - **CPU test mode** `--fake_live` (launch config `gradio-test`, :7861): an uploaded eval100 FloodNet photo gets its cached prediction, labelled "TEST MODE … not the model" on screen; anything else is refused. Checked through the Gradio API: a 3000×4000 portrait and a 1365×768 (16:9) version of 7301 both map back to exactly 7301 after `prep_upload`; a RescueNet photo is refused. Headless smoke test: 200 scenes × 2 maps = 400 views, 2,064 house clicks, 116 empty views, no errors.
     - **Live hosting plan (presentation Thu 8 Oct 16:00):**
       1. Main demo = the **static web viewer** (no server, works offline from `docs/`, and from Pages once Batch 12 publishes it). Gradio is the extra "upload your own photo" moment.
       2. **Live Gradio on Kaggle** (prepared, not pushed; needs Marlen's OK): `kaggle/build_rr_code.sh` builds `rr-code` (src + decodes + tiers + viewer index; upload command printed), then `cd kaggle/gradio_live && kaggle kernels push -p .` (notebook `marlenmelis/rr-gradio-live`, 1 T4, internet on). Setup ~10 min, then the log shows `SHARE LINK: https://….gradio.live` (also `/kaggle/working/share_url.txt`); it serves for `HOURS = 3` (≈ 3.2 GPU-h incl. setup), then stops. Push at ~15:15 on Thu, open the link on the presentation laptop by 15:40, upload `demo/uploads/floodnet_7307.jpg` once to warm up. Live inference ≈ 20 s per photo on a T4 (50-shot eval100 took ~18 s/image). Share links last up to a week; the script's HOURS bounds the GPU use. Link-catcher and log-draining logic tested locally with a fake app (the app's output is pumped in a thread, so a full pipe cannot block it).
       3. **First real run = Batch 7** (with the live check): run `rr-gradio-live` briefly (e.g. HOURS = 0.5) to prove the share link + upload end to end, and **record the screen then** (Marlen) as the backup video.
       4. **Fallbacks, in order:** share link down → the recording of step 3; no internet in the room → `docs/` opened locally + the local Gradio from caches (`python src/gradio_app.py`, CPU, no upload).
     - **Demo upload photos** `demo/uploads/` (1024 px, FloodNet test, not in eval100, not near-duplicates of the supports; README with the expected statuses): 7307 flooded street (best; 10 of 13 houses agree), 6844 dry cul-de-sac (10/10), 11483 river + houses (20/20).
   0m. **Batch 7 done (7 Oct 02:32): GPU before the deadline** (Marlen's OK incl. the optional flips, 6 Oct ~22:28). Kaggle use: rr-gpu-flips 2×T4 22:30–02:25 (≈ 3.9 session-h), rr-gradio-live 1×T4 ≈ 0.8 h; week total ≈ 14 of 30 h.
     - **rr-code rebuilt** with `kaggle/build_rr_code.sh` (commit 4ef281f: src, splits incl. `floodnet_test_rest348_{a,b}` (full test minus eval100, halves; SHA256SUMS updated), supports, kaggle/live, decodes, tiers, viewer index).
     - **Live check passed** (`kaggle/live/live_check.py`, run on GPU 1 of rr-gpu-flips before its flips; `results/kaggle_v2/report/live/live_check/`): the app's `LiveModel` reproduces the eval100 cache **exactly** (max |diff| 0, decode agreement 1.0) on 7301 / 7241 / 9107, 14.7–14.9 s per photo, model load 85 s; a 4000×3000 upload through `prep_upload` 15.1 s.
     - **Live Gradio with a public share link** (`rr-gradio-live`, HOURS 0.75): setup 0.5 min (Kaggle mounts notebook outputs unpacked), support encoding 28 s, link up ~22:35. Uploaded `demo/uploads/floodnet_7307.jpg` through the link from the Mac (gradio_client): live inference + routing **20.6 s**; 14 houses found on the prediction (7 unreachable by road, 7 "check first"); view saved `results/kaggle_v2/report/live/gradio_live_upload_7307.jpg`. Marlen notified to record (link up until ~23:25). `run_gradio_live.py` set back to HOURS 3.0 for Thursday.
     - **Test-time flips (optional), FloodNet 50-shot:** `predict.py --flip h|v|hv` (flip the photo, flip the probabilities back; the 384/192 tile grid is symmetric at 768×576), 9 caches (val30 + full test, 3 flips; 17 s per photo per view), `src/tta.py` averages them with the unflipped caches into run `floodnet_50shot_s0_stratified_tta4`, `decode_tune.py` re-tunes its decode on val30, `tta.py report` compares (paired bootstrap, B 2000; `results/kaggle_v2/report/tta/tta.md`). Dry run first with copies of the unflipped caches as "flips": reproduces the headline exactly. Each flipped view alone scores like the plain model (eval100 .450–.458 vs .457), so the flips are registered correctly.
       - Results, own decodes: val30 tuned .484 → .498 (+.015 [−.006, +.032]); **full test tuned .472 → .455 (−.017 [−.027, −.007])**, eval100 tuned .492 → .481; at τ 0.5: full .420 → .431 (+.011 [+.003, +.019]). TTA's decode (τ .3/.3/.5/.4/.5 + building vote) lowered the road thresholds: full-test drivable precision .830 → .765. Routing (full test): stranded F1 .759 → .742 (own tuned), .748 → .766 (τ 0.5).
       - **Decision (rule fixed before looking at test): not adopted**; the headline stays the plain model. Post-hoc only (would be test-set selection): TTA probabilities with the plain model's decode give full .485 (+.013 [+.006, +.020]), eval100 .501, drivable precision .829. For the write-up: averaging views helps slightly at a fixed decode; tuning a decode on 30 images undid it; 4× inference cost.
   0n. **Batch 8 done (7 Oct 03:16): engineering.**
     - **One-command CPU rebuild** `src/rebuild_all.py` (`-j` parallel steps with dependencies, `--only`, `--list`, `--check`; logs in `results/rebuild_logs/`): 21 steps from the caches: results tables + curve, decode tuning + `report/decode.md`, baselines, transfer, labelled-map routing stats, eval100 routing (gated and gate-free), full-test routing, gate sweeps, physical-rule ablation, tiers, sensitivity, near-duplicates, bootstrap CIs, failures, figures, test-time-flip report (+ its routing), 200 viewer scenes, `docs/`, Python + JS tests; then a check of 14 headline numbers. **Full run 14.9 min, all headline numbers as published.** Verified harder: every JSON/MD result file and every scene file was snapshotted before and compared after (timing fields ignored): 907 / 916 identical; the rest explained and fixed: `results.md/json` was being corrupted by `collect_results.py` (it took the zero-label-transfer and full-test-half metrics as runs; fixed: only a run's own val30 summary and same-dataset eval100 count) and `dupes.json` lost its hand-recorded `manual` list (now written by the script); both identical after the fix. Remaining intended differences: the TTA report (now fully script-made: post-hoc, routing, decision), `gate.md` layout (one format for both datasets, same numbers), `gt_rescuenet_*` (recorded gate 0.3 → 0.2, unused for labelled maps; results identical).
     - **New scripts for steps that had been one-off commands:** `src/gate_sweep.py` (both gate sweeps; P/R/F1 identical to the published tables), `src/physical_ablation.py` (identical table and parameters), `decode_tune.py --table` (`decode.md` byte-identical). Bug fixed in `route_eval.py`: `--point_vehicles` together with `--router` crashed when recording the parameters (the routing itself was right). `route_report.py` skips the TTA run.
     - **Tests:** `src/tests/test_metrics.py` (12: scalar and per-channel τ, building vote, wet-context rule, ctx over vote, label mapping, confusion, `summarize` on a hand-computed case and a perfect one, bootstrap point/CI/seeding, TTA averaging) and `test_cache_pack.py`; **40 Python tests** + JS parity on 200 scenes.
     - **README.md** (own, for readers: what, headline numbers, layout, setup, reproduce, run, licences) and `requirements.txt` (pinned CPU env). The course starter's README is kept locally as `README_starter.md` (untracked). `.gitignore` allowlist: + README.md, requirements.txt.
     - **Data zip spec** (§2.6 rewritten) + **`src/cache_pack.py`**: lossless packing of the 16 needed caches as WebP inside a tar with a SHA-256 manifest; `unpack` restores bit-identical npz (all 16 verified): 497 → 264 MB. Measured total ≈ 380 MB > 300 MB target: options (a)–(c) in §2.6, Marlen to decide with the KLMS limit.
     - **PLAN cleanup:** a "how to read" note at the top; *[Superseded]* notes on §0 (Colab → Kaggle), §2.6, §10 (with the remaining schedule) and §12; §15 reordered (Quick summary → improvement batches → chronological log); the stale "waiting for OK" items removed.
   0o. **Batch 9 done (7 Oct 05:08): final notebook + G3.** Marlen's OK = "Do Batch 9" (6 Oct plan: the final T4 run needs OK).
     - **Notebook** `notebook/rescue_router.ipynb`, generated by `notebook/build_notebook.py` (cells as code, versionable); the submission copy with all outputs is `notebook/rescue_router_executed.ipynb` (8.4 MB, 20 code cells, 8 figures, 0 errors; to be renamed `20260876_<NAME>_code.ipynb` in Batch 12). Sections as §7: setup (flags `RUN_TRAINING=False`, `RUN_DEMO_TRAINING`, `RUN_LIVE`, `RUN_FULL_REBUILD`, `RUN_APP`; environment detection Kaggle / Colab / local; course `generalist.ckpt` from the shared Drive folder on Colab as in the starter, notebook input on Kaggle; our data bundle from `/content`, My Drive or `RR_DATA_ZIP`), data (split hashes, mapping, samples), formulation (tiling), fine-tuning (logs of the 7 runs; **a real 60-step 5-shot run** of `finetune.py`: 7.3 min, val30 mIoU .016 after 60 steps vs .314 for the real 500-step run: shows the code trains, not a result), results (**live** `predict.py` for FloodNet 50- and 20-shot on the first 10 eval100 photos vs the caches; FloodNet 50 decode re-tuned on val30 = the shipped one; curve with CIs recomputed (`bootstrap_ci.py`); full-test headline; baselines; near-duplicates; TTA decision), failures, cross-disaster, routing (**route_eval live** on eval100 / full test / RescueNet; tiers, gate, ablation, sensitivity tables; RescueNet 12195 scene), app (**live upload** of `demo/uploads/floodnet_7307.jpg` through the app's `LiveModel` + router, 16 s), reproducibility. The changed-paradigm note: 50-shot inference fits a T4 (12 GB), so the §0.5 fallback to 20 shots only applies below 14 GB.
     - **Data bundle** `src/make_bundle.py` (layout = the repository: code at a commit, `rr_data/` with exactly the photos/labels the notebook and scripts read, checkpoints, decodes, logs, packed caches, all reports, README_data, LICENSES, BUNDLE.json): **363 MB** (caches 264, checkpoints 35, data 44, reports 18). Checked locally first: every CPU step and `rebuild_all.py --check` run from inside the bundle with only its data. Uploaded as private Kaggle dataset `marlenmelis/rr-bundle` (zip `20260876_NAME_data.zip`).
     - **G3 on a fresh Kaggle T4** (`marlenmelis/rr-final-notebook`, notebook type; Python 3.13, torch 2.11, Tesla T4 14.6 GB): v1 failed at §4 on my string-concatenation bug, v2 at §9 (`finetune.setup_chameleon` `os.chdir`s, so relative paths broke after the in-process app model: now absolute + `chdir` back), v3 **COMPLETE in 26.5 min**. Kaggle's CLI does not return a notebook's outputs, so the same notebook was executed once more with `jupyter nbconvert --execute` inside a script kernel (`kaggle/notebook_exec/`, `marlenmelis/rr-final-notebook-exec`): rc 0, **25.8 min**, the saved copy. Both runs: **FloodNet 50-shot and 20-shot live = cache bit for bit (max |diff| 0, decode agreement 1.0)**, 4.2 / 1.8 min for 10 photos; full test .420 [.386, .455] / .472 [.437, .506]; stranded F1 .835 / .759 / .818 recomputed. Evidence: `results/kaggle_v2/report/g3/` (logs, HTML). Kaggle GPU this batch ≈ 1.5 h (week ≈ 15.5 of 30).
     - **Not tested:** the Colab path (Drive mount + course folder + zip in `/content`): it needs Marlen to approve the Drive popup; the code paths are written to the starter's conventions and avoid walking the whole Drive.
   0p. **Batch 10 done (7 Oct 05:17): write-up** `writeup/20260876_MarlenMelis_writeup.pdf` (LaTeX `writeup/writeup.tex`, figures `writeup/make_figures.py`, build `writeup/build.sh`; 3 pages: body ≈ 2.5 pages + references, within "2–3 pages, references not counted"). One section per graded point (problem and motivation, formulation, data and resources, method and setup, results and analysis), "Track A + Track B" in the title line, every pretrained weight with its link (Chameleon generalist from the course Drive/repo, containing its BEiT-v2-L image encoder and ViT-L label encoder; nothing else), all data disclosed (FloodNet + RescueNet with links/licences; "we labelled nothing ourselves"; supports/val30/eval100/full test counts).
     - **Fact checks done for it:** Seoul 8 Aug 2022: 141.5 mm in one hour, highest since 1907 (NPR); Osong 15 Jul 2023: 17 vehicles, 14 dead (Korea Times); "about one hour per photo" of labelling is in both dataset papers (FloodNet §3.1 "about one hour to annotate each image"; RescueNet "approximately one hour per image") → **"50 photos ≈ 50 labeller-hours, 3.5 % of FloodNet's training labels"** replaces "hours, not weeks". Chameleon authors from arXiv (Kim, Cho, Kim, Luo, Hong). Trainable parameters checked in the starter: custom-task config has all optional tuning flags off → biases, matching temperature, LayerNorm; optimiser Adam, weight decay 0 (the table says so). The 10-class variant is listed as a rejected alternative, not as a run ablation (it was never run).
     - Leads with the full-test headline and CIs, states plainly that it is far below PSPNet (flooded roads .29 vs .82), the labelled map = our router on human labels (not ground truth), confidence tiers (98.7 % vs 74 %), cross-disaster (.32 vs .63), failure analysis (shallow water as land → the gate), TTA not adopted, live = cache in the notebook, limitations (US vs Korea, single frames, router assumptions, one seed, val30 reused, rough ETAs).
     - Figures: learning curve with bootstrap CIs + full-test diamonds + RescueNet + PSPNet line (`figs/curve.pdf`); FloodNet 7301 prediction vs labelled map + RescueNet 12195 truck→boat→foot (`figs/routes.jpg`, drawn by the Gradio renderer).
   0q. **Batch 11 done (7 Oct 05:25): slides, GIF backup, recording storyboard.** Marlen chose the Claude Slides artifact (private; https://claude.ai/artifact/YYgLzsuLXTeRiQmWUtpyJw; downloads as PDF/PPTX). Course rule: title + ≤ 6 slides, 5-minute talk. **Title** (name, ID, "Track A + Track B", drone photo) → **1 problem + best results** (Seoul 2022 / Osong 2023, 1 h labelling per photo, big numbers .472 / .759 / 98.7 %) → **2 formulation** (photo + 5-channel label, channel table, router, rejected alternatives) → **3 data, weights, training** (no own labels; only the Chameleon checkpoint; recipe; notebook = cache) → **4 Track A** (curve with CIs, 1→5 / 5→10 / 20→50, flooded roads .29 vs .82, transfer .32 vs .63) → **5 Track B** (7301 prediction vs labelled map, 12195 truck → boat → foot; F1 .759 / .835 / .818; 98.7 % vs 74 %; viewer + Gradio 16 s) → **6 tried and learned** (7241 honest failure with check-first, gate .74 → .84, dropped TTA / 5→10 / smoothing, limits). Palette navy #0F1B2D / paper #F6F4EF / blue #1F6FB2 / orange #B85A12; Rubik + IBM Plex Sans. Speaker notes ≈ 4:40 of talk. Sources in `slides/deck/` (deck.json + one HTML per slide), images by `slides/make_slide_assets.py` (`slides/assets/`, uploaded as artifact assets).
     - **GIF backup** `slides/demo_route.gif` (RescueNet 12195 house 8, truck → boat → foot drawn step by step; 63 frames, 0.9 MB).
     - **Recording storyboard** `slides/recording_script.md`: talk timing per slide, a 75-s screen recording shot list with exact viewer links (7301 swipe + house 2, 12195 house 8, 7241 water-probability layer, Gradio upload of 7307) and voice-over, macOS recording steps, the Thursday share-link procedure and fallbacks.
     - **For Marlen:** download the deck as PDF → `slides/20260876_MarlenMelis_slides.pdf` (needed in the submission zip, Batch 12); record `recording.mp4` when convenient.
   0r. **Batch 12 done (7 Oct 05:46): submission zip + public demo.** Marlen's answers (05:38): KLMS allows **3 GB**, so the data zip ships whole (363 MB); **OK to create and publish** the demo repo; the slides were exported as **PPTX** (`Rescue Router.pptx`, 7 slides with notes; the course accepts PPTX) → `slides/20260876_MarlenMelis_slides.pptx`; no recording yet → the GIF goes in the app zip.
     - **Checks first:** 40 Python tests, JS router 200/200 bit-exact, `rebuild_all.py --check` all 14 headline numbers as published.
     - **Public demo:** `bash src/publish_demo.sh --create` → repo https://github.com/MarlenMM/rescue-router-demo (public, only `docs/` + README), Pages built: **https://marlenmm.github.io/rescue-router-demo/**. Checked in the browser pane (fresh profile, no storage): desktop RescueNet 12195 house 8 (truck 20 m → boat 30 m → foot 10 m), 375 px phone width FloodNet 7301/9107 (no horizontal scroll, no console errors). On a phone the photo is ~230 px wide between the title card and the controls sheet (Batch 5 layout, usable; a house card covers its lower part).
     - **Data zip = the exact bundle G3 ran on** (`~/rr_push/bundle/20260876_NAME_data.zip`, commit e4ec713, md5 9be98ccf…, = Kaggle dataset rr-bundle), renamed; no rebuild because its code (src, splits, supports, demo, kaggle/live, requirements) is unchanged since e4ec713 (the build script checks this) and the notebook finds `20260876_*_data.zip`.
     - **App zip** (84 MB): `viewer/` (= docs/), `src/` (git HEAD), requirements, `kaggle/gradio_live/` (Colab + Kaggle live-link scripts), `demo/uploads/`, `README_app.md` (`release/README_app.md`: Pages URL, local server, Gradio from the data zip, live command), `demo_route.gif`, and **the 98 RescueNet eval100 photos the data zip leaves out**: tested from a clean unzip, the Gradio app failed on RescueNet 10794 with the data zip alone (FileNotFoundError: the bundle has only the 2 showcase RescueNet photos); with the app zip's `rr_data/` copied in it shows them. FloodNet works from the data zip alone.
     - **Project zip** `~/rr_push/submission/20260876_MarlenMelis_project.zip` (458 MB, stored, outside iCloud): exactly the 5 course names; recursive check: one `.ipynb` (`20260876_MarlenMelis_code.ipynb` = `notebook/rescue_router_executed.ipynb`, 0 errors), no `__MACOSX`/`._*`/.DS_Store; `unzip -t` clean on all three zips. Builder: `release/build_release.sh` (~1 min).
     - Slides numbers vs notebook outputs: all present (.472, .759, 98.7 %, .835, .818, .29/.82, .32/.63, +.14/+.10, gate .74 → .84, TTA −.017, 1232 houses, upload 16.1 s; PSPNet as 84.8).
     - **Thursday live link moved to Colab** (GPU rule): `kaggle/gradio_live/colab_gradio_live.py`, one cell: mounts Drive, finds the data zip (+ app zip) in My Drive and the course `generalist.ckpt`, unpacks the caches, pins gradio 6.29.1, starts `gradio_app.py --live --share` in the background, prints the share link and returns; serves 3 h. **Not run yet** (needs GPU OK + the zips in My Drive; Google Drive for desktop is not installed, so Marlen drags them in at drive.google.com).
     - README.md: Pages URL, `notebook/ writeup/ slides/ release/` rows, "Submission" section. `.gitignore`: `release/` tracked.
   0s. **After Batch 12 (7 Oct): web viewer inside Gradio** (`src/gradio_viewer.py`; viewer no-labels mode for uploads `#s=up/<token>`; classic view as 2nd tab; test `src/tests/test_gradio_viewer.py`, JS = Python statuses on an upload; checked in CPU test mode in the browser, and from the unzipped data zip + app zip layout as on Colab). Colab live cell now runs it when the app zip is in My Drive. Project zip rebuilt 06:35 (still one .ipynb). Pages NOT republished (needs OK; change is invisible there). Not yet run on a GPU.
   0t. **Colab check of the notebook (7 Oct, Marlen ran it, ~13:40–14:10 KST): PASSED.** The clean notebook (`notebook/rescue_router.ipynb`, same cells as the submitted copy) uploaded to Colab, fresh T4 runtime, Run all: 20/20 code cells, 0 errors, **30.0 min**; Drive mount + course checkpoint shortcut + data zip in `AIC200-my_work` found automatically. Every number identical to the submitted run (curve, CIs, full test .420/.472, baselines, transfer, routing F1 .835/.759/.818, tiers, upload 16.1 s). One difference: the live recompute on Colab is **not bit-exact** (50-shot max |diff| 3/255, tuned-decode pixel agreement 0.9999; 20-shot 14/255, 0.9995), while on Kaggle it was 0 (the caches were made on Kaggle; same T4 + torch 2.11, other driver/cuDNN kernels). Nothing downstream changes (all reported numbers come from the caches). Also: Colab now has an empty `/kaggle/input`, so the notebook prints "environment: Kaggle" and works in /tmp/rr_nb (harmless). Copy kept locally: `results/kaggle_v2/report/g3/colab_t4_run_7oct.ipynb`. **Wording fixed (Marlen's OK, 16:05):** the notebook's §5.1 text (markdown only: code, outputs and metadata of the executed copy unchanged; the clean copy regenerated, same cells) and the write-up System paragraph now say identical on Kaggle, within 3/255 (99.99 %) on Colab; write-up still 3 pages; project zip rebuilt. Slide 3 changed in the deck (16:10, version 8): "about 30 minutes; match exactly on Kaggle and on 99.99 % of pixels on Colab" + notes; the PPTX in slides/ and the project zip still hold the old slide until Marlen re-exports the PPTX (zip not rebuilt, on Marlen's word).
   0u. **Batch 13.1 done (7 Oct 18:15 KST): infrastructure + free (CPU) experiments; round 1 launched.**
     - **Kaggle quota** (read from Kaggle's own quota endpoint in Marlen's Chrome; the Kaggle connector is unauthenticated): 17.2 of 30 GPU-h used at 18:00 → **12.8 h left**, resets Sat 10 Oct 09:00 KST. **Measured: a 2×T4 session costs 1× wall time** (93.7 s of quota over 93 s), so 2 lanes per session; the plan's per-GPU cut was not needed.
     - **Code (defaults bit-exact):** `finetune.py` `--tag`, `--tune label_decoder|relpos|head` (I5–I7) + `--lr_pretrained`, `--separate_alpha` (I8), `--lora r` (I9, `rr/lora.py`: adapters on the image-encoder qkv; stage-2 loader for such checkpoints), `--accum` (I11), `--pos_weight` (I12), `--rare_crop` (I13), `--dice` (I14), `--rot_blur` (I15), `--attn_dropout` (I18), `--copy_paste` (I19), `--ema/--ema_from` (I17, EMA snapshot `<run>_ema_step<N>.ckpt`), `--snap_from`, `--pseudo_list/--pseudo_dir` (I20); **`--freeze_rest`** (new efficiency item: requires_grad only on the trained tensors; the same updates, since nothing else is optimised). `rr/data.py` support-crop options (I1 `support_view`, I2 `crop_weights`) and the augmentations/loss (`rr_loss`). `predict.py` `--views/--crop_weights/--stride/--scales/--compile/--tag` (I1, I2, I24, I25, I36; eval_shot from the checkpoint for I20 runs). **`src/infer_sweep.py`** (I33: one model load, many variants; I3 self-support variant: the photo's own confident crop as a 51st support, `rr/infer.py add_self_support`). `src/avg_ckpt.py` (I23), `src/balanced_supports.py` (I4), `src/pseudo_labels.py` (I20), `src/sam_refine.py` (I31), `src/postproc_sweep.py` (I22, I27–I30, I32), `src/b13_decide.py` (13.3 scoring), `select_ckpt.py --tag/--min_step/--steps` (+ EMA rows). Checks: `FloodDataset.postprocess` md5 over 20 training crops + the support crop identical to the pre-Batch-13 code; loss defaults = the starter's BCE exactly; **49 Python tests**, JS router 200/200.
     - **`--freeze_rest` probe (Colab T4, 30 steps):** training peak **5.7 GB (was 10.1)**, **2.5 s/step (was 3.44 on Kaggle)**; I10 with 2+2 episodes (gbs 4) fits: 10.4 GB, 5.4 s/step. Loss values as in the published run's first steps. Used in every Batch 13 run (it changes no update, only memory/time; the control run b13ctl measures the rest).
     - **CPU results on val30** (current model; own decode re-tuned on val30; reference tuned .4835, drivable precision .569; `results/b13/postproc/postproc.md`): I27 bias offsets .4798 (−.004, **dropped**); **I28 two-step decode** (wet vs dry first: wet iff max(p1,p2,p4) ≥ t_w·(W+D), then class by tau) **.4982 (+.015, kept**, flat for t_w ≤ .3); **I29 clean-up** (fill components < 1600 px from the nearest larger one + road/water closing) **.4862 (+.003, kept)**; **I30 guided filter** (grey photo as guide; dense CRF: pydensecrf does not build for Python 3.12) r 4, eps 1e-3 **.4904 (+.007, kept)**, r ≥ 16 hurts; **stacked (I30 r4/1e-3 → I28 t_w .2 → I29 k 1600 r 1): .5037 (+.020), drivable precision .569 → .599** (no running cost: CPU post-processing ≈ 0.1 s/photo) → **adopted as the decode for all further comparisons**. I22 snapshot averaging (2–4× prediction cost, needs +.01): steps 1500+2000 .5011 (+.018), 1000+1500+2000 .4998 (+.016), 1000+1500 .4976 (+.014), 500+…+2000 .4972, 500+1000+1500 .4830 → passes the costly-change bar on the current model; to be re-checked on the final model's own snapshots (cost 2×). I32: p_wet ECE .054 on val30; the gates are val30-tuned thresholds, so a monotone recalibration changes no status: no change unless the model changes (then gates re-tuned).
     - **I31 SAM** (SAM ViT-B, Meta, `sam_vit_b_01ec64.pth` 375 MB from dl.fbaipublicfiles.com, Apache-2.0; prompted with each predicted building's box; Chameleon keeps the class; `results/b13/sam/sam.md`): on tuned tau: replace .4878, **grow .4917 (+.008)**, shrink .4794; on the stacked decode: grow **.5138 (+.0099 over .5038)**, replace .5082, shrink .4980. Dry-building IoU .386 → .430. Running cost (second model) needs +.010: **misses by .0001 → not adopted for now**; re-scored on the final model in 13.3.
     - **I4 class-balanced supports** (`supports/floodnet_balanced_s0.txt`, greedy on sqrt of per-channel pixel counts, train pool only): dry-building share of labelled pixels 3.5 % → 11.2 %, flooded road 24 → 31 %, water 43 → 38 %; photos with dry buildings 16 → 27; 3 of 50 shared with the stratified list.
     - **I20 pool** (`supports/floodnet_pseudo100_s0.txt`): 100 of the 115 train photos that are not supports and lie > 3 frame numbers from every val/test photo (near-duplicate guard); their labels are never read.
     - **I37 done:** `ON_KAGGLE = os.path.isdir('/kaggle/input') and not IN_COLAB` in `notebook/build_notebook.py` (clean notebook regenerated; the executed submission notebook untouched until 13.6).
     - **Not run as singles (budget, logged):** I7 head tuning (the head is part of the label decoder, so I5 covers it); I11 accumulation alone (×k compute; I10's 2+2 episodes test the bigger batch at equal images); I9 LoRA and I21 per-photo adaptation → 13.4 if the budget allows; I26 dropped (Marlen).
     - **Round 1 launched (13.2):** Kaggle `rr-b13-r1a` (2×T4, from 18:06; lane 0: b13ctl = I34 control, b13rare = I12+I13, b13dice = I14; lane 1: b13ld = I5, b13reg = I15+I18(0.7)+I19, b13lr2 = I16 2×) and `rr-b13-r1b` (from 18:14; lane 0: b13rp = I6, b13bal = I4, then I24 stride 128; lane 1: b13ep4 = I10 (gbs 4, 500 steps), b13sa = I8, then I25 scales 1+1.25). Every run: 1000 steps (I34), cosine, fp16, grad ckpt, `--freeze_rest`, EMA .998 (I17), snapshots 750 + 1000 + EMA 1000 validated on val30. Both sessions first failed once on a Kaggle dataset-mount error (new rr-code version still settling; ~1 min each), then ran. Colab T4 (CLI, session `b13`, from 18:01): the probe above, then `infer_sweep.py` on val30 with the published checkpoint: I2 crop weights 1,3,3,3,1, I23 weight average 1500+2000, I1 two views, I36 compile.
     - **13.2 incidents (19:10):** `rr-b13-r1b` failed to mount the rr-code dataset twice (versions 1–2); version 3 embeds the code tarball (base64) in the script, no dataset input: running since 18:28 (code 3fa600e = b43e36b + PLAN). **Colab session b13 vanished at ~18:37** (free tier), during the I1 two-view variant; the finished caches were on the VM and are lost. Surviving numbers (val30, τ .5 only, vs .4361): **I2 crop weights 1,3,3,3,1 .4438 (+.008)**, **I23 weight average 1500+2000 .4195 (−.017)**; 17.2 s/photo, 11.95 GB peak. The four prediction variants (I1, I2, I23, I36) move to Kaggle round 2 (~45 min in one lane, one model load).
     - **SAM decision (Marlen, 7 Oct 19:28): adopt SAM now** (override of the +.010 bar for a +.0099 gain; logged as a post-hoc decision). SAM "grow" (min area 50) on top of the stacked decode is part of the final setup and must pass the paired full-test check together with everything else; the write-up reports Chameleon-only and +SAM side by side, and lists SAM ViT-B as the second pretrained model (Chameleon decides every class; SAM only adds building pixels inside each predicted building's box). SAM runs on the decoded maps, so for 13.5 it runs on the Mac CPU (~1.3 s/photo, no Kaggle quota); the notebook and the live app get it in 13.6 (T4, ~1 s/photo + 375 MB download).
   0v. **Batch 13.2 done (round 1, Kaggle; outputs in `results/b13/r1a`, `r1b`) and 13.3 decided (7 Oct 23:15).** Session A ran 18:06–22:30 (3 pairs per lane; I14 Dice skipped by the 3-h start limit), session B 18:28–21:44 (its I24/I25 sweeps skipped by the start limit). Per run: training 60–66 min (1000 steps, ~2.6 s/step on Kaggle, 11.4 GB peak reported), val30 of 750 + 1000 + EMA 28–30 min. **Quota after round 1: 5.2 h left** (89,198 s of 108,000 used).
     - **val30, τ .5 / own tuned τ** (reference: published model .436 / .4835; `results/b13/decide/decide.json`): **I34 control (1000-step cosine + `--freeze_rest`) step 1000: .296 / .5137 (+.030)**, step 750 .300 / .5066, EMA .236 / .4828; I5 label decoder .464 / .4938 (750), .462 / .4861 (1000); I6 relpos .371 / .5064; I8 separate alpha .376 / .5076 (1000), .369 / .5095 (750); I10 2+2 episodes (500 steps) .375 / .4834, step 375 .4880; I12+I13 rare-class weights + crops .317 / .4661; I15+I18+I19 regularisers .261 / .4165; I16 2× lr .304 / .4602; I4 class-balanced supports .382 / .4074; **I17 EMA (decay .998/.996): vs the same run's last step, better in 3 of 9 runs (I12+13 +.013, I5 +.003, I4 +.0005) and worse in 6 (control −.031, I6 −.033, I8 −.049, I15+18+19 −.056, I16 −.018, I10 −.002)** (corrected 8 Oct 03:20; earlier written as 'worse in all 10'). Pattern: the new schedule shifts calibration (τ .5 numbers fall, drivable precision ≈ .95) while the tuned decode gains; the τ .5 snapshot choice of select_ckpt picks the worse snapshot for the control (750), so **snapshots are chosen by own tuned val30 from now on** (same data, the decode actually used).
     - **Decision (applied against the control, since every run shares the I34 schedule): no training variant beats the control** (best: I8 .5095 at 750 = −.004, I6 −.007); all dropped; no combined recipe needed. **New base model = b13ctl step 1000** (published recipe + I34 + `--freeze_rest`). I14 Dice, I9 LoRA, I25 two scales (3× prediction cost: a final full-test pass would not fit the 5.2 h) and I21: **not run (GPU budget)**; loss/augmentation changes (I12/I13, I15/I18/I19) all hurt, which lowers I14's prior.
     - The stacked no-cost decode does not transfer with the old model's parameters (t_w .2: control .5066 → .4766); `b13_decide.py` now re-tunes the whole stack (filter, t_w, tau, clean-up) per model (running).
     - **Round 2 (13.4) running: `rr-b13-r2`** (from 23:10; 1 session, ~2 h): lane 0 = I33 sweep on the control step 1000 (I2 crop weights 1,3,3,3,1; I23 weight average 750+1000; I1 two views; I24 stride 128; I3 self-support (tau = the control's tuned τ, hi .7); I36 compile); lane 1 = I20: predict the 100 pseudo photos with the control → pseudo-labels (tuned τ, conf ≥ .7, other < .2) → control recipe on 50 supports + 100 pseudo-labelled photos (EMA off) → val30 at 750/1000. Kaggle mounting failed again for a kernel-output input (r1a, twice); the two control snapshots go in through a small new private dataset `marlenmelis/rr-b13-ckpt` (mounted fine). I22 snapshot average of the control (750 + 1000) is scored on CPU.
     - **The new model's own decode (8 Oct 00:50, `results/b13/postproc_ctl/`, `src/decode_ctl_check.py`, val30):** the stacked decode of the old model does not transfer: re-tuned per model it gives the control .5112 < its tuned τ .5137 (b13_decide: no variant wins under it either). Piece by piece on the control: I28 two-step −.008 (dropped), I27 −.029 (dropped), **I29 clean-up +.0037, I30 guided filter r4/1e-3 +.0021; together: guided filter r4 eps 1e-3 → tuned τ [.25, .25, .55, .15, .2] → clean-up k 1600 r 2 = .5182 (+.0045), drivable precision .745 → .764 → adopted (no cost)**. **SAM grow on top: .5237 (+.0054)** (on plain tuned τ: .5198, +.006); dry-building IoU .364 → .383. **Current val30: .4835 → .5237 (+.040)** = control model + own decode + SAM. I3 self-support: **CUDA OOM on a T4** (51 supports do not fit next to the cached 50 at 14.6 GB; not feasible here).
   0w. **Batch 13.4 done (round 2, `rr-b13-r2`, 23:10–~01:00; outputs `results/b13/r2`, scores `results/b13/r2/score_r2.json`) and 13.5 launched (8 Oct 01:13).** val30, own tuned τ (control step 1000 = .5137, 16.8 s/photo): **I23 weight average 750+1000 .5153 (+.0016, drivable P .745 → .754, no cost) → kept: final checkpoint `floodnet_50shot_s0_stratified_b13ctl_wavg750-1000`**; I22 prediction average 750+1000 .5166 (+.003, 2× cost) dropped; I24 stride 128 .5236 (+.0099, 34 s/photo) and under the full decode chain .5285 vs .5182 (+.0103): **passes the +.010 bar by .0003 but cut for the GPU budget** (2× full test + transfer ≈ 3.1 h of the 3.3 h left, no margin; also 2× slower live uploads in the talk); I1 two views .5108 (−.003) dropped; I2 crop weights .5135 (−.0002) dropped; I36 torch.compile .5136, outputs within 1/255 but no speed-up (16.9 vs 16.8 s/photo) dropped; **I20 pseudo-labels .4437 (−.070) dropped**; I3 OOM. Final decode of the weight-averaged model (val30): guided filter r2 eps 1e-3 → τ [.25, .25, .45, .1, .15] → clean-up k 1600 r 0 = **.5231** (`results/b13/r2/decode_wavg.json`); + SAM grow to be scored (control: +.0054). Quota after round 2: 3.29 h.
     - **13.5 running: `rr-b13-final`** (checkpoint via new private dataset `marlenmelis/rr-b13-final-ckpt`, made locally by `avg_ckpt.py` from the round-1 snapshots, same CPU float32 mean as round 2's; a val30 pass checks it): lane 0 val30 + full test part 1; lane 1 full test part 2 + RescueNet eval100 transfer. ~1.5 h.
   0x. **Batch 13.5 done (8 Oct 02:55): final predictions + the paired full-test check: PASSED.** `rr-b13-final` (01:13–~02:45; lanes: val30 10.7 min + full part 1 62 min; part 2 66 min + RescueNet transfer 30 min; outputs `results/b13/final_run`). Consistency: the val30 cache of the locally averaged checkpoint = round 2's bit for bit. Check `src/b13_final_check.py` (`results/b13/final/final.md`, B 2000 paired, same resamples): **full test mIoU .472 → .506 [.473, .538] with SAM, Δ +.034 [+.013, +.056]**; Chameleon only .502 [.470, .533], Δ +.030 [+.009, +.052]; **both CIs above 0 → switch (rule)**. Drivable precision .830 → .842. Per class (full test, old → new +SAM): road clear .544 → .660, road flooded .291 → .309, building flooded .577 → .582, building dry .500 → .496, water .448 → .485. eval100 (subset): .492 → .515 (Δ +.024 [−.009, +.054]; Chameleon only .513). Quota left ≈ 1.7 h. **Next = 13.6 (full switch: `src/b13_switch.py install`, rebuild, gates re-tuned (I32), routing/tiers/viewer/Gradio, notebook re-run on a T4, write-up, slides numbers, zips): waits for Marlen's go; replacing the submission zip and the Pages republish need Marlen's OK.**
   0y. **Batch 13.6: full switch to the Batch 13 model (8 Oct 03:25–, Marlen asleep; blanket permission "do anything you see as needed", incl. the zip and Pages).**
     - **Install** (`src/b13_switch.py install`, reversible: `restore`): published FloodNet 50-shot files archived to `results/published_7oct.tar` (680 MB) and kept as run `floodnet_50shot_s0_stratified_pub` (caches, decode, metrics, logs); the new model's caches under the published names (val30, full test + parts, eval100 = its subset, RescueNet transfer); checkpoint = the weight average; decode JSON `frozen` (decode_tune keeps it) with the chain and **precomputed SAM class maps** (`results/kaggle_v2/rr/classmaps/`, per split); every consumer decodes via `rr/final_decode.decode_photo` (verified bit-exact on the published model before: all .md tables, reports and 200 scenes byte-identical).
     - **Correction found during the rebuild: the clean-up (I29) deletes and merges small houses** (building-detection F1 eval100 .853 → .736, recall .654; on val30 .680 → .653). New guard on val30 (like drivable precision): the decode may not detect fewer buildings than plain tuned τ. Every clean-up variant failed (incl. protecting buildings: merging comes from filling the gaps between houses); **final chain = guided filter (r 2, ε 1e-3) → τ [.25, .25, .45, .1, .15] → SAM grow**: val30 .5186 (det F1 .684 ≥ .680, drv P .756). **Paired full-test check of this chain: .472 → .507 [.474, .538], Δ +.035 [+.013, +.057]** (Chameleon only .503, Δ +.031 [+.010, +.052]) → switch stands (slightly better than with the clean-up, .506). `src/b13_final_check.py` now reads the old model from the `_pub` run after the switch (a first run compared new with new by mistake; caught).
     - **I32 gates re-tuned on val30:** off-road gate 0.3 unchanged (val30 F1 .980); cautious gate 0.1 → **0.05** (same rule; `route_eval.PRED_CAUTIOUS_MAXWET`).
     - **Rebuild (all 21 steps + new step `b13` = `src/b13_report.py`):** full test .507 [.474, .538] (τ .5 .278: calibration shift of the shorter schedule); eval100 .519 [.462, .570]; per class (full) road clear .662, road flooded .307, bldg flooded .594, bldg dry .498, water .472; drivable P .845; **routing stranded F1 full .752 [.629, .851], eval100 .819 [.722, .893]** (vs first model −.007 [−.048, +.026] and −.015 [−.060, +.020]: within noise, honestly reported); tiers: 208 confident 100 %, 287 check-first 79 %; building detection F1 .806 (first .853); flips report unchanged (reads the `_pub` caches; .455). `rebuild_all.py` EXPECTED updated to these values (`--check`: all as published). 50 Python tests, JS 200/200.
     - **Notebook** (`notebook/build_notebook.py`): intro + §4 recipe + §5.1 live check of the full chain incl. SAM vs the stored class maps + decode-aware metrics + new **§5.4 Batch 13** (`b13_report.py`, `writeup/batch13_improvements.md`) + SAM in setup (pip `segment-anything`, weights downloaded from Meta) + gate text + weights list. **Data bundle** rebuilt (`make_bundle.py`: + class maps, `_pub` caches packed, all 448 FloodNet test photos (the filter reads the photo), the improvements file): **554 MB**, tested from a clean unzip (cache SHA-256s verified, `b13_report` reproduces, check passes); Kaggle dataset `marlenmelis/rr-bundle-b13` (new dataset: mounts fine). **G3 run `rr-final-notebook-b13` launched 04:52** (Kaggle T4).
     - **Live paths:** Gradio upload path verified on CPU (filter + SAM live 4.5 s; live chain = stored class map 100 %); `colab_gradio_live.py` installs SAM + downloads the weights; the Kaggle fallback `run_gradio_live.py` now reads the data bundle (it pulled the OLD model's checkpoint/caches before, with the new decode: fixed).
     - **Write-up** (3 pages, builds): new headline, SAM as 4th pretrained weight, compute ~29 GPU-h, improvement-round paragraph, per-class table (first vs final), routing honest, tiers, limitations; figure 1 = final model (τ .5 shown up to 20 shots; first model as open markers). **README**, `release/README_app.md` (SAM for live), `release/build_release.sh` (new bundle 1b5d176). **Slides:** `slides/patch_b13.py` writes `slides/20260876_MarlenMelis_slides.pptx` from Marlen's latest export (`Rescue Router.pptx`, untouched): 21 text replacements (slides + speaker notes, lengths kept for the auto-shrinking boxes) + 4 regenerated images (curve, 7301 pred/gt, 7241); checked by rendering with LibreOffice. The Claude Slides deck gets the same edits. **Pages republished** (05:00) with the new scenes (viewer checked: no console errors).
     - **G3 PASSED (8 Oct 05:36):** `rr-final-notebook-b13` v2 (v1 stopped in §5.1: SAM on the GPU inside the notebook process left too little memory for the 20-shot subprocess; fix: SAM pre-loaded on the CPU in the notebook process, `sh()` errors now include the command's output): **21 code cells, 0 errors, 27.9 min** on a Kaggle T4; live 50- and 20-shot = cache exactly (max |diff| 0), **full decode chain incl. SAM live = stored class maps 100 %**, upload through the app 30.5 s. A local CPU dry run (throwaway env, bundle only) also passed all CPU sections first. `notebook/rescue_router_executed.ipynb` = this run (HTML: `results/kaggle_v2/report/g3/b13_kaggle_t4_run_8oct.html`).
     - **Submission rebuilt (05:38): `~/rr_push/submission/20260876_MarlenMelis_project.zip`, 640 MB** (code.ipynb 8.4, writeup.pdf 1.0, slides.pptx 1.7, data.zip 545, app.zip 84), one .ipynb (the data zip had a second one, `report/g3/colab_t4_run_7oct.ipynb`: deleted from the zip with `zip -d`; `make_bundle.py` now skips .ipynb), `unzip -t` clean; md5 51fe3085… (05:50, final). **Fallback (7 Oct 16:01) kept at `~/rr_push/submission_fallback_7oct/` (md5 423664e3…).** Slides in the zip = `slides/20260876_MarlenMelis_slides.pptx` (Marlen's 16:06 export patched: numbers, notes, 4 images); the Claude Slides deck has the same edits (version 10). Pages republished (SAM credited in the viewer and the demo README).
     - **Live app on a Kaggle T4 (`rr-b13-livetest`, 05:47; the talk's live-link setup): model + supports 101 s, then 3 demo uploads 16.1–17.3 s each end to end (Chameleon ~15 s, decode + SAM + routing ~1 s), SAM on the GPU, peak 12.3 GB** (so the new CPU fallback for SAM on CUDA OOM is a safety net, not needed on a T4). Slide 5/6 now says 17 s (the notebook's 31 s has SAM on the CPU). After-switch fixes: the data zip's README counts 19 caches; the release check excludes Pages-only/test-only/live-app-only changes (documented in the script). **Final zip md5 51fe3085…** (05:50). Deck version 11.
   0z. **After 13.6 (8 Oct ~06:00): demo video planned, not yet recorded.** Marlen wants a **90-s video for the slides**: MP4 1080p 30 fps, no audio/voice-over, no title or closing card, short burned-in captions, fast pace; Gradio only briefly (CPU test mode, captioned as such); accessibility and share links left out. Scene choice checked on the final model with the auto entry point: FloodNet 7301 and 7320 have no real truck leg (7320's boat houses start in the boat; truck ≤ 30 m), so the route scene uses **FloodNet 7577 house 1: truck 100 m → boat 80 m → on foot 40 m** (scan of all 100 FloodNet eval100 scenes: 7577 has 11 houses with truck 100 m + boat ≥ 30 m; next 7476, truck 70 m → boat 40 m); on the labelled map the same house is truck-reachable over a 260 m dry road (the prediction is too cautious there: the disagreement / ghost-route scene). Full 9-scene script and the new-chat prompt: `NEW_CHAT_PROMPT_VIDEO.md` (in the project folder; git-ignored, not in the repo). Output to be `slides/rescue_router_demo.mp4`. The submission zip (md5 51fe3085…) does not need the video.
   0za. **Demo video DONE (8 Oct 11:30): `slides/rescue_router_demo.mp4`** — 90.0 s, 1920×1080, 30 fps CFR (2,700 frames), H.264 CRF 18, no audio, 16.2 MB; the 9 scenes and captions of 0z. Made by `slides/demo_video/record_demo.py`: headless Chromium (Playwright 1.63 in a scratch venv; Chromium from the Playwright cache) drives the real viewer (`docs/` on :8766) and the Gradio app in CPU test mode (`src/gradio_app.py --fake_live` on :7861); frames from the DevTools screencast (JPEG + timestamps, ~37–45 fps) → ffmpeg (Homebrew, installed for this; it has no drawtext) → 30 fps. Caption bar and cursor (with click ripples) are injected DOM; the viewer's layout only ends above the caption bar, and for the video the map is fitted into the area left free by the title, route card and panel (`fitBounds` padding + zoomSnap 0.05; otherwise house 1 and its route sat under the cards). Viewer 1280×720 CSS × 1.5; Gradio 1120×630 × 1.714 so its result map fits. Choices checked on stills and a 1-fps contact sheet of the encode: scene 4 entry point (640, 460) in 7577 (searched: with the risk-aware toggle house 1's route changes, truck 60 → boat 80 → foot 40 m vs truck 80 → boat 50 m), 9107 house 3 (card: "100 % of such houses were truly truck-reachable"), 7241 house 6 (check first, "only 79 %"), browse = sort "most houses cut off" → 7294 (27/35) → 7300 (17/18), Gradio upload = eval100 7476 (screen: "TEST MODE: cached prediction of eval100 photo 7476, not the model"), clicks on house 7 (boat 50 m) and house 3 (check first). Re-record: start both servers, then `<venv with playwright>/bin/python slides/demo_video/record_demo.py` (`--only viewer|gradio|encode`, `--shots` for stills). Noticed, not changed: the viewer's About text still says mIoU .47 and "99 % / 74 %" (current tier stats: 100 % / 79 %); not visible in the video.
   0zb. **Talk deck with the video (8 Oct 14:12): `slides/20260876_MarlenMelis_slides.pptx` (19.4 MB)**, built by `slides/make_final_deck.py` from `slides/deck_base_b13.pptx` (the 05:48 deck). Same title + 6 slides and design; added: the 90-s video on the Track B slide (click to play; its three numbers are stat cards: .752 F1, 208/208, 17 s) + QR code to the public demo (also on the title), a photo → flood map → routes strip on the problem slide (frames of the video), prediction next to the human label on Track A (7320, mean IoU .84) and on the failure case (7241). Speaker notes rewritten for 5 min with the video (~760 words, ~200 of them spoken over the video). The submission zip (51fe3085) still holds the 05:48 deck: rebuild only on Marlen's OK.
   0zc. **Live Gradio link for the talk (8 Oct 16:01–19:05 KST, Colab CLI T4).** Launcher `kaggle/gradio_live/colab_cli_live.py` (CLI copy of colab_gradio_live.py: no Drive, app detached with `timeout` until 19:05 KST, re-run = restart only) and one-shot `kaggle/gradio_live/colab_cli_up.sh` (new session → gdown ckpt → zips up in 40 MB parts → setup → link). Events (KST):
     - 16:01 `colab new -s live --gpu T4` (T4 15 GB); gdown generalist.ckpt 2,967,114,775 B in 66 s.
     - 16:02 `colab upload` of the whole zips fails (HTTP 500, too large); 40 MB parts also failed until `/content/parts` existed (the contents API does not create folders). Parts 4 at a time: 630 MB in 70 s; joined zips md5 = Mac (data edf1cc05, app c632c6d1).
     - 16:14 setup 2.7 min → **share link up** (gradio_viewer.py, live GPU). 16:16 first upload (eval100 7476, browser): ~60 s incl. model load; 10 houses, 8 truck / 2 boat = cached scene; 2.4 % of map pixels differ (upload path = 1024-px JPEG resized, Colab GPU). Link sent to Marlen, in LIVE_LINK.txt (git-ignored).
     - 16:18–16:20 the two unseen OpenAerialMap photos (first time the model sees them; images in `demo/live_8oct/`): **Vijayawada** = water excellent (every flooded street is "road flooded"), houses fail (flat concrete roofs almost all missed: 2 houses, both truck); **Jamaica** = lagoon found as water, the intact hip-roof house found, but the dry dirt road and part of a white roof called water/flooded road and the wrecked houses missed (2 houses, both truck). Domain gap as expected (FloodNet = Texas suburbs). Each upload injected by script ran inference twice (`.change` fires twice for a scripted file set; ~20 s each); not changed during the talk.
     - 16:21 watchdog `kaggle/gradio_live/colab_cli_watch.sh` (every 5 min: curl link, `colab status`, VM check `colab_cli_hc.py`; two failures 45 s apart → restart the app, or a new session via colab_cli_up.sh; events → Monitor).
     - 16:55 Marlen asked for more test photos: `demo/uploads/floodnet_unused/` = 10 never-used FloodNet val photos with flooded streets and houses (≥ 3 frames from any training photo; most are 1–2 frames from a test photo: same flight). Watchdog all OK so far; 12 uploads served by 16:52 (5 of them mine).
     - 18:15 one false alarm (VM check saw no app process while the link answered 200; the recheck 45 s later passed; no restart). All other 5-min checks OK.
     - 19:05 the app stopped by its own `timeout` (GPU 0 MiB at 19:21). 19:21 `colab stop -s live`, `colab status` = no sessions, link 404, LIVE_LINK.txt deleted, watchdog stopped. **Uptime 16:14–19:05 (2 h 51 min), one link, 0 restarts, 15 uploads served (5 mine, 10 from visitors).** Next time: the one-shot `colab_cli_up.sh` + `colab_cli_watch.sh` work as is.
   0zd. **Final submission check (8 Oct 20:20–20:35, after the talk).** Checked against the course README: write-up 3 pages incl. references, all 5 sections, every pretrained weight with a link, data disclosed, tracks named; slides = the 14:12 talk deck (title + 6, name/ID/tracks, video embedded), numbers consistent with the write-up; notebook 21 code cells, outputs saved, 0 errors, .507 / .752 shown, the only .ipynb in all zips; data zip = the G3 bundle (md5 edf1cc05). **Fixed:** viewer About text still had the first model's numbers (mIoU .47, cautious gate 0.1, 99 % / 74 % → .51, 0.05, 100 % / 79 %; app/ and docs/, Pages not republished); README_app and demo/uploads README current (live-checked outside photos, floodnet_unused, CLI scripts); name on the notebook title line (markdown only, outputs unchanged). **Second pass against the TA session slides (AIC200_TA_Session_MiniProject_Chameleon.pdf, pp. 18–35) and the starter README** (identical to GitHub upstream, last commit f21c55b 30 Sep): all rules met; added a grader "How to run" in the notebook's Setup cell (markdown only: code cells + outputs identical to the G3-verified notebook bf1e2a8, checked) and the 90-s screen recording in the app zip ("a screen recording is welcome"). **Final: `~/rr_push/final/20260876_MarlenMelis_project/` (5 files) and `~/rr_push/final/20260876_MarlenMelis_project.zip` (677 MB, md5 a30d45b8)**; old 05:49 zip in `~/rr_push/submission/` kept as fallback. **Third check (20:49–21:28): the submitted notebook run on a fresh Colab T4** through the Colab CLI (cells executed one by one in the Colab kernel; only Drive mount skipped and the working directory carried between execs, both CLI artefacts): 21 cells, 0 errors, 31.5 min; every result number identical to the submitted outputs (.507 [.474, .538], routing F1 .752, ...), live recompute 99.98 % / 99.95 % of pixels (max 5/255, 14/255), only the 60-step demo run differs (.019 vs .016); log `release/notebook_colab_t4_run_8oct.log`; §5.1 Colab sentence updated (markdown only). **21:39 public demo republished** (Marlen's OK; `src/publish_demo.sh`, demo commit 2689232: only index.html, the About text with the final numbers; JS router parity 200/200; Pages built and checked live, no console errors).
   *Steps 1–5 below are the original product plan of 6 Oct ~16:00; 1–3 are done (0–0c, extended in Batches 1–6), 4 (deploy) is Batch 12, 5 (recording) is Marlen's.*
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
