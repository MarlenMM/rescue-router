# Batch 13: every improvement idea and what it actually changed

FloodNet 50-shot model, 7–8 Oct 2026. Every number below is a measured value from the result files in
`results/b13/` (Kaggle/Colab T4 runs + CPU scoring). Where an idea was not run, it says so and gives no number.

## How to read the numbers

- **val30** = the 30 tuning photos (FloodNet val). All choices were made on val30 only; eval100 and the full test
  were never used to choose anything.
- **mIoU (tuned)** = 5-class mIoU with the model's **own decode thresholds re-tuned on val30** (the published
  decode-tuning rule: drivable precision may not drop below the τ 0.5 decode's). This is the number the adoption
  rule uses. **τ .5** = the plain threshold 0.5 decode (shown because it moved a lot).
- **drv P** = drivable precision (how often a "clear road" pixel is really clear; the router's safety number).
- **Two references:**
  - **published model**: val30 tuned **.4835** (τ .5 .436, drv P .569); full test **.472**;
  - **control run** (the published recipe with the 1000-step schedule + `--freeze_rest`, step 1000): val30 tuned **.5137**
    (τ .5 .296, drv P .745). Every training variant below was trained exactly like the control except for its idea, so
    **Δ vs control** is the effect of the idea itself.
- **Noise:** val30 has 30 photos; differences below about ±.01 are within run-to-run noise. Every training variant is
  one run (one seed). Bundled ideas (e.g. I12+I13) were trained together and cannot be separated.
- **Adoption rule (fixed before any result):** no running cost → keep if Δ ≥ 0 and drv P does not drop; extra running
  cost (slower prediction, extra model) → keep only if Δ ≥ +.010. Final switch only if the paired full-test
  difference has its 95 % bootstrap CI above 0.

## Final result

| setup | val30 (tuned) | full test mIoU (448 photos) | Δ full test [95 % CI] | full-test drv P |
|---|---|---|---|---|
| published model | .4835 | .472 | | .830 |
| new model, Chameleon only | .5232 | .502 | +.030 [+.009, +.052] | .841 |
| **new model + SAM** | **.5255** | **.506** | **+.034 [+.013, +.056]** | **.842** |

eval100 (100-photo subset of the full test): .492 → .513 (Chameleon only), .515 (+SAM); Δ +.024 [−.009, +.054].
Per class on the full test (published → new + SAM): road clear .544 → .660, road flooded .291 → .309,
building flooded .577 → .582, building dry .500 → .496, water .448 → .485.

New model = published recipe + **I34** 1000-step schedule (+ `--freeze_rest`) + **I23** weight average of steps 750
and 1000 + decode **I30** guided filter (r 2) → re-tuned τ → **I29** clean-up → **I31** SAM.

## All ideas (I1–I37)

### 1. Support set

| # | idea | measured on | result (val30 mIoU, own tuned decode unless noted) | Δ | other effects | decision |
|---|---|---|---|---|---|---|
| I1 | two different support crops per labelled photo, predictions averaged | control model | .5108 | **−.0029** vs control | τ .5 .2945; drv P .745 → .725; **2× prediction time** (34.2 s vs 16.8 s per photo) | dropped |
| I2 | support-crop channel weights 1,3,3,3,1 (more dry-building pixels in the crop) | control model | .5135 | **−.0002** vs control | τ .5 .2952; drv P .744. On the published model: τ .5 .4361 → .4438 (**+.0077**), tuned number lost (Colab session reclaimed) | dropped |
| I3 | the photo's own most confident crop added as a 51st support | control model | **no number: CUDA out of memory on a T4** (51 supports do not fit next to the cached 50) | – | – | infeasible |
| I4 | class-balanced choice of the 50 labelled photos (dry-building pixels 3.5 % → 11.2 % of the support pixels) | own training run | .4074 (step 1000); .4049 (750) | **−.1063** vs control | τ .5 .3816; drv P .767 | dropped |

### 2. What is trained

| # | idea | result | Δ vs control | other effects | decision |
|---|---|---|---|---|---|
| I5 | also train the label decoder | .4938 (step 750); .4861 (1000) | **−.0199** | τ .5 .4639 (the highest τ .5 of all runs); drv P .700 | dropped |
| I6 | also train the relative-position tables | .5064 (1000); .4930 (750) | **−.0073** | τ .5 .3705; drv P .740 | dropped |
| I7 | also train the decoder head | **not run separately**: the head is part of the label decoder, so I5 includes it | – | – | not run |
| I8 | one matching temperature per channel | .5095 (750); .5076 (1000) | **−.0042** | τ .5 .3687; drv P .797 | dropped |
| I9 | LoRA adapters in the image-encoder attention | **not run** (coded and unit-tested; no GPU budget left) | – | – | not run |

### 3. How it is trained

| # | idea | result | Δ vs control | other effects | decision |
|---|---|---|---|---|---|
| **I34** | **1000-step cosine schedule** (published: 2000 steps, best at step 1000) | **.5137** (step 1000); .5066 (750) | **+.0302 vs published** | τ .5 .436 → .296 (calibration shift that the tuned decode removes); drv P .569 → .745; half the training time | **kept** |
| (new) | `--freeze_rest`: gradients only for the tensors that are trained (same updates by construction) | inside the control run; its effect on accuracy cannot be separated from I34 | – | Colab probe (30 steps): training peak **5.7 GB vs 10.1 GB**; Kaggle runs: **2.6 s/step vs 3.44**, but they logged an 11.4 GB peak over 1000 steps, so the memory saving was not confirmed in the long runs | kept (speed) |
| I10 | authors' episodes: 2 support + 2 query photos per step, same number of images (500 steps) | .4880 (step 375); .4834 (500) | **−.0257** | τ .5 .3636; drv P .778 | dropped |
| I11 | gradient accumulation (bigger batch) | **not run** alone (×k compute; I10 tested the bigger batch at equal images) | – | – | not run |
| I12 + I13 | rare-class loss weights (positive weight 3 for flooded road, dry building) + crops centred on rare-class pixels (p .5) | .4753 (750); .4661 (1000) | **−.0384** | drv P .574 (lowest of the useful runs) | dropped |
| I14 | Dice term added to the loss | **not run** (skipped by the session's start limit, then no GPU budget) | – | – | not run |
| I15 + I18 + I19 | rotation ±30° + blur, matching-attention dropout 0.7 (published 0.5), rare-class copy-paste (p .3) | .4165 (1000); .3910 (750) | **−.0972** | drv P .574 | dropped |
| I16 | 2× learning rate (2e-3) | .4732 (1000); .4602 (750) | **−.0405** | drv P .656 | dropped |
| I17 | exponential moving average of the trained weights (decay .998; .996 for I10), scored at the last step | EMA vs the same run's last step without EMA: control .4828 vs .5137 (**−.0309**), I6 −.0331, I8 −.0494, I15+18+19 −.0561, I16 −.0179, I10 −.0018, **I12+13 +.0130, I5 +.0034, I4 +.0005** | better in 3 of 9 runs, worse in 6; on the control (the model kept) −.0309 | – | dropped |

### 4. Unlabelled data

| # | idea | result | Δ vs control | other effects | decision |
|---|---|---|---|---|---|
| I20 | pseudo-labels: the control model labels 100 unused train photos (confident pixels only), retrained on 50 real + 100 pseudo | .4437 (1000); .4296 (750) | **−.0700** | drv P .683 | dropped |
| I21 | per-photo test-time adaptation | **not run** (GPU budget; also slow and unstable by design) | – | – | not run |

### 5. Prediction

| # | idea | measured on | result | Δ | other effects | decision |
|---|---|---|---|---|---|---|
| I22 | average the predictions of several snapshots | published model (CPU, existing caches) | 1500+2000: .5011 (**+.0176**); 1000+1500+2000: .4998 (+.0163); 1000+1500: .4976 (+.0141); 500…2000: .4972 (+.0137); 500+1000+1500: .4830 (−.0005) | vs .4835 | 2–4× prediction time | – |
| I22 | same | control model, 750+1000 | .5166 | **+.0029** vs control | drv P .753; 2× prediction time (needs +.010) | dropped |
| **I23** | **average the weights of snapshots (one model, no extra cost)** | control model, 750+1000 | **.5153** (own final decode: .5231 vs .5182) | **+.0016** vs control (+.0049 with the final decode) | drv P .745 → .754 | **kept (final checkpoint)** |
| I23 | same | published model, 1500+2000 | τ .5 .4361 → .4195 (**−.0166**); tuned number lost (Colab session reclaimed) | | | – |
| I24 | denser tiles (stride 128, 12 tiles instead of 6) | control model | .5236; with the full decode chain .5285 vs .5182 | **+.0099** (tuned τ); **+.0103** (full decode) | **2× prediction time** (34.1 s per photo) | passed the +.010 bar under the full decode, **not adopted for the GPU budget** (the final full-test pass at 2× would have used all remaining quota) |
| I25 | two scales (1 + 1.25) | **not run** (3× prediction: a final pass could not fit the GPU budget) | – | – | – | not run |
| I26 | three training seeds | **not run** (dropped before Batch 13: too slow) | – | – | – | not run |

### 6. Probabilities → classes (CPU decode)

| # | idea | on the published model (vs .4835) | on the control model (vs .5137) | decision |
|---|---|---|---|---|
| I27 | per-class additive bias offsets | .4798 (**−.0037**) | .4851 (**−.0286**) | dropped |
| I28 | two-step decode: wet vs dry first, then the class | .4982 (**+.0146**) | .5058 (**−.0079**) | dropped (helped only the old model) |
| I29 | clean-up: specks < 1600 px filled from the neighbour, road/water closing | .4862 (**+.0027**), drv P .586 | .5174 (**+.0037**), drv P .760 | **kept** |
| I30 | guided filter (edge-aware, guided by the photo) | best r 4 ε 1e-3: .4904 (**+.0069**); r 16 ε .01: +.0008; r 16 ε .1: −.0095; r 32: −.0159 | best r 4 ε 1e-3: .5158 (**+.0021**); r 16 ε .01: −.0091; r 32: −.0280 | **kept** |
| I29+I30 together | filter → τ → clean-up | (with I28: .5037, **+.0202**, drv P .599) | .5182 (**+.0045**), drv P .764; on the final checkpoint .5231 vs .5153 (**+.0078**) | **kept** |
| I31 | SAM ViT-B redraws building outlines ("grow": adds building pixels inside each predicted building's box; Chameleon keeps the class) | on tuned τ .4917 (**+.0082**); on the stacked decode .5138 (**+.0099**) | on tuned τ .5198 (**+.0061**); on the final decode .5237 (**+.0054**); final checkpoint .5255 vs .5232 (**+.0023**); **full test .502 → .506 (+.004)** | **kept** (Marlen's decision; it missed the +.010 bar for an extra model by .0001 on the old model) |
| I31 other modes | "replace" outline / "shrink" only | replace +.0043 / shrink −.0041 | replace +.0005, shrink −.0052 (final decode) | not used |
| I32 | calibration of the router's wet probability | ECE .054 on val30; the router gates are val30-tuned thresholds, so recalibration changes no status | gates to be re-tuned for the new model in 13.6 | re-tune |

### 7. Efficiency

| # | idea | measured | decision |
|---|---|---|---|
| I33 | one process runs all prediction variants (the 3 GB generalist loads once) | model load 29 s once instead of per variant; 6 variants in one 68-min process (one of them, I3, ran out of memory) | **kept** |
| I34 | 1000-step schedule | see section 3: +.0302 and half the training time | **kept** |
| I35 | save encoded supports for the app | **not done yet** (part of 13.6) | open |
| I36 | `torch.compile` of the encoder blocks | outputs within 1/255 of the uncompiled ones (val30 .5136 vs .5137), but **slower**: 16.9 vs 16.8 s per photo | dropped |
| I37 | Colab detection fix in the notebook (`/kaggle/input` also exists on Colab) | no metric change (correct environment label and work folder on Colab) | **kept** |

## Caveats

- val30 is small: the per-idea differences of ±.01 are not individually significant; only the final full-test
  comparison has a confidence interval, and it excludes zero.
- Training variants are single runs; some losses (e.g. I4, I15+I18+I19) are large enough to be real, the small ones
  (I6, I8, I23) are not distinguishable from noise.
- The 1000-step schedule changed the calibration (τ .5 numbers fell sharply while the tuned numbers rose); every
  variant was therefore compared with its own re-tuned decode.
- SAM is a second pretrained model (Meta, Apache-2.0): it is reported separately (Chameleon only .502, + SAM .506).
