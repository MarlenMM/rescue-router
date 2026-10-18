# Model card: Rescue Router flood mappers and router v3

## Two models on the same labelled photos (17 Oct 2026; [`notes/PLAN.md`](notes/PLAN.md) §16.8–16.9)

The project started with Chameleon: "submitted" below means that model as handed in for the course on 8 Oct 2026
(tag `submitted-2026-10-08`), and most of this card describes it. A pre-registered comparison
(`experiments/phase2/PROTOCOL.md`) then found a standard segmenter trained on the same photos better on FloodNet, so the
viewer now opens FloodNet on it and keeps Chameleon one click away; RescueNet opens on Chameleon, which is better there.

| FloodNet, full test (448 photos) | DeepLabV3 ensemble (FloodNet default) | Chameleon (submitted) |
|---|---|---|
| what | 3 × DeepLabV3-ResNet50 (COCO-pretrained), training seeds 0-2, test-time flips averaged | Chameleon generalist + bias / matching-module fine-tune, SAM building outlines |
| labels | the same 50 training photos + the same 30 tuning photos (thresholds) | 50 + 30 (checkpoint, thresholds, filters) |
| mIoU [95 % CI] | **.617** [.583, .652] (paired vs Chameleon +.110 [+.079, +.144]) | .507 [.474, .538] |
| stranded-house F1 (router v3) | .817 | .752 |
| false-safe rate (shown truck-reachable, not) | **.068** | .111 |
| "confident truck" houses truly truck-reachable (eval100) | 99.4 % of 361 | 100 % of 208 |
| hardware, time per photo | CPU, ~4 s | GPU (T4), ~17 s |
| stability (5 training runs at 50 photos, eval100) | single network .555 ± .014 | .487 ± .054 |

RescueNet (Hurricane Michael, 20 labelled photos): Chameleon .421 vs the same DeepLabV3 ensemble .373 (eval100; paired
−.048 [−.083, −.014]; one network .360), so the viewer opens RescueNet on Chameleon (`results/baselines/report/
rescuenet_ensemble.md`); over three support draws with one recipe Chameleon (+30 tuning photos) .439 ± .037 vs one
network's .396 ± .013 on the full test. Its lead depends on the 30 tuning photos: a DeepLabV3 trained on all 50 photos (20 + 30) reaches .479.
Reproduce: `src/ensemble.py --paired --baselines_only`, `experiments/product_options.py`, `src/baseline_report.py`.

## The in-browser model ("Try your own drone photo")

LR-ASPP-MobileNetV3 (torchvision, COCO-pretrained, 13 MB), fine-tuned on the same 50 labelled FloodNet photos plus
pseudo-labels of the 1,395 other FloodNet training photos (only pixels where the 50-photo DeepLabV3 was at least
90 % sure; no evaluation photo is among them). Full FloodNet test: mIoU .609 with argmax, .601 with its thresholds
tuned on the 30 tuning photos (the registered rule picked the tuned decode on val30 + confirm100). Exported to ONNX
with its normalization and softmax inside the graph (`src/export_onnx.py`); onnxruntime matches PyTorch to 1e-5 in
probability. The viewer builds the routing scene from its output in JavaScript (`app/js/scenebuild.js`), checked
array for array against the Python pipeline in CI. An uploaded photo has no labels, so its houses and entry point come
from the model's own map, and the photo never leaves the browser.

## What it is (Chameleon, as submitted)

- **Segmentation:** [Chameleon](https://github.com/GitGyun/chameleon) (Kim et al., ECCV 2024), the generalist
  meta-trained by the AIC200 course (`generalist.ckpt`, 2.97 GB), fine-tuned on 50 labelled FloodNet drone photos for five
  per-pixel channels: clear road, flooded road, flooded building, dry building, water (anything else = "other").
  Only Chameleon's bias terms and its matching module's temperature and layer-norm tensors are trained (the course
  starter's stage-1 recipe; the checkpoint is ~5 MB on top of the generalist). Building outlines are then redrawn by
  [SAM ViT-B](https://github.com/facebookresearch/segment-anything) (Apache-2.0), which never changes a class.
- **Routing:** router v3 (`src/router.py`, JS port `app/js/router.js`, bit-exact): truck on dry roads (and yards at
  4× cost where the model's water probability is < 0.3), then boat across water, then the last stretch on foot, never
  back into a vehicle; vehicle widths, parked-car gaps, tree canopy, corner rule. Status per house: truck / boat /
  on foot / no route, and a confidence tier for truck houses (the same router with a cautious gate of 0.05).

## Training and tuning data (label budget: 80 photos)

| use | photos | source |
|---|---|---|
| fine-tuning (support set) | 50 (`supports/floodnet_stratified_s0.txt`, first 50) | FloodNet train |
| checkpoint schedule, weight average, decode thresholds, guided filter, SAM mode, router gates | 30 (`splits/floodnet_val_tune30.txt`) | FloodNet val |

Recipe: 1,000 steps, cosine lr 1e-3, batch 2, fp16, 384-px crops of 768×576 photos, weight average of steps 750 and
1,000 (`experiments/phase2/PROTOCOL.md` "Fixed recipe"). Prediction: 6 overlapping 384-px tiles, 17 s per photo on a T4.

## Results (FloodNet test, 448 photos unless noted; 95 % bootstrap intervals over photos)

| | value |
|---|---|
| mIoU, final decode (guided filter → val30-tuned thresholds → SAM) | **.507** [.474, .538] |
| mIoU, thresholds from the 50 training labels only (no tuning labels) | .468 |
| mIoU, plain threshold 0.5 | .278 |
| per class (final): clear road / flooded road / flooded building / dry building / water | .662 / .307 / .594 / .498 / .472 |
| PSPNet trained on all 1,445 labelled photos (FloodNet paper, same classes) | .848 |
| houses cut off by road, F1 (router on prediction vs router on labels) | .752 [.629, .851] |
| **false-safe rate** (shown truck-reachable, not truck-reachable on the labels) | **.111** [.054, .176] |
| "confident truck" houses truly truck-reachable (eval100, 208 houses) | 100 % |
| RescueNet (Hurricane Michael), 20-shot model, eval100 mIoU / stranded F1 | .421 / .818 |
| **same 50 photos, DeepLabV3-ResNet50 (COCO), no tuning labels**: mIoU / stranded F1 / false safe | **.581** / .813 / .081 |
| without the 11 test photos whose labels mark dry lawns as water (`splits/floodnet_label_errors.tsv`): this model's false safe / stranded F1 | .071 / .806 (baseline .036 / .870) |

The last row (14 Oct) is better than this model on all three (paired intervals exclude zero, `results/baselines/report/`):
on FloodNet at 50 labels, Chameleon's few-shot adaptation does not beat ordinary fine-tuning of a pretrained segmenter.

**Phase 2 (16-17 Oct; pre-registered, [`experiments/phase2/PROTOCOL.md`](experiments/phase2/PROTOCOL.md), [`notes/PLAN.md`](notes/PLAN.md) §16.8).**
- **Is val30 over-fitted (F2)?** No: on 100 other validation photos (confirm100) this model scores .502 vs .516 on
  val30 (registered limit .03).
- **Training variance (A3, eval100):** five 50-photo runs with the same recipe (support draws 0-2, training seeds 0-2)
  give .487 ± .054; this model (.519) is the second best of the five; one training seed gave .396 (normal training
  log). The baseline's five runs: .555 ± .014, ahead in every one (+.032 to +.152).
- **Learning curve with one recipe, full tests, baseline − Chameleon (Chameleon also tuned on 30 labels):** FloodNet 1
  / 5 / 10 / 20 photos −.122 / −.023 / −.020 (n.s.) / +.028; RescueNet 5 / 10 / 20 photos −.008 (n.s.) / −.023 / −.006
  (draw 0) and −.064 / −.057 (draws 3, 4), 50 photos +.060 (eval100). Chameleon's lead lives at 1-5 FloodNet photos and
  at 10-20 photos of the second disaster; with the same labels for both it keeps only FloodNet 1-5 and two of three
  RescueNet 20-photo draws, and when the baseline also trains on the 30 tuning photos it leads everywhere. The
  published RescueNet 20-photo point (.421, eval100) came from a longer recipe with val30 checkpoint selection; the
  one recipe gives .358 on that draw (three draws .420 ± .056).
- **Product-model choice (registered rule on val30 + confirm100):** a 50/50 probability mix of this model and the
  self-trained DeepLabV3 with flips (full test .591 [.556, .627], stranded F1 .844, false safe .052 [.015, .107]); the
  CPU-only alternative (the baselines' seed ensemble with flips, the registered baseline-only choice) .617 [.583, .652],
  false safe .068; the two are statistically level. The router's gates re-chosen on val30 + confirm100 stay at 0.3 and
  0.05. Since 17 Oct the viewer opens FloodNet on the CPU-only ensemble (the table at the top); this card's other
  sections describe the submitted Chameleon model.

The [web viewer](https://marlenmm.github.io/rescue-router/) shows every test scene with either model (Chameleon or the baseline), each with its own
statuses, tiers and measured false-safe rate, and marks the houses the two models route differently.

## Intended use

Research and teaching: how far few-shot adaptation gets on a new disaster, and what a routing layer on top of a
segmentation needs (widths, tiers, honest uncertainty). The web viewer is a demonstration of that pipeline.

**Not for operational rescue decisions.** In particular:
- About one in nine houses shown as truck-reachable is not (false-safe 11 %), one in fourteen (7 %) once 11 test
  photos with labelling errors are set aside (FloodNet marks dry lawns and river banks as water in them; 42 of the 105
  false-safe houses are in two of these photos); in most of the rest the route crosses water the model left
  unlabelled. Only the "confident" tier was reliable in testing, and it covers fewer houses.
- Shallow or muddy water read as dry ground is the dangerous error; flooded roads are the weakest class (.31).
- Adapted on Hurricane Harvey suburbs (Texas, 2017), nadir drone photos at FloodNet's height. Other places look
  different: on unseen OpenAerialMap photos (Vijayawada, Jamaica) water was found well but flat or damaged roofs were
  mostly missed (`demo/uploads/README.md`); the same-budget baseline found the roofs but read parts of the flooded
  street as clear road. With no RescueNet labels, both transfer equally poorly to Hurricane Michael (shared classes,
  mean IoU .223 vs .226; water .10).
- Single photos, no georeference: routes start at the image edge, minutes and metres assume FloodNet's ground scale.
- The "labelled map" the routing is scored against is the same router on human pixel labels, not observed
  reachability.

## Ethics and data

Public datasets (FloodNet CDLA-Permissive-1.0, RescueNet figshare CC0) of disaster areas; no people are identifiable
at this resolution. See `DATA.md`.
