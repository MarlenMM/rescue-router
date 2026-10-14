# Model card: Rescue Router flood mapper (FloodNet 50-shot) and router v3

## What it is

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

The last row (14 Oct) is better than this model on all three (paired intervals exclude zero, `results/baselines/report/`):
on FloodNet at 50 labels, Chameleon's few-shot adaptation does not beat ordinary fine-tuning of a pretrained segmenter.
Where it might (fewer labels, another disaster) is being measured.

Training variance (seeds, choice of the 50 photos) is not yet measured: every number is one run (Phase 2, A3).

## Intended use

Research and teaching: how far few-shot adaptation gets on a new disaster, and what a routing layer on top of a
segmentation needs (widths, tiers, honest uncertainty). The web viewer is a demonstration of that pipeline.

**Not for operational rescue decisions.** In particular:
- About one in nine houses shown as truck-reachable is not (false-safe 11 %); in most of these the route crosses
  water the model left unlabelled. Only the "confident" tier was reliable in testing, and it covers fewer houses.
- Shallow or muddy water read as dry ground is the dangerous error; flooded roads are the weakest class (.31).
- Adapted on Hurricane Harvey suburbs (Texas, 2017), nadir drone photos at FloodNet's height. Other places look
  different: on unseen OpenAerialMap photos (Vijayawada, Jamaica) water was found well but flat or damaged roofs were
  mostly missed (`demo/uploads/README.md`).
- Single photos, no georeference: routes start at the image edge, minutes and metres assume FloodNet's ground scale.
- The "labelled map" the routing is scored against is the same router on human pixel labels, not observed
  reachability.

## Ethics and data

Public datasets (FloodNet CDLA-Permissive-1.0, RescueNet figshare CC0) of disaster areas; no people are identifiable
at this resolution. See `DATA.md`.
