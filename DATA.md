# Data card

## Sources

| dataset | event | photos used | licence |
|---|---|---|---|
| [FloodNet](https://github.com/BinaLab/FloodNet-Supervised_v1.0) (Rahnemoonfar et al., 2021) | Hurricane Harvey, Texas, 2017 | train 1,445 (support pool), val 450, test 448 | CDLA-Permissive-1.0 |
| [RescueNet](https://springernature.figshare.com/collections/RescueNet_A_High_Resolution_UAV_Semantic_Segmentation_Benchmark_Dataset_for_Natural_Disaster_Damage_Assessment/6647354/1) (Rahnemoonfar et al., 2023) | Hurricane Michael, Florida, 2018 | val (support pool), test | figshare release CC0 (we use this one; the GitHub release states CC BY-NC-ND 4.0) |
| [OpenAerialMap](https://openaerialmap.org) crops (2 photos, live-demo only, never trained or tuned on) | Vijayawada floods (Bhuvan/ISRO); Hurricane Melissa, Jamaica (HOT-PH) | 2 | CC BY 4.0; **CC BY-NC 4.0** (Jamaica) |

## Processing (`src/prep_data.py`)

- Photos and labels resized to 768×576 (labels nearest-neighbour); FloodNet photos also at 1024×768 for the viewer.
- Labels mapped to five channels ([`notes/PLAN.md`](notes/PLAN.md) §3.3–3.4): FloodNet road-clear → clear road, road-flooded → flooded road,
  building-flooded / building-non-flooded → flooded / dry building, water → water; everything else → "other" (254),
  unlabelled → ignore (255). RescueNet: road-clear, road-blocked, building no/minor damage → intact, major/total →
  damaged, water.

## Splits (`splits/`, `supports/`, SHA-256 in `splits/SHA256SUMS`; rebuilt from the labels by `src/splits_local.py regen`)

| file | photos | role |
|---|---|---|
| `floodnet_test_full` | 448 | headline test |
| `floodnet_test_eval100` | 100 | fixed stratified subset of the test (half with a flooded class); the viewer's photos |
| `floodnet_val_tune30` | 30 | all tuning (checkpoint, thresholds, gates, Batch 13 choices) |
| `floodnet_val_confirm100` | 100 | disjoint from val30; confirms choices only (14 Oct) |
| `floodnet_stratified_s{0,1,2}` | 100 each | nested support orders (the k-shot set is the first k), three independent draws |
| `rescuenet_*` | same roles | RescueNet supports come from its val split; `rescuenet_val_tune30` = 30 photos of that pool outside the draw-0 supports |
| `rescuenet_stratified_s{3,4}` | 100 each | RescueNet support draws made without val30 (16 Oct, `src/splits_local.py rn_draws`) |

RescueNet's published draws 1 and 2 (`rescuenet_stratified_s{1,2}`) contain 8 and 5 val30 photos, because val30 was
cut outside draw 0 only; they were never used, and draws 3 and 4 replace them. `finetune.py` and `baseline_seg.py`
refuse any support photo that is in the dataset's val30 or confirm100 (`rr.runs.check_supports`).

Unlabelled use (14 Oct, PROTOCOL amendment 1 item 4): the self-trained baseline candidates also train on the other
1,395 FloodNet train photos **without their labels**, with pseudo-labels from the 50-shot baseline
(`src/baseline_pseudo.py`; evaluation photos excluded).

Near-duplicates: 14 of 448 FloodNet test photos are near-duplicate frames of a support photo (same flight);
removing them changes mIoU by ≤ .004 (`results/kaggle_v2/report/dupes/`).

Label errors (14 Oct): 11 FloodNet test photos and 2 validation photos (val30 7769, confirm100 7762) label large dry
lawns or river banks as water, in FloodNet's own annotation (raw class 5; e.g. 9344: 58 % of the photo "water", no
grass class). Found because two of them (9344, 9348) held 42 of the submitted model's 105 false-safe houses; then every
evaluation photo whose labelled water is ≥ 30 % vegetation-green (32 candidates, `src/label_audit.py`) was checked by
eye on the photo and the label only, never on a prediction, so the list favours no model. List with reasons:
`splits/floodnet_label_errors.tsv`; impact on both models: `results/kaggle_v2/report/label_audit/impact.md`
(published numbers stay on all photos; results are also given without these photos). The same check on RescueNet's
evaluation photos found none: its "green water" candidates are green sea and canal water with correct outlines.

## Storage

Preprocessed data in `$RR_DATA` (default `~/rr_data`); results in `results/` (not in git). Both are backed up
off-site as checksummed parts (`src/backup.py`, `release/backup/MANIFEST.tsv`).
