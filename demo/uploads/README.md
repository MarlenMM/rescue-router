# Photos for the live-upload demo (Gradio `--live`)

FloodNet test photos (CDLA-Permissive-1.0), 1024×768, that are **not** in eval100 (so the web viewer does not
show them) and are **not** near-duplicates of the 50 support photos (`results/kaggle_v2/report/dupes/`). The
model has never seen them; the numbers below are from the full-test routing run (`results/routing/full/`, FloodNet
50-shot, tuned decode, router v3), so you know what to expect.

| file | scene | labelled map (truck / boat / foot / none) | prediction | houses agreeing |
|---|---|---|---|---|
| `floodnet_7307.jpg` | **flooded street (best demo)** | 6 / 5 / 2 / 0 | 7 / 3 / 3 / 0 | 10 of 13 |
| `floodnet_6844.jpg` | dry cul-de-sac, all by truck | 10 / 0 / 0 / 0 | 10 / 0 / 0 / 0 | 10 of 10 |
| `floodnet_11483.jpg` | higher, wider shot: river and a row of houses | 18 / 1 / 1 / 0 | 18 / 1 / 1 / 0 | 20 of 20 |

An upload has no labels, so the app finds the houses and the entry point on the prediction itself: counts can
differ slightly from the table (which uses the labelled houses).
