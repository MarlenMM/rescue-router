# 10 FloodNet photos the project never used (8 Oct, for live-link tests)

From the FloodNet **val** split, 768×576 (the size the model sees). They're not among the 50 supports, the balanced or pseudo-label pools, val_tune30, or the 448 test photos, and none is within 3 frame numbers of a training photo.
Every one has flooded streets and flooded houses (labels in `labels/`, FloodNet raw classes: 1 building flooded,
2 building dry, 3 road flooded, 4 road dry, 5 water).

Caveat: FloodNet is video frames, and the flooded part of the flight (frames ~7260–7610) is shared by all splits, so
most of these sit 1–2 frames from a **test** photo: similar scenes, not new places.

| photo | flooded road | flooded houses | dry houses | water | note |
|---|---|---|---|---|---|
| 7309 | 21 % | 4 | 11 | – | mixed dry / flooded street |
| 7337 | 62 % | 8 | 3 | – | wide flooded road, power line |
| 7369 | 5 % | 5 | 0 | 74 % | big flooded field, tennis court |
| 7407 | 13 % | 12 | 1 | – | flooded backyards |
| 7417 | 17 % | 22 | 0 | 2 % | dense block, flooded street |
| 7516 | 10 % | 15 | 1 | – | furthest from any used frame (7 / 25) |
| 7519 | 14 % | 16 | 0 | – | block next to a dry path |
| 7576 | 17 % | 20 | 0 | – | flooded cul-de-sac |
| 7592 | 20 % | 3 | 12 | – | crossroads, dry houses on flooded streets |
| 7599 | 25 % | 18 | 0 | – | deep street flooding |
