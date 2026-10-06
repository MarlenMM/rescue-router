Continue the Rescue Router project (KAIST AIC200, Chameleon, Track A + B). Read PLAN.md fully first; it is the source of truth. §15 "Status / handoff" says where we are; §5.4 and §6.1 describe the router as built; §6.2–6.4 describe the product. Also check your memory notes (project status, Kaggle operations, overnight permissions).

Where we are (Tue 6 Oct, ~17:45 KST): all GPU work and all Track A analysis are finished.
- All 7 runs are done: FloodNet 1/5/10/20/50, RescueNet 20/50. Every run has val30 and eval100 prediction caches, tuned decodes and a routing evaluation.
- Everything is local: results in `results/kaggle_v2/rr/` (preds, ckpt, metrics, decode, logs), routing in `results/routing/`, data (768×576) in `~/rr_data/`, Python env `~/.venvs/rr`.
- Headline model: FloodNet 50-shot, eval100 mIoU .457 (τ 0.5) / .492 (tuned); full 448-image test .420 / .472.
- Track A analysis is done (PLAN §15 "Track A completion"): PSPNet table, zero-label transfer to RescueNet, failure analysis with a gallery, and figures in results/kaggle_v2/report/figures/. Overlay colours are in src/viz.py.
- The router (`src/router.py`, 16 unit tests) and the routing metrics already work. Off-road truck ×4; on predicted maps, off-road only where p_wet < 0.3 (the gate, tuned on val30; ground-truth maps have no gate). The JS port must match this.
- Deadline: Thu 8 Oct 12:30 KST (present 16:00).

This phase is Track B only: build the product so I can see it. No write-up or slides yet. Good showcase candidates come from the failure analysis (results/kaggle_v2/report/failures/) and the qualitative grid. Follow PLAN §15 "Next" item 5, in this order:
1. Pick 8 showcase scenes from eval100 (6 FloodNet with the 50-shot model, 2 RescueNet) and write `export_web_assets` in `src/`: photo, pred/GT overlays, cell grid, buildings with predicted and GT status, meta. CPU only, from the caches. Show me the 8 scenes (a contact sheet) before building on them.
2. The web viewer in `app/`, built into `docs/`: Leaflet, no build step. Click a house → a JavaScript port of `src/router.py` that must reproduce the Python statuses and costs on the exported grids (test it) → animated truck then boat. Layer toggles, swipe, prediction/GT swap, "Unreachable by road" counter, disagreements toggle, legend, attribution. Serve it locally and show it to me in the browser; check phone width too.
3. The Gradio app (§6.3): showcase scenes from caches on CPU. Live inference on an upload needs a GPU. Prepare it, but ask me before using any Kaggle GPU time.
4. Deploy: a separate public repo `MarlenMM/rescue-router-demo` holding only the built viewer, on GitHub Pages. The main repo `rescue-router` stays private until grading, then gets merged. Ask for my explicit OK before creating the public repo or publishing anything.

Rules: PLAN §0.2 (no repeated runs, no GPU unless a step needs it). Update PLAN.md §15 and your memory notes as things change; commit and push after each step. If a choice is genuinely mine (scene picks, look and feel, anything public), ask me with short options instead of guessing.
