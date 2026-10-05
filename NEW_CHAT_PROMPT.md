Continue the Rescue Router project (KAIST AIC200, Chameleon, Track A + B). Read PLAN.md fully first; it is the source of truth. §15 "Status / handoff" says exactly where we are; §13 records the starter API, the T4/fp16 measurements and the starter bugs (worked around in src/). Follow §0.2: no repeated runs, no GPU unless a step needs it. Also check your memory notes (project status, Kaggle operations, Colab/Drive operations).

Situation: I have no Colab subscription, so all GPU work now runs on Kaggle (free 30 GPU-h/week, 2× T4), driven from the Mac with the Kaggle CLI at ~/.venvs/kaggle/bin/kaggle. The core queue (FloodNet 1/5/10/20/50-shot and RescueNet 20/50-shot, recipe v2: fp16 + grad ckpt, cosine lr, best-val30 checkpoint, eval100 caches) is running as the Kaggle notebook marlenmelis/rr-gpu-queue, launched Tue 6 Oct 02:40 KST, expected to finish around 11:30–12:00 KST. The deadline is Thu 8 Oct 12:30 KST (present 16:00).

Do the next steps, in this order:
1. Check the queue status (`kaggle kernels status marlenmelis/rr-gpu-queue`). If it is still running, do not wait for it: work on Track B on the Mac, i.e. src/router.py (PLAN §6.1) with unit tests on synthetic grids, and the routing metrics on ground-truth maps (§5.4). If it failed or stopped early, read the live log (PLAN §15 says how) and tell me before re-launching anything.
2. When it is complete: download the outputs into results/kaggle_v2, check the queue logs for failed or unfinished jobs, and report the val30 + eval100 results per run (per-class IoU, mIoU, drivable precision/recall, wet IoU) as a table, with the data-efficiency curve.
3. Decode tuning on val30 from the caches (CPU only): per-channel τ and a dry-building fix; apply it to eval100 and report before/after.
4. Recommend whether the optional dry-building ablation run (PLAN §15 step 4) is worth the time and quota, and wait for my answer.
Update PLAN.md §15 and your memory notes as things change, commit and push after each step.
