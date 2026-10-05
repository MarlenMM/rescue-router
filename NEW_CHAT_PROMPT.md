Continue the Rescue Router project (KAIST AIC200, Chameleon, Track A + B). Read PLAN.md fully first; it is the source of truth. §15 "Status / handoff" says exactly where we are, and §13 records the starter API, the T4 measurements and the 3 starter bugs (already worked around in src/). Follow §0.2 strictly: no repeated runs, no GPU unless the step needs it, release runtimes when done. Also check your memory notes (Colab/Drive operations, project status).

Situation: data prep, splits, supports and the meta-checkpoint copy are done on Google Drive. Code is in src/ (private repo github.com/MarlenMM/rescue-router). I only have CPU, T4 and TPU v5e-1 on Colab, and the planned 384-px fp32 5-channel training runs out of memory on the T4. We agreed to stay on the T4 and add fp16 mixed precision plus gradient checkpointing. You may use my Google Drive connector whenever needed.

Do the next step, in this order:
1. On CPU only: implement --amp fp16 (autocast + GradScaler, scaler state saved for resume) in src/finetune.py, an fp16 option for inference in src/rr/infer.py and src/predict.py, and an fp16 flag in src/probe.py, as described in PLAN §15. Commit, push, and upload the new code tarball to Drive.
2. Connect to Colab. I will set the runtime to T4 and approve the Drive popup. Run the setup cell, then probe fp16 (+ grad ckpt) at 5 and 3 channels per step, and 20-/50-shot inference, each in a fresh process.
3. Pick the cheapest recipe that fits (PLAN §15 order), record it in PLAN.md, and launch the G1 run (FloodNet 20-shot s0) in the background. Check it at milestones only, not in a loop.
4. Report the G1 result and a time estimate for the rest of the queue before launching more runs, and release the GPU when idle.
