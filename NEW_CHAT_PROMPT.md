Continue the Rescue Router project (KAIST AIC200 mini-project, Chameleon few-shot flood mapping + rescue routing, Track A + B). I am Marlen Melis, 20260876; file names are `20260876_MarlenMelis_*`. Timeline: the talk is Thu 8 Oct 16:00 and needs only the PPTX; the notebook, write-up and zips are submitted by 23:59 Thu. Model work may run until 15:00 Thu.

Read first, in this order:
1. PLAN.md §15 "Status / handoff": the "Quick summary (read this first)", then the **"Batch 13: model improvements before the deadline"** block (the 36 ideas I1–I37, the adoption rule, the GPU budget and batches 13.1–13.6), then log entries 0r–0t (Batch 12, Colab check, wording fixes).
2. Your memory notes (project status, workflow, GPU platform choice, Colab and Kaggle operations, "notify after each batch").
3. The course's rules: `README_starter.md` ("IMPORTANT NOTES", "What to Submit").

State: Batches 1–12 are done. The submission is built and verified: `~/rr_push/submission/20260876_MarlenMelis_project.zip` (built 7 Oct 16:01; it is the fallback and must stay untouched until a better one passes). Public viewer https://marlenmm.github.io/rescue-router-demo/. The notebook passed on Colab (30 min, 0 errors). Slide 3 changed in the deck; the PPTX in the zip is still the old one until I re-export it.

Batch 13 = improve the FloodNet 50-shot model before the deadline: try every idea I1–I37 except I26, adopt whatever improves or does not hurt (rule in PLAN), always apply the efficiency items I33–I37, and switch to the new model only if it passes the full-test check (talk material by ~15:00 Thu, everything else by 23:59). Test ideas unpacked wherever the GPU budget allows (see "Unpacking" in the Batch 13 block). Run 13.1 → 13.5 back to back without waiting for a new prompt from me; notify me only when blocked or when 13.5 is done.

GPU: Kaggle ~12–13 h left this week (check the quota first), Colab ~1 h today. You never need to ask me before a GPU run; just stay within that budget. Still ask before publishing anything public (Pages republish) and before replacing the submission zip. For Colab use the official **Colab CLI** (`~/.local/bin/colab`, already logged in: `colab new -s <name> --gpu T4`, `colab exec`, `colab upload`/`download`, `colab stop`; see your Colab memory note), not the MCP browser tools. Don't mount Drive (it needs my consent every session): upload files from the Mac and fetch the course checkpoint on the VM with gdown from the course's public folder (id in the memory note). Always `colab stop` when a job ends.

How I work with you:
- For each batch: do it whole, test it, update PLAN §15 (mark it done, record results and decisions, log every variant with its val30 numbers) and your memory, commit and push, give me a short summary in chat. PushNotification (one line, lead with the result) only when blocked on me or when 13.5 is done (not after every overnight batch).
- If a choice is genuinely mine (anything public, look and feel, replacing the zip), ask with short options.
- Do not regenerate the final notebook (`notebook/rescue_router_executed.ipynb`) without re-running it on a T4.
- Run `date` before writing any time into PLAN or memory.
- The shell is zsh: `$VAR` holding several arguments is not split (use `${=VAR}` or a small Python driver); avoid backticks inside unquoted heredocs.
- Checks: `~/.venvs/rr/bin/python -m pytest src/tests -q`, `node --test app/test/router.test.mjs` (bit-exact on 200 scenes), `~/.venvs/rr/bin/python src/rebuild_all.py --check` (headline numbers), `src/build_viewer.py` before publishing `docs/`. New options must keep today's defaults bit-exact.
- Local previews: `.claude/launch.json` has `viewer` (app/, :8765), `viewer-docs` (docs/, :8766), `gradio` (:7860) and `gradio-test` (:7861, CPU test mode).

Batch 13.1
