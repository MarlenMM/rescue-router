# Rescue Router: talk plan, recording storyboard, demo backup

Deck: the Claude Slides artifact <https://claude.ai/artifact/YYgLzsuLXTeRiQmWUtpyJw> (private; title + 6 slides, the
full talk text is in each slide's speaker notes). For KLMS: download it as PDF (or PPTX) and save it as
`slides/20260876_MarlenMelis_slides.pdf`.

## 1. The 5-minute talk (course rule: ≤ 5:00, title + ≤ 6 slides)

| slide | time | say in one line (full text: speaker notes) |
|---|---|---|
| Title | 0:00–0:15 | Rescue Router: few-shot flood maps (Track A) + rescue routing system (Track B) |
| 1 Problem + results | 0:15–1:05 | Seoul 2022, Osong 2023 → "can a truck get there?"; 1 h of labelling per photo; .507 / .752 / 100 % (first model .472) |
| 2 Formulation | 1:05–1:45 | 5 dense channels chosen for the router; tiles; rejected alternatives |
| 3 Data + setup | 1:45–2:25 | FloodNet + RescueNet, no own labels; only the Chameleon checkpoint; T4 recipe; notebook = cache |
| 4 Track A | 2:25–3:15 | curve with intervals; flooded roads are the gap (.29 vs .82); new disaster needs its own labels |
| 5 Track B | 3:15–4:05 | router truck → boat → foot; F1 .752; confident 100 % (208 houses) vs check first 79 %; viewer + Gradio |
| 6 Learned | 4:05–4:50 | the dangerous error and the gate (.74 → .84); confidence > IoU points; dropped ideas; limits |

Rehearse with a timer: the notes are ~4:40 at a calm pace. If the live demo is shown, take it from slide 5's time
(play the 75-second recording or the GIF instead when the network is bad).

## 2. Screen recording (backup video, 60–90 s, for the app zip as `recording.mp4`)

**Setup (once):**

```bash
python3 -m http.server 8766 --directory docs
```

Open the links below in Chrome, full screen (⌃⌘F), zoom 100 %. Record with **⇧⌘5 → Record Selected Portion** (or
QuickTime → New Screen Recording), microphone on. For the Gradio part, the share link must be running (§4).

| time | screen | action | voice-over |
|---|---|---|---|
| 0:00–0:08 | <http://localhost:8766/#s=floodnet_7301> | nothing; let the map settle | "A drone photo after Hurricane Harvey, mapped by Chameleon from fifty labelled photos." |
| 0:08–0:20 | same | click **Swipe**, drag the handle left and right | "Left the prediction, right the labelled map: the same router on the human labels." |
| 0:20–0:34 | <http://localhost:8766/#s=floodnet_7301&h=2> | click house 2 (top row, orange badge) and let the animation run | "The truck drives to the water's edge, a boat takes over. About this many minutes." |
| 0:34–0:46 | <http://localhost:8766/#s=rescuenet_12195&h=8&m=gt> | click house 8 again to replay | "Another hurricane: truck to the canal, boat across, the last stretch on foot." |
| 0:46–0:58 | <http://localhost:8766/#s=floodnet_7241&l=pwet> | point at the yellow badges, click house 4 | "An honest failure: shallow water read as land. But these houses say *check first*; yellow is where the model is unsure." |
| 0:58–1:20 | the Gradio share link | upload `demo/uploads/floodnet_7307.jpg`, wait (~16–20 s, cut or speed up the wait), click a boat house | "And a new photo, live on a GPU: map, houses and routes in under twenty seconds." |
| 1:20–1:25 | same | stop | "Rescue Router. Thank you." |

If the Gradio link is not up, record 0:00–0:58 only (still 60 s) and add the GIF.

## 3. Backup GIF

`slides/demo_route.gif` (768×576, 0.9 MB, 63 frames): RescueNet 12195, labelled map, house 8, truck → boat → foot drawn
step by step (made by `slides/make_slide_assets.py`). Drop it into the slides or play it if both the network and the
recording fail.

## 4. Live Gradio link on Thursday (PLAN §15 0l)

About 15:15:

```bash
cd kaggle/gradio_live && ~/.venvs/kaggle/bin/kaggle kernels push -p .
```

Then about 15:30, read the link from the log (`SHARE LINK: https://….gradio.live`, also `/kaggle/working/share_url.txt`).
The run serves for 3 h (≈ 3.2 GPU-h). Open the link on the presentation laptop by 15:40 and upload
`floodnet_7307.jpg` once to warm it up. Fallbacks: the recording, then `docs/` opened locally (works offline).
