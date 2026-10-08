#!/bin/zsh
# Watchdog for the live link until 19:05 KST: every 5 min curl the link, `colab status`, VM health check.
# Prints one line per event only (failures, restarts, new link, hourly heartbeat); fixes what it can by itself:
#   app dead or link dead, VM alive -> restart the app (colab_cli_live.py); session gone -> colab_cli_up.sh.
# Usage: nohup zsh kaggle/gradio_live/colab_cli_watch.sh <project dir> <work dir> >> <work dir>/events.log &
export PATH="$HOME/.local/bin:$PATH"
P=$1; W=$2; H=$P/kaggle/gradio_live; SES=live
END=$(date -j -f "%Y-%m-%d %H:%M" "2026-10-08 19:05" +%s)
link() { head -1 $P/LIVE_LINK.txt 2>/dev/null; }
check() {   # 0 = healthy
  local u=$(link) c st hc
  c=$(curl -s -m 20 -o /dev/null -w "%{http_code}" "$u/")
  st=$(colab status 2>&1)
  if ! print -r -- "$st" | grep -q "^\[$SES\]"; then echo "SESSION_GONE"; return 1; fi
  hc=$(colab exec -s $SES --timeout 60 -f $H/colab_cli_hc.py 2>&1)
  print -r -- "$hc" > $W/last_hc.txt
  if ! print -r -- "$hc" | grep -q "gradio_viewer.py --live --share\|gradio_app.py --live --share"; then echo "APP_DEAD http=$c"; return 1; fi
  if [[ $c != 200 ]]; then echo "LINK_DEAD http=$c"; return 1; fi
  echo "OK http=$c $(print -r -- "$hc" | grep -o 'uploads total [0-9]*')"; return 0
}
newlink() {
  local u=$(grep -o 'https://[A-Za-z0-9.-]*\.gradio\.live' $1 | tail -1)
  if [[ -n $u ]]; then
    printf '%s\nLive Rescue Router (Colab T4, real GPU inference). Restarted %s KST, until 19:05 KST.\n' $u "$(date +%H:%M)" > $P/LIVE_LINK.txt
    echo "$(date +%H:%M:%S) NEW_LINK $u"
  else
    echo "$(date +%H:%M:%S) RESTART_FAILED see $1"
  fi
}
n=0
while (( $(date +%s) < END - 120 )); do
  r=$(check); rc=$?
  if (( rc != 0 )); then
    sleep 45; r2=$(check); rc=$?           # confirm: one blip is not a failure
    if (( rc != 0 )); then
      echo "$(date +%H:%M:%S) FAIL $r / $r2"
      if [[ $r2 == SESSION_GONE* ]]; then
        L=$W/up_$(date +%H%M).log; echo "$(date +%H:%M:%S) NEW_SESSION starting (~10 min)"
        TMPDIR=$W zsh $H/colab_cli_up.sh $SES > $L 2>&1; newlink $L
      else
        L=$W/restart_$(date +%H%M).log; echo "$(date +%H:%M:%S) APP_RESTART starting"
        colab exec -s $SES --timeout 1800 -f $H/colab_cli_live.py > $L 2>&1; newlink $L
      fi
    fi
  fi
  print -r -- "$(date +%H:%M:%S) $r" >> $W/watch.log
  (( n++ % 12 == 0 )) && echo "$(date +%H:%M:%S) heartbeat $r"
  sleep 300
done
echo "$(date +%H:%M:%S) WATCH_END"
