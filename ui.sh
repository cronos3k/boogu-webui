#!/usr/bin/env bash
# ui.sh - start / stop / status for the Boogu Gradio web UI.
#   ./ui.sh start   # (re)launch, detached; prints the LAN URL
#   ./ui.sh stop
#   ./ui.sh status
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-start}" in
  start)
    pkill -f "python.*app.py" 2>/dev/null || true
    sleep 1
    source .venv/bin/activate
    setsid nohup python app.py > app.log 2>&1 &
    sleep 6
    ip=$(hostname -I | awk '{print $1}')
    url=$(grep -oE "http://0.0.0.0:[0-9]+" app.log | head -1 | sed "s#0.0.0.0#$ip#")
    echo "Boogu UI: ${url:-<see app.log>}"
    grep -i "error" app.log | tail -3 || true
    ;;
  stop)
    pkill -f "python.*app.py" && echo "stopped" || echo "not running"
    ;;
  status)
    if pgrep -f "python.*app.py" >/dev/null; then
      echo "running (pid $(pgrep -f "python.*app.py" | tr '\n' ' '))"
      grep -oE "http://0.0.0.0:[0-9]+" app.log | head -1 | sed "s#0.0.0.0#$(hostname -I | awk '{print $1}')#"
    else echo "not running"; fi
    ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac
