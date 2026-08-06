#!/bin/zsh
cd /Users/ritwickraj/abc-credit-model
until [ -f reports/tuning_results.json ] && ! pgrep -f "03_tune.py" >/dev/null; do sleep 20; done
echo "=== tuning done, running 04_final_eval ==="
.venv/bin/python 04_final_eval.py
echo "=== running 05_explain ==="
.venv/bin/python 05_explain.py
echo "=== FINAL CHAIN COMPLETE ==="
