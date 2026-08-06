#!/bin/zsh
cd /Users/ritwickraj/abc-credit-model
echo "=== 09_diagnostics ==="
.venv/bin/python 09_diagnostics.py
echo "=== 10_improve ==="
.venv/bin/python 10_improve.py
echo "=== IMPROVE CHAIN COMPLETE ==="
