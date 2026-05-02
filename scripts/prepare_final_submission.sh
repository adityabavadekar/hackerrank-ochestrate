#!/usr/bin/env bash
# prepare_final_submission.sh
# Assembles the three required submission artefacts into final_submission/:
#   1. code.zip          - code/ directory only (no venvs, caches, data corpus, CSVs)
#   2. output.csv        - agent predictions for support_tickets/support_tickets.csv
#   3. log.txt           - chat transcript from ~/hackerrank_orchestrate/log.txt

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FINAL_DIR="$REPO_ROOT/final_submission"
LOG_SRC="$HOME/hackerrank_orchestrate/log.txt"
OUTPUT_SRC="$REPO_ROOT/support_tickets/output.csv"

echo "==> Preparing final submission in: $FINAL_DIR"
rm -rf "$FINAL_DIR"
mkdir -p "$FINAL_DIR"

# ---------------------------------------------------------------------------
# 1. code.zip  — full project (tracked files), excluding corpus / CSVs / output dirs
# ---------------------------------------------------------------------------
echo "==> Building code.zip ..."

cd "$REPO_ROOT"

# All tracked files, minus:
#   support_tickets/  — CSVs not needed in the code zip
#   data/             — corpus is too large; evaluator provides it separately
#   final_submission/ — would cause a recursive zip
#   submission/       — old packaging artefacts
CODE_FILES="$(
  git ls-files | grep -Ev '^(support_tickets/|data/|final_submission/|submission/)' 
)"

if [ -z "$CODE_FILES" ]; then
  echo "ERROR: no tracked files found — did you run 'git add'?" >&2
  exit 1
fi

TMP_LIST="$(mktemp)"
echo "$CODE_FILES" > "$TMP_LIST"

zip -q "$FINAL_DIR/code.zip" --names-stdin < "$TMP_LIST"
rm -f "$TMP_LIST"

CODE_COUNT="$(echo "$CODE_FILES" | wc -l | tr -d ' ')"
CODE_SIZE="$(du -sh "$FINAL_DIR/code.zip" | cut -f1)"
echo "    code.zip     : $CODE_SIZE  ($CODE_COUNT files)"

# ---------------------------------------------------------------------------
# 2. output.csv — agent predictions
# ---------------------------------------------------------------------------
echo "==> Copying output.csv ..."

if [ ! -f "$OUTPUT_SRC" ]; then
  echo "ERROR: $OUTPUT_SRC not found. Run the pipeline first:" >&2
  echo "       uv run python -m code.main" >&2
  exit 1
fi

cp "$OUTPUT_SRC" "$FINAL_DIR/output.csv"
CSV_ROWS="$(( $(wc -l < "$FINAL_DIR/output.csv") - 1 ))"
echo "    output.csv   : $CSV_ROWS prediction rows"

# ---------------------------------------------------------------------------
# 3. log.txt — chat transcript
# ---------------------------------------------------------------------------
echo "==> Copying log.txt ..."

if [ ! -f "$LOG_SRC" ]; then
  echo "ERROR: log.txt not found at $LOG_SRC" >&2
  exit 1
fi

cp "$LOG_SRC" "$FINAL_DIR/log.txt"
LOG_SIZE="$(du -sh "$FINAL_DIR/log.txt" | cut -f1)"
echo "    log.txt      : $LOG_SIZE"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "==> final_submission/ is ready:"
ls -lh "$FINAL_DIR"
