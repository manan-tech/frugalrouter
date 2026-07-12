#!/bin/bash
# Live status of FrugalRouter eval runs.
#   ./eval/status.sh            -> latest 5 runs, any branch
#   ./eval/status.sh <run-id>   -> full health + token breakdown for one run
set -uo pipefail
R=manan-tech/frugalrouter

if [ $# -eq 0 ]; then
  echo "=== recent eval runs ==="
  gh run list -R "$R" --workflow=eval --limit 5 \
    --json databaseId,status,conclusion,headBranch,createdAt \
    -q '.[] | "  \(.databaseId)  \(.status)/\(.conclusion // "-")  \(.headBranch)  \(.createdAt)"'
  echo
  echo "watch live (GitHub streams logs in the browser):"
  gh run list -R "$R" --workflow=eval --limit 2 --json databaseId \
    -q '.[] | "  https://github.com/manan-tech/frugalrouter/actions/runs/\(.databaseId)"'
  echo
  echo "detail: ./eval/status.sh <run-id>"
  exit 0
fi

ID="$1"
ST=$(gh run view "$ID" -R "$R" --json status -q .status)
echo "=== run $ID : $ST ==="
if [ "$ST" != "completed" ]; then
  echo "  still running — logs stream in the browser:"
  echo "  https://github.com/manan-tech/frugalrouter/actions/runs/$ID"
  exit 0
fi

D=$(mktemp -d)
gh run download "$ID" -R "$R" -n eval-output -D "$D" >/dev/null 2>&1 || { echo "  no artifact"; exit 1; }

echo
echo "--- LOCAL HEALTH ---"
grep -E "warmup workout|server died|LOCAL PATH|LOCAL DIED|ner_onnx|mass fallback|WATCHDOG" "$D/agent.log" | head -8

echo
echo "--- TOKEN EGRESS (every billed call) ---"
grep -E "escalated\[|batch escalated\[|early batch|escalation skipped|degrading" "$D/agent.log"

echo
echo "--- TOTAL ---"
grep -E "^\[.*done:" "$D/agent.log"

echo
echo "--- ACCURACY ---"
sed -n '/Per category/,/TOTAL/p' "$D/judge.log"
grep -E "FAIL" "$D/judge.log" | head -5
rm -rf "$D"
