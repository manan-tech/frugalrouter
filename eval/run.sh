#!/usr/bin/env bash
# Run the container exactly like the grading harness: 2 vCPU, 4 GB RAM,
# linux/amd64, /input + /output mounts. Times the whole run.
#
# Usage: eval/run.sh [tasks.json] [image] [extra docker args...]
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS="${1:-eval/variants.json}"
IMAGE="${2:-frugalrouter:dev}"
shift 2 2>/dev/null || shift $# 2>/dev/null || true

WORK="$(mktemp -d)"
mkdir -p "$WORK/input" output
# strip rubric/category fields — the container sees only task_id + prompt
python3 - "$TASKS" "$WORK/input/tasks.json" <<'EOF'
import json, sys
tasks = json.load(open(sys.argv[1]))
slim = [{"task_id": t["task_id"], "prompt": t["prompt"]} for t in tasks]
json.dump(slim, open(sys.argv[2], "w"), indent=1)
EOF
rm -f output/results.json

ENV_ARGS=()
if [[ -f .env ]]; then ENV_ARGS+=(--env-file .env); fi

echo ">>> running $IMAGE on $TASKS ($(python3 -c "import json,sys;print(len(json.load(open('$WORK/input/tasks.json'))))" ) tasks)"
START=$(date +%s)
docker run --rm --platform linux/amd64 --cpus=2 --memory=4g \
  "${ENV_ARGS[@]}" \
  -v "$WORK/input:/input:ro" \
  -v "$PWD/output:/output" \
  "$@" \
  "$IMAGE"
RC=$?
END=$(date +%s)
echo ">>> exit=$RC  wall=$((END - START))s"
test -f output/results.json && echo ">>> results.json written ($(python3 -c "import json;print(len(json.load(open('output/results.json'))))") answers)"
