#!/bin/bash
# Run the agent on a REAL 2 vCPU / 4 GB amd64 box — the exact grader spec.
#
# WHY EC2 AND NOT CI OR THE LAPTOP:
#   - GitHub runners give a 4GB cgroup on a ~16GB host: page cache and memory
#     bandwidth come from OUTSIDE the cgroup, so they hide the very pressure
#     that kills us (FAILURES.md §6).
#   - The laptop is arm64 EMULATING amd64 — a large, unquantifiable penalty,
#     and it overheats.
#   - c6i.large is 2 vCPU / 4 GB / real x86 / no swap. ~$0.10 a session.
#
# ON YOUR MACHINE:
#   aws ec2 run-instances --image-id <AL2023-x86_64> --instance-type c6i.large \
#       --key-name <key> --security-group-ids <sg> --count 1
#   scp -i <key.pem> eval/ec2_test.sh ec2-user@<ip>:~
#   ssh -i <key.pem> ec2-user@<ip> 'bash ec2_test.sh <IMAGE> <FIREWORKS_KEY>'
#
# Then TERMINATE it. (Last time we left one running.)
set -euo pipefail

IMAGE="${1:?usage: ec2_test.sh <ghcr-image> <fireworks-key>}"
KEY="${2:?need the Fireworks API key}"

echo "== box =="
nproc | sed 's/^/  vCPU: /'
free -m | awk '/^Mem:/{print "  RAM : "$2" MiB"}'
swapon --show | grep -q . && echo "  ⚠️  SWAP IS ON — the grader has none. Disabling." && sudo swapoff -a || echo "  swap: off ✓"
grep -m1 'model name' /proc/cpuinfo | sed 's/^/  /'

echo "== docker =="
command -v docker >/dev/null || { sudo dnf install -y -q docker; sudo systemctl start docker; sudo usermod -aG docker "$USER"; }
sudo docker --version | sed 's/^/  /'

echo "== pulling $IMAGE =="
sudo docker pull "$IMAGE" >/dev/null
sudo docker images "$IMAGE" --format '  size: {{.Size}}'

mkdir -p work/input work/output
cat > work/input/tasks.json <<'TASKS'
__TASKS_JSON__
TASKS
if grep -q __TASKS_JSON__ work/input/tasks.json; then
  echo "  !! tasks.json placeholder not filled — scp eval/rehearsal19.json across and rebuild it"
  exit 1
fi

echo "== running under the EXACT grading contract =="
echo "   2 vCPU / 4 GB / no swap / 600s hard kill / ALLOWED_MODELS injected"
START=$(date +%s)
timeout --signal=KILL 600 sudo docker run --rm \
  --cpus=2 --memory=4g --memory-swap=4g \
  -e FIREWORKS_API_KEY="$KEY" \
  -e FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1 \
  -e ALLOWED_MODELS="accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/minimax-m3" \
  -e EMERGENCY_BUDGET_TOKENS=12000 \
  -v "$PWD/work/input:/input:ro" -v "$PWD/work/output:/output" \
  "$IMAGE" 2>&1 | tee run.log | grep -E "warmup|CALIB|escalat|done:|WATCHDOG" || true
END=$(date +%s)

echo
echo "════════════════ VERDICT ════════════════"
echo "  wall-clock : $((END-START))s   (soft 445s / flush 505s / hard 535s / grader kills at 600s)"
grep -oE "decode ~[0-9.]+ tok/s, prompt ~[0-9]+ tok/s" run.log | tail -1 | sed 's/^/  speed      : /'
grep -oE "tokens spent=[0-9]+" run.log | tail -1 | sed 's/^/  /'
grep -c "timed out" run.log | sed 's/^/  chat timeouts: /'
python3 - <<'PY'
import json
try:
    r = json.load(open("work/output/results.json"))
    n = len(r); blank = sum(1 for v in r.values() if not str(v).strip() or "could not determine" in str(v).lower())
    print(f"  answers    : {n}  ({blank} are fallback text)")
except Exception as e:
    print(f"  answers    : NONE — {e}")
PY
echo "  full log in ./run.log ; answers in ./work/output/results.json"
echo "  >>> scp them back and judge with: python3 eval/judge.py results.json eval/rehearsal19.json"
echo "  >>> THEN TERMINATE THE INSTANCE."
