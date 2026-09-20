#!/usr/bin/env bash
# Every load-test arm against a freshly started server, all reports kept.
#
# The single-user arms are not optional: with a multi-second transport a
# concurrent run's percentiles measure queueing as much as the service, so
# only a 1-user arm answers "how long is the user waiting".
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a
mkdir -p reports

# Locust gets its own dependency tree: it pins requests<2.32.5, which
# conflicts with tabpfn-client's requests>=2.32.5. It only speaks HTTP to the
# running API, so PYTHONPATH puts that tree first for the generator alone.
LOADTEST_LIBS=.venv-loadtest/libs
if [ ! -d "$LOADTEST_LIBS" ]; then
  echo "installing the load generator into $LOADTEST_LIBS ..."
  .venv/bin/python -m pip install -q --target "$LOADTEST_LIBS" -r requirements-loadtest.txt
fi

stop_server () {
  # uvicorn --workers spawns children that outlive a kill of the parent, so
  # stop the whole process group and wait for the port to be released.
  local pgid=$1
  kill -TERM -- "-$pgid" 2>/dev/null
  for _ in $(seq 1 20); do
    curl -fsS -m 1 http://localhost:8000/health >/dev/null 2>&1 || return 0
    sleep 0.5
  done
  kill -KILL -- "-$pgid" 2>/dev/null
}

run_arm () {
  local name=$1 transport=$2 users=$3 rate=$4 duration=$5
  echo "=== arm '$name': transport=$transport users=$users duration=$duration ==="

  setsid env TABPFN_TRANSPORT="$transport" SERVING_ARTIFACT_SOURCE=registry \
      WEB_CONCURRENCY="${WEB_CONCURRENCY:-2}" \
      .venv/bin/python scripts/serve.py > "/tmp/serve_${name}.log" 2>&1 &
  local pgid=$!

  for _ in $(seq 1 60); do
    curl -fsS http://localhost:8000/ready >/dev/null 2>&1 && break
    sleep 2
  done
  if ! curl -fsS http://localhost:8000/ready >/dev/null 2>&1; then
    echo "  server never became ready; see /tmp/serve_${name}.log"
    stop_server "$pgid"; return 1
  fi

  PYTHONPATH="$LOADTEST_LIBS" .venv/bin/python -m locust -f loadtest/locustfile.py \
      --host http://localhost:8000 --headless -u "$users" -r "$rate" -t "$duration" \
      --csv="reports/${name}" --only-summary 2>&1 | tail -14
  curl -s http://localhost:8000/metrics | grep -E "payout_calls|cache_(hits|misses)"

  stop_server "$pgid"
}

run_arm surrogate_1u   surrogate 1  1 45s
run_arm surrogate_20u  surrogate 20 5 45s
run_arm live_1u        live      1  1 45s
run_arm stub_20u       stub      20 5 45s
echo "ALL ARMS DONE"
