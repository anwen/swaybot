#!/usr/bin/env bash
# Run a single tau3-bench task with the SwayBot agent + remote glm endpoint.
# Model name comes from .env (SWAYBOT_MODEL) so we never hardcode it here.
set -euo pipefail

cd "$(dirname "$0")/../.."

set -a
source .env
set +a

export TAU2_DATA_DIR="$HOME/.cache/modelscope/datasets/evalscope--tau3-bench-data/snapshots/master"
# litellm (user_simulator) builds its own OpenAI client from these env vars.
export OPENAI_API_KEY="$SWAYBOT_API_KEY"
export OPENAI_API_BASE="$SWAYBOT_API_BASE"
# Bypass the macOS system proxy (127.0.0.1:7890) for the gateway host so
# litellm's httpx client does not corrupt TLS.
export NO_PROXY="api.example.invalid,127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

DOMAIN="${1:-banking_knowledge}"
TASK_IDS="${2:-task_001}"
RETRIEVAL="${3:-golden_retrieval}"
MAX_STEPS="${4:-50}"

python -m benchmarks.tau3bench \
  --domain "$DOMAIN" \
  --task-ids "$TASK_IDS" \
  --max-steps "$MAX_STEPS" \
  --retrieval-config "$RETRIEVAL" \
  --llm-agent "$SWAYBOT_MODEL" \
  --llm-user "openai/$SWAYBOT_MODEL"
