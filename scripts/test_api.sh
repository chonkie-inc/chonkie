#!/usr/bin/env bash
# Smoke-test every Chonkie OSS API endpoint.
# Usage:  bash scripts/test_api.sh [BASE_URL]
# Default BASE_URL: http://localhost:8000

set -euo pipefail

BASE="${1:-http://localhost:8000}"
PASS=0
FAIL=0
SKIP=0

# ── helpers ──────────────────────────────────────────────────────────────────

green() { printf '\033[0;32m%s\033[0m\n' "$*"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$*"; }
yellow(){ printf '\033[0;33m%s\033[0m\n' "$*"; }

check() {
    local label="$1"; shift
    local http_code
    http_code=$(eval "$@" -o /dev/null -w '%{http_code}' -s 2>&1) || true
    if [[ "$http_code" =~ ^2 ]]; then
        green "  PASS  $label (HTTP $http_code)"
        (( PASS++ )) || true
    else
        red   "  FAIL  $label (HTTP $http_code)"
        (( FAIL++ )) || true
    fi
}

skip() {
    yellow "  SKIP  $1 ($2)"
    (( SKIP++ )) || true
}

# ── wait for the service ──────────────────────────────────────────────────────

echo "Waiting for $BASE/health …"
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null || true)
    if [[ "$code" == "200" ]]; then
        echo "Service is up."
        break
    fi
    if [[ $i -eq 30 ]]; then
        red "Service did not start within 60 s. Aborting."
        exit 1
    fi
    sleep 2
done

echo ""
echo "══════════════════════════════════════════════"
echo " Meta"
echo "══════════════════════════════════════════════"

check "GET /" \
    curl -f "$BASE/"

check "GET /health" \
    curl -f "$BASE/health"

echo ""
echo "══════════════════════════════════════════════"
echo " Chunking"
echo "══════════════════════════════════════════════"

SAMPLE_TEXT='The quick brown fox jumps over the lazy dog. This is a second sentence for testing. And here is a third one.'

check "POST /v1/chunk/token" \
    curl -f -X POST "$BASE/v1/chunk/token" \
         -H 'Content-Type: application/json' \
         -d "{\"text\": \"$SAMPLE_TEXT\", \"chunk_size\": 20}"

check "POST /v1/chunk/sentence" \
    curl -f -X POST "$BASE/v1/chunk/sentence" \
         -H 'Content-Type: application/json' \
         -d "{\"text\": \"$SAMPLE_TEXT\", \"chunk_size\": 50}"

check "POST /v1/chunk/recursive" \
    curl -f -X POST "$BASE/v1/chunk/recursive" \
         -H 'Content-Type: application/json' \
         -d "{\"text\": \"$SAMPLE_TEXT\", \"chunk_size\": 50, \"recipe\": \"default\"}"

check "POST /v1/chunk/semantic" \
    curl -f -X POST "$BASE/v1/chunk/semantic" \
         -H 'Content-Type: application/json' \
         -d "{\"text\": \"$SAMPLE_TEXT\", \"chunk_size\": 50}"

CODE_SAMPLE='def hello():\n    print(\"hello world\")\n\ndef goodbye():\n    print(\"goodbye\")'

check "POST /v1/chunk/code" \
    curl -f -X POST "$BASE/v1/chunk/code" \
         -H 'Content-Type: application/json' \
         -d "{\"text\": \"$CODE_SAMPLE\", \"language\": \"python\", \"chunk_size\": 50}"

echo ""
echo "══════════════════════════════════════════════"
echo " Refineries"
echo "══════════════════════════════════════════════"

# Build a small chunk list from the token chunker to use as refinery input
CHUNKS=$(curl -s -X POST "$BASE/v1/chunk/token" \
    -H 'Content-Type: application/json' \
    -d "{\"text\": \"$SAMPLE_TEXT\", \"chunk_size\": 20}")

if [[ "$CHUNKS" == "[]" || -z "$CHUNKS" ]]; then
    # Fallback hard-coded chunk
    CHUNKS='[{"text":"The quick brown fox","start_index":0,"end_index":19,"token_count":19}]'
fi

check "POST /v1/refine/overlap" \
    curl -f -X POST "$BASE/v1/refine/overlap" \
         -H 'Content-Type: application/json' \
         -d "{\"chunks\": $CHUNKS, \"context_size\": 5}"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
    check "POST /v1/refine/embeddings" \
        curl -f -X POST "$BASE/v1/refine/embeddings" \
             -H 'Content-Type: application/json' \
             -d "{\"chunks\": $CHUNKS, \"embedding_model\": \"text-embedding-3-small\"}"
else
    skip "POST /v1/refine/embeddings" "OPENAI_API_KEY not set"
fi

echo ""
echo "══════════════════════════════════════════════"
echo " Pipelines"
echo "══════════════════════════════════════════════"

PIPELINE_NAME="smoke-test-$(date +%s)"

# Create
CREATE_RESP=$(curl -s -X POST "$BASE/v1/pipelines" \
    -H 'Content-Type: application/json' \
    -d "{
      \"name\": \"$PIPELINE_NAME\",
      \"description\": \"Smoke-test pipeline\",
      \"steps\": [
        {\"type\": \"chunk\", \"chunker\": \"token\", \"config\": {\"chunk_size\": 30}},
        {\"type\": \"refine\", \"refinery\": \"overlap\", \"config\": {\"context_size\": 5}}
      ]
    }")

HTTP_CREATE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/pipelines" \
    -H 'Content-Type: application/json' \
    -d "{
      \"name\": \"${PIPELINE_NAME}-check\",
      \"description\": \"Smoke-test pipeline check\",
      \"steps\": [
        {\"type\": \"chunk\", \"chunker\": \"token\", \"config\": {\"chunk_size\": 30}}
      ]
    }")

if [[ "$HTTP_CREATE" =~ ^2 ]]; then
    green "  PASS  POST /v1/pipelines (HTTP $HTTP_CREATE)"
    (( PASS++ )) || true
else
    red   "  FAIL  POST /v1/pipelines (HTTP $HTTP_CREATE)"
    (( FAIL++ )) || true
fi

# Extract id from first create response
PIPELINE_ID=$(echo "$CREATE_RESP" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

check "GET /v1/pipelines" \
    curl -f "$BASE/v1/pipelines"

if [[ -n "$PIPELINE_ID" ]]; then
    check "GET /v1/pipelines/{id}" \
        curl -f "$BASE/v1/pipelines/$PIPELINE_ID"

    check "PUT /v1/pipelines/{id}" \
        curl -f -X PUT "$BASE/v1/pipelines/$PIPELINE_ID" \
             -H 'Content-Type: application/json' \
             -d "{\"description\": \"updated description\"}"

    check "POST /v1/pipelines/{id}/execute" \
        curl -f -X POST "$BASE/v1/pipelines/$PIPELINE_ID/execute" \
             -H 'Content-Type: application/json' \
             -d "{\"text\": \"$SAMPLE_TEXT\"}"

    check "DELETE /v1/pipelines/{id}" \
        curl -f -X DELETE "$BASE/v1/pipelines/$PIPELINE_ID"
else
    skip "GET  /v1/pipelines/{id}"    "could not extract pipeline id from create response"
    skip "PUT  /v1/pipelines/{id}"    "could not extract pipeline id from create response"
    skip "POST /v1/pipelines/{id}/execute" "could not extract pipeline id from create response"
    skip "DELETE /v1/pipelines/{id}"  "could not extract pipeline id from create response"
fi

echo ""
echo "══════════════════════════════════════════════"
echo " Results"
echo "══════════════════════════════════════════════"
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
echo "  SKIP: $SKIP"

if [[ $FAIL -gt 0 ]]; then
    red "Some endpoints failed."
    exit 1
fi

green "All endpoints passed."
