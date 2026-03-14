#!/usr/bin/env bash
# Cancel all queued and in_progress runs for "Clear cache on production deploy".
# Usage: export GITHUB_TOKEN=ghp_xxx && ./scripts/cancel-queued-runs.sh
# Rate limit: ~5000/hour; at 1.2s per cancel, ~30k runs ≈ 10 hours. Run in background:
#   nohup env GITHUB_TOKEN=ghp_xxx ./scripts/cancel-queued-runs.sh > cancel.log 2>&1 &
# Then: tail -f cancel.log

set -e
REPO="${REPO:-cheeksmagunda/basketball}"
WORKFLOW_FILE="clear-cache-on-deploy.yml"
PER_PAGE=100
# Stay under 5000/hour: ~1.3/sec. Use 1.2s between cancels to be safe.
SLEEP_BETWEEN_CANCELS=1.2

if [ -z "$GITHUB_TOKEN" ]; then
  echo "Set GITHUB_TOKEN (e.g. export GITHUB_TOKEN=ghp_xxx)"
  exit 1
fi

echo "Getting workflow ID for $WORKFLOW_FILE ..."
WORKFLOW_ID=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
  "https://api.github.com/repos/$REPO/actions/workflows" \
  | jq -r ".workflows[] | select(.path == \".github/workflows/$WORKFLOW_FILE\") | .id")

if [ -z "$WORKFLOW_ID" ] || [ "$WORKFLOW_ID" = "null" ]; then
  echo "Could not find workflow. Check token and repo."
  exit 1
fi

echo "Workflow ID: $WORKFLOW_ID"
CANCELED=0

for STATUS in queued in_progress; do
  while true; do
    JSON=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      "https://api.github.com/repos/$REPO/actions/workflows/$WORKFLOW_ID/runs?status=$STATUS&per_page=$PER_PAGE&page=1")
    if echo "$JSON" | jq -e '.message' >/dev/null 2>&1; then
      echo "API response: $(echo "$JSON" | jq -r '.message') — rate limit or error. Wait and retry or run script again later."
      exit 1
    fi
    COUNT=$(echo "$JSON" | jq '.workflow_runs | length')
    [ "$COUNT" -eq 0 ] && break
    echo "Cancelling batch of $COUNT $STATUS runs ..."
    for id in $(echo "$JSON" | jq -r '.workflow_runs[].id'); do
      curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/$REPO/actions/runs/$id/cancel" > /dev/null
      CANCELED=$((CANCELED + 1))
      [ $((CANCELED % 500)) -eq 0 ] && echo "  $CANCELED canceled ..."
      sleep "$SLEEP_BETWEEN_CANCELS"
    done
    echo "  Batch done — $CANCELED total canceled so far."
  done
done

echo "Done. Canceled $CANCELED runs."
