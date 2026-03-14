#!/usr/bin/env bash
# Cancel all queued and in_progress runs for "Auto-merge to main".
# Use this to clear the backlog so your next push can actually merge.
# Usage: export GITHUB_TOKEN=ghp_xxx && ./scripts/cancel-auto-merge-queue.sh

set -e
REPO="${REPO:-cheeksmagunda/basketball}"
WORKFLOW_FILE="auto-merge-to-main.yml"
PER_PAGE=100
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
      echo "API response: $(echo "$JSON" | jq -r '.message') — rate limit or error. Wait and retry later."
      exit 1
    fi
    COUNT=$(echo "$JSON" | jq '.workflow_runs | length')
    [ "$COUNT" -eq 0 ] && break
    echo "Cancelling batch of $COUNT $STATUS runs ..."
    for id in $(echo "$JSON" | jq -r '.workflow_runs[].id'); do
      curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" \
        "https://api.github.com/repos/$REPO/actions/runs/$id/cancel" > /dev/null
      CANCELED=$((CANCELED + 1))
      [ $((CANCELED % 100)) -eq 0 ] && echo "  $CANCELED canceled ..."
      sleep "$SLEEP_BETWEEN_CANCELS"
    done
    echo "  Batch done — $CANCELED total canceled so far."
  done
done

echo "Done. Canceled $CANCELED runs. Push to claude/* again and one auto-merge should run."
