# Recovering from GitHub Actions Queue Backlog (historical)

**Current setup:** The **auto-merge to main** and **clear cache on production deploy** workflows have been removed. Deploy by merging to `main` (PR or local merge + push); Vercel builds from `main`. Cache bust logic lives in the app (`/api/refresh`); no GitHub Action triggers it.

If you ever see a huge backlog of queued runs (e.g. after re-adding similar workflows), use the steps below.

## Get your latest code onto main and production

Your latest work is on a branch. Merge it yourself:

**Option A – GitHub UI**

1. Open **https://github.com/cheeksmagunda/basketball**
2. **Pull requests** → **New pull request** (base: `main`, compare: your branch)
3. **Merge pull request**
4. Vercel deploys from `main` in a minute or two.

**Option B – Local**

```bash
cd /path/to/basketball
git fetch origin
git checkout main
git pull origin main
git merge origin/your-branch-name
git push origin main
```

## Cancel scripts (if you re-add workflows and get a queue)

- `scripts/cancel-auto-merge-queue.sh` – cancels queued runs for `auto-merge-to-main.yml` (workflow must exist in repo).
- `scripts/cancel-queued-runs.sh` – cancels queued runs for `clear-cache-on-deploy.yml` (workflow must exist in repo).

Usage: `export GITHUB_TOKEN=ghp_xxx && ./scripts/cancel-auto-merge-queue.sh` (or the other script). Run in background with `nohup ... &` for large queues.

## Workflows that remain

- **Sync model config** – Runs on push to `main` (code changes only; path filter excludes `data/` and `.github/`). Syncs config defaults into `data/model-config.json`.
- **Retrain NBA Model** – Scheduled daily + manual dispatch. Trains LightGBM and pushes updated model to `main`.
