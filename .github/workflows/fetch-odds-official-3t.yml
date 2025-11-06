name: Fetch 3T Odds (official)

on:
  workflow_dispatch:
    inputs:
      date:
        description: "対象日 (YYYYMMDD)"
        required: true
        default: "20250101"
      mode:
        description: "single or all"
        required: true
        default: "single"
      pid:
        description: "01..24（single時のみ）"
        required: false
        default: "01"
      race:
        description: "1..12（single時のみ）"
        required: false
        default: "1"
      parallel:
        description: "同時実行数（allモードのみ）"
        required: false
        default: "8"

permissions:
  contents: write

jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        run: |
          set -euo pipefail
          if [ -f package.json ]; then
            npm ci --no-audit --no-fund || npm i --no-audit --no-fund
          else
            npm init -y
            npm pkg set type=module
            npm i cheerio@^1.0.0-rc.12 --no-audit --no-fund
          fi

      - name: Fetch odds (single/all with parallel)
        env:
          DATE: ${{ github.event.inputs.date }}
          MODE: ${{ github.event.inputs.mode }}
          PID:  ${{ github.event.inputs.pid }}
          RACE: ${{ github.event.inputs.race }}
          PARALLEL: ${{ github.event.inputs.parallel }}
        run: |
          set -euo pipefail
          echo "MODE=$MODE DATE=$DATE PID=$PID RACE=$RACE PARALLEL=${PARALLEL:-8}"

          if [ "$MODE" = "all" ]; then
            # 01..24 × 1..12 の( pid race )ペアを列挙して並列実行
            # xargs: -P 同時実行数, -n 2 で2引数ずつ渡す
            { for p in $(printf "%02d\n" $(seq 1 24)); do
                for r in $(seq 1 12); do
                  printf "%s %s\n" "$p" "$r"
                done
              done
            } | xargs -P "${PARALLEL:-8}" -n 2 sh -c '
                  pid="$1"; race="$2";
                  echo "=== $DATE pid=$pid race=$race ===";
                  node scripts/fetch-odds-official-3t.js "$DATE" "$pid" "$race" || true
                ' _
          else
            node scripts/fetch-odds-official-3t.js "$DATE" "$PID" "$RACE"
          fi

      - name: Commit and push results
        run: |
          set -euo pipefail
          if [ -n "$(git status --porcelain)" ]; then
            git config user.name  "github-actions[bot]"
            git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
            DATE="${{ github.event.inputs.date }}"
            MODE="${{ github.event.inputs.mode }}"
            PID="${{ github.event.inputs.pid }}"
            RACE="${{ github.event.inputs.race }}"
            if [ "$MODE" = "single" ]; then
              MSG="odds3t: date=${DATE} pid=${PID} race=${RACE}"
            else
              MSG="odds3t: date=${DATE} (all pids & races, parallel=${{ github.event.inputs.parallel || '8' }})"
            fi
            git add -A
            git commit -m "$MSG"
            git push
          else
            echo "No new files to commit."
          fi
