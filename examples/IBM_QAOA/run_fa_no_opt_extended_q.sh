#!/usr/bin/env bash
# Re-run FA† (FA_PP_no_opt) campaigns at p=2,3,4,5 with Q grid extended to 500K
# so the curves reach the same right boundary as PT (T_max ~89s at Q=300K).
# FA† is zero-training (pure sampling), so this runs in minutes not hours.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Extended Q grid: original 10 pts + 3 anchors beyond 100K, up to 500K
FA_Q_VALUES="100,250,500,1000,2500,5000,10000,25000,50000,100000,200000,350000,500000"

echo "FA† extended Q grid (13 pts): ${FA_Q_VALUES}"
echo "Starting 4 FA† campaigns..."
echo ""

run_campaign() {
    local method="$1" p="$2" qvals="$3"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') | START ${method} p=${p} ==="
    METHOD_NAME="${method}" P_VALUES="${p}" Q_VALUES="${qvals}" MAX_PARALLEL_THREADS=4 \
        bash "${SCRIPT_DIR}/run_job.sh" "$@"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') | DONE  ${method} p=${p} ==="
    echo ""
}

run_campaign FA_PP_no_opt 2 "${FA_Q_VALUES}" --restart
run_campaign FA_PP_no_opt 3 "${FA_Q_VALUES}" --restart
run_campaign FA_PP_no_opt 4 "${FA_Q_VALUES}" --restart
run_campaign FA_PP_no_opt 5 "${FA_Q_VALUES}" --restart

echo "All 4 FA† campaigns complete."
