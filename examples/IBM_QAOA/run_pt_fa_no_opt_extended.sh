#!/usr/bin/env bash
# Re-run all PT and FA_no_opt campaigns with extended Q grids.
# FA_no_opt p=2,3,4,5: Q up to 100,000  (t_shot large enough, ~87K needed for 44s)
# PT p=2,3,5,6,7:      Q up to 300,000  (t_shot small at p=7, ~302K needed for 44s)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# FA†: original 7 points + 3 high-Q anchors = 10 points total
FA_Q_VALUES="100,250,500,1000,2500,5000,10000,25000,50000,100000"

# PT: original dense grid + extend to 300K
PT_Q_VALUES="$(python3 -c '
import numpy as np
values = np.unique(np.rint(np.geomspace(100, 300000, 100)).astype(int))
print(",".join(str(int(v)) for v in values))
')"

echo "FA† Q grid (10 pts): ${FA_Q_VALUES}"
echo "PT  Q grid (100→300K, 100pts): ${PT_Q_VALUES}"
echo "Starting 9 campaigns..."
echo ""

run_campaign() {
    local method="$1" p="$2" qvals="$3"
    shift 3
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') | START ${method} p=${p} ==="
    METHOD_NAME="${method}" P_VALUES="${p}" Q_VALUES="${qvals}" MAX_PARALLEL_THREADS=4 \
        bash "${SCRIPT_DIR}/run_job.sh" "$@"
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') | DONE  ${method} p=${p} ==="
    echo ""
}

# FA† (no optimization) — Q up to 100,000
run_campaign FA_PP_no_opt 2 "${FA_Q_VALUES}"
run_campaign FA_PP_no_opt 3 "${FA_Q_VALUES}"
run_campaign FA_PP_no_opt 4 "${FA_Q_VALUES}"
run_campaign FA_PP_no_opt 5 "${FA_Q_VALUES}"

# PT (parameter transfer) — Q up to 300,000
run_campaign PT_PP_AAA 2 "${PT_Q_VALUES}"
run_campaign PT_PP_AAA 3 "${PT_Q_VALUES}"
run_campaign PT_PP_AAA 5 "${PT_Q_VALUES}"
run_campaign PT_PP_AAA 6 "${PT_Q_VALUES}"
run_campaign PT_PP_AAA 7 "${PT_Q_VALUES}"

echo "All 9 campaigns complete."
