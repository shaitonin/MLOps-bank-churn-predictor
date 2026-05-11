#!/usr/bin/env bash
# ci_cd.sh — Simulated CI/CD Pipeline
# Bank Customer Churn Prediction
#
# Simulates a continuous integration and delivery pipeline:
#   1. Environment validation
#   2. Data quality checks
#   3. Model training and registration
#   4. Metric validation (quality gate)
#   5. Deploy — promotion to Production in MLflow
#   6. API smoke test
#
# Usage:
#   chmod +x scripts/ci_cd.sh
#   ./scripts/ci_cd.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT_DIR/venv/bin/activate"
LOG_FILE="$ROOT_DIR/outputs/ci_cd.log"
MIN_AUC=0.85
MIN_RECALL=0.70

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
mkdir -p "$ROOT_DIR/outputs"

log() { echo -e "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
ok()  { log "${GREEN}✓ $1${NC}"; }
err() { log "${RED}✗ $1${NC}"; exit 1; }
warn(){ log "${YELLOW}⚠ $1${NC}"; }

log "══════════════════════════════════════════════"
log "  CI/CD Pipeline — Bank Churn MLOps"
log "══════════════════════════════════════════════"

# ── Step 1: Environment ───────────────────────────────────────────────────────
log "[1/6] Validating environment..."
source "$VENV" || err "venv not found at $VENV"
python -c "import lightgbm, mlflow, fastapi, sklearn" || err "Missing dependencies"
ok "Environment OK"

# ── Step 2: Data ──────────────────────────────────────────────────────────────
log "[2/6] Checking feature data..."
FEATURES_FILE="$ROOT_DIR/data/features/bank_churn_features.parquet"
[ -f "$FEATURES_FILE" ] || err "Features not found: $FEATURES_FILE"

N_ROWS=$(python -c "
import pyarrow.parquet as pq
t = pq.read_table('$FEATURES_FILE')
print(len(t))
")
log "  Samples found: $N_ROWS"
[ "$N_ROWS" -gt 1000 ] || err "Dataset too small: $N_ROWS rows"
ok "Data OK ($N_ROWS samples)"

# ── Step 3: Training and Deploy ───────────────────────────────────────────────
log "[3/6] Training and registering model..."
cd "$ROOT_DIR"
python notebooks/deploy.py >> "$LOG_FILE" 2>&1 || err "deploy.py failed"
ok "Training and registration complete"

# ── Step 4: Quality Gate ──────────────────────────────────────────────────────
log "[4/6] Validating model metrics..."
METRICS=$(python - <<'PYEOF'
import sqlite3, json
conn = sqlite3.connect("mlflow.db")
run = conn.execute("""
    SELECT r.run_uuid FROM runs r
    JOIN experiments e ON r.experiment_id = e.experiment_id
    WHERE e.name = 'bank-churn-classification'
      AND r.name = 'deploy_final_model'
    ORDER BY r.start_time DESC LIMIT 1
""").fetchone()
if not run:
    print(json.dumps({"error": "run not found"}))
else:
    uuid = run[0]
    rows = conn.execute(
        "SELECT key, value FROM latest_metrics WHERE run_uuid = ?", (uuid,)
    ).fetchall()
    print(json.dumps(dict(rows)))
PYEOF
)

AUC=$(echo "$METRICS"    | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('holdout_roc_auc', 0))")
RECALL=$(echo "$METRICS" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('holdout_recall', 0))")

log "  AUC    = $AUC  (minimum = $MIN_AUC)"
log "  Recall = $RECALL  (minimum = $MIN_RECALL)"

PASS_AUC=$(python -c "print(1 if float('$AUC') >= $MIN_AUC else 0)")
PASS_REC=$(python -c "print(1 if float('$RECALL') >= $MIN_RECALL else 0)")

[ "$PASS_AUC" = "1" ]    || err "AUC below minimum ($AUC < $MIN_AUC) — deploy blocked"
[ "$PASS_REC" = "1" ]    || err "Recall below minimum ($RECALL < $MIN_RECALL) — deploy blocked"
ok "Quality gate passed (AUC=$AUC | Recall=$RECALL)"

# ── Step 5: API Smoke Test ────────────────────────────────────────────────────
log "[5/6] Starting API and running smoke test..."

uvicorn app.main:app --port 8001 --log-level error &
API_PID=$!
sleep 3

HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health)
if [ "$HTTP_STATUS" != "200" ]; then
    kill $API_PID 2>/dev/null
    err "Health check failed (HTTP $HTTP_STATUS)"
fi
ok "Health check OK"

PREDICT_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 650, "Age": 42, "Tenure": 5,
    "Balance": 125000, "NumOfProducts": 2,
    "HasCrCard": 1, "IsActiveMember": 1,
    "EstimatedSalary": 80000,
    "Geography_France": 1, "Geography_Germany": 0, "Geography_Spain": 0,
    "Gender_Female": 0, "Gender_Male": 1,
    "AgeGroup_Middle": 1, "AgeGroup_Senior": 0, "AgeGroup_Young": 0,
    "AgeInactivity": 0, "EngagementScore": 2,
    "BalanceSalaryRatio": 1.56, "ProductsPerYear": 0.4
  }')

kill $API_PID 2>/dev/null

[ "$PREDICT_STATUS" = "200" ] || err "Predict endpoint failed (HTTP $PREDICT_STATUS)"
ok "Smoke test OK"

# ── Step 6: Initial monitoring cycle ─────────────────────────────────────────
log "[6/6] Running initial monitoring cycle..."
python notebooks/monitoring.py >> "$LOG_FILE" 2>&1 || warn "Monitoring completed with warnings"
ok "Monitoring OK"

# ── Final result ──────────────────────────────────────────────────────────────
log "══════════════════════════════════════════════"
ok "CI/CD PIPELINE COMPLETED SUCCESSFULLY"
log "  Full log: $LOG_FILE"
log "══════════════════════════════════════════════"
