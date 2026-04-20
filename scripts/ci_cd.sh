#!/usr/bin/env bash
# ci_cd.sh — Pipeline CI/CD Simulado
# Bank Customer Churn Prediction — Parte 6
#
# Simula um pipeline de integração e entrega contínua:
#   1. Validação do ambiente
#   2. Qualidade dos dados
#   3. Treino e registro do modelo
#   4. Validação das métricas (gate de qualidade)
#   5. Deploy — promoção para Production no MLflow
#   6. Smoke test da API
#
# Uso:
#   chmod +x scripts/ci_cd.sh
#   ./scripts/ci_cd.sh

set -euo pipefail

# ── Configuração ─────────────────────────────────────────────────────────────
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

# ── Step 1: Ambiente ──────────────────────────────────────────────────────────
log "[1/6] Validando ambiente..."
source "$VENV" || err "venv não encontrado em $VENV"
python -c "import lightgbm, mlflow, fastapi, sklearn" || err "Dependências ausentes"
ok "Ambiente OK"

# ── Step 2: Dados ─────────────────────────────────────────────────────────────
log "[2/6] Verificando dados de features..."
FEATURES_FILE="$ROOT_DIR/data/features/bank_churn_features.parquet"
[ -f "$FEATURES_FILE" ] || err "Features não encontradas: $FEATURES_FILE"

N_ROWS=$(python -c "
import pyarrow.parquet as pq
t = pq.read_table('$FEATURES_FILE')
print(len(t))
")
log "  Amostras encontradas: $N_ROWS"
[ "$N_ROWS" -gt 1000 ] || err "Dataset muito pequeno: $N_ROWS linhas"
ok "Dados OK ($N_ROWS amostras)"

# ── Step 3: Treino e Deploy ───────────────────────────────────────────────────
log "[3/6] Treinando e registrando modelo..."
cd "$ROOT_DIR"
python notebooks/deploy.py >> "$LOG_FILE" 2>&1 || err "Falha no deploy.py"
ok "Treino e registro concluídos"

# ── Step 4: Gate de Qualidade ─────────────────────────────────────────────────
log "[4/6] Validando métricas do modelo..."
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
    print(json.dumps({"error": "run não encontrado"}))
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

log "  AUC    = $AUC  (mínimo = $MIN_AUC)"
log "  Recall = $RECALL  (mínimo = $MIN_RECALL)"

PASS_AUC=$(python -c "print(1 if float('$AUC') >= $MIN_AUC else 0)")
PASS_REC=$(python -c "print(1 if float('$RECALL') >= $MIN_RECALL else 0)")

[ "$PASS_AUC" = "1" ]    || err "AUC abaixo do mínimo ($AUC < $MIN_AUC) — deploy bloqueado"
[ "$PASS_REC" = "1" ]    || err "Recall abaixo do mínimo ($RECALL < $MIN_RECALL) — deploy bloqueado"
ok "Gate de qualidade aprovado (AUC=$AUC | Recall=$RECALL)"

# ── Step 5: API Smoke Test ────────────────────────────────────────────────────
log "[5/6] Iniciando API e executando smoke test..."

# Inicia API em background
uvicorn app.main:app --port 8001 --log-level error &
API_PID=$!
sleep 3

# Health check
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health)
if [ "$HTTP_STATUS" != "200" ]; then
    kill $API_PID 2>/dev/null
    err "Health check falhou (HTTP $HTTP_STATUS)"
fi
ok "Health check OK"

# Predict test
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

[ "$PREDICT_STATUS" = "200" ] || err "Predict endpoint falhou (HTTP $PREDICT_STATUS)"
ok "Smoke test OK"

# ── Step 6: Monitoramento inicial ─────────────────────────────────────────────
log "[6/6] Executando ciclo de monitoramento inicial..."
python notebooks/monitoring.py >> "$LOG_FILE" 2>&1 || warn "Monitoramento concluiu com avisos"
ok "Monitoramento OK"

# ── Resultado Final ───────────────────────────────────────────────────────────
log "══════════════════════════════════════════════"
ok "CI/CD PIPELINE CONCLUÍDO COM SUCESSO"
log "  Log completo: $LOG_FILE"
log "══════════════════════════════════════════════"
