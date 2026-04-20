"""
streamlit_app.py — Interface visual de predição de churn

Iniciar:
  cd /Users/Shaiane/Projeto_MLOps_Bank_Churn
  source venv/bin/activate
  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.inference import ChurnPredictor

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title='Bank Churn Predictor',
    page_icon='🏦',
    layout='wide',
)

# ── Carregar modelo (uma vez, em cache) ───────────────────────────────────────
@st.cache_resource
def load_predictor():
    return ChurnPredictor()

predictor = load_predictor()

# ── Título ────────────────────────────────────────────────────────────────────
st.title('🏦 Bank Customer Churn Predictor')
st.caption(f'Modelo: `{predictor.source}` · Threshold: `{predictor.threshold}`')
st.divider()

# ── Layout: formulário + resultado ───────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap='large')

with col_form:
    st.subheader('Dados do Cliente')

    c1, c2 = st.columns(2)
    with c1:
        credit_score    = st.slider('Credit Score', 300, 850, 650)
        age             = st.slider('Idade', 18, 92, 42)
        tenure          = st.slider('Tempo como cliente (anos)', 0, 10, 5)
        num_products    = st.selectbox('Nº de Produtos', [1, 2, 3, 4], index=1)
        has_cr_card     = st.toggle('Possui cartão de crédito', value=True)
        is_active       = st.toggle('Membro ativo', value=True)

    with c2:
        balance         = st.number_input('Saldo (€)', 0, 300000, 50000, step=5000)
        salary          = st.number_input('Salário estimado (€)', 10000, 250000, 80000, step=5000)
        geography       = st.selectbox('País', ['France', 'Germany', 'Spain'])
        gender          = st.selectbox('Gênero', ['Male', 'Female'])

    predict_btn = st.button('🔮 Prever Churn', use_container_width=True, type='primary')

# ── Montar features ───────────────────────────────────────────────────────────
def build_features(credit_score, age, tenure, balance, num_products,
                   has_cr_card, is_active, salary, geography, gender) -> dict:
    age_group_young  = int(age < 30)
    age_group_middle = int(30 <= age < 55)
    age_group_senior = int(age >= 55)
    engagement       = int(is_active) + int(has_cr_card) + min(num_products, 2) - 1
    balance_ratio    = round(balance / max(salary, 1), 4)
    products_year    = round(num_products / max(tenure, 1), 4)
    age_inactivity   = age * int(not is_active)

    return {
        'CreditScore':       credit_score,
        'Age':               age,
        'Tenure':            tenure,
        'Balance':           balance,
        'NumOfProducts':     num_products,
        'HasCrCard':         int(has_cr_card),
        'IsActiveMember':    int(is_active),
        'EstimatedSalary':   salary,
        'Geography_France':  int(geography == 'France'),
        'Geography_Germany': int(geography == 'Germany'),
        'Geography_Spain':   int(geography == 'Spain'),
        'Gender_Female':     int(gender == 'Female'),
        'Gender_Male':       int(gender == 'Male'),
        'AgeGroup_Middle':   age_group_middle,
        'AgeGroup_Senior':   age_group_senior,
        'AgeGroup_Young':    age_group_young,
        'AgeInactivity':     age_inactivity,
        'EngagementScore':   engagement,
        'BalanceSalaryRatio': balance_ratio,
        'ProductsPerYear':   products_year,
    }

# ── Resultado ─────────────────────────────────────────────────────────────────
with col_result:
    st.subheader('Resultado da Predição')

    if predict_btn:
        features = build_features(
            credit_score, age, tenure, balance, num_products,
            has_cr_card, is_active, salary, geography, gender
        )
        result = predictor.predict_single(features)

        prob       = result['churn_probability']
        flag       = result['churn_flag']
        risk       = result['risk_level']
        pct        = int(prob * 100)

        # Cor e ícone por nível de risco
        if risk == 'high':
            color, icon, label = '#d32f2f', '🔴', 'ALTO RISCO DE CHURN'
        elif risk == 'medium':
            color, icon, label = '#f57c00', '🟡', 'RISCO MODERADO'
        else:
            color, icon, label = '#388e3c', '🟢', 'BAIXO RISCO'

        # Card principal
        st.markdown(f"""
        <div style="
            background:{color}18; border:2px solid {color};
            border-radius:12px; padding:24px; text-align:center;
        ">
            <div style="font-size:48px">{icon}</div>
            <div style="font-size:22px; font-weight:700; color:{color}">{label}</div>
            <div style="font-size:52px; font-weight:800; color:{color}; margin:8px 0">{pct}%</div>
            <div style="color:#555; font-size:14px">probabilidade de churn</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('')

        # Barra de progresso
        st.progress(prob, text=f'Probabilidade: {prob:.4f}')

        # Métricas secundárias
        m1, m2, m3 = st.columns(3)
        m1.metric('Predição', '⚠ Churn' if flag else '✓ Retido')
        m2.metric('Nível de risco', risk.capitalize())
        m3.metric('Threshold', f'{result["threshold_used"]:.2f}')

        # Fatores de risco identificados
        st.markdown('**Fatores de risco identificados:**')
        risks = []
        if age >= 50:            risks.append(f'• Idade elevada ({age} anos)')
        if not is_active:        risks.append('• Cliente inativo')
        if num_products == 1:    risks.append('• Apenas 1 produto')
        if balance > 100000 and not is_active:
                                 risks.append(f'• Saldo alto sem engajamento (€{balance:,.0f})')
        if geography == 'Germany': risks.append('• País: Germany (maior taxa histórica de churn)')
        if credit_score < 500:   risks.append(f'• Credit score baixo ({credit_score})')

        if risks:
            for r in risks:
                st.markdown(r)
        else:
            st.markdown('• Nenhum fator de risco crítico identificado')

    else:
        st.info('Preencha os dados do cliente e clique em **Prever Churn**.')

# ── Seção: batch de clientes ──────────────────────────────────────────────────
st.divider()
st.subheader('📋 Predição em Lote')
st.caption('Cole dados de múltiplos clientes (CSV) ou use o exemplo abaixo.')

EXAMPLE_CSV = """CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography,Gender
400,55,2,180000,1,0,0,60000,Germany,Female
800,30,8,50000,2,1,1,120000,France,Male
620,45,4,95000,1,1,0,75000,Spain,Female
750,28,6,0,2,1,1,95000,France,Male"""

csv_input = st.text_area('Dados CSV:', value=EXAMPLE_CSV, height=140)

if st.button('🔮 Prever Lote', use_container_width=False):
    try:
        import io
        raw = pd.read_csv(io.StringIO(csv_input))
        rows = []
        for _, row in raw.iterrows():
            f = build_features(
                row['CreditScore'], row['Age'], row['Tenure'],
                row['Balance'], row['NumOfProducts'],
                bool(row['HasCrCard']), bool(row['IsActiveMember']),
                row['EstimatedSalary'], row['Geography'], row['Gender']
            )
            r = predictor.predict_single(f)
            rows.append({
                'País':        row['Geography'],
                'Gênero':      row['Gender'],
                'Idade':       int(row['Age']),
                'Produtos':    int(row['NumOfProducts']),
                'Ativo':       '✓' if row['IsActiveMember'] else '✗',
                'Prob. Churn': f"{r['churn_probability']:.2%}",
                'Risco':       r['risk_level'].capitalize(),
                'Flag':        '⚠ Churn' if r['churn_flag'] else '✓ Retido',
            })

        df_out = pd.DataFrame(rows)

        def _color_risk(val):
            if val == 'High':   return 'background-color:#ffebee; color:#c62828'
            if val == 'Medium': return 'background-color:#fff8e1; color:#e65100'
            return 'background-color:#e8f5e9; color:#2e7d32'

        st.dataframe(
            df_out.style.map(_color_risk, subset=['Risco']),
            use_container_width=True,
        )

        n_churn = sum(1 for r in rows if r['Flag'] == '⚠ Churn')
        st.metric('Clientes com risco de churn', f'{n_churn} / {len(rows)}')

    except Exception as e:
        st.error(f'Erro ao processar CSV: {e}')
