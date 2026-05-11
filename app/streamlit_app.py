"""
streamlit_app.py — Visual churn prediction interface

Start:
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
from src.features import build_features

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title='Bank Churn Predictor',
    page_icon='🏦',
    layout='wide',
)

# ── Load model (cached once) ─────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return ChurnPredictor()

predictor = load_predictor()

# ── Title ───────────────────────────────────────────────────────────────────
st.title('🏦 Bank Customer Churn Predictor')
st.caption(f'Model: `{predictor.source}` · Threshold: `{predictor.threshold}`')
st.divider()

# ── Layout: form + result ───────────────────────────────────────────────────
col_form, col_result = st.columns([1, 1], gap='large')

with col_form:
    st.subheader('Customer Data')

    c1, c2 = st.columns(2)
    with c1:
        credit_score    = st.slider('Credit Score', 300, 850, 650)
        age             = st.slider('Age', 18, 92, 42)
        tenure          = st.slider('Client tenure (years)', 0, 10, 5)
        num_products    = st.selectbox('Number of Products', [1, 2, 3, 4], index=1)
        has_cr_card     = st.toggle('Has credit card', value=True)
        is_active       = st.toggle('Active member', value=True)

    with c2:
        balance         = st.number_input('Balance (€)', 0, 300000, 50000, step=5000)
        salary          = st.number_input('Estimated salary (€)', 10000, 250000, 80000, step=5000)
        geography       = st.selectbox('Country', ['France', 'Germany', 'Spain'])
        gender          = st.selectbox('Gender', ['Male', 'Female'])

    predict_btn = st.button('🔮 Predict Churn', use_container_width=True, type='primary')

# ── Prediction result ────────────────────────────────────────────────────────
with col_result:
    st.subheader('Prediction Result')

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

        # Color and icon by risk level
        if risk == 'high':
            color, icon, label = '#d32f2f', '🔴', 'HIGH CHURN RISK'
        elif risk == 'medium':
            color, icon, label = '#f57c00', '🟡', 'MODERATE RISK'
        else:
            color, icon, label = '#388e3c', '🟢', 'LOW RISK'

        # Main card
        st.markdown(f"""
        <div style="
            background:{color}18; border:2px solid {color};
            border-radius:12px; padding:24px; text-align:center;
        ">
            <div style="font-size:48px">{icon}</div>
            <div style="font-size:22px; font-weight:700; color:{color}">{label}</div>
            <div style="font-size:52px; font-weight:800; color:{color}; margin:8px 0">{pct}%</div>
            <div style="color:#555; font-size:14px">churn probability</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('')

        # Progress bar
        st.progress(prob, text=f'Probability: {prob:.4f}')

        # Secondary metrics
        m1, m2, m3 = st.columns(3)
        m1.metric('Prediction', '⚠ Churn' if flag else '✓ Retained')
        m2.metric('Risk level', risk.capitalize())
        m3.metric('Threshold', f'{result["threshold_used"]:.2f}')

        # Identified risk factors
        st.markdown('**Identified risk factors:**')
        risks = []
        if age >= 50:            risks.append(f'• Older age ({age} years)')
        if not is_active:        risks.append('• Inactive customer')
        if num_products == 1:    risks.append('• Only 1 product')
        if balance > 100000 and not is_active:
                                 risks.append(f'• High balance without engagement (€{balance:,.0f})')
        if geography == 'Germany': risks.append('• Country: Germany (historically higher churn)')
        if credit_score < 500:   risks.append(f'• Low credit score ({credit_score})')

        if risks:
            for r in risks:
                st.markdown(r)
        else:
            st.markdown('• No critical risk factors identified')

    else:
        st.info('Fill in the customer data and click **Predict Churn**.')

# ── Batch prediction section ─────────────────────────────────────────────────
st.divider()
st.subheader('📋 Batch Prediction')
st.caption('Paste multiple customer records (CSV) or use the example below.')

EXAMPLE_CSV = """CreditScore,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary,Geography,Gender
400,55,2,180000,1,0,0,60000,Germany,Female
800,30,8,50000,2,1,1,120000,France,Male
620,45,4,95000,1,1,0,75000,Spain,Female
750,28,6,0,2,1,1,95000,France,Male"""

csv_input = st.text_area('CSV data:', value=EXAMPLE_CSV, height=140)

if st.button('🔮 Predict Batch', use_container_width=False):
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
                'Country':      row['Geography'],
                'Gender':       row['Gender'],
                'Age':          int(row['Age']),
                'Products':     int(row['NumOfProducts']),
                'Active':       '✓' if row['IsActiveMember'] else '✗',
                'Churn Prob.':  f"{r['churn_probability']:.2%}",
                'Risk':         r['risk_level'].capitalize(),
                'Flag':         '⚠ Churn' if r['churn_flag'] else '✓ Retained',
            })

        df_out = pd.DataFrame(rows)

        def _color_risk(val):
            if val == 'High':   return 'background-color:#ffebee; color:#c62828'
            if val == 'Medium': return 'background-color:#fff8e1; color:#e65100'
            return 'background-color:#e8f5e9; color:#2e7d32'

        st.dataframe(
            df_out.style.map(_color_risk, subset=['Risk']),
            use_container_width=True,
        )

        n_churn = sum(1 for r in rows if r['Flag'] == '⚠ Churn')
        st.metric('Customers at churn risk', f'{n_churn} / {len(rows)}')

    except Exception as e:
        st.error(f'Error processing CSV: {e}')
