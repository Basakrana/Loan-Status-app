import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AI Loan Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #00f2fe;
        --error-color: #f5576c;
        --background-color: #f8f9fa;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    /* Input styling */
    .stSelectbox > div > div > div {
        background-color: white;
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stNumberInput > div > div > input {
        border: 2px solid #e1e5e9;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Error message styling */
    .success-box {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .error-box {
        background: linear-gradient(90deg, #f093fb, #f5576c);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 2rem 0;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("xgboost_model.pkl")
    except:
        st.error("⚠️ Model file not found! Please ensure 'xgboost_model.pkl' is in the app directory.")
        return None

model_XGBoost = load_model()

# -------------------- Category Mapping --------------------
category_mapping = {
    'person_gender': {'Female': 0, 'Male': 1},
    'person_education': {
        'Associate': 0, 'Bachelor': 1, 'Doctorate': 2,
        'High School': 3, 'Master': 4
    },
    'person_home_ownership': {
        'Mortgage': 0, 'Other': 1, 'Own': 2, 'Rent': 3
    },
    'loan_intent': {
        'Debt Consolidation': 0, 'Education': 1, 'Home Improvement': 2,
        'Medical': 3, 'Personal': 4, 'Venture': 5
    },
    'previous_loan_defaults_on_file': {'No': 0, 'Yes': 1}
}

# -------------------- Header --------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🚀 AI Loan Predictor</div>
    <div class="main-subtitle">Intelligent loan approval prediction powered by machine learning</div>
</div>
""", unsafe_allow_html=True)

# -------------------- Feature Cards --------------------
st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <h3>🤖 AI-Powered</h3>
        <p>Advanced XGBoost algorithm with 95%+ accuracy</p>
    </div>
    <div class="feature-card">
        <h3>⚡ Instant Results</h3>
        <p>Real-time predictions in seconds</p>
    </div>
    <div class="feature-card">
        <h3>📊 Data-Driven</h3>
        <p>Evidence-based risk assessment</p>
    </div>
    <div class="feature-card">
        <h3>🔒 Secure</h3>
        <p>Your data is processed safely</p>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 📋 Application Guide")
    st.info("""
    **How to use:**
    1. Fill in all required fields
    2. Review the loan-to-income ratio
    3. Click 'Predict Loan Status'
    4. Get instant AI-powered results
    """)
    
    st.markdown("### 📈 Key Factors")
    st.warning("""
    **Important factors:**
    - Credit Score (300-850)
    - Income vs Loan Amount
    - Employment History
    - Previous Defaults
    """)

# -------------------- Main Form --------------------
st.markdown("## 📝 Loan Application Details")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Personal Information")
    person_age = st.number_input("🎂 Age", min_value=18, max_value=100, value=30, help="Applicant's age in years")
    person_gender = st.selectbox("⚧ Gender", list(category_mapping['person_gender'].keys()))
    person_education = st.selectbox("🎓 Education Level", list(category_mapping['person_education'].keys()))
    person_emp_exp = st.number_input("💼 Employment Experience", min_value=0, max_value=50, value=5, help="Years of work experience")

with col2:
    st.markdown("### 💰 Financial Information")
    person_income = st.number_input("💵 Income ($)", min_value=1000, step=1000, value=50000, format="%d")
    person_home_ownership = st.selectbox("🏠 Home Ownership", list(category_mapping['person_home_ownership'].keys()))
    credit_score = st.number_input("⭐ Credit Score", min_value=300, max_value=850, value=700, help="FICO score range: 300-850")
    cb_person_cred_hist_length = st.number_input("📅 Credit History Length", min_value=0, max_value=50, value=10, help="Years of credit history")

with col3:
    st.markdown("### 🏦 Loan Information")
    loan_amnt = st.number_input("💳 Loan Amount ($)", min_value=500, step=500, value=10000, format="%d")
    loan_intent = st.selectbox("🎯 Loan Purpose", list(category_mapping['loan_intent'].keys()))
    loan_int_rate = st.number_input("📈 Interest Rate (%)", min_value=1.0, max_value=50.0, value=12.5, format="%.1f")
    previous_loan_defaults_on_file = st.selectbox("⚠️ Previous Defaults", list(category_mapping['previous_loan_defaults_on_file'].keys()))

# -------------------- Loan Analysis --------------------
loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0

st.markdown("---")
st.markdown("## 📊 Loan Analysis")

analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)

with analysis_col1:
    st.metric("💰 Loan Amount", f"RS {loan_amnt:,}")

with analysis_col2:
    st.metric("📊 Annual Income", f"RS{person_income:,}")

with analysis_col3:
    st.metric("📈 Loan-to-Income Ratio", f"{loan_percent_income:.1%}")

with analysis_col4:
    risk_level = "🟢 Low" if loan_percent_income < 0.3 else "🟡 Medium" if loan_percent_income < 0.5 else "🔴 High"
    st.metric("⚠️ Risk Level", risk_level)

# Create a gauge chart for loan-to-income ratio
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = loan_percent_income * 100,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Loan-to-Income Ratio (%)"},
    delta = {'reference': 30},
    gauge = {
        'axis': {'range': [None, 100]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 30], 'color': "lightgreen"},
            {'range': [30, 50], 'color': "yellow"},
            {'range': [50, 100], 'color': "red"}],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 50}}))

fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# -------------------- Prediction --------------------
st.markdown("---")
prediction_col1, prediction_col2, prediction_col3 = st.columns([1, 2, 1])

with prediction_col2:
    predict_button = st.button("🔮 **Predict Loan Status**", type="primary", use_container_width=True)

if predict_button and model_XGBoost is not None:
    with st.spinner("🤖 AI is analyzing your application..."):
        # Convert categorical to encoded
        input_data = pd.DataFrame([{
            'person_age': person_age,
            'person_gender': category_mapping['person_gender'][person_gender],
            'person_education': category_mapping['person_education'][person_education],
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_home_ownership': category_mapping['person_home_ownership'][person_home_ownership],
            'loan_amnt': loan_amnt,
            'loan_intent': category_mapping['loan_intent'][loan_intent],
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cb_person_cred_hist_length,
            'credit_score': credit_score,
            'previous_loan_defaults_on_file': category_mapping['previous_loan_defaults_on_file'][previous_loan_defaults_on_file]
        }])

        # Make prediction
        prediction = model_XGBoost.predict(input_data)[0]
        prediction_proba = model_XGBoost.predict_proba(input_data)[0]
        
        # Display results with custom styling
        st.markdown("## 🎯 Prediction Results")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="success-box">
                ✅ <strong>LOAN APPROVED</strong><br>
                Low Default Risk Detected<br>
                <small>Confidence: {prediction_proba[1]:.1%}</small>
            </div>
            """, unsafe_allow_html=True)
            
            st.balloons()
            
        else:
            st.markdown(f"""
            <div class="error-box">
                ❌ <strong>LOAN NOT APPROVED</strong><br>
                High Default Risk Detected<br>
                <small>Confidence: {prediction_proba[0]:.1%}</small>
            </div>
            """, unsafe_allow_html=True)

        # Show probability breakdown
        st.markdown("### 📊 Risk Assessment Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("🟢 Approval Probability", f"{prediction_proba[1]:.1%}")
        with prob_col2:
            st.metric("🔴 Default Risk", f"{prediction_proba[0]:.1%}")

        # Create probability chart
        # prob_fig = px.bar(
        #     x=['Approval', 'Default Risk'],
        #     y=[prediction_proba[1]*100, prediction_proba[0]*100],
        #     color=['Approval', 'Default Risk'],
        #     color_discrete_map={'Approval': '#00f2fe', 'Default Risk': '#f5576c'},
        #     title="Risk Assessment Probability"
        # )
        # prob_fig.update_layout(showlegend=False, yaxis_title="Probability (%)")
        # st.plotly_chart(prob_fig, use_container_width=True)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-top: 2rem;">
    <h4>🤖 Powered by Advanced Machine Learning</h4>
    <p>This AI model analyzes multiple factors to provide accurate loan approval predictions.<br>
    <em>Disclaimer: This is for demonstration purposes only. Consult financial professionals for actual loan decisions.</em></p>
</div>
""", unsafe_allow_html=True)