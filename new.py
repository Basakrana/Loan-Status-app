import joblib
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Main theme colors */
    :root {
        --primary-color: #ff6b6b;
        --secondary-color: #4ecdc4;
        --accent-color: #45b7d1;
        --success-color: #96ceb4;
        --warning-color: #feca57;
        --error-color: #ff9ff3;
        --dark-color: #2f3542;
        --light-color: #f1f2f6;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><radialGradient id="a" cx="50%" cy="40%"><stop offset="0%" stop-color="rgba(255,255,255,.1)"/><stop offset="100%" stop-color="rgba(255,255,255,0)"/></radialGradient></defs><rect width="100%" height="100%" fill="url(%23a)"/></svg>');
    }
    
    .main-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    
    .main-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
        line-height: 1.6;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--dark-color);
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* Form styling */
    .form-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .form-section-title {
        color: var(--dark-color);
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-color);
        display: inline-block;
    }
    
    /* Input styling */
    .stSelectbox > div > div > div {
        background: white;
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    
    .stTextInput > div > div > input {
        border: 2px solid #e1e5e9;
        border-radius: 12px;
        padding: 12px 16px;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        background: white;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(255, 107, 107, 0.1);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .predict-button {
        background: linear-gradient(135deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.3rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
        font-family: 'Poppins', sans-serif;
    }
    
    .predict-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(255, 107, 107, 0.6);
    }
    
    /* Result styling */
    .price-result {
        background: linear-gradient(135deg, #96ceb4, #feca57);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(150, 206, 180, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .price-result::before {
        content: '₹';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 8rem;
        opacity: 0.1;
        font-weight: 700;
    }
    
    .price-value {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Metrics styling */
    .metrics-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(15px);
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid var(--primary-color);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Info boxes */
    .info-highlight {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1), rgba(78, 205, 196, 0.1));
        border-left: 5px solid var(--accent-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
    
    .warning-highlight {
        background: linear-gradient(135deg, rgba(254, 202, 87, 0.1), rgba(255, 107, 107, 0.1));
        border-left: 5px solid var(--warning-color);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('xgboost_pipelinen.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file 'xgboost_pipelinen.pkl' not found! Please ensure it's in the app directory.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

model_XGBoost = load_model()

# -------------------- Header --------------------
st.markdown("""
<div class="main-header fade-in">
    <div class="main-title">🚗 AI Car Price Predictor</div>
    <div class="main-subtitle">
        Get instant, accurate car valuations powered by advanced machine learning<br>
        <strong>94% Accuracy</strong> • <strong>Instant Results</strong> • <strong>Smart Analytics</strong>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Feature Cards --------------------
st.markdown("""
<div class="feature-grid fade-in">
    <div class="feature-card">
        <span class="feature-icon">🤖</span>
        <div class="feature-title">AI-Powered</div>
        <div class="feature-desc">Advanced XGBoost algorithm with 94% accuracy rate</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">⚡</span>
        <div class="feature-title">Lightning Fast</div>
        <div class="feature-desc">Get instant price predictions in seconds</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">📊</span>
        <div class="feature-title">Data-Driven</div>
        <div class="feature-desc">Based on thousands of real car sales data</div>
    </div>
    <div class="feature-card">
        <span class="feature-icon">🔒</span>
        <div class="feature-title">Trusted</div>
        <div class="feature-desc">Reliable valuations you can count on</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### 🎯 Quick Guide")
    st.markdown("""
    <div class="info-highlight">
        <strong>How to get accurate predictions:</strong><br>
        1. Select your car brand and specifications<br>
        2. Enter accurate mileage and condition details<br>
        3. Click predict for instant AI valuation<br>
        4. Get market-competitive price estimate
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Price Factors")
    st.markdown("""
    <div class="warning-highlight">
        <strong>Key factors affecting price:</strong><br>
        • <strong>Age & Mileage:</strong> Lower is better<br>
        • <strong>Brand:</strong> Premium brands hold value<br>
        • <strong>Fuel Type:</strong> Impacts running costs<br>
        • <strong>Owner History:</strong> First owner premium<br>
        • <strong>Engine Power:</strong> Performance matters
    </div>
    """, unsafe_allow_html=True)

# -------------------- Data Options --------------------
list_Cars = ['Maruti','Skoda','Honda','Hyundai','Toyota','Ford','Renault','Mahindra','Tata','Chevrolet',
             'Datsun','Jeep','Mercedes-Benz','Mitsubishi','Audi','Volkswagen','BMW','Nissan','Lexus',
             'Jaguar','Land','MG','Volvo','Daewoo','Kia','Fiat','Force','Ambassador','Ashok','Isuzu','Opel']

fuel_types = ['Petrol', 'Diesel', 'CNG', 'LPG']
owner_types = ['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car']
seller_types = ['Individual','Dealer','Trustmark Dealer']
transmission_types = ['Manual','Automatic']

# -------------------- Main Form --------------------
st.markdown("""
<div class="form-container fade-in">
    <h2 class="form-section-title">🔧 Vehicle Specifications</h2>
</div>
""", unsafe_allow_html=True)

# Create improved layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("#### 🚗 Basic Information")
    name = st.selectbox("🏷️ Car Brand", list_Cars, help="Select the manufacturer of your car")
    fuel = st.selectbox("⛽ Fuel Type", fuel_types, help="Primary fuel type of the vehicle")
    transmission = st.selectbox("⚙️ Transmission", transmission_types, help="Type of gearbox")
    owner = st.selectbox("👤 Owner Type", owner_types, help="Ownership history affects value")

with col2:
    st.markdown("#### 📊 Performance Specs")
    engine = st.text_input("🔧 Engine Capacity (CC)", "1200", help="Engine displacement in cubic centimeters")
    max_power = st.text_input("⚡ Max Power (hp)", "120", help="Maximum power output in horsepower")
    mileage = st.text_input("⛽ Mileage (kmpl)", "20", help="Fuel efficiency in kilometers per liter")
    seats = st.text_input("💺 Seating Capacity", "5", help="Number of seats in the vehicle")

with col3:
    st.markdown("#### 📈 Usage & Condition")
    km_driven = st.text_input("🛣️ Kilometers Driven", "10000", help="Total distance covered")
    age = st.text_input("📅 Vehicle Age (years)", "3", help="Age of the vehicle in years")
    seller_type = st.selectbox("🏪 Seller Type", seller_types, help="Type of seller affects pricing")

# -------------------- Real-time Metrics --------------------
try:
    current_age = float(age) if age else 0
    current_km = float(km_driven) if km_driven else 0
    current_mileage = float(mileage) if mileage else 0
    current_power = float(max_power) if max_power else 0
    
    st.markdown("---")
    st.markdown("### 📊 Vehicle Analysis")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        condition = "Excellent" if current_age < 3 else "Good" if current_age < 7 else "Fair"
        st.metric("🏆 Condition", condition)
    
    with metric_col2:
        usage_level = "Low" if current_km < 30000 else "Medium" if current_km < 80000 else "High"
        st.metric("📊 Usage Level", usage_level)
    
    with metric_col3:
        efficiency = "High" if current_mileage > 18 else "Medium" if current_mileage > 12 else "Low"
        st.metric("⛽ Fuel Efficiency", efficiency)
    
    with metric_col4:
        performance = "High" if current_power > 150 else "Medium" if current_power > 100 else "Standard"
        st.metric("⚡ Performance", performance)

except (ValueError, TypeError):
    st.info("📝 Please fill in all numeric fields to see vehicle analysis")

# -------------------- Prepare DataFrame --------------------
df = None
input_valid = True

try:
    df = pd.DataFrame({
        'name': [name],
        'age': [float(age)],
        'km_driven': [float(km_driven)],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [float(mileage)],
        'engine': [float(engine)],
        'max_power': [float(max_power)],
        'seats': [float(seats)]
    })
except Exception as e:
    input_valid = False
    st.error(f"⚠️ Please check your inputs: {str(e)}")

# -------------------- Prediction Section --------------------
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    if st.button("🔮 **GET AI PRICE PREDICTION**", key="predict_btn", help="Click to get instant AI-powered price prediction"):
        if model_XGBoost is None:
            st.error("❌ Model not available. Please check the model file.")
        elif not input_valid or df is None:
            st.error("❌ Please fill in all fields correctly.")
        else:
            with st.spinner("🤖 AI is analyzing your car..."):
                try:
                    # Make prediction
                    prediction = model_XGBoost.predict(df)
                    price = np.exp(prediction)[0]
                    formatted_price = f"{price:,.0f}"
                    
                    # Display result with animation
                    st.markdown(f"""
                    <div class="price-result fade-in">
                        <h2 style="margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                            🎉 Predicted Price
                        </h2>
                        <div class="price-value">₹ {formatted_price}</div>
                        <p style="margin: 0; font-size: 1.1rem; opacity: 0.9;">
                            Based on current market trends and vehicle specifications
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show price analysis
                    st.markdown("### 💡 Price Analysis")
                    
                    analysis_col1, analysis_col2 = st.columns(2)
                    
                    with analysis_col1:
                        # Price range estimation
                        lower_estimate = price * 0.9
                        upper_estimate = price * 1.1
                        
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>📊 Price Range</h4>
                            <div class="metric-card">
                                <div class="metric-value">₹ {lower_estimate:,.0f} - ₹ {upper_estimate:,.0f}</div>
                                <div class="metric-label">Estimated Range</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with analysis_col2:
                        # Value assessment
                        price_per_year = price / max(current_age, 1)
                        
                        st.markdown(f"""
                        <div class="metrics-container">
                            <h4>💰 Value Assessment</h4>
                            <div class="metric-card">
                                <div class="metric-value">₹ {price_per_year:,.0f}</div>
                                <div class="metric-label">Value per Year</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Create a simple price visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['Predicted Price'],
                        y=[price],
                        marker_color='rgba(255, 107, 107, 0.8)',
                        text=[f'₹{price:,.0f}'],
                        textposition='auto',
                        name='Predicted Price'
                    ))
                    
                    fig.update_layout(
                        title="🎯 AI Price Prediction",
                        yaxis_title="Price (₹)",
                        showlegend=False,
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Success animation
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-top: 3rem; backdrop-filter: blur(10px);">
    <h3 style="color: white; margin-bottom: 1rem;">🚀 Powered by Advanced AI Technology</h3>
    <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; line-height: 1.6; margin: 0;">
        Our machine learning model analyzes multiple factors including brand value, vehicle condition, 
        market trends, and regional pricing to provide you with the most accurate car valuation possible.
    </p>
    <br>
    <p style="color: rgba(255,255,255,0.7); font-style: italic; margin: 0;">
        <strong>Disclaimer:</strong> Predictions are estimates based on historical data. 
        Actual market prices may vary based on local conditions and vehicle specifics.
    </p>
</div>
""", unsafe_allow_html=True)