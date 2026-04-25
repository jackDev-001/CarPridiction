import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; font-weight: 800; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.8; margin-top: 0.5rem; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-card p  { font-size: 0.85rem; opacity: 0.85; margin: 0; }

    .price-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.35);
    }
    .price-box h2 { font-size: 3rem; font-weight: 900; margin: 0; }
    .price-box p  { font-size: 1rem; opacity: 0.9; margin-top: 0.3rem; }

    .info-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        color: #1a1a2e;
        font-weight: 600;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Sample Dataset (realistic Indian used car data) ────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 800

    brands = ["Maruti", "Hyundai", "Honda", "Tata", "Toyota",
              "Ford", "Volkswagen", "BMW", "Audi", "Mercedes"]
    fuels  = ["Petrol", "Diesel", "CNG", "Electric"]
    trans  = ["Manual", "Automatic"]
    owners = ["First Owner", "Second Owner", "Third Owner"]
    sellers= ["Individual", "Dealer", "Trustmark Dealer"]

    brand       = np.random.choice(brands, n)
    fuel        = np.random.choice(fuels,  n, p=[0.5, 0.35, 0.1, 0.05])
    transmission= np.random.choice(trans,  n, p=[0.65, 0.35])
    owner       = np.random.choice(owners, n, p=[0.6, 0.3, 0.1])
    seller_type = np.random.choice(sellers,n, p=[0.4, 0.4, 0.2])
    year        = np.random.randint(2005, 2024, n)
    km_driven   = np.random.randint(5000, 200000, n)
    mileage     = np.random.uniform(10, 35, n).round(1)
    engine      = np.random.choice([800,1000,1200,1400,1500,1600,1800,2000,2500,3000], n)
    max_power   = np.random.uniform(50, 350, n).round(1)
    seats       = np.random.choice([2, 4, 5, 6, 7, 8], n, p=[0.02,0.02,0.6,0.05,0.25,0.06])

    # Price logic with realistic factors
    base = {
        "Maruti":500_000,"Hyundai":600_000,"Honda":700_000,"Tata":550_000,
        "Toyota":750_000,"Ford":600_000,"Volkswagen":800_000,
        "BMW":3_000_000,"Audi":3_500_000,"Mercedes":4_000_000
    }
    price = np.array([base[b] for b in brand], dtype=float)
    price *= (1 + (year - 2005) * 0.04)
    price *= (1 - km_driven / 1_000_000)
    price *= np.where(fuel == "Diesel", 1.15, 1.0)
    price *= np.where(fuel == "Electric", 1.3, 1.0)
    price *= np.where(transmission == "Automatic", 1.1, 1.0)
    price *= np.where(owner == "First Owner", 1.0,
              np.where(owner == "Second Owner", 0.82, 0.68))
    price *= (1 + max_power / 2000)
    price += np.random.normal(0, 30_000, n)
    price = np.clip(price, 50_000, 15_000_000)

    df = pd.DataFrame({
        "brand": brand, "year": year, "km_driven": km_driven,
        "fuel": fuel, "transmission": transmission, "owner": owner,
        "seller_type": seller_type, "mileage": mileage,
        "engine": engine, "max_power": max_power, "seats": seats,
        "selling_price": price.astype(int)
    })
    return df

# ─── Train Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df, model_name="Random Forest"):
    encoders = {}
    df_enc = df.copy()
    cat_cols = ["brand","fuel","transmission","owner","seller_type"]
    for col in cat_cols:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col])
        encoders[col] = le

    X = df_enc.drop("selling_price", axis=1)
    y = df_enc["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression()
    }
    model = models[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    return model, encoders, X.columns.tolist(), mae, r2, X_test, y_test, preds

# ─── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🚗 Car Price Predictor</h1>
    <p>ML-powered used car valuation — built with Random Forest & Gradient Boosting</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Data ──────────────────────────────────────────────────────────────────
df = generate_dataset()

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    model_choice = st.selectbox("Choose ML Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    st.markdown("---")
    st.markdown("## 🚘 Car Details")

    brand        = st.selectbox("Brand",        sorted(df["brand"].unique()))
    year         = st.slider("Year of Manufacture", 2005, 2024, 2018)
    km_driven    = st.number_input("KM Driven", 500, 300000, 30000, step=1000)
    fuel         = st.selectbox("Fuel Type",    df["fuel"].unique().tolist())
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner        = st.selectbox("Ownership",    ["First Owner", "Second Owner", "Third Owner"])
    seller_type  = st.selectbox("Seller Type",  ["Individual", "Dealer", "Trustmark Dealer"])
    mileage      = st.slider("Mileage (kmpl)", 8.0, 40.0, 18.0, step=0.5)
    engine       = st.selectbox("Engine (CC)",  [800,1000,1200,1400,1500,1600,1800,2000,2500,3000])
    max_power    = st.slider("Max Power (bhp)", 40.0, 400.0, 90.0, step=5.0)
    seats        = st.selectbox("Seats",        [2,4,5,6,7,8])

    predict_btn = st.button("🔮 Predict Price", use_container_width=True, type="primary")

# ─── Train ───────────────────────────────────────────────────────────────────────
with st.spinner(f"Training {model_choice}..."):
    model, encoders, feature_cols, mae, r2, X_test, y_test, preds = train_model(df, model_choice)

# ─── Tabs ────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📈 Data Insights"])

# ── Tab 1: Prediction ────────────────────────────────────────────────────────────
with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{len(df):,}</h3><p>Training Samples</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{r2*100:.1f}%</h3><p>Model Accuracy (R²)</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>₹{mae/1000:.0f}K</h3><p>Avg Error (MAE)</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if predict_btn:
        input_dict = {
            "brand": encoders["brand"].transform([brand])[0],
            "year": year,
            "km_driven": km_driven,
            "fuel": encoders["fuel"].transform([fuel])[0],
            "transmission": encoders["transmission"].transform([transmission])[0],
            "owner": encoders["owner"].transform([owner])[0],
            "seller_type": encoders["seller_type"].transform([seller_type])[0],
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        predicted_price = model.predict(input_df)[0]
        predicted_price = max(50_000, predicted_price)

        low  = int(predicted_price * 0.92)
        high = int(predicted_price * 1.08)

        st.markdown(f"""
        <div class="price-box">
            <p>Estimated Market Price</p>
            <h2>₹ {predicted_price:,.0f}</h2>
            <p>Range: ₹{low:,.0f} — ₹{high:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">✅ Prediction made using <b>{model_choice}</b> | Model R² Score: {r2*100:.1f}%</div>', unsafe_allow_html=True)

        # Input summary
        st.markdown("### 📋 Your Car Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Brand","Year","KM Driven","Fuel","Transmission","Owner","Mileage","Engine","Max Power","Seats"],
            "Value": [brand, year, f"{km_driven:,} km", fuel, transmission, owner, f"{mileage} kmpl", f"{engine} CC", f"{max_power} bhp", seats]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    else:
        st.info("👈 Fill in the car details in the sidebar and click **Predict Price**")

# ── Tab 2: Model Performance ─────────────────────────────────────────────────────
with tab2:
    st.markdown("### Model Evaluation Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("R² Score",  f"{r2*100:.2f}%", help="Higher is better. 100% = perfect.")
        st.metric("Mean Absolute Error", f"₹ {mae:,.0f}", help="Lower is better.")
    with c2:
        st.metric("Model Used", model_choice)
        st.metric("Test Set Size", f"{len(X_test)} samples")

    st.markdown("### Actual vs Predicted Prices")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test/1e5, preds/1e5, alpha=0.4, color="#667eea", s=20)
    lims = [min(y_test.min(), preds.min())/1e5, max(y_test.max(), preds.max())/1e5]
    ax.plot(lims, lims, 'r--', lw=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (₹ Lakhs)")
    ax.set_ylabel("Predicted Price (₹ Lakhs)")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Feature importance (for tree models)
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.markdown("### Feature Importance")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        importances.plot(kind="barh", ax=ax2, color="#764ba2")
        ax2.set_title("Feature Importance")
        ax2.set_xlabel("Importance Score")
        ax2.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

# ── Tab 3: Data Insights ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Dataset Overview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Price Distribution by Fuel Type")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        df.groupby("fuel")["selling_price"].median().div(1e5).sort_values().plot(
            kind="bar", ax=ax3, color=["#f5576c","#667eea","#43e97b","#f093fb"], edgecolor="white"
        )
        ax3.set_ylabel("Median Price (₹ Lakhs)")
        ax3.set_title("Median Price by Fuel")
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=0)
        ax3.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with c2:
        st.markdown("### Price vs Year (Trend)")
        trend = df.groupby("year")["selling_price"].median().div(1e5)
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.plot(trend.index, trend.values, color="#667eea", lw=2.5, marker='o', markersize=4)
        ax4.fill_between(trend.index, trend.values, alpha=0.15, color="#667eea")
        ax4.set_ylabel("Median Price (₹ Lakhs)")
        ax4.set_title("Price Trend by Year")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("### Top Brands by Avg Price")
    brand_price = df.groupby("brand")["selling_price"].mean().div(1e5).sort_values(ascending=False)
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    brand_price.plot(kind="bar", ax=ax5, color="#0f3460", edgecolor="white")
    ax5.set_ylabel("Avg Price (₹ Lakhs)")
    ax5.set_title("Average Selling Price by Brand")
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=30, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:gray; font-size:13px;'>Built with ❤️ using Streamlit + Scikit-learn | Jack Ammunition</center>",
    unsafe_allow_html=True
)
