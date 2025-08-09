# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

st.set_page_config(page_title="Car Price Estimator (2025)", layout="centered")


st.markdown("""
<style>
/* Background Image */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1586375103528-8bc247aa8110?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjYzfHxjYXJzfGVufDB8fDB8fHww");
    background-size: cover;
    background-position: right;
    background-repeat: no-repeat;
}

/* Top-right Header */
#top-header {
    position: fixed;
    top: 80px;  /* 👈 Yahi line hai jo update karni thi */
    right: 20px;
    background-color: rgba(0,0,0,0.5);
    padding: 14px 18px;
    border-radius: 8px;
    color: white;
    font-size: 18px;
    font-weight: bold;
    z-index: 90;
}

/* Bottom-left Footer */
#bottom-footer {
    position: fixed;
    bottom: 10px;
    left:300px;
    background-color: rgba(0,0,0,0.5);
    padding: 6px 14px;
    border-radius: 6px;
    color: white;
    font-size: 14px;
    z-index: 100;
}
</style>

<div id="top-header">Respected Sir Shahzaib & Sir Ali Hamza</div>
<div id="bottom-footer">Developed by Faraz Hussain</div>
""", unsafe_allow_html=True)

# --------- Helper functions (same logic as backend) ----------
def extract_number(value):
    """Extract first numeric occurrence from a string; return np.nan if none."""
    if pd.isnull(value):
        return np.nan
    nums = re.findall(r"[\d.]+", str(value))
    if nums:
        try:
            return float(nums[0])
        except:
            return np.nan
    return np.nan

def clean_price_display(price):
    """Format numeric price to human readable with $ and commas."""
    try:
        p = float(price)
        return f"${p:,.0f}"
    except:
        return str(price)

# --------- Load model, encoders, feature order ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_model.pkl")
    encoders = joblib.load("car_price_encoders.pkl")
    feature_order = joblib.load("car_price_features.pkl")
    return model, encoders, feature_order

try:
    model, encoders, FEATURE_ORDER = load_artifacts()
except Exception as e:
    st.error("Model files not found or failed to load. Make sure these files are in the app folder:\n"
             "car_price_model.pkl, car_price_encoders.pkl, car_price_features.pkl")
    st.stop()

# --------- Sidebar: Optional sample data preview & instructions ----------
st.sidebar.title("Instructions")
st.sidebar.markdown("""
1. Fill car specs on the left.  
2. Click **Predict Price**.  
3. Make sure backend files (`.pkl`) are in same folder.  
""")
if st.sidebar.button("Show sample dataset (first 10 rows) if exists"):
    try:
        sample_df = pd.read_csv("cars dataset 2025.csv", encoding="latin1")
        st.sidebar.dataframe(sample_df.head(10))
    except Exception as e:
        st.sidebar.write("Sample CSV not found or can't be read.")

# --------- Main UI ----------
st.title("🚗 Car Price Estimator — 2025")
st.markdown("Enter car specs (ya model ke approximate numbers) — app will predict estimated price.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("Company Name (e.g., TOYOTA, BMW)", value="Toyota")
        car_name = st.text_input("Car Model Name (optional)", value="Corolla")
        engine = st.text_input("Engine Type / Name (e.g., V6, I4)", value="I4")
        cc_raw = st.text_input("CC / Battery Capacity (e.g., 1998 cc or 60 kWh)", value="1998 cc")
        hp_raw = st.text_input("HorsePower (e.g., 150 hp)", value="150 hp")
        speed_raw = st.text_input("Top Speed (e.g., 220 km/h)", value="220 km/h")

    with col2:
        perf_raw = st.text_input("0-100 km/h (e.g., 8.5 sec)", value="8.5 sec")
        torque_raw = st.text_input("Torque (e.g., 240 Nm)", value="240 Nm")
        seats = st.number_input("Seats", min_value=1, max_value=10, value=5, step=1)
        fuel_type = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric", "Hybrid", "plug in hyrbrid", "Other"], index=0)
        predict_button = st.form_submit_button("Predict Price")

# --------- Prepare single-row dataframe for prediction ----------
def prepare_input_row():
    row = {}
    # Numeric extraction
    row["CC/Battery Capacity"] = extract_number(cc_raw)
    row["HorsePower"] = extract_number(hp_raw)
    row["Total Speed"] = extract_number(speed_raw)
    row["Performance(0 - 100 )KM/H"] = extract_number(perf_raw)
    row["Torque"] = extract_number(torque_raw)
    # Seats numeric
    row["Seats"] = seats

    # Categorical fields: encode using label encoders if possible; if unseen, add a special value
    # We must match columns used during training
    # The backend encoded: "Company Names", "Cars Names", "Engines", "Fuel Types"
    # If unseen category appears, we handle by adding it to encoder classes dynamically (map to -1)
    def safe_encode(col_name, value):
        le = encoders.get(col_name)
        if le is None:
            return 0
        try:
            return int(le.transform([value])[0])
        except Exception:
            # unseen label -> try to expand classes (not modifying original encoder permanently)
            # best simple fallback: map to a new integer not used by encoder (e.g., -1)
            return -1

    row["Company Names"] = safe_encode("Company Names", company_name)
    row["Cars Names"] = safe_encode("Cars Names", car_name)
    row["Engines"] = safe_encode("Engines", engine)
    row["Fuel Types"] = safe_encode("Fuel Types", fuel_type)

    # Create DataFrame in the same column order expected by model
    df_row = pd.DataFrame([row])
    # Ensure all FEATURE_ORDER columns exist; if missing fill with 0/NaN
    for col in FEATURE_ORDER:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[FEATURE_ORDER]
    return df_row

# --------- Prediction and Display ----------
if predict_button:
    input_df = prepare_input_row()

    # Check for NaNs in numeric important fields
    if input_df[["CC/Battery Capacity", "HorsePower", "Total Speed", "Performance(0 - 100 )KM/H", "Torque"]].isnull().any(axis=None):
        st.warning("Kuch numeric fields missing ya wrong format — ensure CC, HorsePower, Top Speed, Acceleration, Torque me numeric values hain (e.g., '220 km/h', '150 hp'). Prediction may be less accurate.")
    try:
        pred = model.predict(input_df)[0]
        st.success("✅ Estimated Price: " + clean_price_display(pred))
        st.write("**Raw value:**", pred)

        # Extra: show a small confidence-ish range using simple tree-ensemble deviation (not true CI)
        try:
            # if RandomForest, we can get per-estimator predictions
            all_preds = np.asarray([est.predict(input_df)[0] for est in model.estimators_])
            low = np.percentile(all_preds, 10)
            high = np.percentile(all_preds, 90)
            st.info(f"Estimated range (10th-90th percentile across trees): {clean_price_display(low)} — {clean_price_display(high)}")
        except Exception:
            pass

        # Show input summary
        with st.expander("Input summary"):
            st.write(input_df.T)

        # Optional: Suggest similar priced cars from sample CSV if available
        try:
            df_full = pd.read_csv("cars dataset 2025.csv", encoding="latin1")
            # clean Cars Prices like backend quickly
            def _clean_price(p):
                if pd.isnull(p): return np.nan
                s = str(p).replace("$","").replace(",","")
                if "-" in s:
                    parts = s.split("-")
                    try:
                        return (float(parts[0]) + float(parts[1]))/2
                    except:
                        return np.nan
                try:
                    return float(s)
                except:
                    return np.nan
            df_full["CleanPrice"] = df_full["Cars Prices"].apply(_clean_price)
            sample_close = df_full.iloc[(df_full["CleanPrice"] - pred).abs().argsort()[:5]][["Company Names","Cars Names","Cars Prices","HorsePower"]]
            st.write("Similar priced models (from dataset):")
            st.dataframe(sample_close.reset_index(drop=True))
        except Exception:
            pass

    except Exception as e:
        st.error("Prediction failed: " + str(e))

st.markdown("---")
st.caption("deploy on streamlit.")
