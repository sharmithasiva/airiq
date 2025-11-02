import streamlit as st

st.set_page_config(page_title="Air Quality Dashboard", layout="wide", page_icon="🌤")

import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import joblib
import numpy as np
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download


# -----------------------------
# CONFIG
# -----------------------------
OPENWEATHER_API_KEY = "9329b83f589b5f626195a675dc0f080d"

AQI_CONVERSION = {
    1: (25, "Good", "Air quality is good. Safe for outdoor activities.", "#4CAF50", "☀"),
    2: (75, "Satisfactory", "Air quality is acceptable. Minor risk for sensitive groups.", "#8BC34A", "🌤"),
    3: (150, "Moderate", "Sensitive groups should reduce outdoor exertion.", "#FFC107", "⛅"),
    4: (250, "Poor", "Health effects possible. Limit outdoor activity.", "#FF9800", "🌥"),
    5: (400, "Severe", "Serious health risk. Stay indoors.", "#F44336", "🌪"),
}

POLLUTANT_CARD_COLOR = "#1976D2"
FEATURE_COLUMNS = ["Year","Month","Day","Hour","DayOfWeek","City_Encoded","PM2_5","PM10","NO2","SO2","CO","O3"]

# -----------------------------
# REQUESTS SESSION WITH RETRIES
# -----------------------------
def make_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session = make_session()

def safe_get_json(url, params=None, timeout=10):
    try:
        resp = session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e} - URL: {url} - params: {params}")
        return None

def get_coordinates(city):
    url = "https://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city, "limit": 1, "appid": OPENWEATHER_API_KEY}
    resp = safe_get_json(url, params=params)
    if not resp:
        return None, None
    if isinstance(resp, list) and len(resp) > 0:
        return resp[0].get("lat"), resp[0].get("lon")
    return None, None

def get_air_quality(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    return safe_get_json(url, params=params)

def get_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    return safe_get_json(url, params=params)

# -----------------------------
# PAGE CONFIG & STYLE
# -----------------------------

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0f7fa, #b2ebf2);
    color: #000;
}
.card {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    border-left: 6px solid #00BCD4;
}
.card h2 {
    background-color: #00BCD4;
    color: white;
    padding: 10px;
    border-radius: 10px 10px 0 0;
    margin-top: -20px;
    margin-bottom: 15px;
}
.aqi-card {
    text-align: center;
    border-radius: 15px;
    padding: 20px;
    color: white;
    margin-bottom: 20px;
}
.pollutant-card {
    background: """ + POLLUTANT_CARD_COLOR + """;
    color: white;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    margin-bottom: 10px;
}
.weather-card {
    padding: 15px;
    border-radius: 12px;
    background: rgba(33,150,243,0.1);
    text-align: center;
    margin-bottom: 20px;
}
.stButton>button {
    color: white !important;
    background-color: #00BCD4 !important;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🌤 Air Quality Prediction & Advisory System")
city = st.text_input("Enter City Name", "Delhi")

# ---------- AQI SECTION ----------
if st.button("Get Air Quality Data"):
    lat, lon = get_coordinates(city)
    if lat is None or lon is None:
        st.error(f"⚠ Could not get coordinates for '{city}'.")
    else:
        current_data = get_air_quality(lat, lon)
        if not current_data or "list" not in current_data or len(current_data["list"]) == 0:
            st.error("⚠ Could not fetch air quality data.")
        else:
            aqi_index = current_data["list"][0]["main"]["aqi"]
            aqi_value, category, message, color, icon = AQI_CONVERSION.get(aqi_index, (0, "Unknown", "", "#888", "❓"))
            components = current_data["list"][0].get("components", {})

            weather_data = get_weather(lat, lon)
            temp = weather_data["main"].get("temp","N/A") if weather_data else "N/A"
            humidity = weather_data["main"].get("humidity","N/A") if weather_data else "N/A"
            wind = weather_data["wind"].get("speed","N/A") if weather_data else "N/A"
            condition = weather_data["weather"][0].get("description","").title() if weather_data else "N/A"

            # AQI Card
            st.markdown(f'<div class="card">', unsafe_allow_html=True)
            col1, col2 = st.columns([2,1])
            with col1:
                st.markdown(f"""
                    <div class="aqi-card" style="background-color:{color}">
                        <h1>{icon} {aqi_value}</h1>
                        <h3>Air Quality: {category}</h3>
                        <p>{message}</p>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                    <div class="weather-card">
                        <p>🌡 Temperature: {temp}°C</p>
                        <p>💧 Humidity: {humidity}%</p>
                        <p>💨 Wind: {wind} km/h</p>
                        <p>🌥 Condition: {condition}</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Pollutant Cards
            st.subheader("Pollutant Levels (µg/m³)")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            cols = st.columns(3)
            for i, (name, value) in enumerate(components.items()):
                col = cols[i % 3]
                val_str = f"{value:.2f}" if isinstance(value, (float, int)) else str(value)
                col.markdown(f"""
                    <div class="pollutant-card">
                        <h4>{name.upper()}</h4>
                        <p>{val_str}</p>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # AQI Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=aqi_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"AQI - {category}"},
                gauge={
                    'axis': {'range': [0, 500]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': '#4CAF50'},
                        {'range': [51, 100], 'color': '#8BC34A'},
                        {'range': [101, 200], 'color': '#FFC107'},
                        {'range': [201, 300], 'color': '#FF9800'},
                        {'range': [301, 400], 'color': '#F44336'},
                        {'range': [401, 500], 'color': '#9C27B0'}
                    ]
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models_and_encoder():
    repo_id = "sharmithas151005/aqi-prediction-models"

    # List of all model files in your HF repo
    model_files = {
        "PM2_5": "PM2_5_model.joblib",
        "PM10": "PM10_model.joblib",
        "NO2": "NO2_model.joblib",
        "SO2": "SO2_model.joblib",
        "CO": "CO_model.joblib",
        "O3": "O3_model.joblib",
        "AQI": "AQI_model.joblib",
        "ENCODER": "city_label_encoder.joblib"
    }

    models = {}

    # Download & load each file from HF
    for key, filename in model_files.items():
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        models[key] = joblib.load(file_path)

    # Encoder is stored in models dict
    encoder = models.pop("ENCODER")

    return models, encoder


models, encoder = load_models_and_encoder()


def get_aqi_category(aqi):
    if aqi <= 50: return 'Good'
    if aqi <= 100: return 'Satisfactory'
    if aqi <= 200: return 'Moderate'
    if aqi <= 300: return 'Poor'
    if aqi <= 400: return 'Very Poor'
    return 'Severe'

# ---------- FORECAST ----------
def forecast_pollutants_and_aqi(city_name, hours=24, diurnal=True):
    try:
        df_data = pd.read_csv('dataset/aqi_8_cities_2021_2023.csv')
        df_data.rename(columns={"PM2.5": "PM2_5", "Ozone": "O3"}, inplace=True)
        if 'Datetime' not in df_data.columns:
            df_data['Datetime'] = pd.to_datetime(df_data['From Date'], errors='coerce')
        else:
            df_data['Datetime'] = pd.to_datetime(df_data['Datetime'], errors='coerce')
        if city_name not in df_data['City'].unique():
            st.error(f"City '{city_name}' not found in dataset.")
            return None
        last_data = df_data[df_data['City'] == city_name].sort_values('Datetime').iloc[-1]
        pollutants_list = ['PM2_5','PM10','NO2','SO2','CO','O3']
        city_encoded = encoder.transform([city_name])[0]
        preds = []
        current_time = datetime.now()
        last_pollutant_values = {p: last_data[p] for p in pollutants_list}

        for h in range(1, hours + 1):
            future_time = current_time + timedelta(hours=h)
            row = {
                'Year': future_time.year,
                'Month': future_time.month,
                'Day': future_time.day,
                'Hour': future_time.hour,
                'DayOfWeek': future_time.weekday(),
                'City_Encoded': city_encoded
            }
            for p in pollutants_list:
                input_dict = {k: v for k, v in {**row, **last_pollutant_values}.items() if k in FEATURE_COLUMNS}
                input_df = pd.DataFrame([input_dict])
                pred = models[p].predict(input_df)[0]
                if diurnal:
                    pred *= (1 + 0.1 * np.sin(2 * np.pi * future_time.hour / 24))
                row[p] = max(pred, 0.1)
                last_pollutant_values[p] = row[p]

            aqi_input = pd.DataFrame([{k: v for k, v in row.items() if k in FEATURE_COLUMNS}])
            aqi_val = round(models['AQI'].predict(aqi_input)[0])
            row['Predicted_AQI'] = aqi_val
            row['AQI_Category'] = get_aqi_category(aqi_val)
            preds.append(row)

        df_forecast = pd.DataFrame(preds)
        df_forecast['Datetime'] = pd.to_datetime(df_forecast[['Year','Month','Day','Hour']])
        return df_forecast
    except Exception as e:
        st.error(f"Error during forecast: {e}")
        return None

# ---------- PREDICTION SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("<h2>Forecast Future Air Quality & Pollutants</h2>", unsafe_allow_html=True)
st.markdown("<p>Get the predicted AQI and pollutant levels for the next 24, 48, or 72 hours.</p>", unsafe_allow_html=True)

# Load available cities from dataset
try:
    city_df = pd.read_csv('dataset/aqi_8_cities_2021_2023.csv')
    available_cities = sorted(city_df['City'].dropna().unique())
except Exception:
    available_cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore", "Hyderabad"]

# --- City selector & forecast slider ---
col1, col2 = st.columns([2, 1])
with col1:
    city = st.selectbox("🏙️ Select City", available_cities, index=available_cities.index("Delhi") if "Delhi" in available_cities else 0)
with col2:
    forecast_hours = st.slider("⏱️ Forecast Duration (Hours)", 24, 72, 24, step=24)

# --- Forecast Button ---
if st.button("🚀 Generate Forecast"):
    forecast_df = forecast_pollutants_and_aqi(city, hours=forecast_hours)
    if forecast_df is not None:
        st.success(f"Forecast generated for {city} - Next {forecast_hours} hours")

        # AQI Forecast Chart
        fig_aqi = px.area(
            forecast_df,
            x="Datetime",
            y="Predicted_AQI",
            title=f"AQI Forecast ({forecast_hours} Hours) - {city}",
            color_discrete_sequence=["#0288D1"]
        )
        fig_aqi.update_traces(line_color="#0288D1", fillcolor="rgba(2,136,209,0.3)")
        fig_aqi.update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(255,255,255,0.3)",
        paper_bgcolor="rgba(255,255,255,0.0)",
        font=dict(color="black", size=14),
        title=dict(text=f"AQI Forecast ({forecast_hours} Hours) - {city}", font=dict(size=18, color="#01579B")),
        xaxis=dict(
        title=dict(text="Datetime", font=dict(size=16, color="black")),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        tickangle=-45,
        tickfont=dict(color="black", size=12)
        ),
        yaxis=dict(
        title=dict(text="AQI Value", font=dict(size=16, color="black")),
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)",
        tickfont=dict(color="black", size=12)
        )
        )

        st.plotly_chart(fig_aqi, use_container_width=True)

        # Pollutant Trend Chart
        fig_poll = px.line(
            forecast_df,
            x="Datetime",
            y=['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'],
            title="Predicted Pollutant Trends",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_poll.update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(255,255,255,0.3)",
            paper_bgcolor="rgba(255,255,255,0.0)",
            font=dict(color="black", size=14),
            title=dict(text="Predicted Pollutant Trends", font=dict(size=18, color="#01579B")),
            xaxis=dict(
            title=dict(text="Datetime", font=dict(size=16, color="black")),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            tickangle=-45,
            tickfont=dict(color="black", size=12)
            ),
            yaxis=dict(
                title=dict(text="Pollutant Concentration (µg/m³)", font=dict(size=16, color="black")),
                showgrid=True,
                gridcolor="rgba(0,0,0,0.1)",
                tickfont=dict(color="black", size=12)
            ),
            legend=dict(
                title="Pollutants",
                bgcolor="rgba(255,255,255,0.6)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1,
                font=dict(size=12)
            )
        )
        st.plotly_chart(fig_poll, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if 'forecast_df' in locals() and forecast_df is not None:
          styled_df = forecast_df[['Datetime', 'Predicted_AQI', 'AQI_Category', 'PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']].round(2)

def color_aqi(val):
    if val < 50:
        color = "#81C784"   # Green
    elif val < 100:
        color = "#FFF176"   # Yellow
    elif val < 200:
        color = "#FFB74D"   # Orange
    else:
        color = "#E57373"   # Red
    return f'background-color: {color}; color: black;'

if 'forecast_df' in locals() and forecast_df is not None:
   styled_html = (
    styled_df.style
        .applymap(color_aqi, subset=["Predicted_AQI"])
        .set_table_styles([
            {"selector": "th",
             "props": [("background-color", "#0288D1"), ("color", "white"),
                       ("font-weight", "bold"), ("padding", "8px"), ("text-align", "center")]},
            {"selector": "td",
             "props": [("padding", "6px"), ("text-align", "center"), ("border", "1px solid #ddd")]},
            {"selector": "tr:nth-child(even)",
             "props": [("background-color", "rgba(240, 248, 255, 0.6)")]},
            {"selector": "table",
             "props": [("border-collapse", "collapse"),
                       ("border-radius", "12px"),
                       ("overflow", "hidden"),
                       ("width", "100%")]}
        ])
        .hide(axis="index")
        .to_html(escape=False)
)

# ✅ Properly centered container
if 'forecast_df' in locals() and forecast_df is not None:
   st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-top: 30px;
    ">
        <div style="
            background-color: rgba(255,255,255,0.85);
            border-radius: 15px;
            padding: 25px;
            width: 80%;
            max-width: 950px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        ">
            <h4 style="text-align:center; color:#01579B; margin-bottom: 20px;">Forecast Data Table</h4>
            {styled_html}
    """,
    unsafe_allow_html=True
)
