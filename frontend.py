"""
frontend.py
-----------
Streamlit dashboard for the Bosch Rexroth AG Predictive Maintenance System.

Tabs:
  1. Fleet Overview  — health status of all 10 HPU machines
  2. Live Prediction — single machine sensor input + instant results

Prediction modes (toggle in sidebar):
  - API Mode     : calls FastAPI /predict endpoint (requires uvicorn running)
  - Direct Mode  : loads models directly from S3 (standalone, no API needed)

Run with:
  streamlit run frontend.py
"""

import sys
import json
import time
import joblib
import boto3
import tempfile
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Bosch Rexroth — Predictive Maintenance",
    page_icon  = "⚙️",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
# Palette: industrial steel blue as base, amber for warning states,
# crimson for critical alerts, muted graphite for backgrounds.
# Typography: monospaced data values to reinforce the sensor/telemetry feel.
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header bar */
    .main-header {
        background: linear-gradient(135deg, #0f2744 0%, #1a3a5c 60%, #0d3460 100%);
        padding: 28px 36px 22px;
        border-radius: 12px;
        margin-bottom: 28px;
        border-left: 5px solid #f59e0b;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.3px;
        margin: 0 0 4px 0;
    }
    .main-header p {
        color: #94b4cc;
        font-size: 13px;
        font-family: 'JetBrains Mono', monospace;
        margin: 0;
    }

    /* Alert banners */
    .alert-critical {
        background: linear-gradient(90deg, #7f1d1d, #991b1b);
        color: #fff; padding: 16px 20px; border-radius: 8px;
        border-left: 5px solid #ef4444; font-weight: 600; font-size: 15px;
    }
    .alert-warning {
        background: linear-gradient(90deg, #78350f, #92400e);
        color: #fff; padding: 16px 20px; border-radius: 8px;
        border-left: 5px solid #f59e0b; font-weight: 600; font-size: 15px;
    }
    .alert-advisory {
        background: linear-gradient(90deg, #1c3a1c, #14532d);
        color: #fff; padding: 16px 20px; border-radius: 8px;
        border-left: 5px solid #22c55e; font-weight: 600; font-size: 15px;
    }
    .alert-healthy {
        background: linear-gradient(90deg, #0c1a2e, #0f2744);
        color: #94b4cc; padding: 16px 20px; border-radius: 8px;
        border-left: 5px solid #3b82f6; font-weight: 500; font-size: 15px;
    }

    /* Metric cards */
    .metric-card {
        background: #0f1e30;
        border: 1px solid #1e3a52;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-card .label {
        color: #64748b;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    .metric-card .value {
        color: #e2e8f0;
        font-size: 24px;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card .sub {
        color: #475569;
        font-size: 11px;
        margin-top: 4px;
    }

    /* Fleet table rows */
    .fleet-critical { background-color: rgba(239,68,68,0.12) !important; }
    .fleet-warning  { background-color: rgba(245,158,11,0.12) !important; }
    .fleet-advisory { background-color: rgba(34,197,94,0.10) !important; }
    .fleet-healthy  { background-color: rgba(59,130,246,0.08) !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0a1628;
        border-right: 1px solid #1e3a52;
    }
    [data-testid="stSidebar"] label {
        color: #94b4cc !important;
        font-size: 12px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #0f1e30;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #64748b;
        font-weight: 500;
        font-size: 13px;
        border-radius: 6px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1a3a5c !important;
        color: #f59e0b !important;
    }

    /* Divider */
    hr { border-color: #1e3a52; }

    /* Section labels */
    .section-label {
        color: #f59e0b;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 12px;
    }

    /* Mode toggle */
    .mode-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    .mode-api    { background: #1e3a52; color: #38bdf8; border: 1px solid #38bdf8; }
    .mode-direct { background: #1a2e1a; color: #4ade80; border: 1px solid #4ade80; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
API_URL     = os.getenv("API_URL", "http://localhost:8000/")
BUCKET_NAME = os.getenv("BUCKET_NAME", "grp-feature-engineered-bucket")
MACHINES    = [f"HPU_{i:02d}" for i in range(1, 11)]

FAILURE_COLORS = {
    "No Failure"    : "#22c55e",
    "pump_wear"     : "#ef4444",
    "valve_leakage" : "#f59e0b",
    "contamination" : "#a855f7",
    "cylinder_drift": "#3b82f6",
}

ALERT_COLORS = {
    "CRITICAL": "#ef4444",
    "WARNING" : "#f59e0b",
    "ADVISORY": "#22c55e",
    "HEALTHY" : "#3b82f6",
}

# ── Session state defaults ─────────────────────────────────────────────────────
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "prediction_mode" not in st.session_state:
    st.session_state.prediction_mode = "API"
if "direct_models" not in st.session_state:
    st.session_state.direct_models = None


# ══════════════════════════════════════════════════════════════════════════════
# DIRECT MODE: load models from S3
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading models from S3...")
def load_models_from_s3():
    """
    Download and cache classifier, regressor, and label encoder from S3.
    Uses st.cache_resource so models are loaded once per session,
    not on every user interaction.
    """
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name           = os.getenv("AWS_DEFAULT_REGION", "eu-west-1"),
        )

        def download_joblib(key):
            buf = BytesIO()
            s3.download_fileobj(BUCKET_NAME, key, buf)
            buf.seek(0)
            return joblib.load(buf)

        # Load all three artefacts
        clf     = download_joblib("models/xgb_classifier.joblib")
        reg     = download_joblib("models/xgb_regressor.joblib")
        le      = download_joblib("models/label_encoder.joblib")

        # Load feature column list
        resp    = s3.get_object(Bucket=BUCKET_NAME, Key="models/feature_cols.json")
        feat_cols = json.loads(resp["Body"].read().decode("utf-8"))

        return {"clf": clf, "reg": reg, "le": le, "feature_cols": feat_cols}

    except Exception as e:
        st.error(f"Failed to load models from S3: {e}")
        return None


def predict_direct(reading: dict, models: dict) -> dict:
    """
    Run inference using locally loaded S3 models.
    Mirrors the logic in api/predictor.py for consistency.
    """
    feat_cols = models["feature_cols"]
    clf       = models["clf"]
    reg       = models["reg"]
    le        = models["le"]

    # Compute cross-sensor features
    p    = reading.get("pressure_bar", 0)
    flow = reading.get("flow_lpm", 1)
    vx   = reading.get("vibration_x_g", 0)
    vy   = reading.get("vibration_y_g", 0)
    t    = reading.get("temp_celsius", 0)

    reading["pressure_flow_ratio"]   = p / (flow + 1e-6)
    reading["vibration_magnitude"]   = np.sqrt(vx**2 + vy**2)
    reading["thermal_hydraulic_idx"] = t * p / 1000

    row  = {col: reading.get(col, 0.0) for col in feat_cols}
    X_in = np.array([[row[c] for c in feat_cols]], dtype=np.float32)

    failure_pred  = int(clf.predict(X_in)[0])
    failure_probs = clf.predict_proba(X_in)[0]
    rul_pred      = float(max(0, reg.predict(X_in)[0]))
    failure_mode  = le.classes_[failure_pred]

    if rul_pred <= 24:
        alert  = "CRITICAL"
        action = "Halt machine — emergency maintenance required immediately"
    elif rul_pred <= 72:
        alert  = "WARNING"
        action = "Schedule maintenance within 3 days. Prepare spare parts."
    elif rul_pred <= 168:
        alert  = "ADVISORY"
        action = "Increase monitoring frequency. Pre-order components."
    else:
        alert  = "HEALTHY"
        action = "Continue standard monitoring schedule."

    return {
        "machine_id"            : reading.get("machine_id", "Unknown"),
        "predicted_failure_mode": failure_mode,
        "failure_probabilities" : {
            cls: round(float(p), 4)
            for cls, p in zip(le.classes_, failure_probs)
        },
        "model_confidence"      : round(float(failure_probs.max()), 4),
        "predicted_rul_hours"   : round(rul_pred, 1),
        "alert_level"           : alert,
        "health_status"         : f"{alert} — {rul_pred:.0f} hours remaining",
        "recommended_action"    : action,
    }


# ══════════════════════════════════════════════════════════════════════════════
# API MODE: call FastAPI endpoint
# ══════════════════════════════════════════════════════════════════════════════

def predict_via_api(payload: dict) -> dict:
    """Call the FastAPI /predict endpoint and return the response dict."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json    = payload,
            timeout = 8,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(
            f"Cannot reach FastAPI at {API_URL}. "
            "Start the API with: `uvicorn main.app:app --reload --port 8000`"
        )
        return {}
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return {}
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return {}


def check_api_health() -> bool:
    """Return True if FastAPI /health endpoint responds successfully."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
# FLEET SIMULATION (placeholder until live telemetry pipeline is connected)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=30)
def get_fleet_status() -> pd.DataFrame:
    """
    Simulate fleet status for all 10 machines.
    Replace this function body with a real S3/API call when live
    telemetry is available. TTL=30 refreshes every 30 seconds.
    """
    np.random.seed(int(time.time()) // 30)  # changes every 30s
    records = []
    for machine in MACHINES:
        rul   = int(np.random.choice(
            [np.random.randint(0, 24),
             np.random.randint(24, 72),
             np.random.randint(72, 168),
             np.random.randint(168, 500)],
            p=[0.05, 0.10, 0.15, 0.70]
        ))
        failure = np.random.choice(
            list(FAILURE_COLORS.keys()),
            p=[0.75, 0.07, 0.07, 0.06, 0.05]
        )
        if rul <= 24:   alert = "CRITICAL"
        elif rul <= 72: alert = "WARNING"
        elif rul <= 168:alert = "ADVISORY"
        else:           alert = "HEALTHY"

        records.append({
            "Machine"       : machine,
            "Alert"         : alert,
            "RUL (hrs)"     : rul,
            "Failure Mode"  : failure,
            "Confidence"    : round(np.random.uniform(0.72, 0.99), 2),
            "Last Reading"  : datetime.now().strftime("%H:%M:%S"),
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>⚙️ Bosch Rexroth AG — Predictive Maintenance</h1>
    <p>Hydraulic System Health Monitor &nbsp;|&nbsp; Failure Classification &amp; RUL Prediction &nbsp;|&nbsp; 10 HPU Units</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    # ── Prediction mode toggle ─────────────────────────────────────────────
    st.markdown('<p class="section-label">Prediction Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        label     = "Source",
        options   = ["API Mode (FastAPI)", "Direct Mode (S3 Models)"],
        index     = 0,
        label_visibility = "collapsed",
    )
    st.session_state.prediction_mode = "API" if "API" in mode else "DIRECT"

    if st.session_state.prediction_mode == "API":
        api_ok = check_api_health()
        status_color = "#22c55e" if api_ok else "#ef4444"
        status_text  = "Online" if api_ok else "Offline"
        st.markdown(
            f'<span class="mode-badge mode-api">API &nbsp;●&nbsp; '
            f'<span style="color:{status_color}">{status_text}</span></span>',
            unsafe_allow_html=True
        )
        if not api_ok:
            st.caption("Start API: `uvicorn main.app:app --port 8000`")
    else:
        st.markdown('<span class="mode-badge mode-direct">Direct — S3 Models</span>',
                    unsafe_allow_html=True)
        if st.session_state.direct_models is None:
            if st.button("Load Models from S3", use_container_width=True):
                st.session_state.direct_models = load_models_from_s3()
                if st.session_state.direct_models:
                    st.success("Models loaded.")

    st.markdown("---")

    # ── Machine selector ───────────────────────────────────────────────────
    st.markdown('<p class="section-label">Machine</p>', unsafe_allow_html=True)
    selected_machine = st.selectbox(
        "Select HPU", MACHINES, label_visibility="collapsed")

    st.markdown("---")

    # ── Sensor inputs ──────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Live Sensor Readings</p>',
                unsafe_allow_html=True)

    pressure    = st.slider("Pressure (bar)",      0,   400, 125)
    temperature = st.slider("Temperature (°C)",    0,    60,  52)
    flow        = st.slider("Flow (L/min)",        20,  120,  88)
    vibration   = st.slider("Vibration X (g)",      0,   10,   1, step=1)
    pump_rpm    = st.slider("Pump RPM",           800, 1600, 1480)

    st.markdown("---")
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True, type="primary")


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

if predict_btn:
    payload = {
        "machine_id"   : selected_machine,
        "pressure_bar" : float(pressure),
        "temp_celsius" : float(temperature),
        "flow_lpm"     : float(flow),
        "vibration_x_g": float(vibration),
        "vibration_y_g": float(vibration),
        "pump_rpm"     : float(pump_rpm),
    }

    with st.spinner("Running inference..."):
        if st.session_state.prediction_mode == "API":
            result = predict_via_api(payload)
        else:
            if st.session_state.direct_models is None:
                st.error("Load models from S3 first using the sidebar button.")
                result = {}
            else:
                result = predict_direct(payload, st.session_state.direct_models)

    if result:
        st.session_state.prediction_result = result


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_fleet, tab_predict = st.tabs(["🏭 Fleet Overview", "🔬 Live Prediction"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FLEET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

with tab_fleet:
    fleet_df = get_fleet_status()

    # ── Summary KPIs ──────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    counts = fleet_df["Alert"].value_counts()

    kpis = [
        ("TOTAL MACHINES", len(fleet_df),                   "units online"),
        ("CRITICAL",       counts.get("CRITICAL", 0),       "immediate action"),
        ("WARNING",        counts.get("WARNING",  0),       "within 3 days"),
        ("ADVISORY",       counts.get("ADVISORY", 0),       "monitor closely"),
        ("HEALTHY",        counts.get("HEALTHY",  0),       "normal operation"),
    ]
    kpi_colors = ["#e2e8f0", "#ef4444", "#f59e0b", "#22c55e", "#3b82f6"]

    for col, (label, value, sub), color in zip(
        [col1, col2, col3, col4, col5], kpis, kpi_colors
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{color}">{value}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── RUL bar chart across fleet ─────────────────────────────────────────
    chart_col, table_col = st.columns([1.2, 1])

    with chart_col:
        st.markdown('<p class="section-label">Remaining Useful Life — All Machines</p>',
                    unsafe_allow_html=True)
        bar_colors = [ALERT_COLORS[a] for a in fleet_df["Alert"]]
        fig_fleet = go.Figure(go.Bar(
            x           = fleet_df["Machine"],
            y           = fleet_df["RUL (hrs)"],
            marker_color= bar_colors,
            text        = fleet_df["RUL (hrs)"].astype(str) + "h",
            textposition= "outside",
            hovertemplate=(
                "<b>%{x}</b><br>"
                "RUL: %{y} hours<br>"
                "<extra></extra>"
            ),
        ))
        fig_fleet.add_hline(y=24,  line_dash="dot", line_color="#ef4444",
                            annotation_text="Critical (24h)")
        fig_fleet.add_hline(y=72,  line_dash="dot", line_color="#f59e0b",
                            annotation_text="Warning (72h)")
        fig_fleet.add_hline(y=168, line_dash="dot", line_color="#22c55e",
                            annotation_text="Advisory (168h)")
        fig_fleet.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(15,30,48,0.6)",
            font          = dict(color="#94b4cc", family="JetBrains Mono"),
            xaxis         = dict(gridcolor="#1e3a52", tickfont=dict(size=11)),
            yaxis         = dict(gridcolor="#1e3a52", title="Hours Remaining"),
            margin        = dict(t=20, b=20, l=10, r=10),
            height        = 320,
        )
        st.plotly_chart(fig_fleet, use_container_width=True)

    with table_col:
        st.markdown('<p class="section-label">Fleet Status Table</p>',
                    unsafe_allow_html=True)

        # Colour-code the Alert column
        def style_alert(val):
            colors = {
                "CRITICAL": "color:#ef4444;font-weight:700",
                "WARNING" : "color:#f59e0b;font-weight:700",
                "ADVISORY": "color:#22c55e;font-weight:600",
                "HEALTHY" : "color:#3b82f6;font-weight:500",
            }
            return colors.get(val, "")

        styled = (
            fleet_df.style
            .applymap(style_alert, subset=["Alert"])
            .format({"Confidence": "{:.0%}"})
            .hide(axis="index")
        )
        st.dataframe(styled, use_container_width=True, height=320)

    st.markdown("---")

    # ── Failure mode distribution donut ───────────────────────────────────
    dist_col, trend_col = st.columns(2)

    with dist_col:
        st.markdown('<p class="section-label">Failure Mode Distribution</p>',
                    unsafe_allow_html=True)
        mode_counts = fleet_df["Failure Mode"].value_counts().reset_index()
        mode_counts.columns = ["Failure Mode", "Count"]
        fig_donut = go.Figure(go.Pie(
            labels      = mode_counts["Failure Mode"],
            values      = mode_counts["Count"],
            hole        = 0.55,
            marker_colors = [FAILURE_COLORS.get(m, "#64748b")
                             for m in mode_counts["Failure Mode"]],
            textfont    = dict(family="JetBrains Mono", size=11),
        ))
        fig_donut.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            font          = dict(color="#94b4cc"),
            legend        = dict(font=dict(size=11, color="#94b4cc")),
            margin        = dict(t=10, b=10, l=10, r=10),
            height        = 260,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with trend_col:
        st.markdown('<p class="section-label">Alert Level Breakdown</p>',
                    unsafe_allow_html=True)
        alert_counts = fleet_df["Alert"].value_counts().reindex(
            ["CRITICAL","WARNING","ADVISORY","HEALTHY"], fill_value=0)
        fig_alerts = go.Figure(go.Bar(
            x            = alert_counts.index,
            y            = alert_counts.values,
            marker_color = [ALERT_COLORS[a] for a in alert_counts.index],
            text         = alert_counts.values,
            textposition = "outside",
        ))
        fig_alerts.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(15,30,48,0.6)",
            font          = dict(color="#94b4cc", family="JetBrains Mono"),
            xaxis         = dict(gridcolor="#1e3a52"),
            yaxis         = dict(gridcolor="#1e3a52", title="Machines"),
            margin        = dict(t=20, b=10, l=10, r=10),
            height        = 260,
        )
        st.plotly_chart(fig_alerts, use_container_width=True)

    # ── Auto-refresh notice ────────────────────────────────────────────────
    st.caption("Fleet data refreshes every 30 seconds. "
               "Connect live telemetry by replacing `get_fleet_status()` "
               "in frontend.py with your S3/API polling logic.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — LIVE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

with tab_predict:

    result = st.session_state.prediction_result

    if result is None:
        st.markdown("""
        <div style="
            text-align:center;
            padding:60px 20px;
            color:#475569;
            border:1px dashed #1e3a52;
            border-radius:12px;
            margin-top:20px;
        ">
            <div style="font-size:40px;margin-bottom:12px">⚙️</div>
            <div style="font-size:16px;font-weight:600;color:#64748b;">
                No prediction yet
            </div>
            <div style="font-size:13px;margin-top:6px;">
                Adjust sensor readings in the sidebar and click
                <strong style="color:#f59e0b">Run Prediction</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        alert  = result.get("alert_level", "HEALTHY")
        rul    = result.get("predicted_rul_hours", 0)
        mode   = result.get("predicted_failure_mode", "Unknown")
        conf   = result.get("model_confidence", 0)
        action = result.get("recommended_action", "")
        probs  = result.get("failure_probabilities", {})

        # ── Alert banner ───────────────────────────────────────────────────
        css_class = f"alert-{alert.lower()}"
        st.markdown(f"""
        <div class="{css_class}">
            {alert} &nbsp;|&nbsp; {result.get("machine_id")} &nbsp;|&nbsp;
            {mode.replace("_"," ").title()} &nbsp;|&nbsp;
            {rul:.0f} hours remaining
            <div style="font-weight:400;font-size:13px;margin-top:6px;opacity:0.85;">
                {action}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── KPI row ────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        kpi_data = [
            (k1, "MACHINE",        result.get("machine_id","—"),      "selected unit"),
            (k2, "FAILURE MODE",   mode.replace("_"," ").title(),     "predicted class"),
            (k3, "RUL ESTIMATE",   f"{rul:.0f} hrs",                  "remaining useful life"),
            (k4, "CONFIDENCE",     f"{conf*100:.1f}%",                "model certainty"),
        ]
        for col, label, value, sub in kpi_data:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Charts row ─────────────────────────────────────────────────────
        gauge_col, prob_col = st.columns([1, 1.2])

        with gauge_col:
            st.markdown('<p class="section-label">RUL Gauge</p>',
                        unsafe_allow_html=True)
            alert_color = ALERT_COLORS.get(alert, "#3b82f6")
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = rul,
                title = dict(
                    text = f"{result.get('machine_id')} — Remaining Hours",
                    font = dict(color="#94b4cc", size=13, family="JetBrains Mono")
                ),
                number = dict(
                    suffix    = " hrs",
                    font      = dict(color=alert_color, size=36,
                                     family="JetBrains Mono")
                ),
                gauge = dict(
                    axis  = dict(
                        range    = [0, 500],
                        tickfont = dict(color="#64748b", size=10),
                        tickcolor= "#1e3a52",
                    ),
                    bar   = dict(color=alert_color, thickness=0.25),
                    bgcolor = "rgba(15,30,48,0.8)",
                    bordercolor = "#1e3a52",
                    steps = [
                        dict(range=[0,   24],  color="rgba(239,68,68,0.15)"),
                        dict(range=[24,  72],  color="rgba(245,158,11,0.12)"),
                        dict(range=[72,  168], color="rgba(34,197,94,0.10)"),
                        dict(range=[168, 500], color="rgba(59,130,246,0.07)"),
                    ],
                    threshold = dict(
                        line      = dict(color="#ef4444", width=3),
                        thickness = 0.8,
                        value     = 24,
                    ),
                )
            ))
            fig_gauge.update_layout(
                paper_bgcolor = "rgba(0,0,0,0)",
                font          = dict(color="#94b4cc"),
                height        = 280,
                margin        = dict(t=40, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with prob_col:
            st.markdown('<p class="section-label">Failure Probability Distribution</p>',
                        unsafe_allow_html=True)
            if probs:
                prob_df = (
                    pd.DataFrame(list(probs.items()),
                                 columns=["Failure Mode", "Probability"])
                    .sort_values("Probability", ascending=True)
                )
                bar_colors = [FAILURE_COLORS.get(m, "#64748b")
                              for m in prob_df["Failure Mode"]]
                fig_prob = go.Figure(go.Bar(
                    x            = prob_df["Probability"],
                    y            = prob_df["Failure Mode"],
                    orientation  = "h",
                    marker_color = bar_colors,
                    text         = [f"{p:.1%}" for p in prob_df["Probability"]],
                    textposition = "outside",
                    textfont     = dict(
                        family="JetBrains Mono", size=11, color="#94b4cc"),
                ))
                fig_prob.update_layout(
                    paper_bgcolor = "rgba(0,0,0,0)",
                    plot_bgcolor  = "rgba(15,30,48,0.6)",
                    font          = dict(color="#94b4cc",
                                        family="JetBrains Mono"),
                    xaxis = dict(
                        range      = [0, 1.08],
                        tickformat = ".0%",
                        gridcolor  = "#1e3a52",
                        tickfont   = dict(size=10),
                    ),
                    yaxis = dict(
                        gridcolor  = "#1e3a52",
                        tickfont   = dict(size=11),
                    ),
                    margin = dict(t=10, b=10, l=10, r=60),
                    height = 280,
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        # ── Raw sensor echo ────────────────────────────────────────────────
        with st.expander("📡 Raw Sensor Inputs Submitted"):
            sensor_echo = {
                "machine_id"   : selected_machine,
                "pressure_bar" : pressure,
                "temp_celsius" : temperature,
                "flow_lpm"     : flow,
                "vibration_x_g": vibration,
                "vibration_y_g": vibration,
                "pump_rpm"     : pump_rpm,
            }
            echo_df = pd.DataFrame(
                list(sensor_echo.items()),
                columns=["Sensor", "Value"]
            )
            st.dataframe(echo_df, use_container_width=True, hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='color:#334155;font-size:11px;font-family:JetBrains Mono,monospace;"
    "text-align:center;'>"
    "Bosch Rexroth AG Predictive Maintenance v1.0 &nbsp;|&nbsp; "
    "XGBoost Classifier + RUL Regressor &nbsp;|&nbsp; "
    "MLflow · DagsHub · AWS S3 · FastAPI"
    "</p>",
    unsafe_allow_html=True
)
