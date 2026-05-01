# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import time
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_image(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return data

def show_preloader(logo):
    loader = st.empty()
    loader.markdown(f"""
    <style>
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    .loader-container {{ text-align:center; padding:40px; }}
    .loader-logo {{ width:110px; animation: spin 5s linear infinite; }}
    </style>
    <div class="loader-container">
        <img src="data:image/png;base64,{logo}" class="loader-logo">
        <h3>Wait! AI Analytical Engine Processing Student Records...</h3>
    </div>
    """, unsafe_allow_html=True)
    return loader

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached so it only loads once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model         = joblib.load("models/model.pkl")
    feature_names = joblib.load("encoders/feature_names.pkl")
    return model, feature_names

@st.cache_data
def load_importance():
    path = "data/feature_importance_rankings.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

model, FEATURE_NAMES = load_model()
importance_df        = load_importance()

# ─────────────────────────────────────────────────────────────────────────────
# ENCODING MAPS  — must match ml_pipeline.py exactly
# ─────────────────────────────────────────────────────────────────────────────
ENCODE = {
    "Gender":                      {"Male": 1, "Female": 0},
    "Entry_Mode":                  {"UTME": 0, "Direct Entry": 1, "Transfer": 2, "Part-Time": 3},
    "Socioeconomic_Status":        {"Low": 0, "Middle": 1, "High": 2},
    "Tuition_Payment_Consistency": {"Defaulter": 0, "Irregular": 1, "Consistent": 2},
    "Study_Mode":                  {"Full-Time": 0, "Distance/Part-Time": 1},
    "Marital_Status":              {"Single": 0, "Married": 1},
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictive System",
    page_icon=r"assets\logo.png",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (your original styles — untouched)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color:#f3f7fb; }
.main { background-color:#f3f7fb; }

.hero-banner {
height:260px; width:100%;
background-image:url("assets/ban");
background-size:cover; background-position:center;
display:flex; align-items:center; justify-content:center;
flex-direction:column; color:white; text-align:center;
border-radius:0px 0px 18px 18px;
box-shadow:0px 6px 18px rgba(0,0,0,0.2);
}
.hero-title { font-size:42px; font-weight:700; }
.hero-sub   { font-size:18px; opacity:0.9; }

[data-testid="stSidebar"] { background-color:#0b2545; color:white; }
.sidebar-logo { text-align:center; padding-top:10px; padding-bottom:20px; }
.sidebar-logo img { width:120px; }

.kpi-card {
background:white; padding:20px; border-radius:12px;
box-shadow:0px 4px 14px rgba(0,0,0,0.08); text-align:center;
}
.kpi-title { font-size:14px; color:#777; }
.kpi-value { font-size:32px; font-weight:bold; color:#002147; }

.ai-panel {
background:white; padding:25px; border-radius:12px;
box-shadow:0px 5px 20px rgba(0,0,0,0.08);
}
.ai-header  { font-size:20px; font-weight:bold; margin-bottom:15px; color:#002147; }
.risk-high  { color:#d62828; font-weight:bold; font-size:22px; }
.risk-low   { color:#2a9d8f; font-weight:bold; font-size:22px; }
.gauge      { font-size:40px; font-weight:bold; color:#003366; }
.feature-panel { margin-top:10px; font-size:15px; }

.chart-box {
background:white; padding:20px; border-radius:12px;
box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

/* XAI explanation cards */
.xai-good {
background:#e8f8f5; border-left:4px solid #2a9d8f;
padding:10px 14px; border-radius:6px;
font-size:14px; color:#145a4c; margin:5px 0;
}
.xai-warn {
background:#fef9e7; border-left:4px solid #f0a500;
padding:10px 14px; border-radius:6px;
font-size:14px; color:#7d6608; margin:5px 0;
}
.xai-risk {
background:#fdecea; border-left:4px solid #d62828;
padding:10px 14px; border-radius:6px;
font-size:14px; color:#7b241c; margin:5px 0;
}

.footer {
text-align:center; margin-top:40px; color:#555; font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO BANNER  (your original — untouched)
# ─────────────────────────────────────────────────────────────────────────────
banner_img = load_image("assets/banner3.jfif")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Exo+2:wght@300;400;600&display=swap');
.block-container {{ padding-top: 3rem !important; }}
.hero-wrap {{
    width:100%; height:280px; position:relative; border-radius:20px;
    overflow:hidden; margin-bottom:2.0rem;
    box-shadow: 0 0 0 1px rgba(0,229,255,0.14),
                0 8px 40px rgba(0,0,0,0.55),
                0 0 80px rgba(41,121,255,0.10);
    font-family:'Exo 2',sans-serif;
}}
.hero-photo {{
    position:absolute; inset:0;
    background: linear-gradient(135deg,rgba(2,6,23,0.91) 0%,rgba(4,14,46,0.81) 45%,rgba(0,30,40,0.77) 100%),
                url("data:image/jfif;base64,{banner_img}");
    background-size:cover; background-position:center;
    transform:scale(1.05);
    animation:hero-zoom 20s ease-in-out infinite alternate;
}}
@keyframes hero-zoom {{ from{{transform:scale(1.05);}} to{{transform:scale(1.30);}} }}
.hero-scan {{
    position:absolute; inset:0;
    background:repeating-linear-gradient(180deg,transparent,transparent 3px,rgba(0,229,255,0.025) 3px,rgba(0,229,255,0.025) 4px);
    animation:scan-move 4s linear infinite; pointer-events:none;
}}
@keyframes scan-move {{ from{{background-position:0 0;}} to{{background-position:0 40px;}} }}
.hero-orb {{ position:absolute; border-radius:50%; filter:blur(64px); pointer-events:none; }}
.hero-orb-a {{ width:260px;height:260px;background:rgba(41,121,255,0.20);top:-70px;left:-50px;animation:orb-a 9s ease-in-out infinite alternate; }}
.hero-orb-b {{ width:200px;height:200px;background:rgba(0,229,255,0.16);bottom:-55px;right:80px;animation:orb-b 11s ease-in-out infinite alternate; }}
.hero-orb-c {{ width:150px;height:150px;background:rgba(29,233,182,0.14);top:10px;right:-30px;animation:orb-c 8s ease-in-out infinite alternate; }}
@keyframes orb-a {{ from{{transform:translate(0,0) scale(1);}} to{{transform:translate(22px,16px) scale(1.09);}} }}
@keyframes orb-b {{ from{{transform:translate(0,0);}} to{{transform:translate(-20px,-12px);}} }}
@keyframes orb-c {{ from{{transform:translate(0,0);}} to{{transform:translate(-14px,20px);}} }}
.hero-corner {{ position:absolute;width:520px;height:60px;border-color:#00e5ff;border-style:solid;opacity:0.60;animation:hud-pulse 3.2s ease-in-out infinite; }}
.hud-tl{{top:14px;left:14px;border-width:2px 0 0 2px;}} .hud-tr{{top:14px;right:14px;border-width:2px 2px 0 0;}}
.hud-bl{{bottom:36px;left:14px;border-width:0 0 2px 2px;}} .hud-br{{bottom:36px;right:14px;border-width:0 2px 2px 0;}}
@keyframes hud-pulse {{ 0%,100%{{opacity:0.30;}} 50%{{opacity:1.00;}} }}
.hero-content {{ position:absolute;inset:0;bottom:30px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:0 7%;text-align:center;gap:11px; }}
.hero-badge {{ display:inline-flex;align-items:center;gap:7px;background:rgba(0,229,255,0.08);border:1px solid rgba(0,229,255,0.28);border-radius:40px;padding:4px 14px 4px 10px;font-family:'Exo 2',sans-serif;font-size:10.5px;font-weight:600;letter-spacing:0.13em;text-transform:uppercase;color:#00e5ff;animation:entry-down 0.7s ease both; }}
.hero-badge-dot {{ width:7px;height:7px;border-radius:50%;background:#1de9b6;box-shadow:0 0 7px #1de9b6;animation:dot-blink 1.9s ease-in-out infinite; }}
@keyframes dot-blink {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.2;}} }}
.hero-title {{ font-family:'Rajdhani',sans-serif;font-size:clamp(20px,4vw,44px);font-weight:700;line-height:1.12;letter-spacing:0.01em;background:linear-gradient(100deg,#ffffff 0%,#b8f0ff 28%,#ffffff 52%,#9ecfff 78%,#ffffff 100%);background-size:220% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;filter:drop-shadow(0 0 16px rgba(0,229,255,0.48)) drop-shadow(0 0 38px rgba(41,121,255,0.32));animation:entry-up 0.8s 0.15s ease both,title-shimmer 5s 1s linear infinite; }}
@keyframes title-shimmer {{ from{{background-position:220% center;}} to{{background-position:-220% center;}} }}
.hero-divider {{ height:2px;width:0;background:linear-gradient(90deg,transparent,#00e5ff,transparent);border-radius:2px;box-shadow:0 0 10px rgba(0,229,255,0.55);animation:entry-up 0.8s 0.30s ease both,divider-open 0.9s 0.85s ease forwards; }}
@keyframes divider-open {{ from{{width:0;}} to{{width:120px;}} }}
.hero-ticker-wrap {{ width:100%;max-width:900px;overflow:hidden;animation:entry-up 0.8s 0.45s ease both; }}
.hero-ticker {{ display:inline-block;white-space:nowrap;font-family:'Exo 2',sans-serif;font-size:clamp(11px,1.4vw,14.5px);font-weight:400;letter-spacing:0.055em;color:rgba(200,238,255,0.80);text-shadow:0 0 14px rgba(0,229,255,0.22);animation:ticker-run 30s linear infinite; }}
@keyframes ticker-run {{ from{{transform:translateX(100%);}} to{{transform:translateX(-100%);}} }}
.hero-bar {{ position:absolute;bottom:0;left:0;right:0;height:32px;background:rgba(0,8,28,0.70);backdrop-filter:blur(6px);border-top:1px solid rgba(0,229,255,0.10);display:flex;align-items:center;justify-content:space-between;padding:0 18px;animation:entry-up 0.8s 0.65s ease both; }}
.hero-stat {{ display:flex;align-items:center;gap:6px;font-family:'Exo 2',sans-serif;font-size:10.5px;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;color:rgba(0,229,255,0.65); }}
.hero-stat-pip {{ width:6px;height:6px;border-radius:50%;background:#d5e91d;box-shadow:0 0 5px #1de9b6;animation:dot-blink 2.5s ease-in-out infinite; }}
@keyframes entry-down {{ from{{opacity:0;transform:translateY(-10px);}} to{{opacity:1;transform:translateY(0);}} }}
@keyframes entry-up {{ from{{opacity:0;transform:translateY(12px);}} to{{opacity:1;transform:translateY(0);}} }}
@media (max-width:640px) {{ .hero-wrap{{height:210px;}} .hero-bar{{display:none;}} .hero-badge{{font-size:9px;}} .hud-bl,.hud-br{{bottom:14px;}} }}
</style>

<div class="hero-wrap">
  <div class="hero-photo"></div>
  <div class="hero-scan"></div>
  <div class="hero-orb hero-orb-a"></div>
  <div class="hero-orb hero-orb-b"></div>
  <div class="hero-orb hero-orb-c"></div>
  <div class="hero-corner hud-tl"></div>
  <div class="hero-corner hud-tr"></div>
  <div class="hero-corner hud-bl"></div>
  <div class="hero-corner hud-br"></div>
  <div class="hero-content">
    <div class="hero-badge"><span class="hero-badge-dot"></span> AI-Powered Academic Intelligence</div>
    <div class="hero-title">Student Performance &amp; Retention<br>Prediction System</div>
    <div class="hero-divider"></div>
    <div class="hero-ticker-wrap">
      <span class="hero-ticker">
        &diams;&nbsp;&nbsp;An AI-Driven Early Warning Platform &nbsp;&middot;&nbsp;
        Designed by UNILORIN Educational Technology Dept. &nbsp;&middot;&nbsp;
        Built for Nigerian Universities &nbsp;&middot;&nbsp;
        Real-Time Risk Detection &nbsp;&middot;&nbsp;
        Predictive Academic Analytics &nbsp;&middot;&nbsp;
        Early Intervention Intelligence &nbsp;&nbsp;&diams;
      </span>
    </div>
  </div>
  <div class="hero-bar">
    <div class="hero-stat"><span class="hero-stat-pip"></span> System Online</div>
    <div class="hero-stat"><span class="hero-stat-pip"></span> Model Active</div>
    <div class="hero-stat"><span class="hero-stat-pip"></span> UNILORIN EdTech &middot; v2.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  (your original layout — untouched)
# ─────────────────────────────────────────────────────────────────────────────
logo = load_image(r"assets\logo.png")

st.sidebar.markdown(f"""
<div style="text-align:center; padding-bottom:20px;">
    <img src="data:image/jfif;base64,{logo}" width="120">
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Student Input Variables")

prediction_mode = st.sidebar.radio(
    "Prediction Mode",
    ["Individual Prediction", "Bulk Prediction"]
)

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
if prediction_mode == "Individual Prediction":

    # ── Sidebar inputs ────────────────────────────────────────────────────────
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age    = st.sidebar.slider("Age", 16, 50, 20)

    entry_mode_display = st.sidebar.selectbox(
        "Entry Mode",
        ["UTME (100L)", "Direct Entry (200L)", "Transfer", "Part-Time"]
    )
    # Map display label → model label
    entry_mode_map = {
        "UTME (100L)": "UTME", "Direct Entry (200L)": "Direct Entry",
        "Transfer": "Transfer", "Part-Time": "Part-Time"
    }
    entry_mode = entry_mode_map[entry_mode_display]
    entry_level = 200 if entry_mode == "Direct Entry" else 100

    tuition_display = st.sidebar.selectbox(
        "Financial Status",
        ["Full Payment", "Partial Payment", "Outstanding"]
    )
    # Map display label → model label
    tuition_map = {
        "Full Payment": "Consistent", "Partial Payment": "Irregular", "Outstanding": "Defaulter"
    }
    tuition = tuition_map[tuition_display]

    ses = st.sidebar.selectbox(
        "Socioeconomic Status", ["Low", "Middle", "High"]
    )

    study_mode = st.sidebar.selectbox(
        "Study Mode", ["Full-Time", "Distance/Part-Time"]
    )

    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married"])

    current_cgpa = st.sidebar.number_input("Current CGPA", 0.0, 5.0, 3.0, step=0.01)

    portal_logins = st.sidebar.slider("Monthly Portal Logins", 0, 200, 45)

    attendance = st.sidebar.slider("Attendance Rate (%)", 0, 100, 75)

    o_level = st.sidebar.slider("O'Level Credits", 4, 9, 6)

    jamb = st.sidebar.number_input(
        "JAMB Score (0 if not applicable)", 0, 400, 220
    )

    carryovers = st.sidebar.number_input("Carryover Courses", 0, 20, 0)

    assignment_rate = st.sidebar.slider("Assignment Submission Rate (%)", 0, 100, 80)

    entry_year = st.sidebar.number_input("Entry Year", 2016, 2024, 2020)

    st.sidebar.markdown("**Semester GPAs** *(0.00 – 5.00)*")
    sem_gpas = []
    for i in range(8):
        g = st.sidebar.number_input(
            f"Semester {i+1} GPA", 0.0, 5.0,
            value=round(current_cgpa + np.random.uniform(-0.3, 0.3), 2),
            step=0.01, key=f"sem_{i}"
        )
        sem_gpas.append(g)

    avg_credits = st.sidebar.number_input(
        "Avg Credit Units/Semester", 10.0, 30.0, 18.0, step=0.5
    )

    # ── Predict button ────────────────────────────────────────────────────────
    if st.sidebar.button("Predict Student Outcome"):

        preloader    = show_preloader(logo)
        progress_bar = st.progress(0)
        status_text  = st.empty()

        stages = [
            "Loading student profile data...",
            "Validating academic records...",
            "Extracting behavioural features...",
            "Running predictive model...",
            "Evaluating academic risk indicators...",
            "Generating AI prediction report..."
        ]

        for i in range(100):
            percent     = i + 1
            stage_index = min(int((percent / 100) * len(stages)), len(stages) - 1)
            progress_bar.progress(percent)
            status_text.text(f"{stages[stage_index]}  {percent}%")
            time.sleep(0.25)   # faster — model is doing real work now

        preloader.markdown(
            "<h3 style='text-align:center;color:green;'>Prediction Completed....100%</h3>",
            unsafe_allow_html=True
        )
        status_text.success("Prediction Successful!")
        st.divider()

        # ── BUILD FEATURE VECTOR ──────────────────────────────────────────────
        cgpa_computed = round(np.mean(sem_gpas), 2)
        total_cu      = int(avg_credits * 8)

        feature_vector = np.array([[
            ENCODE["Gender"][gender],
            age,
            ENCODE["Entry_Mode"][entry_mode],
            entry_level,
            entry_year,
            ENCODE["Socioeconomic_Status"][ses],
            ENCODE["Tuition_Payment_Consistency"][tuition],
            ENCODE["Study_Mode"][study_mode],
            ENCODE["Marital_Status"][marital],
            o_level,
            jamb,
            sem_gpas[0], sem_gpas[1], sem_gpas[2], sem_gpas[3],
            sem_gpas[4], sem_gpas[5], sem_gpas[6], sem_gpas[7],
            cgpa_computed,
            avg_credits,
            total_cu,
            attendance,
            portal_logins,
            assignment_rate,
            carryovers,
        ]])

        # ── RUN MODEL ─────────────────────────────────────────────────────────
        retention_pred  = model.predict(feature_vector)[0]
        retention_proba = model.predict_proba(feature_vector)[0]
        retained        = retention_pred == 1
        retain_pct      = round(retention_proba[1] * 100, 1)
        withdraw_pct    = round(retention_proba[0] * 100, 1)
        confidence      = round(retention_proba[retention_pred] * 100, 1)

        # Performance class from CGPA
        if cgpa_computed >= 4.5:   perf_label = "First Class"
        elif cgpa_computed >= 3.5: perf_label = "Second Class Upper"
        elif cgpa_computed >= 2.5: perf_label = "Second Class Lower"
        elif cgpa_computed >= 1.5: perf_label = "Third Class"
        else:                      perf_label = "Fail / At-Risk"

        performance_score = "High Achieving" if cgpa_computed >= 3.0 else "At Risk"

        # Risk tier
        if withdraw_pct >= 60:   risk_label = "🔴 HIGH RISK"
        elif withdraw_pct >= 35: risk_label = "🟡 MODERATE RISK"
        else:                    risk_label = "🟢 LOW RISK"

        # ── KPI DASHBOARD — updated with real model metrics ───────────────────
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">Model Accuracy</div>
        <div class="kpi-value">91.67%</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">F1 Score</div>
        <div class="kpi-value">0.9409</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">Precision</div>
        <div class="kpi-value">0.9343</div>
        </div>
        """, unsafe_allow_html=True)

        col4.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">AUC-ROC</div>
        <div class="kpi-value">0.9689</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PREDICTION RESULTS ────────────────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
            st.subheader("Performance Prediction")
            if performance_score == "High Achieving":
                st.success(f"✅ {performance_score}  —  {perf_label}")
            else:
                st.error(f"⚠️ {performance_score}  —  {perf_label}")
            st.write(f"**Computed CGPA:** {cgpa_computed:.2f}")
            st.write("Prediction driven by CGPA trajectory, tuition consistency, and engagement behaviour.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ai-panel">', unsafe_allow_html=True)
            st.subheader("Retention Prediction")
            st.metric(
                label="Probability of Continued Enrolment",
                value=f"{retain_pct}%",
                delta=f"Withdrawal risk: {withdraw_pct}%"
            )
            if retained:
                st.success(f"✅ RETAINED  |  Confidence: {confidence}%  |  {risk_label}")
            else:
                st.error(f"⚠️ WITHDRAWAL RISK  |  Confidence: {confidence}%  |  {risk_label}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── XAI — WHY THIS PREDICTION ─────────────────────────────────────────
        st.subheader("🧠 Why This Prediction?  (Explainability)")
        xai_col1, xai_col2 = st.columns(2)

        protective = []
        risks_list = []

        if tuition == "Consistent":
            protective.append("✅ <b>Tuition payment is Consistent</b> — strongest positive retention signal (25.9% importance).")
        elif tuition == "Irregular":
            risks_list.append("⚠️ <b>Tuition payment is Irregular</b> — moderate dropout risk. Consider payment plan.")
        else:
            risks_list.append("🔴 <b>Tuition payment Outstanding</b> — highest single dropout risk factor.")

        if ses == "High":
            protective.append("✅ <b>Socioeconomic status is High</b> — reduces financial vulnerability.")
        elif ses == "Low":
            risks_list.append("⚠️ <b>Socioeconomic status is Low</b> — increases dropout vulnerability.")

        if cgpa_computed >= 3.5:
            protective.append(f"✅ <b>CGPA {cgpa_computed:.2f}</b> is strong — students above 3.5 persist at high rates.")
        elif cgpa_computed < 2.0:
            risks_list.append(f"🔴 <b>CGPA {cgpa_computed:.2f}</b> is critically low — immediate academic intervention needed.")
        elif cgpa_computed < 2.5:
            risks_list.append(f"⚠️ <b>CGPA {cgpa_computed:.2f}</b> is below 2.5 threshold — academic support recommended.")

        if carryovers == 0:
            protective.append("✅ <b>No carryover courses</b> — academic progression is on track.")
        elif carryovers > 4:
            risks_list.append(f"🔴 <b>{carryovers} carryover courses</b> — significant disengagement signal.")
        elif carryovers > 0:
            risks_list.append(f"⚠️ <b>{carryovers} carryover course(s)</b> — monitor academic progress closely.")

        if attendance >= 75:
            protective.append(f"✅ <b>Attendance {attendance}%</b> is above the 75% benchmark.")
        elif attendance < 50:
            risks_list.append(f"🔴 <b>Attendance {attendance}%</b> is critically low.")
        else:
            risks_list.append(f"⚠️ <b>Attendance {attendance}%</b> is below the 75% benchmark.")

        if sem_gpas[2] >= 3.0:
            protective.append(f"✅ <b>Semester 3 GPA {sem_gpas[2]:.2f}</b> — early trajectory is positive.")
        elif sem_gpas[2] < 2.0:
            risks_list.append(f"🔴 <b>Semester 3 GPA {sem_gpas[2]:.2f}</b> — critical early warning signal.")

        with xai_col1:
            st.markdown("**Protective Factors**")
            if protective:
                for p in protective:
                    st.markdown(f'<div class="xai-good">{p}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="xai-warn">No strong protective factors detected.</div>', unsafe_allow_html=True)

        with xai_col2:
            st.markdown("**Risk Factors — Intervention Needed**")
            if risks_list:
                for r in risks_list:
                    box = "xai-risk" if "🔴" in r else "xai-warn"
                    st.markdown(f'<div class="{box}">{r}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="xai-good">✅ No significant risk factors detected.</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── INTERVENTION RECOMMENDATIONS ──────────────────────────────────────
        st.subheader("📌 Recommended Interventions")
        interventions = []
        if tuition in ["Irregular", "Defaulter"]:
            interventions.append("💰 Connect student with bursary office, scholarship opportunities, or structured payment plan.")
        if ses == "Low":
            interventions.append("🤝 Refer to student welfare services for socioeconomic support assessment.")
        if cgpa_computed < 2.5:
            interventions.append("📚 Enrol in peer tutoring programme; schedule regular academic advisor meetings.")
        if carryovers > 2:
            interventions.append("📋 Review and restructure credit load to prevent further carryover accumulation.")
        if attendance < 75:
            interventions.append("🏫 Flag for attendance improvement programme; investigate barriers to attendance.")
        if assignment_rate < 60:
            interventions.append("📝 Faculty to provide structured assignment support and track submission compliance.")
        if not interventions:
            interventions.append("✅ No urgent interventions required. Continue routine monitoring.")

        for item in interventions:
            st.markdown(f'<div class="xai-good">{item}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE CHART — now uses real model data ──────────────────
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)

    if importance_df is not None:
        chart_data = importance_df.head(10).copy()
        chart_data["Feature"] = chart_data["Feature"].str.replace("_", " ")
        fig = px.bar(
            chart_data,
            x="Importance_Pct",
            y="Feature",
            orientation="h",
            color="Importance_Pct",
            color_continuous_scale="Blues",
            labels={"Importance_Pct": "Importance (%)", "Feature": "Feature"},
            title="Top 10 Predictors of Student Retention (Random Forest — Real Model)"
        )
    else:
        # Fallback to placeholder if CSV not found
        data = pd.DataFrame({
            "Feature": ["Tuition Consistency","Socioeconomic Status","Carryover Courses",
                        "Cumulative GPA","Semester 3 GPA"],
            "Importance": [25.9, 9.74, 8.96, 7.06, 5.45]
        })
        fig = px.bar(data, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")

    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BULK PREDICTION  — now uses real model
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("Bulk Prediction Upload")
    

    uploaded_file = st.file_uploader("Upload Student Dataset (format- file.csv).", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        if st.button("Run Bulk Prediction"):

            preloader    = show_preloader(logo)
            progress_bar = st.progress(0)
            status_text  = st.empty()

            stages = [
                "Loading student profile data...",
                "Validating academic records...",
                "Extracting behavioural features...",
                "Running predictive model...",
                "Evaluating academic risk indicators...",
                "Generating AI prediction report..."
            ]

            for i in range(100):
                percent     = i + 1
                stage_index = min(int((percent / 100) * len(stages)), len(stages) - 1)
                progress_bar.progress(percent)
                status_text.text(f"{stages[stage_index]}  {percent}%")
                time.sleep(0.25)

            preloader.markdown(
                "<h3 style='text-align:center;color:green;'>Prediction Completed....100%</h3>",
                unsafe_allow_html=True
            )
            status_text.success("Prediction Successful!")

            # ── Run real model on uploaded data ───────────────────────────────
            try:
                # Select only the feature columns the model expects
                available = [f for f in FEATURE_NAMES if f in df.columns]
                missing   = [f for f in FEATURE_NAMES if f not in df.columns]

                if missing:
                    st.warning(f"⚠️ {len(missing)} expected column(s) not found in upload: {missing}. "
                               "Filling with 0. For best results, upload the ML-ready CSV.")

                X_bulk = pd.DataFrame(0, index=df.index, columns=FEATURE_NAMES)
                for col in available:
                    X_bulk[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

                preds  = model.predict(X_bulk.values)
                probas = model.predict_proba(X_bulk.values)[:, 1]

                df["Retention_Prediction"] = np.where(preds == 1, "Retained", "Withdrawn")
                df["Retain_Probability_%"] = (probas * 100).round(1)
                df["Risk_Level"]           = pd.cut(
                    probas,
                    bins=[0, 0.40, 0.65, 1.0],
                    labels=["🔴 High Risk", "🟡 Moderate Risk", "🟢 Low Risk"]
                )

                # Performance class from CGPA if available
                if "Cumulative_GPA" in df.columns:
                    conditions = [
                        df["Cumulative_GPA"] >= 4.5,
                        df["Cumulative_GPA"] >= 3.5,
                        df["Cumulative_GPA"] >= 2.5,
                        df["Cumulative_GPA"] >= 1.5,
                    ]
                    choices = ["First Class", "Second Class Upper", "Second Class Lower", "Third Class"]
                    df["Performance_Class"] = np.select(conditions, choices, default="Fail/At-Risk")

                # Style results
                def highlight_risk(row):
                    if row["Retention_Prediction"] == "Withdrawn":
                        return ["background-color:#a63e32"] * len(row)
                    return [""] * len(row)

                styled_df = df.style.apply(highlight_risk, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                # Summary charts
                col1, col2 = st.columns(2)

                with col1:
                    fig1 = px.histogram(
                        df, x="Retention_Prediction",
                        title="Retention Prediction Distribution",
                        color="Retention_Prediction",
                        color_discrete_map={"Retained": "#2a9d8f", "Withdrawn": "#d62828"}
                    )
                    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    if "Performance_Class" in df.columns:
                        fig2 = px.histogram(
                            df, x="Performance_Class",
                            title="Academic Performance Class Distribution",
                            color="Performance_Class"
                        )
                        fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig2, use_container_width=True)

                # Download results
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Prediction Results as CSV",
                    data=csv_out,
                    file_name="bulk_prediction_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER  (your original — untouched)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
Designed &amp; Developed by<br>
<b>FABUNMI Kazeem Olaiya - 15/68TC001</b>
<br>Department of Educational Technology
University of Ilorin
<br>© 2026
</div>
""", unsafe_allow_html=True)
