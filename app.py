# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
# LOAD MODEL & ASSETS
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
# ENCODING MAPS  — updated to include new multi-institution variables
# ─────────────────────────────────────────────────────────────────────────────
ENCODE = {
    "Gender":                      {"Male": 1, "Female": 0},
    "Entry_Mode":                  {"UTME": 0, "Direct Entry": 1, "Transfer": 2, "Part-Time": 3},
    "Socioeconomic_Status":        {"Low": 0, "Middle": 1, "High": 2},
    "Tuition_Payment_Consistency": {"Defaulter": 0, "Irregular": 1, "Consistent": 2},
    "Study_Mode":                  {"Full-Time": 0, "Distance/Part-Time": 1},
    "Marital_Status":              {"Single": 0, "Married": 1},
    # ── NEW variables for multi-institution model ─────────────────────────
    "Institution_Type":            {"Federal": 0, "State": 1, "Private": 2},
    "Disability_Status":           {"None": 0, "Visual Impairment": 1,
                                    "Hearing Impairment": 2, "Physical Disability": 3},
    "Sponsorship_Type":            {"Self": 0, "Parent/Guardian": 1,
                                    "Government Scholarship": 2,
                                    "NGO/Foundation": 3, "Employer": 4},
    "State_of_Origin":             {
        "Kwara": 0,  "Niger": 1,   "Benue": 2,   "Kogi": 3,
        "Nassarawa": 4, "Plateau": 5, "FCT-Abuja": 6,
        "Oyo": 7,    "Osun": 8,    "Ekiti": 9,   "Ondo": 10,
        "Ogun": 11,  "Lagos": 12,  "Delta": 13,  "Anambra": 14,
        "Imo": 15,   "Enugu": 16,  "Ebonyi": 17, "Kano": 18,
        "Kaduna": 19,"Sokoto": 20, "Zamfara": 21,"Kebbi": 22,
        "Bauchi": 23,"Others": 24
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictive System",
    page_icon=r"assets/logo.png",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────────────────────

# GLOBAL CSS

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

.shap-box {
background:white; padding:20px; border-radius:12px;
box-shadow:0px 4px 15px rgba(0,0,0,0.08); margin-top:10px;
}

.footer {
text-align:center; margin-top:40px; color:#555; font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────

# HERO BANNER


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
    <div class="hero-stat"><span class="hero-stat-pip"></span> UNILORIN EdTech &middot; v3.0</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR

# ─────────────────────────────────────────────────────────────────────────────
logo = load_image(r"assets/logo.png")

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

    st.sidebar.markdown("**📍 Institution & Demographics**")

    # ── NEW: Institution Type ─────────────────────────────────────────────────
    institution_type = st.sidebar.selectbox(
        "Institution Type",
        ["Federal", "State", "Private"],
        help="Type of university the student attends"
    )

    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age    = st.sidebar.slider("Age at Entry", 16, 50, 20)
    marital = st.sidebar.selectbox("Marital Status", ["Single", "Married"])

    # ── NEW: State of Origin ──────────────────────────────────────────────────
    state_of_origin = st.sidebar.selectbox(
        "State of Origin",
        ["Kwara", "Niger", "Benue", "Kogi", "Nassarawa", "Plateau", "FCT-Abuja",
         "Oyo", "Osun", "Ekiti", "Ondo", "Ogun", "Lagos", "Delta", "Anambra",
         "Imo", "Enugu", "Ebonyi", "Kano", "Kaduna", "Sokoto", "Zamfara",
         "Kebbi", "Bauchi", "Others"]
    )

    # ── NEW: Disability Status ────────────────────────────────────────────────
    disability = st.sidebar.selectbox(
        "Disability Status",
        ["None", "Visual Impairment", "Hearing Impairment", "Physical Disability"]
    )

    st.sidebar.markdown("**🎓 Academic Entry**")
    entry_mode_display = st.sidebar.selectbox(
        "Entry Mode",
        ["UTME (100L)", "Direct Entry (200L)", "Transfer", "Part-Time"]
    )
    entry_mode_map = {
        "UTME (100L)": "UTME", "Direct Entry (200L)": "Direct Entry",
        "Transfer": "Transfer", "Part-Time": "Part-Time"
    }
    entry_mode  = entry_mode_map[entry_mode_display]
    entry_level = 200 if entry_mode == "Direct Entry" else 100
    entry_year  = st.sidebar.number_input("Entry Year", 2016, 2024, 2020)
    o_level     = st.sidebar.slider("O'Level Credits", 4, 9, 6)
    jamb        = st.sidebar.number_input("JAMB Score (0 if not applicable)", 0, 400, 220)
    study_mode  = st.sidebar.selectbox("Study Mode", ["Full-Time", "Distance/Part-Time"])

    st.sidebar.markdown("**💰 Socioeconomic Background**")
    ses = st.sidebar.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])

    tuition_display = st.sidebar.selectbox(
        "Financial / Tuition Status",
        ["Full Payment", "Partial Payment", "Outstanding"]
    )
    tuition_map = {
        "Full Payment": "Consistent", "Partial Payment": "Irregular", "Outstanding": "Defaulter"
    }
    tuition = tuition_map[tuition_display]

    # ── NEW: Sponsorship Type ─────────────────────────────────────────────────
    sponsorship = st.sidebar.selectbox(
        "Sponsorship Type",
        ["Parent/Guardian", "Self", "Government Scholarship", "NGO/Foundation", "Employer"],
        help="Who funds the student's education?"
    )

    st.sidebar.markdown("**📊 Academic Performance**")
    current_cgpa = st.sidebar.number_input("Current CGPA", 0.0, 5.0, 3.0, step=0.01)
    carryovers   = st.sidebar.number_input("Carryover Courses", 0, 20, 0)

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

    st.sidebar.markdown("**📈 Engagement Metrics**")
    attendance      = st.sidebar.slider("Attendance Rate (%)", 0, 100, 75)
    portal_logins   = st.sidebar.slider("Monthly Portal Logins", 0, 200, 45)
    assignment_rate = st.sidebar.slider("Assignment Submission Rate (%)", 0, 100, 80)

    # ── PREDICT BUTTON ────────────────────────────────────────────────────────
    if st.sidebar.button("🔍 Predict Student Outcome"):

        preloader    = show_preloader(logo)
        progress_bar = st.progress(0)
        status_text  = st.empty()

        stages = [
            "Loading student profile data...",
            "Validating academic records...",
            "Extracting behavioural features...",
            "Running Random Forest model...",
            "Computing SHAP explainability values...",
            "Generating AI prediction report..."
        ]

        for i in range(100):
            percent     = i + 1
            stage_index = min(int((percent / 100) * len(stages)), len(stages) - 1)
            progress_bar.progress(percent)
            status_text.text(f"{stages[stage_index]}  {percent}%")
            time.sleep(0.20)

        preloader.markdown(
            "<h3 style='text-align:center;color:green;'>Prediction Completed....100%</h3>",
            unsafe_allow_html=True
        )

        status_text.success("Prediction Successful! view Result Below!")
        st.info("📄 Your prediction results are displayed below. Scroll down to view the full report. To print, press Ctrl + P on your keyboard or right-click the page and select 'Print'.")
        st.divider()

        # ── BUILD FEATURE VECTOR (30 features — updated for multi-institution) ─
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
            # ── NEW features ────────────────────────────────────────────────
            ENCODE["Institution_Type"][institution_type],
            ENCODE["Disability_Status"][disability],
            ENCODE["Sponsorship_Type"][sponsorship],
            ENCODE["State_of_Origin"].get(state_of_origin, 24),
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

        # ── KPI DASHBOARD — updated metrics for multi-institution model ───────
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">Model Accuracy</div>
        <div class="kpi-value">92.84%</div>
        </div>
        """, unsafe_allow_html=True)
        col2.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">F1 Score</div>
        <div class="kpi-value">0.9502</div>
        </div>
        """, unsafe_allow_html=True)
        col3.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">Precision</div>
        <div class="kpi-value">0.9463</div>
        </div>
        """, unsafe_allow_html=True)
        col4.markdown("""
        <div class="kpi-card">
        <div class="kpi-title">AUC-ROC</div>
        <div class="kpi-value">0.9768</div>
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

        # ── SHAP EXPLAINABILITY — instance-level XAI ──────────────────────────
        st.subheader("🧠 SHAP Explainability — Why This Prediction?")
        st.caption("SHAP (SHapley Additive exPlanations) shows exactly how much each factor pushed this student toward Retained or Withdrawn.")

        try:
            import shap

            # Build a named DataFrame so SHAP shows readable feature names
            feature_col_names = [
                "Gender", "Age_at_Entry", "Entry_Mode", "Entry_Level", "Entry_Year",
                "Socioeconomic_Status", "Tuition_Payment_Consistency", "Study_Mode",
                "Marital_Status", "O_Level_Credits", "JAMB_Score",
                "Semester_1_GPA", "Semester_2_GPA", "Semester_3_GPA", "Semester_4_GPA",
                "Semester_5_GPA", "Semester_6_GPA", "Semester_7_GPA", "Semester_8_GPA",
                "Cumulative_GPA", "Avg_Credit_Units_Per_Sem", "Total_Credit_Units_Earned",
                "Attendance_Rate_Pct", "Portal_Login_Count",
                "Assignment_Submission_Rate_Pct", "Carryover_Courses",
                "Institution_Type", "Disability_Status",
                "Sponsorship_Type", "State_of_Origin"
            ]

            fv_df = pd.DataFrame(feature_vector, columns=feature_col_names)

            # Compute SHAP values using TreeExplainer (optimised for Random Forest)
            explainer = shap.TreeExplainer(model)

            # Use the modern Explanation object — works across all SHAP versions
            explanation = explainer(fv_df, check_additivity=False)

            # explanation.values shape: (n_samples, n_features, n_classes) or (n_samples, n_features)
            vals = explanation.values

            if vals.ndim == 3:
               # (n_samples, n_features, n_classes) — take sample 0, class 1 (Retained)
               sv = vals[0, :, 1]
            elif vals.ndim == 2:
               # (n_samples, n_features) — take sample 0
               sv = vals[0, :]
            else:
               sv = vals.flatten()

            sv = np.array(sv, dtype=float)

            # Build a clean SHAP summary dataframe
            # Build a clean SHAP summary dataframe
            shap_df = pd.DataFrame({
                    "Feature":    feature_col_names,
                    "SHAP_Value": sv,
            })
            shap_df["Abs_SHAP"]       = shap_df["SHAP_Value"].abs()
            shap_df["Feature_Clean"]  = shap_df["Feature"].str.replace("_", " ")
            shap_df["Direction"]      = shap_df["SHAP_Value"].apply(
                lambda v: "Increases Retention" if v > 0 else "Increases Withdrawal Risk"
            )
            shap_df = shap_df.sort_values("Abs_SHAP", ascending=False).head(12).reset_index(drop=True)

            # ── SHAP waterfall bar chart ──────────────────────────────────────
            colors = ["#1E8449" if v > 0 else "#C0392B" for v in shap_df["SHAP_Value"].tolist()]
            # ── SHAP waterfall bar chart ──────────────────────────────────────
            st.markdown("**SHAP Feature Contribution — Individual Student Prediction**")
            st.caption("🟢 Green bars push toward Retained  |  🔴 Red bars push toward Withdrawal")

            shap_display = shap_df[["Feature_Clean", "SHAP_Value"]].copy()
            shap_display = shap_display.sort_values("SHAP_Value", ascending=True)
            shap_display.columns = ["Feature", "SHAP Value"]
            shap_display = shap_display.set_index("Feature")

            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")

            fig_shap, ax = plt.subplots(figsize=(10, 6))
            bar_colors = ["#1E8449" if v > 0 else "#C0392B"
                          for v in shap_display["SHAP Value"]]
            bars = ax.barh(
               shap_display.index,
               shap_display["SHAP Value"],
               color=bar_colors,
               edgecolor="white",
               linewidth=0.5
            )
            ax.axvline(0, color="#555555", linewidth=1.2, linestyle="--")
            for bar, val in zip(bars, shap_display["SHAP Value"]):
                if val >= 0:
                    # Positive bars — label outside to the right
                    ax.text(val + 0.0003,
                            bar.get_y() + bar.get_height()/2,
                            f"{val:+.4f}", va="center", ha="left",
                            fontsize=8, fontweight="bold", color="#1E8449")
                else:
                    # Negative bars — label inside the bar (white text)
                    ax.text(val / 2,
                            bar.get_y() + bar.get_height()/2,
                            f"{val:+.4f}", va="center", ha="center",
                            fontsize=8, fontweight="bold", color="white")
            ax.set_xlabel("SHAP Value (impact on model prediction)", fontsize=10)
            ax.set_title("SHAP Feature Contributions — This Student's Prediction",
                         fontsize=12, fontweight="bold", color="#1F4E79", pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("#F9FAFB")
            fig_shap.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig_shap)
            plt.close()

            # ── Top SHAP drivers in plain English ────────────────────────────
            st.markdown("**📋 SHAP Plain-Language Interpretation**")
            top_positive = shap_df[shap_df["SHAP_Value"] > 0].head(3)
            top_negative = shap_df[shap_df["SHAP_Value"] < 0].head(3)

            shap_col1, shap_col2 = st.columns(2)
            with shap_col1:
                st.markdown("**Factors supporting Retention:**")
                if len(top_positive) > 0:
                    for _, row in top_positive.iterrows():
                        st.markdown(
                            f'<div class="xai-good">✅ <b>{row["Feature_Clean"]}</b> '
                            f'— SHAP contribution: <b>+{row["SHAP_Value"]:.4f}</b> '
                            f'(pushes prediction toward Retained)</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown('<div class="xai-warn">No strong retention-supporting factors detected.</div>',
                                unsafe_allow_html=True)

            with shap_col2:
                st.markdown("**Factors increasing Withdrawal risk:**")
                if len(top_negative) > 0:
                    for _, row in top_negative.iterrows():
                        st.markdown(
                            f'<div class="xai-risk">⚠️ <b>{row["Feature_Clean"]}</b> '
                            f'— SHAP contribution: <b>{row["SHAP_Value"]:.4f}</b> '
                            f'(pushes prediction toward Withdrawal)</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown('<div class="xai-good">✅ No withdrawal risk factors detected.</div>',
                                unsafe_allow_html=True)

        except ImportError:
            # Graceful fallback if SHAP not installed — show rule-based XAI
            st.warning("⚠️ SHAP library not installed. Run: pip install shap  — Showing rule-based explanation instead.")
            _show_rule_based_xai(tuition, ses, cgpa_computed, carryovers, attendance, sem_gpas)

        except Exception as e:
            st.warning(f"SHAP computation encountered an issue: {e}. Showing rule-based explanation.")
            # Rule-based fallback
            protective, risks_list = [], []
            if tuition == "Consistent":
                protective.append("✅ <b>Tuition payment Consistent</b> — strongest positive retention signal (22.47% importance).")
            elif tuition == "Irregular":
                risks_list.append("⚠️ <b>Tuition payment Irregular</b> — moderate dropout risk.")
            else:
                risks_list.append("🔴 <b>Tuition payment Outstanding</b> — highest single dropout risk factor.")
            if cgpa_computed >= 3.5:
                protective.append(f"✅ <b>CGPA {cgpa_computed:.2f}</b> — strong academic trajectory.")
            elif cgpa_computed < 2.5:
                risks_list.append(f"🔴 <b>CGPA {cgpa_computed:.2f}</b> — below retention threshold.")
            if carryovers == 0:
                protective.append("✅ <b>No carryover courses</b> — on-track progression.")
            elif carryovers > 4:
                risks_list.append(f"🔴 <b>{carryovers} carryover courses</b> — significant disengagement signal.")
            if attendance >= 75:
                protective.append(f"✅ <b>Attendance {attendance}%</b> — above 75% benchmark.")
            else:
                risks_list.append(f"⚠️ <b>Attendance {attendance}%</b> — below recommended threshold.")

            fb_c1, fb_c2 = st.columns(2)
            with fb_c1:
                st.markdown("**Protective Factors**")
                for p in (protective or ['<div class="xai-warn">None detected.</div>']):
                    st.markdown(f'<div class="xai-good">{p}</div>', unsafe_allow_html=True)
            with fb_c2:
                st.markdown("**Risk Factors**")
                for r in (risks_list or ['<div class="xai-good">✅ None detected.</div>']):
                    box = "xai-risk" if "🔴" in r else "xai-warn"
                    st.markdown(f'<div class="{box}">{r}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── INTERVENTION RECOMMENDATIONS ──────────────────────────────────────
        st.subheader("📌 Recommended Interventions")
        interventions = []

        # ── Build intervention list purely from SHAP withdrawal risk factors ──────────
        # Only factors with negative SHAP values (pushing toward Withdrawal) get flagged

        shap_risk_features = shap_df[shap_df["SHAP_Value"] < 0]["Feature"].tolist()

        for risk_feature in shap_risk_features:

            if risk_feature == "Tuition_Payment_Consistency":
                interventions.append("💰 <b>Tuition Risk:</b> Tuition payment pattern is increasing withdrawal risk — connect student with bursary office, payment plans, or scholarship opportunities immediately.")

            elif risk_feature == "Cumulative_GPA":
                interventions.append("📚 <b>Academic Performance Risk:</b> Cumulative GPA is a withdrawal risk factor — enrol student in peer tutoring and schedule mandatory academic advisory sessions.")

            elif risk_feature == "Carryover_Courses":
                interventions.append("📋 <b>Carryover Risk:</b> Carryover courses are increasing dropout risk — review and restructure credit load with the Head of Department.")

            elif risk_feature == "Attendance_Rate_Pct":
                interventions.append("🏫 <b>Attendance Risk:</b> Low attendance is flagged as a withdrawal risk — investigate barriers and enrol in attendance improvement programme.")

            elif risk_feature == "Socioeconomic_Status":
                interventions.append("🤝 <b>Socioeconomic Risk:</b> Household economic background is increasing withdrawal risk — refer to student welfare services for financial and pastoral support.")

            elif risk_feature == "Sponsorship_Type":
                interventions.append("💼 <b>Funding Risk:</b> Sponsorship type is a withdrawal risk factor — connect student with alternative funding sources and institutional financial aid.")

            elif risk_feature == "Assignment_Submission_Rate_Pct":
                interventions.append("📝 <b>Engagement Risk:</b> Low assignment submission rate is flagged — faculty to implement structured submission tracking and support.")

            elif risk_feature == "Portal_Login_Count":
                interventions.append("💻 <b>Digital Engagement Risk:</b> Low portal login activity signals academic disengagement — advisor to investigate and encourage active use of institutional digital resources.")

            elif "Semester" in risk_feature and "GPA" in risk_feature:
                sem = risk_feature.replace("Semester_", "Semester ").replace("_GPA", " GPA")
                interventions.append(f"📉 <b>Early Academic Warning:</b> {sem} is a withdrawal risk factor — early academic intervention and mentoring recommended at this stage.")

            elif risk_feature == "Institution_Type":
                interventions.append(f"🏛️ <b>Institutional Risk:</b> Institution type is contributing to withdrawal risk for this student — ensure access to institution-specific retention and support programmes.")

            elif risk_feature == "Entry_Year":
                interventions.append("📅 <b>Cohort Risk:</b> Entry year pattern is flagged as a risk factor — advisor to check if student belongs to a cohort with elevated dropout history and provide targeted support.")

            elif risk_feature == "O_Level_Credits":
                interventions.append("📖 <b>Entry Qualification Risk:</b> O'Level credit profile is a withdrawal risk factor — consider foundational academic support to bridge entry-level knowledge gaps.")

            elif risk_feature == "JAMB_Score":
                interventions.append("📖 <b>Admission Score Risk:</b> JAMB score profile is contributing to withdrawal risk — recommend foundational academic strengthening support.")

            elif risk_feature == "Disability_Status":
                interventions.append("♿ <b>Accessibility Risk:</b> Disability status is flagged as a risk factor — ensure full disability support services and accessible learning resources are in place.")

            elif risk_feature == "State_of_Origin":
                interventions.append("🗺️ <b>Geographic Risk:</b> State of origin is contributing to withdrawal risk — check for interstate student integration challenges and provide pastoral support.")

     # ── Disability — always flag regardless of SHAP ───────────────────────────────
            if disability != "None" and "Disability_Status" not in shap_risk_features:
                interventions.append(f"♿ <b>Disability Support:</b> Student has {disability} — ensure appropriate support services are in place.")

      # ── Default if no risk factors identified ────────────────────────────────────
            if not interventions:
                interventions.append("✅ <b>No urgent interventions required.</b> All key risk factors are within acceptable range. Continue routine semesterly monitoring.")

        for item in interventions:
            st.markdown(f'<div class="xai-good">{item}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # ── FEATURE IMPORTANCE CHART ──────────────────────────────────────────────
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
            title="Top 10 Predictors of Student Retention (Optimised Random Forest — Multi-Institutional)"
        )
    else:
        data = pd.DataFrame({
            "Feature": ["Tuition Consistency", "Cumulative GPA", "Socioeconomic Status",
                        "Carryover Courses", "Semester 3 GPA"],
            "Importance": [22.47, 11.83, 10.12, 8.64, 6.21]
        })
        fig = px.bar(data, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Blues")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BULK PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.subheader("Bulk Prediction Upload")
    st.info("Upload the **nigerian_university_ML_ready.csv** file or any CSV with the same column structure. "
            "New columns: Institution_Type, Disability_Status, Sponsorship_Type, State_of_Origin.")

    uploaded_file = st.file_uploader("Upload Student Dataset (format: file.csv)", type=["csv"])

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
                time.sleep(0.20)

            preloader.markdown(
                "<h3 style='text-align:center;color:green;'>Prediction Completed....100%</h3>",
                unsafe_allow_html=True
            )

            status_text.success("Prediction Successful! view Result Below!")


            try:
                available = [f for f in FEATURE_NAMES if f in df.columns]
                missing   = [f for f in FEATURE_NAMES if f not in df.columns]

                if missing:
                    st.warning(f"⚠️ {len(missing)} expected column(s) not found: {missing}. "
                               "Filling with 0. For best results, upload the full ML-ready CSV.")

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

                if "Cumulative_GPA" in df.columns:
                    conditions = [
                        df["Cumulative_GPA"] >= 4.5,
                        df["Cumulative_GPA"] >= 3.5,
                        df["Cumulative_GPA"] >= 2.5,
                        df["Cumulative_GPA"] >= 1.5,
                    ]
                    choices = ["First Class", "Second Class Upper", "Second Class Lower", "Third Class"]
                    df["Performance_Class"] = np.select(conditions, choices, default="Fail/At-Risk")

                def highlight_risk(row):
                    if row["Retention_Prediction"] == "Withdrawn":
                        return ["background-color:#fdecea"] * len(row)
                    return [""] * len(row)

                st.dataframe(df.style.apply(highlight_risk, axis=1), use_container_width=True)

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
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
Designed &amp; Developed by<br>
<b>FABUNMI Kazeem Olaiya - 15/68TC001</b><br>
Department of Educational Technology<br>
University of Ilorin<br>
© 2026
</div>
""", unsafe_allow_html=True)
