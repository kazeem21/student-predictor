# =============================================================================
# app.py — Student Performance & Retention Predictor
# Streamlit Web Application
# Development of a Random Forest Model to Predict Students' Performance
# and Retention in Nigerian Universities
# FABUNMI, Kazeem Olaiya | Ph.D. Research | University of Ilorin
# =============================================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Retention Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F4F8FC; }

    /* Header banner */
    .header-box {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-box h1 { color: white; font-size: 1.8rem; margin: 0; }
    .header-box p  { color: #D6E4F0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Result cards */
    .card-retained {
        background: linear-gradient(135deg, #1a7a4a, #27ae60);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
    }
    .card-withdrawn {
        background: linear-gradient(135deg, #922b21, #e74c3c);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
    }
    .card-neutral {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        padding: 1.5rem; border-radius: 12px; color: white; text-align: center;
    }
    .card-title  { font-size: 0.85rem; opacity: 0.85; margin-bottom: 0.3rem; }
    .card-value  { font-size: 2rem; font-weight: 800; margin: 0; }
    .card-sub    { font-size: 0.8rem; opacity: 0.8; margin-top: 0.2rem; }

    /* Section headers */
    .section-title {
        color: #1F4E79; font-weight: 700; font-size: 1.05rem;
        border-left: 4px solid #C9A84C; padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem 0;
    }

    /* Info box */
    .info-box {
        background: #EBF5FB; border-left: 4px solid #2E75B6;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.88rem; color: #1F4E79; margin: 0.5rem 0;
    }

    /* Warning box */
    .warn-box {
        background: #FEF9E7; border-left: 4px solid #C9A84C;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.88rem; color: #7D6608; margin: 0.5rem 0;
    }

    /* Risk box */
    .risk-box {
        background: #FDEDEC; border-left: 4px solid #E74C3C;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.88rem; color: #922B21; margin: 0.5rem 0;
    }

    /* Footer */
    .footer {
        text-align: center; color: #7F8C8D;
        font-size: 0.78rem; margin-top: 2rem;
        padding-top: 1rem; border-top: 1px solid #D5D8DC;
    }

    /* Hide Streamlit default footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL & ASSETS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load("models/model.pkl")
    feature_names = joblib.load("encoders/feature_names.pkl")
    return model, feature_names

@st.cache_data
def load_importance():
    path = "data/feature_importance_rankings.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

try:
    model, FEATURE_NAMES = load_model()
    importance_df = load_importance()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model files. Ensure models/ and encoders/ folders exist.\n\n{e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# ENCODING MAPS  (must match ml_pipeline.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
ENCODE = {
    "Gender":                    {"Male": 1, "Female": 0},
    "Entry_Mode":                {"UTME": 0, "Direct Entry": 1, "Transfer": 2, "Part-Time": 3},
    "Socioeconomic_Status":      {"Low": 0, "Middle": 1, "High": 2},
    "Tuition_Payment_Consistency": {"Defaulter": 0, "Irregular": 1, "Consistent": 2},
    "Study_Mode":                {"Full-Time": 0, "Distance/Part-Time": 1},
    "Marital_Status":            {"Single": 0, "Married": 1},
}

PERFORMANCE_LABELS = {4: "First Class", 3: "Second Class Upper",
                      2: "Second Class Lower", 1: "Third Class", 0: "Fail / At-Risk"}


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 About This Tool")
    st.markdown("""
This application uses an **Optimised Random Forest model** trained on
longitudinal undergraduate data from Nigerian universities (2016–2024).

It predicts:
- Whether a student will be **retained or withdraw**
- The student's likely **academic performance class**
- The **confidence level** of each prediction

---
**Model Performance**
| Metric | Score |
|--------|-------|
| Accuracy | 91.67% |
| F1-Score | 94.09% |
| AUC-ROC  | 96.89% |

---
**Research**
FABUNMI, Kazeem Olaiya
Ph.D. Educational Technology
University of Ilorin, Nigeria

---
""")

    st.markdown("### 🔑 Top Predictors")
    if importance_df is not None:
        for _, row in importance_df.head(5).iterrows():
            pct = row["Importance_Pct"]
            bar = "█" * int(pct / 2)
            st.markdown(f"`{int(row['Rank'])}.` **{row['Feature'].replace('_',' ')}**")
            st.caption(f"{bar} {pct:.1f}%")

    st.markdown("---")
    st.caption("⚠️ This tool is designed to support, not replace, human academic judgment.")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🎓 Student Performance & Retention Predictor</h1>
    <p>Nigerian University Early Warning System — Powered by Optimised Random Forest (RF) Model</p>
    <p>Enter a student's socio-academic profile below to generate a real-time prediction.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📋 Student Profile Input</p>', unsafe_allow_html=True)

with st.form("prediction_form"):

    # ── Row 1: Demographics ──────────────────────────────────────────────────
    st.markdown("**Demographics & Admission**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        age = st.number_input("Age at Entry", min_value=16, max_value=55, value=20)
    with c3:
        marital = st.selectbox("Marital Status", ["Single", "Married"])
    with c4:
        entry_year = st.number_input("Entry Year", min_value=2016, max_value=2024, value=2020)

    # ── Row 2: Academic Entry ────────────────────────────────────────────────
    st.markdown("**Academic Entry Details**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        entry_mode = st.selectbox("Entry Mode",
                                  ["UTME", "Direct Entry", "Transfer", "Part-Time"])
    with c2:
        entry_level = st.selectbox("Entry Level", [100, 200])
    with c3:
        study_mode = st.selectbox("Study Mode", ["Full-Time", "Distance/Part-Time"])
    with c4:
        o_level = st.slider("O'Level Credits", min_value=4, max_value=9, value=6)

    # ── Row 3: Socioeconomic ─────────────────────────────────────────────────
    st.markdown("**Socioeconomic Background**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ses = st.selectbox("Socioeconomic Status",
                           ["Low", "Middle", "High"],
                           help="Household income/wealth category")
    with c2:
        tuition = st.selectbox("Tuition Payment",
                               ["Consistent", "Irregular", "Defaulter"],
                               help="Regularity of school fee payments — strongest predictor")
    with c3:
        jamb = st.number_input("JAMB Score", min_value=0, max_value=400, value=220,
                               help="Enter 0 if not applicable (Direct Entry / Transfer)")
    with c4:
        carryovers = st.number_input("Carryover Courses", min_value=0, max_value=20, value=0)

    # ── Row 4: Academic Performance ──────────────────────────────────────────
    st.markdown("**Semester GPA Records** *(0.00 – 5.00)*")
    gpa_cols = st.columns(8)
    sem_labels = ["Sem 1","Sem 2","Sem 3","Sem 4","Sem 5","Sem 6","Sem 7","Sem 8"]
    sem_gpas = []
    for i, col in enumerate(gpa_cols):
        with col:
            gpa = col.number_input(sem_labels[i], min_value=0.0, max_value=5.0,
                                   value=3.0, step=0.01, format="%.2f")
            sem_gpas.append(gpa)

    # ── Row 5: Engagement metrics ────────────────────────────────────────────
    st.markdown("**Academic Engagement**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        attendance = st.slider("Attendance Rate (%)", 0, 100, 75)
    with c2:
        portal_logins = st.number_input("Portal Login Count", min_value=0,
                                        max_value=300, value=45)
    with c3:
        assignment_rate = st.slider("Assignment Submission Rate (%)", 0, 100, 80)
    with c4:
        avg_credits = st.number_input("Avg Credit Units/Semester",
                                      min_value=10.0, max_value=30.0, value=18.0, step=0.5)

    # ── Submit ───────────────────────────────────────────────────────────────
    st.markdown("")
    submitted = st.form_submit_button("🔍  Generate Prediction",
                                      use_container_width=True,
                                      type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────────────────
if submitted:

    # Build feature vector in exact training order
    cgpa       = round(np.mean(sem_gpas), 2)
    total_cu   = int(avg_credits * 8)

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
        cgpa,
        avg_credits,
        total_cu,
        attendance,
        portal_logins,
        assignment_rate,
        carryovers,
    ]])

    # Predict
    retention_pred  = model.predict(feature_vector)[0]
    retention_proba = model.predict_proba(feature_vector)[0]
    retention_conf  = retention_proba[retention_pred] * 100

    retained = retention_pred == 1

    # Derive performance class from CGPA
    if cgpa >= 4.5:   perf_label = "First Class"
    elif cgpa >= 3.5: perf_label = "Second Class Upper"
    elif cgpa >= 2.5: perf_label = "Second Class Lower"
    elif cgpa >= 1.5: perf_label = "Third Class"
    else:             perf_label = "Fail / At-Risk"

    # Risk level
    withdraw_prob = retention_proba[0] * 100
    if withdraw_prob >= 60:   risk = "🔴 HIGH RISK"
    elif withdraw_prob >= 35: risk = "🟡 MODERATE RISK"
    else:                     risk = "🟢 LOW RISK"

    # ── Results header ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-title">📊 Prediction Results</p>',
                unsafe_allow_html=True)

    # ── Top metric cards ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card_class = "card-retained" if retained else "card-withdrawn"
        verdict    = "RETAINED" if retained else "WITHDRAWN"
        st.markdown(f"""
        <div class="{card_class}">
            <div class="card-title">Retention Prediction</div>
            <div class="card-value">{verdict}</div>
            <div class="card-sub">Primary Outcome</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card-neutral">
            <div class="card-title">Model Confidence</div>
            <div class="card-value">{retention_conf:.1f}%</div>
            <div class="card-sub">Prediction certainty</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card-neutral">
            <div class="card-title">Performance Class</div>
            <div class="card-value" style="font-size:1.3rem">{perf_label}</div>
            <div class="card-sub">Based on CGPA {cgpa:.2f}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="card-neutral">
            <div class="card-title">Dropout Risk Level</div>
            <div class="card-value" style="font-size:1.2rem">{risk}</div>
            <div class="card-sub">Withdrawal probability: {withdraw_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability gauge chart + Feature contribution ───────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-title">📈 Retention Probability Breakdown</p>',
                    unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        categories = ["Withdrawn", "Retained"]
        values     = [retention_proba[0] * 100, retention_proba[1] * 100]
        colours    = ["#E74C3C", "#27AE60"]
        bars = ax.barh(categories, values, color=colours,
                       edgecolor="white", linewidth=1.5, height=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontweight="bold",
                    fontsize=12, color="#2C3E50")
        ax.set_xlim(0, 115)
        ax.axvline(50, color="#BDC3C7", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xlabel("Probability (%)", fontsize=10)
        ax.set_title("Model Output Probabilities", fontsize=11,
                     fontweight="bold", color="#1F4E79")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_facecolor("#F9FAFB")
        fig.patch.set_facecolor("white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown('<p class="section-title">🏆 Global Feature Importance (Top 10)</p>',
                    unsafe_allow_html=True)
        if importance_df is not None:
            top10 = importance_df.head(10).sort_values("Importance")
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            colours2 = ["#C9A84C" if i >= 7 else "#1F4E79"
                        for i in range(len(top10))]
            ax2.barh(top10["Feature"].str.replace("_", " "),
                     top10["Importance_Pct"],
                     color=colours2, edgecolor="white", linewidth=0.5)
            ax2.set_xlabel("Importance (%)", fontsize=9)
            ax2.set_title("Top 10 Predictors of Retention", fontsize=11,
                          fontweight="bold", color="#1F4E79")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.set_facecolor("#F9FAFB")
            fig2.patch.set_facecolor("white")
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

    # ── XAI — Explainability panel ───────────────────────────────────────────
    st.markdown('<p class="section-title">🧠 Explainability — Why This Prediction?</p>',
                unsafe_allow_html=True)

    reasons = []
    risks   = []

    # Tuition
    if tuition == "Consistent":
        reasons.append("✅ **Tuition Payment** is Consistent — strongest positive retention signal.")
    elif tuition == "Irregular":
        risks.append("⚠️ **Tuition Payment** is Irregular — moderate dropout risk factor.")
    else:
        risks.append("🔴 **Tuition Payment** is Defaulter — highest single dropout risk factor.")

    # SES
    if ses == "High":
        reasons.append("✅ **Socioeconomic Status** is High — reduces financial dropout risk.")
    elif ses == "Low":
        risks.append("⚠️ **Socioeconomic Status** is Low — increases vulnerability to dropout.")

    # CGPA
    if cgpa >= 3.5:
        reasons.append(f"✅ **CGPA of {cgpa:.2f}** is strong — students above 3.5 persist at high rates.")
    elif cgpa < 2.0:
        risks.append(f"🔴 **CGPA of {cgpa:.2f}** is critically low — high academic failure risk.")
    elif cgpa < 2.5:
        risks.append(f"⚠️ **CGPA of {cgpa:.2f}** is below the 2.5 threshold — moderate risk.")

    # Carryovers
    if carryovers == 0:
        reasons.append("✅ **No carryover courses** — academic progression is on track.")
    elif carryovers > 4:
        risks.append(f"🔴 **{carryovers} carryover courses** — significant academic disengagement signal.")
    elif carryovers > 0:
        risks.append(f"⚠️ **{carryovers} carryover course(s)** — monitor academic progress closely.")

    # Attendance
    if attendance >= 75:
        reasons.append(f"✅ **Attendance of {attendance}%** is above the 75% benchmark.")
    elif attendance < 50:
        risks.append(f"🔴 **Attendance of {attendance}%** is critically low.")
    else:
        risks.append(f"⚠️ **Attendance of {attendance}%** is below the recommended 75%.")

    # Semester 3 GPA
    if sem_gpas[2] >= 3.0:
        reasons.append(f"✅ **Semester 3 GPA of {sem_gpas[2]:.2f}** — early trajectory is positive.")
    elif sem_gpas[2] < 2.0:
        risks.append(f"🔴 **Semester 3 GPA of {sem_gpas[2]:.2f}** — critical early warning signal.")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.markdown("**Protective Factors (Positive Signals)**")
        if reasons:
            for r in reasons:
                st.markdown(f'<div class="info-box">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">No strong protective factors detected.</div>',
                        unsafe_allow_html=True)

    with exp_col2:
        st.markdown("**Risk Factors (Intervention Needed)**")
        if risks:
            for r in risks:
                box_class = "risk-box" if "🔴" in r else "warn-box"
                st.markdown(f'<div class="{box_class}">{r}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-box">✅ No significant risk factors detected.</div>',
                        unsafe_allow_html=True)

    # ── Intervention Recommendations ─────────────────────────────────────────
    st.markdown('<p class="section-title">📌 Recommended Interventions</p>',
                unsafe_allow_html=True)

    interventions = []
    if tuition in ["Irregular", "Defaulter"]:
        interventions.append("💰 **Financial Support**: Connect student with bursary office, scholarship opportunities, or payment plan arrangements.")
    if ses == "Low":
        interventions.append("🤝 **Welfare Check**: Refer to student welfare services for socioeconomic support assessment.")
    if cgpa < 2.5:
        interventions.append("📚 **Academic Support**: Enrol in peer tutoring programme and schedule regular academic advisor meetings.")
    if carryovers > 2:
        interventions.append("📋 **Course Load Review**: Academic advisor should review and restructure credit load to prevent further accumulation.")
    if attendance < 75:
        interventions.append("🏫 **Attendance Monitoring**: Flag for attendance improvement programme; investigate barriers to physical attendance.")
    if assignment_rate < 60:
        interventions.append("📝 **Assignment Engagement**: Faculty to provide structured assignment support and track submission compliance.")
    if not interventions:
        interventions.append("✅ **No urgent interventions required.** Continue routine monitoring and encourage sustained performance.")

    for item in interventions:
        st.markdown(f'<div class="info-box">{item}</div>', unsafe_allow_html=True)

    # ── Student profile summary ───────────────────────────────────────────────
    with st.expander("📄 View Full Input Profile Summary"):
        summary = {
            "Gender": gender, "Age at Entry": age, "Marital Status": marital,
            "Entry Mode": entry_mode, "Entry Level": entry_level,
            "Entry Year": entry_year, "Study Mode": study_mode,
            "Socioeconomic Status": ses, "Tuition Payment": tuition,
            "JAMB Score": jamb, "O'Level Credits": o_level,
            "Attendance (%)": attendance, "Assignment Rate (%)": assignment_rate,
            "Portal Logins": portal_logins, "Carryover Courses": carryovers,
            "Avg Credits/Semester": avg_credits, "Computed CGPA": cgpa,
            **{f"Semester {i+1} GPA": sem_gpas[i] for i in range(8)}
        }
        profile_df = pd.DataFrame(summary.items(), columns=["Field", "Value"])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Developed for Ph.D. Research — FABUNMI, Kazeem Olaiya | Department of Educational Technology |
    University of Ilorin, Nigeria | 2026<br>
    <em>Model: Optimised Random Forest | Accuracy: 91.67% | AUC-ROC: 96.89% |
    Training Data: 1,500 Nigerian Undergraduate Records (2016–2024)</em>
</div>
""", unsafe_allow_html=True)
