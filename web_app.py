import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Student Performance Analysis", layout="centered")

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #FFF9E6;
}

/* Header title styling */
h1 {
    color: #5A4B00;
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #7A6500;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFF3CC;
}

/* Buttons */
.stButton>button {
    background-color: #E6C200;
    color: black;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #D4AF00;
}

/* Metric boxes */
[data-testid="stMetric"] {
    background-color: #FFFDF5;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #F0E0A0;
}

/* Force all text readable */
body {
    color: #4A3F00 !important;
}

/* Sidebar general text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #4A3F00 !important;
}

/* Sidebar selectbox label */
section[data-testid="stSidebar"] label {
    color: white !important;
    font-weight: 600;
}

/* Force sidebar selectbox label to white */
section[data-testid="stSidebar"] div[data-testid="stSelectbox"] label {
    color: white !important;
    font-weight: 600 !important;
}

/* Markdown text fix */
[data-testid="stMarkdownContainer"] {
    color: #4A3F00 !important;
}

/* ===== FORCE ALL INPUTS DARK ===== */

/* Number input */
div[data-testid="stNumberInput"] input {
    background-color: #2E2E2E !important;
    color: white !important;
}

/* Slider */
div[data-testid="stSlider"] * {
    color: white !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #2E2E2E !important;
    color: white !important;
}

div[data-baseweb="select"] span {
    color: white !important;
}

ul[role="listbox"] {
    background-color: #2E2E2E !important;
}

ul[role="listbox"] li {
    color: white !important;
}

/* Highlight selected navigation item */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span {
    color: white !important;
}
            
/* FORCE Navigation title white (final override) */
section[data-testid="stSidebar"] div[data-baseweb="form-control"] span {
    color: white !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# Load data
df = pd.read_excel("Student_performance_data _.csv.xlsx")

@st.cache_resource

def train_model():
    features = df[["StudyTimeWeekly", "Absences", "Tutoring",
                   "ParentalSupport", "Extracurricular",
                   "Sports", "Music", "Volunteering"]]
    
    target = df["GPA"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, features.columns, mae, r2

model, feature_columns, mae, r2 = train_model()

st.title("🎓 Student Academic Performance Portal")

st.markdown("""
<div style="
background-color:#FFF3CC;
padding:15px;
border-radius:10px;
text-align:center;
font-weight:600;
">
Welcome to the Student Academic Performance Evaluation Portal
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🎓 Academic Intelligence System")
st.sidebar.markdown("Empowering Data-Driven Education")
st.sidebar.markdown("---")

st.markdown("### Empowering Data-Driven Learning Insights")
st.markdown("---")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Analytics Dashboard", "Predict GPA", "About"]
)

if menu == "Dashboard":

    st.markdown("### 📈 Model Overview")
    st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Students", len(df))
    col2.metric("Average GPA", round(df["GPA"].mean(), 2))
    col3.metric("Model Accuracy (MAE)", round(mae, 3))
    col4.metric("Model Strength (R² Score)", round(r2, 3))

    st.markdown("## 📊 Feature Importance Analysis")
    st.caption("Shows how strongly each factor influences GPA prediction")

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_
    })

    importance = importance.sort_values(by="Coefficient", ascending=False)

    st.bar_chart(importance.set_index("Feature"))

if menu == "Analytics Dashboard":

    st.subheader("📊 Academic Data Insights")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### GPA Distribution")
        st.bar_chart(df["GPA"].value_counts().sort_index())

    with col2:
        st.markdown("### Study Hours vs GPA")
        st.scatter_chart(df[["StudyTimeWeekly", "GPA"]])

    st.markdown("### Absences vs GPA")
    st.scatter_chart(df[["Absences", "GPA"]])

if menu == "Predict GPA":

    st.subheader("Enter Student Details")

    study = st.number_input("Study Hours Weekly", min_value=0.0)
    absences = st.number_input("Number of Absences", min_value=0.0)
    tutoring = st.selectbox("Tutoring", [0, 1])
    parental = st.slider("Parental Support", 0, 4)
    extra = st.selectbox("Extracurricular", [0, 1])
    sports = st.selectbox("Sports", [0, 1])
    music = st.selectbox("Music", [0, 1])
    volunteer = st.selectbox("Volunteering", [0, 1])

    if st.button("Predict GPA"):

        new_data = pd.DataFrame([[study, absences, tutoring,
                                  parental, extra, sports,
                                  music, volunteer]],
                                columns=feature_columns)

        predicted_gpa = model.predict(new_data)
        gpa_value = round(max(0, min(4, predicted_gpa[0])), 2)

        # Academic Grade Calculation
        if gpa_value >= 3.7:
            grade = "A+"
            remark = "Outstanding Academic Performance"
        elif gpa_value >= 3.3:
            grade = "A"
            remark = "Excellent Performance"
        elif gpa_value >= 3.0:
            grade = "B"
            remark = "Good Performance"
        elif gpa_value >= 2.5:
            grade = "C"
            remark = "Satisfactory Performance"
        else:
            grade = "D"
            remark = "Needs Academic Improvement"

        st.markdown("""
        <div style="
        background-color:#FFFDF0;
        padding:25px;
        border-radius:12px;
        border:2px solid #E6C200;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        ">
        <h3 style="text-align:center; color:#7A6500;">
        Academic Performance Report
        </h3>
        <hr>

        <p><b>Predicted GPA:</b> {}</p>
        <p><b>Grade:</b> {}</p>
        <p><b>Academic Remark:</b> {}</p>

        </div>
        """.format(gpa_value, grade, remark), unsafe_allow_html=True)

        # Save log
        log = pd.DataFrame([{
            "StudyHours": study,
            "Absences": absences,
            "PredictedGPA": round(predicted_gpa[0], 2)
        }])

        log.to_csv("prediction_log.csv", mode="a", header=False, index=False)

        # Suggestions
        st.subheader("Performance Suggestions")

        if study < 10:
            st.write("• Increase study hours.")

        if absences > 10:
            st.write("• Reduce absences.")

        if parental < 2:
            st.write("• Seek academic support.")

        if tutoring == 0:
            st.write("• Consider tutoring.")

        if study >= 15 and absences <= 5:
            st.write("• Excellent habits! Keep it up.")

if menu == "About":

    st.subheader("About the Academic Intelligence System")
    st.markdown("---")

    st.write("""
The Student Academic Performance Portal is an intelligent analytics platform 
designed to evaluate and predict student academic outcomes using data-driven insights.

This system analyzes key academic and behavioral factors such as study patterns, 
attendance, extracurricular engagement, and support systems to generate 
predictive performance assessments.

The objective of this platform is to assist students, educators, and academic 
institutions in identifying performance trends, strengths, and improvement areas 
through structured analytical modeling.

By integrating predictive analytics with educational insights, the platform 
supports informed academic decision-making and performance enhancement strategies.
""")

# Footer
st.markdown("---")
st.caption("© 2026 Student Academic Intelligence System")