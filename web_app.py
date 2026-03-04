import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Student Performance Analysis", layout="centered")

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

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, features.columns, mae, r2

model, feature_columns, mae, r2 = train_model()

st.title("📊 SPA")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Predict GPA", "About"]
)

if menu == "Dashboard":

    st.subheader("📈 Model Performance")

    st.write(f"MAE: {round(mae, 3)}")
    st.write(f"R² Score: {round(r2, 3)}")

    st.markdown("## 📊 Feature Importance Analysis")
    st.caption("Shows how strongly each factor influences GPA prediction")

    importance = pd.DataFrame({
        "Feature": feature_columns,
        "Coefficient": model.coef_
    })

    importance = importance.sort_values(by="Coefficient", ascending=False)

    st.bar_chart(importance.set_index("Feature"))

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
        predicted_gpa[0] = max(0, min(4, predicted_gpa[0]))

        st.success(f"Predicted GPA: {round(predicted_gpa[0], 2)}")

        # Save log
        log = pd.DataFrame([{
            "StudyHours": study,
            "Absences": absences,
            "PredictedGPA": round(predicted_gpa[0], 2)
        }])

        log.to_csv("prediction_log.csv", mode="a", header=False, index=False)

        # Performance Level
        if predicted_gpa[0] >= 3.5:
            st.success("🎓 Excellent Performance")
        elif predicted_gpa[0] >= 3.0:
            st.info("👍 Good Performance")
        elif predicted_gpa[0] >= 2.5:
            st.warning("⚠ Average Performance")
        else:
            st.error("❗ Needs Improvement")

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
    st.subheader("About This Application")
    st.write("This AI system predicts student GPA using Machine Learning.")
    st.write("Built using Streamlit and Scikit-learn.")