import sys
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, '.')

from src.predict import (
    get_all_predictions,
    find_similar_students,
    predict_cluster,
    project_to_pca,
    get_dataset_pca_projection,
)

st.set_page_config(
    page_title="EduSmart - Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 EduSmart - Student Performance Analytics System")
st.caption(
    "A beginner-friendly demo applying 7 Machine Learning algorithms to the "
    "UCI Student Performance (Math) dataset."
)

with st.expander("ℹ️  How this app maps to our 7 ML algorithms", expanded=False):
    st.markdown(
        """
        | # | Algorithm              | What it predicts here                          |
        |---|-------------------------|-------------------------------------------------|
        | 1 | **Linear Regression**  | Raw final grade (G3), 0-20 scale                |
        | 2 | **Logistic Regression**| Pass / Fail status (G3 >= 10 = Pass)            |
        | 3 | **KNN**                | The 5 most similar students in the dataset      |
        | 4 | **Decision Tree**      | Performance Tier (Low / Mid / High)             |
        | 5 | **Random Forest**      | Risk Level (Low / Medium / High Risk)           |
        | 6 | **K-Means**            | Peer cluster (High Performer / Average / Needs Improvement) |
        | 7 | **PCA**                | 2D visualization of where this student sits among everyone else |
        """
    )

st.divider()

st.sidebar.header("📝 Student Profile")
st.sidebar.caption("Enter student information below")

with st.sidebar.expander("🏫 School & Demographics", expanded=True):
    school = st.selectbox("School", ["GP", "MS"], help="GP = Gabriel Pereira, MS = Mousinho da Silveira")
    sex = st.selectbox("Sex", ["F", "M"])
    age = st.slider("Age", 15, 22, 17)
    address = st.selectbox("Home Address Type", ["U", "R"], help="U = Urban, R = Rural")
    famsize = st.selectbox("Family Size", ["LE3", "GT3"], help="LE3 = 3 or fewer members, GT3 = more than 3")
    Pstatus = st.selectbox("Parents' Cohabitation Status", ["T", "A"], help="T = Living Together, A = Apart")

with st.sidebar.expander("👪 Family Background"):
    Medu = st.slider("Mother's Education", 0, 4, 2, help="0=none, 1=primary, 2=5th-9th grade, 3=secondary, 4=higher")
    Fedu = st.slider("Father's Education", 0, 4, 2, help="0=none, 1=primary, 2=5th-9th grade, 3=secondary, 4=higher")
    Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"])
    guardian = st.selectbox("Guardian", ["mother", "father", "other"])

with st.sidebar.expander("📚 Academics & Study Habits"):
    reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
    traveltime = st.slider("Travel Time to School", 1, 4, 1, help="1=<15min, 2=15-30min, 3=30min-1hr, 4=>1hr")
    studytime = st.slider("Weekly Study Time", 1, 4, 2, help="1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs")
    failures = st.slider("Past Class Failures", 0, 3, 0)
    schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
    famsup = st.selectbox("Family Educational Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extra-Curricular Activities", ["yes", "no"])
    nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Internet Access at Home", ["yes", "no"])

with st.sidebar.expander("🎉 Lifestyle & Social"):
    romantic = st.selectbox("In a Romantic Relationship", ["yes", "no"])
    famrel = st.slider("Quality of Family Relationships", 1, 5, 4, help="1=very bad, 5=excellent")
    freetime = st.slider("Free Time After School", 1, 5, 3, help="1=very low, 5=very high")
    goout = st.slider("Going Out With Friends", 1, 5, 3, help="1=very low, 5=very high")
    Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1, help="1=very low, 5=very high")
    Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1, help="1=very low, 5=very high")
    health = st.slider("Current Health Status", 1, 5, 3, help="1=very bad, 5=very good")
    absences = st.slider("Number of School Absences", 0, 93, 4)

raw_input = {
    "school": school, "sex": sex, "age": age, "address": address,
    "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
    "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
    "traveltime": traveltime, "studytime": studytime, "failures": failures,
    "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
    "activities": activities, "nursery": nursery, "higher": higher,
    "internet": internet, "romantic": romantic, "famrel": famrel,
    "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
    "health": health, "absences": absences,
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Grade & Pass/Fail",
    "👥 Similar Students",
    "⚠️ Risk & Cluster",
    "📊 PCA Visualization",
])

with tab1:
    st.subheader("Predicted Final Grade & Pass/Fail Status")
    st.write(
        "Uses **Linear Regression** to predict the numeric final grade (G3), "
        "and **Logistic Regression** to predict whether the student Passes "
        "or Fails (Pass = G3 >= 10)."
    )

    if st.button("Predict Marks & Pass/Fail", type="primary"):
        results = get_all_predictions(raw_input)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Final Grade (G3)", f"{results['predicted_grade']} / 20")

        with col2:
            if results["pass_fail"] == "Pass":
                st.success(f"✅ Status: **{results['pass_fail']}**")
            else:
                st.error(f"❌ Status: **{results['pass_fail']}**")

with tab2:
    st.subheader("5 Most Similar Students")
    st.write(
        "Uses unsupervised **K-Nearest Neighbors** to find the 5 students "
        "already in the dataset whose profiles are closest to this one."
    )

    if st.button("Find Similar Students", type="primary"):
        similar_students = find_similar_students(raw_input, n_neighbors=5)

        display_df = pd.DataFrame(similar_students)

        display_columns = [
            "row_index", "distance", "age", "sex", "studytime",
            "failures", "absences", "G1", "G2", "G3",
        ]
        display_df = display_df[display_columns].rename(columns={
            "row_index": "Student #",
            "distance": "Similarity Distance",
            "age": "Age",
            "sex": "Sex",
            "studytime": "Study Time",
            "failures": "Past Failures",
            "absences": "Absences",
            "G1": "Period 1 Grade",
            "G2": "Period 2 Grade",
            "G3": "Final Grade",
        })

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("Lower distance = better match")

with tab3:
    st.subheader("Risk Profile & Academic Cluster")
    st.write(
        "Uses a **Decision Tree** for academic Performance Tier, a "
        "**Random Forest** for Risk Level, and **K-Means** to assign the "
        "student to a peer cluster."
    )

    if st.button("Predict Risk & Academic Cluster", type="primary"):
        results = get_all_predictions(raw_input)
        cluster_result = predict_cluster(raw_input)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info(f"📈 **Performance Tier**\n\n{results['performance_tier']}")

        with col2:
            risk_level = results["risk_level"]
            if risk_level == "Low Risk":
                st.success(f"🟢 **Risk Level**\n\n{risk_level}")
            elif risk_level == "Medium Risk":
                st.warning(f"🟡 **Risk Level**\n\n{risk_level}")
            else:
                st.error(f"🔴 **Risk Level**\n\n{risk_level}")

        with col3:
            st.info(f"👥 **Cluster Cohort**\n\n{cluster_result['cluster_label']}")

with tab4:
    st.subheader("2D PCA Visualization")
    st.write(
        "Uses **PCA** to compress all 30 scaled features down to just 2 "
        "dimensions, so the whole student population (colored by actual "
        "final grade) can be visualized on one scatter plot, with this "
        "student's position marked by a red star."
    )

    if st.button("Generate PCA Plot", type="primary"):
        dataset_projection = get_dataset_pca_projection()
        student_point = project_to_pca(raw_input)

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            dataset_projection["pc1"],
            dataset_projection["pc2"],
            c=dataset_projection["G3"],
            cmap="viridis",
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.3,
            label="Students in Dataset",
        )

        ax.scatter(
            student_point["pc1"],
            student_point["pc2"],
            color="red",
            marker="*",
            s=450,
            edgecolors="black",
            linewidths=1.3,
            label="This Student",
            zorder=5,
        )

        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label("Final Grade (G3)")

        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Student Population - PCA Projection")
        ax.legend(loc="best")

        st.pyplot(fig)
        st.caption("Each dot is a student. Red star = current profile.")

st.divider()
st.caption("EduSmart - Student Performance Analytics System | Built with Streamlit & scikit-learn")
