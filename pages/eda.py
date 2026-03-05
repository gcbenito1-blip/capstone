import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="Exploratory Data Analysis",
    layout="wide",
    page_icon=":pencil2:"
)

with st.sidebar:
    st.title("NAT-lytics")
    st.subheader("Model Evaluation Dashboard")
    st.divider()
    st.page_link("app.py", label="Dataset Upload", icon=":material/upload:")
    st.page_link("pages/eda.py", label="Exploratory Data Analysis", icon=":material/analytics:")
    st.page_link("pages/pred.py", label="Prediction Results", icon=":material/psychology:")
    st.page_link("pages/model_eval.py", label="Model Evaluation", icon=":material/monitoring:")
    st.page_link("pages/reports.py", label="Reports & Evaluation", icon=":material/description:")
st.title("Exploratory Data Analysis")
st.text("Examine distributions, correlations, and patterns in the data")

if not st.session_state.get("data_ready", False):
    st.warning("Upload dataset first from Dataset Upload page.")
    st.stop()
else:
    df = st.session_state['uploaded_data']
    with st.container(border=True):
        st.markdown("Filter Options")
        col1,col2,col3 =st.columns(3)
        with col1:
            school = st.multiselect("School", df['School'].unique(), placeholder="Select School", )
        with col2:
            bmi = st.multiselect("BMI", df['BMI/nutrional status'].unique(), placeholder="Select BMI", )
        with col3:
            gender = st.selectbox("Gender", df['sex'].unique(), placeholder="Select Gender", index=None)
            
    # Show basic info
    st.markdown("**Dataset Preview**")
    st.dataframe(df.head(), hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        # Example: BMI / Nutritional Status
        vc = df['BMI/nutrional status'].value_counts()
        plt.pie(vc, labels=vc.index, autopct='%1.1f%%', startangle=90, colors=['cyan','skyblue','royalblue','cornflowerblue', 'lavender'])
        plt.title("BMI / Nutritional Status Distribution")
        plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
        st.pyplot(plt)
        plt.close()
        # st.dataframe(df['BMI/nutrional status'].value_counts())
    with c2:
        sg= df['sex'].value_counts()
        plt.pie(sg, labels=sg.index, autopct='%1.1f%%', startangle=90, colors=['cyan', 'lightcoral'])
        plt.title("Gender Distribution")
        plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
        st.pyplot(plt)
        plt.close()
    # Map columns to subjects
    subjects = {
        "Mathematics": [col for col in df.columns if "math" in col.lower()],
        "Science": [col for col in df.columns if "science" in col.lower()],
        "English": [col for col in df.columns if "english" in col.lower()],
        "Filipino": [col for col in df.columns if "filipino" in col.lower()],
        "Araling Panlipunan": [col for col in df.columns if "araling panlipunan" in col.lower()]
    }

    # Aggregate statistics
    stats_list = []
    for subj, cols in subjects.items():
        all_scores = df[cols].values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]  # Remove NaN if any
        stats_list.append({
            "Subject": subj,
            "Mean": round(np.mean(all_scores), 2),
            "Median": round(np.median(all_scores), 2),
            "Std Dev": round(np.std(all_scores, ddof=1), 2),
            "Min": np.min(all_scores),
            "Max": np.max(all_scores)
        })
    
    stats_df = pd.DataFrame(stats_list)

    st.markdown("**Descriptive Statistics for Academic Performance**")
    st.dataframe(stats_df, hide_index=True)

    # Generate descriptive statistics for numeric columns
    numeric_stats = df.describe().T
    numeric_stats['missing'] = df.isna().sum()
    numeric_stats['unique'] = df.nunique()

    st.write("Grade Distribution per Subject")
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # 3 rows, 2 cols
    axes = axes.flatten()
    
    for i, (subj, cols) in enumerate(subjects.items()):
        all_scores = df[cols].values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]
        axes[i].hist(all_scores, bins=10, color='skyblue', edgecolor='black')
        axes[i].set_title(f"{subj} Grade Distribution")
        axes[i].set_xlabel("Grade")
        axes[i].set_ylabel("Frequency")
    
    # Hide any unused subplot (if subjects < 6)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout(pad=1.0)
    st.pyplot(fig)
    plt.close()
        
    # List of all grade columns
    grade_cols = [col for col in df.columns if "Final ratings" in col]
    
    # Compute average grade per student
    df['avg_grade'] = df[grade_cols].mean(axis=1)
    
    # Group by BMI/nutritional status and compute mean
    bmi_avg = df.groupby('BMI/nutrional status')['avg_grade'].mean().sort_values()
    
    # Plot bar chart
    plt.figure(figsize=(8,5))
    bmi_avg.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Average Academic Performance by BMI Category")
    plt.ylabel("Average Grade")
    plt.xlabel("BMI / Nutritional Status")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    st.pyplot(plt)
    plt.close()

    

    # Compute average grade per subject per student
    for subj, cols in subjects.items():
        df[subj] = df[cols].mean(axis=1)
    
    # Melt the dataframe to long format
    long_df = df.melt(id_vars=['sex'], value_vars=list(subjects.keys()),
                      var_name='Subject', value_name='Grade')
    
    # Group by sex and subject and compute quartiles and mean
    summary = long_df.groupby(['sex', 'Subject'])['Grade'].agg(
        Q1=lambda x: np.percentile(x, 25),
        Median='median',
        Q3=lambda x: np.percentile(x, 75),
        Mean='mean'
    ).reset_index()
    
    # Round values
    summary[['Q1','Median','Q3','Mean']] = summary[['Q1','Median','Q3','Mean']].round(1)
    
    st.markdown("**Performance Distribution by Gender**")
    st.dataframe(summary, hide_index=True)