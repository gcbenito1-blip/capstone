import streamlit as st
import matplotlib.pyplot as plt
import streamlit_shadcn_ui as ui
# dummy data generator
import numpy as np
# --------------------------
# Dummy scatter plot
# --------------------------
x = np.random.rand(50) * 10
y = 2 * x + np.random.randn(50) * 3
fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(x, y, color='blue')
ax.set_xlabel("Actual NAT")
ax.set_ylabel("Predicted NAT")
ax.set_title("Predicted vs. Actual NAT Scores")
fig.tight_layout()  # remove extra padding
fig.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.95)  # adjust margins

# --------------------------
# Dummy feature importance
# --------------------------
features = ['Feature A','Feature B','Feature C','Feature D','Feature E']
importance = [0.35,0.25,0.15,0.15,0.10]
indices = np.argsort(importance)[::-1]
features_sorted = [features[i] for i in indices]
importance_sorted = [importance[i] for i in indices]

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.barh(features_sorted, importance_sorted, color='skyblue')
ax2.set_xlabel("Importance")
ax2.set_title("Feature Importance")
ax2.invert_yaxis()
fig2.tight_layout()
fig2.subplots_adjust(top=0.88, bottom=0.15, left=0.25, right=0.95)  # same top-bottom spacing

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# --------------------------
# Dummy classification data
# --------------------------
y_true = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 2])
y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 1, 2, 2])

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])

# Create figure
fig3, ax3 = plt.subplots(figsize=(5,5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0','Class 1','Class 2'])
disp.plot(cmap=plt.cm.Blues, ax=ax3, colorbar=False)  # embed in ax, no extra colorbar
ax3.set_title("Confusion Matrix")

def render():
    with st.container():
        st.markdown("**Regression Performance**")
        r1, r2, r3 = st.columns(3)
        with r1:
            with st.container():
                st.metric(
                    border=True,
                    label="Root Mean Squared Error",
                    help="This is a helper for this metric",
                    value=1.0,
                )
        with r2:
                st.metric(
                    border=True,
                    label="Mean Absolute Error",
                    help="This is the meaning of this metric",
                    value=2.0,
                )
        with r3:
            st.metric(
                border=True,
                label="R² Score",
                help="This is a helper for this metric",
                value=3.0,
            )

    with st.container():
        st.markdown("**Classification Performance**")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            st.metric(
                border=True,
                label="Accuracy",
                help="This is a helper for this metric",
                value="87%",
            )
        with s2:
            st.metric(
                border=True,
                label="Precision",
                help="This is a helper for this metric",
                value="83%",
            )
        with s3:
            st.metric(
                border=True,
                label="Recall",
                help="This is a helper for this metric",
                value="83%",
            )
        with s4:
            st.metric(
                border=True,
                label="F1 Score",
                help="This is a helper for this metric",
                value="83%",
            )
        with s5:
            st.metric(
                border=True,
                label="ROC-AUC",
                help="This is a helper for this metric",
                value="83%",
            )

    with st.container(border=True):
        st.markdown("**Predicted vs. Actual NAT Scores**")
        st.markdown("""<p class="sub-text">Visual comparison of prediction accuracy</p>""", unsafe_allow_html=True)
        st.pyplot(fig)

    with st.container(border=True):
        st.markdown("**Feature Importance**")
        st.markdown("""<p class="sub-text">Most influential factors in prediction</p>""", unsafe_allow_html=True)
        st.pyplot(fig2)

    with st.container(border=True):
        st.markdown("**Confusion Matrix**")
        st.markdown("""<p class="sub-text">Classification accuracy by proficiency level</p>""", unsafe_allow_html=True)
        st.pyplot(fig3)
