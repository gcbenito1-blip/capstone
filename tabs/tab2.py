# EDA Tab

# import streamlit_shadcn_ui as ui
# import streamlit as st
# import pandas as pd

# def render():
#     if not st.session_state.get("tab1_ready", False):
#         st.warning("Upload Data First")
#     else:
#         df = st.session_state['uploaded_data']
#         is_disabled=True
#         with st.container(border=True):
#             st.markdown("**Dataset Preview**")
#             dataset = st.data_editor(df, num_rows="dynamic")
#             st.write()
#             tab2button = st.button(":material/query_stats: Upload Dataset", type="primary", key="tab2button", width="stretch", disabled=is_disabled)

#             if dataset is not None:
#                 is_disabled=False

#             if tab2button and dataset is not None:
#                 st.session_state['tab2_ready'] = True
#                 st.session_state['dataset'] = df
#                 st.success("Dataset Ready for Exploratory Analysis", icon="✅")

import streamlit as st

def render():
    if not st.session_state.get("tab1_ready", False):
        st.warning("Upload Data First")
        return

    df = st.session_state["uploaded_data"]

    with st.container():
        st.markdown("<p class='sub-text'>Dataset Characteristics & Quality Check</p>", unsafe_allow_html=True)

        # --- Header-only check ---
        if df.shape[0] == 0:
            st.error("The uploaded file contains only column headers and no data rows.")
            st.button(
                ":material/query_stats: Generate Results",
                type="primary",
                disabled=True,
                width="stretch"
            )
            return
        col1, col2, col3 = st.columns(3)
        total_missing = df.isnull().sum().sum()
        empty_rows = df.isnull().all(axis=1).sum()

        col1.metric("Rows", df.shape[0], border=True)
        col2.metric("Columns", df.shape[1], border=True)
        col3.metric("Missing Values", total_missing, border=True)

        issues_detected = False

        # Empty rows
        if empty_rows > 0:
            st.error(f"{empty_rows} completely empty rows detected. Remove them before proceeding.")
            issues_detected = True

        # Missing values
        if total_missing > 0:
            st.error("Missing values detected.")
            issues_detected = True

            rows_with_missing = df[df.isnull().any(axis=1)]

            st.markdown("**Rows with Missing Values**")
            edited_df = st.data_editor(
                rows_with_missing,
                num_rows="fixed",
                key="missing_editor"
            )

            st.markdown("**Missing Value Actions**")
            action = st.selectbox(
                "Select action",
                ["Select", "Remove Rows", "Impute (Mean/Mode)", "Use Edited Version"]
            )

            preview_rows = None
            updated_df = df.copy()

            if action == "Remove Rows":
                preview_rows = df.drop(rows_with_missing.index)
                updated_df = preview_rows.copy()

            elif action == "Impute (Mean/Mode)":
                preview_rows = rows_with_missing.copy()

                for col in preview_rows.columns:
                    if preview_rows[col].isnull().sum() > 0:
                        if df[col].dtype in ["int64", "float64"]:
                            preview_rows[col] = preview_rows[col].fillna(df[col].mean())
                        else:
                            preview_rows[col] = preview_rows[col].fillna(df[col].mode()[0])

                updated_df.update(preview_rows)

            elif action == "Use Edited Version":
                preview_rows = edited_df
                updated_df.update(preview_rows)

            if preview_rows is not None:
                st.markdown("**Preview of Affected Rows**")
                st.dataframe(preview_rows, use_container_width=True)

            # --- Validation conditions ---
            no_action_selected = action == "Select"
            no_difference = preview_rows is not None and preview_rows.equals(rows_with_missing)
            still_has_empty_rows = updated_df.isnull().all(axis=1).sum() > 0

            disable_apply = (
                no_action_selected or
                no_difference or
                still_has_empty_rows
            )

            apply_changes = st.button(
                "Apply Changes",
                type="primary",
                disabled=disable_apply
            )

            if apply_changes:
                st.session_state["uploaded_data"] = updated_df
                st.session_state["changes_applied"] = True
                st.success("Changes applied successfully.")
                st.rerun()

    changes_applied = st.session_state.get("changes_applied", False)

    confirm = st.checkbox(
        "I confirm that the dataset is cleaned and ready for analysis.",
        disabled=not changes_applied and issues_detected
    )

    if issues_detected:
        st.button(
            ":material/query_stats: Generate Results",
            type="primary",
            disabled=True,
            width="stretch"
        )
    else:
        if st.button(
            ":material/query_stats: Generate Results",
            type="primary",
            disabled=not confirm,
            width="stretch"
        ):
            st.session_state["tab2_ready"] = True
            st.session_state["dataset"] = st.session_state["uploaded_data"]
            st.success("Prediction Generated", icon="✅")