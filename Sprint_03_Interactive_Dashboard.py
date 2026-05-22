import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib

def main():

    # load model saved along with pipeline
    pipeline = joblib.load("pipeline_tabpfn_v3.joblib")

    st.set_page_config(layout="wide", page_title="Developer for DACH")

    st.title("Predict Annual Income")

    left_col, spacer_col, right_col = st.columns([2, 0.5, 2])

    with left_col:

        st.header("User Input")

        # instruction to users
        status_placeholder = st.empty()
        status_placeholder.info("Fill in the form below and click Submit:")

        # drop down list allowing only one choice
        country = st.selectbox(
            "Country",
            options=["Austria", "Germany", "Switzerland"],
            index=0
        )

        # free entry numeric field with assist buttons +/-
        work_exp = st.number_input(
            "Work experience (years)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=1.0,
            format="%.1f"
        )

        # printing error messages to users on web page
        if work_exp < 0 or work_exp > 50:
            st.error("Work Experience must be between 0 and 50 years.")

        # radio button allowing only one choice
        role = st.radio(
            "Role",
            options=["Individual contributor", "People manager"]
        )

        submit = st.button("Submit")

        # Update status based on submission
        if submit:
            status_placeholder.success("Input submitted — see prediction on the right.")


    with spacer_col:
        st.write("")


    with right_col:

        st.header("Prediction")
        
        if submit:

            st.success("Predicting for:")

            # echo input received
            st.markdown(f"**Country:** {country}")
            st.markdown(f"**Work experience (years):** {work_exp}")
            st.markdown(f"**Role:** {role}")

            # construct df
            df = pd.DataFrame([[work_exp, country, role]])

            # convert df to numpy array as input for model
            X = df.values

            # print debugging values to console
            # print(type(X))
            # np.set_printoptions(threshold=np.inf)   # show everything
            # print(X)

            # run inference
            prediction = pipeline.predict(X)

            # revert to actual value before log because compensation normalized with log in model
            preds_dollars = np.expm1(prediction)

            st.success("Predicted annual income is:")
            st.markdown(f"{preds_dollars[0]:,.2f} USD")

        else:
            st.info("Waiting for input...")

if __name__ == "__main__":
    main()
