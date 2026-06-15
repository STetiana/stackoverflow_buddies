# Sprint 03 Stack Overflow Buddies: Data app with XGBoost model
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import joblib
import math

# columns as ordered in the prediction model
COLUMNS_IN_MODEL = ['WorkExp','AIAgents','Country','EdLevel','DevType','OrgSize','ICorPM','RemoteWork','Industry','Employment']

# columns and their values (as formulated in the Stack Overflow 2026 Developer Survey) ordered by on screen order
COUNTRY_OPTIONS = [
    "Austria", 
    "Germany", 
    "Switzerland"
]

EDLEVEL_OPTIONS = [
    "Primary/elementary school",    
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",    
    "Associate degree (A.A., A.S., etc.)",    
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",    
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",    
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",    
    "Other"
]

EMPLOYMENT_OPTIONS = [
    "Employed", 
    "Independent contractor, freelancer, or self-employed"
]

INDUSTRY_OPTIONS = [
    "Banking/Financial Services",
    "Energy",
    "Fintech",
    "Healthcare",
    "Internet, Telecomm or Information Services",
    "Manufacturing",
    "Retail and Consumer Services",
    "Software Development",
    "Transportation, or Supply Chain",
    "Other"
]

ORGSIZE_OPTIONS = [
    "Less than 20 employees",
    "20 to 99 employees",
    "100 to 499 employees",
    "500 to 999 employees",
    "1,000 to 4,999 employees",
    "5,000 to 9,999 employees",
    "10,000 or more employees"
]

ICORPM_OPTIONS = [
    "Individual contributor", 
    "People manager"
]

REMOTEWORK_OPTIONS = [
    "In-person",
    "Remote",
    "Hybrid (some remote, leans heavy to in-person)",
    "Hybrid (some in-person, leans heavy to flexibility)",
    "Your choice (very flexible, you can come in when you want or just as needed)"
]

DEVTYPE_OPTIONS = [
    "Developer, full-stack",
    "Developer, back-end",
    "Developer, front-end",
    "Developer, mobile",
    "Developer, desktop or enterprise applications",
    "Developer, embedded applications or devices",
    "DevOps engineer or professional",
    "Architect, software or solutions",
    "Engineering manager"
]

AIAGENTS_OPTIONS = [
    "No, and I don't plan to",
    "No, but I plan to",
    "No, I use AI exclusively in copilot/autocomplete mode",
    "Yes, I use AI agents at work monthly or infrequently",
    "Yes, I use AI agents at work weekly",
    "Yes, I use AI agents at work daily"
 ]

def main():

    # load pipeline from a bundle file
    loaded = joblib.load("xgboost.joblib")
    pipeline = loaded["pipeline"] 

    # page level setup with 3 columns
    st.set_page_config(layout="wide", page_title="Income Potential")
    st.title("Income Potential - DACH Software Developers")
    left_col, middle_col, right_col = st.columns(3, gap="large", vertical_alignment="top", border=False, width="stretch")

    with left_col:

        Country = st.selectbox("Country", options=COUNTRY_OPTIONS, index=1)
        EdLevel = st.selectbox("Education", options=EDLEVEL_OPTIONS, index=3)
        
        WorkExp = st.number_input("Work experience (years)", min_value=1, max_value=50, value=1, step=1, format="%d")
        if WorkExp < 0 or WorkExp > 50:
            st.error("Work experience must be between 0 and 50 years.")

        Employment = st.selectbox("Employment nature", options=EMPLOYMENT_OPTIONS, index=0)

    with middle_col:

        Industry = st.selectbox("Industry", options=INDUSTRY_OPTIONS, index=0)
        OrgSize = st.selectbox("Organization size", options=ORGSIZE_OPTIONS, index=0)
        RemoteWork = st.selectbox("Work location", options=REMOTEWORK_OPTIONS, index=0)
        ICorPM = st.selectbox("Job nature", options=ICORPM_OPTIONS, index=0)

    with right_col:

        DevType = st.selectbox("Developer type", options=DEVTYPE_OPTIONS, index=0)
        AIAgents = st.selectbox("Using AI Agents?", options=AIAGENTS_OPTIONS, index=0)

        submit = st.button("Submit")

        if submit:
        
            # construct pandas data frame with values specified
            df = pd.DataFrame([[WorkExp, AIAgents, Country, EdLevel, DevType, OrgSize, ICorPM, RemoteWork, Industry, Employment]],
                              columns=COLUMNS_IN_MODEL)

            # run inference pipeline
            prediction = pipeline.predict(df)

            # compensation normalized with log in model; revert value to actual value in USD before displaying it
            preds_dollars = np.expm1(prediction)

            # round up to the nearest whole number integer
            st.success("Income potential:")
            value = math.ceil(preds_dollars[0])
            st.markdown(f"{value:,} USD")

if __name__ == "__main__":
    main()
