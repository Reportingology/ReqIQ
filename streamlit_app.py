import streamlit as st
import pandas as pd
from statsIQ import generate_qualifying_segments, score_open_requisitions

# Set up the page
st.set_page_config(page_title="RequisitionIQ", layout="wide")
st.title("RequisitionIQ - Statistical Engine")

# Constants
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/09c4c7426249be40171b37cce2c2e2a75fde39e0/filled_requisitions.csv"
SEGMENTS_CSV_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/main/generate_qualifying_segments.csv"
OPEN_REQS_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/main/open_requisitions.csv"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
filled_reqs_df = load_data(GITHUB_CSV_URL)
segments_df = load_data(SEGMENTS_CSV_URL) 
open_reqs_df = load_data(OPEN_REQS_URL)

# Step 1: Generate Sample Populations
if not filled_reqs_df.empty:
    st.success(f"Loaded {len(filled_reqs_df)} filled requisitions")
    
    metric_column = st.selectbox("Select Metric", ["DaysToFirst Screen"])
    
    if st.button("Generate Sample Populations"):
        with st.spinner("Analyzing..."):
            results_df = generate_qualifying_segments(filled_reqs_df, metric_column)
        
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Sample Populations CSV",
                data=csv,
                file_name=f"reqiq_sample_populations_{metric_column.replace(' ', '_')}.csv",
                mime="text/csv"
            )

# Step 2: Score Open Requisitions  
if not segments_df.empty and not open_reqs_df.empty:
    if st.button("Score Open Requisitions"):
        with st.spinner("Calculating z-scores..."):
            scored_df = score_open_requisitions(
                open_reqs_df, 
                segments_df, 
                filled_reqs_df, 
                "DaysToFirst Screen"
            )
        
        if not scored_df.empty:
            st.dataframe(scored_df, use_container_width=True)
            
            scored_csv = scored_df.to_csv(index=False)
            st.download_button(
                label="Download Scored Requisitions CSV",
                data=scored_csv,
                file_name="reqiq_open_requisitions_DaysToFirst_Screen.csv",
                mime="text/csv"
            )
