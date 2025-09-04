import streamlit as st
import pandas as pd
from statsIQ import generate_qualifying_segments

# Set up the page
st.set_page_config(page_title="RequisitionIQ", layout="wide")
st.title("RequisitionIQ - Statistical Engine")

# Constants
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/09c4c7426249be40171b37cce2c2e2a75fde39e0/filled_requisitions.csv"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Main interface
df = load_data(GITHUB_CSV_URL)

if not df.empty:
    st.success(f"Loaded {len(df)} filled requisitions")
    
    metric_column = st.selectbox("Select Metric", ["DaysToFirst Screen"])
    
    if st.button("Generate Sample Populations"):
        with st.spinner("Analyzing..."):
            results_df = generate_qualifying_segments(df, metric_column)
        
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"reqiq_sample_populations_{metric_column}.csv",
                mime="text/csv"
            )
        else:
            st.error("No qualifying segments found")
