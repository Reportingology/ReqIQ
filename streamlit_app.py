import streamlit as st
import pandas as pd
from statsIQ import generate_qualifying_segments, score_open_requisitions

# Set up the page
st.set_page_config(page_title="RequisitionIQ", layout="wide")
st.title("RequisitionIQ - Statistical Engine")

# Constants
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/09c4c7426249be40171b37cce2c2e2a75fde39e0/filled_requisitions.csv"
SEGMENTS_CSV_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/main/generate_qualifying_segments.csv"

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Main interface
st.subheader("Step 1: Generate Sample Populations")

# Load filled requisitions data
filled_reqs_df = load_data(GITHUB_CSV_URL)

if not filled_reqs_df.empty:
    st.success(f"Loaded {len(filled_reqs_df)} filled requisitions")
    
    metric_column = st.selectbox("Select Metric", ["DaysToFirst Screen"])
    
    if st.button("Generate Sample Populations"):
        with st.spinner("Analyzing..."):
            results_df = generate_qualifying_segments(filled_reqs_df, metric_column)
        
        if not results_df.empty:
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Sample Populations CSV",
                data=csv,
                file_name=f"reqiq_sample_populations_{metric_column.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        else:
            st.error("No qualifying segments found")

# Separator
st.divider()

# Step 2: Score Open Requisitions
st.subheader("Step 2: Score Open Requisitions")

# Load existing segments
segments_df = load_data(SEGMENTS_CSV_URL)

if not segments_df.empty:
    st.success(f"Loaded {len(segments_df)} qualifying segments")
    
    # File uploader for open requisitions
    uploaded_file = st.file_uploader(
        "Upload Open Requisitions CSV", 
        type=['csv'],
        help="CSV should contain: RequisitionID, Recruiter, Country, JobGroup, JobFamily, JobLevel, RequisitionOpenedDate, DaysToFirst Screen, DateFirstScreen"
    )
    
    if uploaded_file is not None:
        # Load uploaded file
        open_reqs_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(open_reqs_df)} open requisitions")
        
        # Show preview
        with st.expander("Preview Open Requisitions"):
            st.dataframe(open_reqs_df.head())
        
        # Select metric for scoring
        available_metrics = [col for col in open_reqs_df.columns if 'DaysTo' in col or 'Days to' in col or 'Days To' in col]
        if not available_metrics:
            available_metrics = ["DaysToFirst Screen"]  # fallback
            
        scoring_metric = st.selectbox("Select Metric to Score", available_metrics)
        
        if st.button("Score Open Requisitions"):
            with st.spinner("Calculating z-scores..."):
                scored_df = score_open_requisitions(
                    open_reqs_df, 
                    segments_df, 
                    filled_reqs_df, 
                    scoring_metric
                )
            
            if not scored_df.empty:
                st.success(f"Scored {len(scored_df)} requisitions")
                st.dataframe(scored_df, use_container_width=True)
                
                # Download button for scored results
                scored_csv = scored_df.to_csv(index=False)
                st.download_button(
                    label="Download Scored Requisitions CSV",
                    data=scored_csv,
                    file_name=f"reqiq_open_requisitions_{scoring_metric.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                
                # Show summary stats
                with st.expander("Scoring Summary"):
                    total_reqs = len(scored_df)
                    recruiter_matches = scored_df['Recruiter_Segment_Match'].notna().sum()
                    org_matches = scored_df['Org_Segment_Match'].notna().sum()
                    
                    st.write(f"**Total Requisitions:** {total_reqs}")
                    st.write(f"**Recruiter Segment Matches:** {recruiter_matches} ({recruiter_matches/total_reqs*100:.1f}%)")
                    st.write(f"**Organization Segment Matches:** {org_matches} ({org_matches/total_reqs*100:.1f}%)")
                    
                    # Show z-score distributions
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Recruiter Z-Scores**")
                        recruiter_z = scored_df['Recruiter_Z_Score'].dropna()
                        if len(recruiter_z) > 0:
                            st.write(f"Mean: {recruiter_z.mean():.2f}")
                            st.write(f"Std: {recruiter_z.std():.2f}")
                            st.write(f"Range: {recruiter_z.min():.2f} to {recruiter_z.max():.2f}")
                    
                    with col2:
                        st.write("**Organization Z-Scores**")
                        org_z = scored_df['Org_Z_Score'].dropna()
                        if len(org_z) > 0:
                            st.write(f"Mean: {org_z.mean():.2f}")
                            st.write(f"Std: {org_z.std():.2f}")
                            st.write(f"Range: {org_z.min():.2f} to {org_z.max():.2f}")
                    
                    with col3:
                        st.write("**Company Z-Scores**")
                        company_z = scored_df['Company_Z_Score'].dropna()
                        if len(company_z) > 0:
                            st.write(f"Mean: {company_z.mean():.2f}")
                            st.write(f"Std: {company_z.std():.2f}")
                            st.write(f"Range: {company_z.min():.2f} to {company_z.max():.2f}")
            else:
                st.error("Failed to score requisitions")
                
else:
    st.warning("No qualifying segments found. Please generate sample populations first.")
