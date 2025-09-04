import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import itertools

# Set up the page configuration - this must be the first Streamlit command
st.set_page_config(page_title="RequisitionIQ", layout="wide")

# Title and description
st.title("RequisitionIQ - Requisition Intelligence Engine")
st.write("Generate all qualifying sample populations (n>=30) for recruiter and org segments")

# Constants - these define our clustering hierarchy and minimum sample size
MIN_SAMPLE_SIZE = 30  # Minimum number of records needed for statistical validity
GITHUB_CSV_URL = "https://raw.githubusercontent.com/Reportingology/ReqIQ/09c4c7426249be40171b37cce2c2e2a75fde39e0/filled_requisitions.csv"

@st.cache_data  # This decorator caches the data so we don't reload it every time
def load_data(url: str) -> pd.DataFrame:
    """
    Load data from GitHub CSV URL and cache it for performance.
    
    Args:
        url: The raw GitHub CSV URL
        
    Returns:
        DataFrame with the requisition data
    """
    try:
        # Read CSV directly from GitHub URL
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def calculate_statistics(df: pd.DataFrame, metric_column: str) -> Dict:
    """
    Calculate mean, standard deviation, and other statistics for a metric.
    
    Args:
        df: DataFrame with the sample population
        metric_column: Column name to calculate statistics for
        
    Returns:
        Dictionary with statistical measures
    """
    if metric_column not in df.columns:
        return {'error': f'Column {metric_column} not found'}
    
    # Remove any null values before calculating statistics
    clean_data = df[metric_column].dropna()
    
    if len(clean_data) == 0:
        return {'error': 'No valid data after removing nulls'}
    
    return {
        'mean': clean_data.mean(),
        'std': clean_data.std(),
        'count': len(clean_data),
        'min': clean_data.min(),
        'max': clean_data.max()
    }

def generate_all_recruiter_segments(df: pd.DataFrame, metric_column: str) -> List[Dict]:
    """
    Generate ALL qualifying recruiter segments using linear backoff fallback.
    
    Args:
        df: Full dataset of filled requisitions
        metric_column: The metric to calculate statistics for
        
    Returns:
        List of dictionaries containing all qualifying recruiter segments
    """
    # Define cluster hierarchy from most specific to least specific
    cluster_hierarchy = [
        ['Recruiter', 'Country', 'JobGroup', 'JobFamily', 'JobLevel'],  # Cluster 5
        ['Recruiter', 'Country', 'JobGroup', 'JobFamily'],              # Cluster 4
        ['Recruiter', 'Country', 'JobGroup'],                           # Cluster 3
        ['Recruiter', 'Country'],                                       # Cluster 2
        ['Recruiter']                                                   # Cluster 1
    ]
    
    qualifying_segments = []
    
    # Get all unique recruiters
    recruiters = df['Recruiter'].unique()
    
    # For each recruiter, find their best qualifying cluster
    for recruiter in recruiters:
        recruiter_data = df[df['Recruiter'] == recruiter].copy()
        
        # Try each cluster level for this recruiter
        for cluster_level, grouping_columns in enumerate(cluster_hierarchy, 1):
            
            # For recruiter segments, we need to check if this recruiter has enough
            # records when grouped by the specified columns
            if len(grouping_columns) == 1:  # Just recruiter
                sample_size = len(recruiter_data)
                sample_data = recruiter_data
            else:
                # Group by all columns and check each group
                grouped = recruiter_data.groupby(grouping_columns)
                
                # Check each group within this recruiter's data
                for group_values, group_data in grouped:
                    sample_size = len(group_data)
                    
                    if sample_size >= MIN_SAMPLE_SIZE:
                        # Calculate statistics for this qualifying group
                        stats = calculate_statistics(group_data, metric_column)
                        
                        if 'error' not in stats:
                            # Create row for results table
                            segment_info = {
                                'Segment_Type': 'Recruiter',
                                'Cluster_Level': f'Cluster_{6-cluster_level}',  # Reverse numbering
                                'Recruiter': recruiter,
                                'Country': group_values[1] if len(group_values) > 1 else 'All',
                                'JobGroup': group_values[2] if len(group_values) > 2 else 'All',
                                'JobFamily': group_values[3] if len(group_values) > 3 else 'All',
                                'JobLevel': group_values[4] if len(group_values) > 4 else 'All',
                                'Grouping': ' + '.join(grouping_columns),
                                'Sample_Size': stats['count'],
                                'Mean': round(stats['mean'], 2),
                                'Std_Dev': round(stats['std'], 2),
                                'Min': stats['min'],
                                'Max': stats['max']
                            }
                            qualifying_segments.append(segment_info)
                continue
            
            # Handle the single recruiter case (Cluster 1)
            if len(grouping_columns) == 1 and sample_size >= MIN_SAMPLE_SIZE:
                stats = calculate_statistics(sample_data, metric_column)
                
                if 'error' not in stats:
                    segment_info = {
                        'Segment_Type': 'Recruiter',
                        'Cluster_Level': 'Cluster_1',
                        'Recruiter': recruiter,
                        'Country': 'All',
                        'JobGroup': 'All', 
                        'JobFamily': 'All',
                        'JobLevel': 'All',
                        'Grouping': 'Recruiter',
                        'Sample_Size': stats['count'],
                        'Mean': round(stats['mean'], 2),
                        'Std_Dev': round(stats['std'], 2),
                        'Min': stats['min'],
                        'Max': stats['max']
                    }
                    qualifying_segments.append(segment_info)
    
    return qualifying_segments

def generate_all_org_segments(df: pd.DataFrame, metric_column: str) -> List[Dict]:
    """
    Generate ALL qualifying organization segments using linear backoff fallback.
    
    Args:
        df: Full dataset of filled requisitions  
        metric_column: The metric to calculate statistics for
        
    Returns:
        List of dictionaries containing all qualifying org segments
    """
    # Define org cluster hierarchy (no recruiter column)
    cluster_hierarchy = [
        ['Country', 'JobGroup', 'JobFamily', 'JobLevel'],   # Cluster 4
        ['Country', 'JobGroup', 'JobFamily'],               # Cluster 3  
        ['Country', 'JobGroup'],                            # Cluster 2
        ['Country']                                         # Cluster 1
    ]
    
    qualifying_segments = []
    
    # For each cluster level, find all qualifying combinations
    for cluster_level, grouping_columns in enumerate(cluster_hierarchy, 1):
        
        # Group data by the current cluster columns
        grouped = df.groupby(grouping_columns)
        
        # Check each group to see if it qualifies
        for group_values, group_data in grouped:
            sample_size = len(group_data)
            
            if sample_size >= MIN_SAMPLE_SIZE:
                # Calculate statistics for this qualifying group
                stats = calculate_statistics(group_data, metric_column)
                
                if 'error' not in stats:
                    # Create row for results table
                    # Handle single value vs tuple for group_values
                    if isinstance(group_values, tuple):
                        country = group_values[0]
                        job_group = group_values[1] if len(group_values) > 1 else 'All'
                        job_family = group_values[2] if len(group_values) > 2 else 'All'
                        job_level = group_values[3] if len(group_values) > 3 else 'All'
                    else:
                        # Single value case (only Country)
                        country = group_values
                        job_group = 'All'
                        job_family = 'All'
                        job_level = 'All'
                    
                    segment_info = {
                        'Segment_Type': 'Organization',
                        'Cluster_Level': f'Cluster_{5-cluster_level}',  # Reverse numbering for org
                        'Recruiter': 'All',
                        'Country': country,
                        'JobGroup': job_group,
                        'JobFamily': job_family,
                        'JobLevel': job_level,
                        'Grouping': ' + '.join(grouping_columns),
                        'Sample_Size': stats['count'],
                        'Mean': round(stats['mean'], 2),
                        'Std_Dev': round(stats['std'], 2),
                        'Min': stats['min'],
                        'Q1': round(stats['q1'], 2),
                        'Median': round(stats['median'], 2),
                        'Q3': round(stats['q3'], 2),
                        'Max': stats['max'],
                        'IQR': round(stats['iqr'], 2)
                    }
                    qualifying_segments.append(segment_info)
    
    return qualifying_segments

# Main application logic starts here
def main():
    """
    Main function that runs the Streamlit application.
    """
    # Load the data from GitHub
    st.subheader("Data Loading")
    with st.spinner("Loading data from GitHub..."):
        df = load_data(GITHUB_CSV_URL)
    
    if df.empty:
        st.error("Failed to load data. Please check the GitHub URL.")
        return
    
    # Show basic data info
    st.success(f"Loaded {len(df)} filled requisitions")
    st.write(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Show first few rows for reference
    with st.expander("Preview Data"):
        st.dataframe(df.head())
    
    # Metric selection (placeholder for future expansion to Core 4 metrics)
    st.subheader("Metric Selection")
    metric_column = st.selectbox("Select Metric", ["DaysToFirst Screen"])
    
    # Analysis button
    if st.button("Generate All Qualifying Sample Populations"):
        
        st.subheader("Qualifying Sample Populations (n >= 30)")
        
        # Add some debugging info first
        st.write("**Debug Info:**")
        st.write(f"- Total records: {len(df)}")
        st.write(f"- Unique recruiters: {df['Recruiter'].nunique()}")
        st.write(f"- Records per recruiter (avg): {len(df) / df['Recruiter'].nunique():.1f}")
        st.write(f"- Available columns: {list(df.columns)}")
        
        # Check if the metric column exists and has valid data
        if metric_column in df.columns:
            valid_metric_data = df[metric_column].dropna()
            st.write(f"- Valid {metric_column} records: {len(valid_metric_data)} / {len(df)}")
            st.write(f"- {metric_column} range: {valid_metric_data.min():.1f} to {valid_metric_data.max():.1f}")
        else:
            st.error(f"Column '{metric_column}' not found in data!")
            return
        
        with st.spinner("Analyzing all recruiter segments..."):
            # Generate all qualifying recruiter segments
            recruiter_segments = generate_all_recruiter_segments(df, metric_column)
        
        st.write(f"Found {len(recruiter_segments)} qualifying recruiter segments")
        
        with st.spinner("Analyzing all organization segments..."):
            # Generate all qualifying org segments  
            org_segments = generate_all_org_segments(df, metric_column)
            
        st.write(f"Found {len(org_segments)} qualifying organization segments")
        
        # Combine all segments into one table
        all_segments = recruiter_segments + org_segments
        
        if all_segments:
            # Convert to DataFrame for display
            results_df = pd.DataFrame(all_segments)
            
            # Sort by segment type, then by sample size descending
            results_df = results_df.sort_values(['Segment_Type', 'Sample_Size'], 
                                              ascending=[True, False])
            
            # Display summary statistics
            st.write(f"**Total Qualifying Segments Found:** {len(all_segments)}")
            st.write(f"- Recruiter Segments: {len(recruiter_segments)}")
            st.write(f"- Organization Segments: {len(org_segments)}")
            
            # Display the full results table
            st.dataframe(results_df, use_container_width=True)
            
            # Option to download results as CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name=f"reqiq_sample_populations_{metric_column}.csv",
                mime="text/csv"
            )
            
            # Show some summary insights
            with st.expander("Summary Insights"):
                st.write("**Recruiter Segment Insights:**")
                if recruiter_segments:
                    recruiter_df = pd.DataFrame(recruiter_segments)
                    st.write(f"- Average sample size: {recruiter_df['Sample_Size'].mean():.0f}")
                    st.write(f"- Largest sample: {recruiter_df['Sample_Size'].max()}")
                    st.write(f"- Most common cluster level: {recruiter_df['Cluster_Level'].mode().iloc[0]}")
                
                st.write("**Organization Segment Insights:**")
                if org_segments:
                    org_df = pd.DataFrame(org_segments)
                    st.write(f"- Average sample size: {org_df['Sample_Size'].mean():.0f}")
                    st.write(f"- Largest sample: {org_df['Sample_Size'].max()}")
                    st.write(f"- Most common cluster level: {org_df['Cluster_Level'].mode().iloc[0]}")
        
        else:
            st.error("No qualifying segments found (all potential clusters have n < 30)")

# Run the application
if __name__ == "__main__":
    main()
