import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
import re

# Set up the page configuration - this must be the first Streamlit command
st.set_page_config(page_title="RequisitionIQ", layout="wide")

# Title and description
st.title("RequisitionIQ - Requisition Intelligence Engine")
st.write("Generate all qualifying sample populations (n>=30) for recruiter and org segments using most recent 30 requisitions")

# Constants - these define our clustering hierarchy and minimum sample size
MIN_SAMPLE_SIZE = 30  # Minimum number of records needed for statistical validity
RECENT_REQ_LIMIT = 30  # Only use most recent 30 requisitions per sample
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

def extract_req_number(req_id: str) -> int:
    """
    Extract numeric part from RequisitionID for sorting by recency.
    
    Args:
        req_id: RequisitionID string (e.g., 'RIQ-6757')
        
    Returns:
        Integer part of the requisition ID
    """
    try:
        # Extract numbers from the RequisitionID
        numbers = re.findall(r'\d+', str(req_id))
        if numbers:
            return int(numbers[-1])  # Take the last number found
        return 0
    except:
        return 0

def get_most_recent_reqs(df: pd.DataFrame, limit: int = RECENT_REQ_LIMIT) -> pd.DataFrame:
    """
    Filter dataframe to only include the most recent requisitions based on RequisitionID numbers.
    
    Args:
        df: DataFrame with requisition data
        limit: Maximum number of recent requisitions to include
        
    Returns:
        DataFrame filtered to most recent requisitions
    """
    # Add a numeric column for sorting
    df_copy = df.copy()
    df_copy['req_number'] = df_copy['RequisitionID'].apply(extract_req_number)
    
    # Sort by req_number descending (most recent first) and take top N
    df_sorted = df_copy.sort_values('req_number', ascending=False)
    recent_df = df_sorted.head(limit)
    
    # Remove the helper column and return
    return recent_df.drop('req_number', axis=1)

def calculate_statistics(df: pd.DataFrame, metric_column: str) -> Dict:
    """
    Calculate basic statistics for a metric.
    
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
    Uses only the most recent 30 requisitions for each sample population.
    
    Args:
        df: Full dataset of filled requisitions
        metric_column: The metric to calculate statistics for
        
    Returns:
        List of dictionaries containing all qualifying recruiter segments
    """
    # Define cluster hierarchy from most specific to least specific
    cluster_hierarchy = [
        (['Recruiter', 'Country', 'JobGroup', 'JobFamily', 'JobLevel'], 'Cluster_5'),
        (['Recruiter', 'Country', 'JobGroup', 'JobFamily'], 'Cluster_4'),
        (['Recruiter', 'Country', 'JobGroup'], 'Cluster_3'),
        (['Recruiter', 'Country'], 'Cluster_2'),
        (['Recruiter'], 'Cluster_1')
    ]
    
    qualifying_segments = []
    
    # Get all unique recruiters
    recruiters = df['Recruiter'].unique()
    
    # For each recruiter, find their best qualifying cluster using linear backoff
    for recruiter in recruiters:
        recruiter_data = df[df['Recruiter'] == recruiter].copy()
        found_qualifying_cluster = False
        
        # Try each cluster level for this recruiter (most specific first)
        for grouping_columns, cluster_name in cluster_hierarchy:
            
            if found_qualifying_cluster:
                break  # Already found qualifying cluster for this recruiter
            
            if len(grouping_columns) == 1:  # Just recruiter (Cluster 1)
                # Get most recent requisitions for this recruiter
                recent_data = get_most_recent_reqs(recruiter_data, RECENT_REQ_LIMIT)
                sample_size = len(recent_data)
                
                if sample_size >= MIN_SAMPLE_SIZE:
                    stats = calculate_statistics(recent_data, metric_column)
                    
                    if 'error' not in stats:
                        segment_info = {
                            'Segment_Type': 'Recruiter',
                            'Cluster_Level': cluster_name,
                            'Recruiter': recruiter,
                            'Country': 'All',
                            'JobGroup': 'All', 
                            'JobFamily': 'All',
                            'JobLevel': 'All',
                            'Grouping': ' + '.join(grouping_columns),
                            'Sample_Size': stats['count'],
                            'Mean': round(stats['mean'], 2),
                            'Std_Dev': round(stats['std'], 2),
                            'Min': stats['min'],
                            'Max': stats['max']
                        }
                        qualifying_segments.append(segment_info)
                        found_qualifying_cluster = True
            else:
                # Group by all columns and check if we get qualifying samples
                grouped = recruiter_data.groupby(grouping_columns[1:])  # Skip 'Recruiter' since we already filtered
                
                # Check each group within this recruiter's data
                for group_values, group_data in grouped:
                    # Get most recent requisitions for this group
                    recent_data = get_most_recent_reqs(group_data, RECENT_REQ_LIMIT)
                    sample_size = len(recent_data)
                    
                    if sample_size >= MIN_SAMPLE_SIZE:
                        stats = calculate_statistics(recent_data, metric_column)
                        
                        if 'error' not in stats:
                            # Handle single value vs tuple for group_values
                            if not isinstance(group_values, tuple):
                                group_values = (group_values,)
                            
                            # Map group values to columns
                            country = group_values[0] if len(group_values) > 0 else 'All'
                            job_group = group_values[1] if len(group_values) > 1 else 'All'
                            job_family = group_values[2] if len(group_values) > 2 else 'All'
                            job_level = group_values[3] if len(group_values) > 3 else 'All'
                            
                            segment_info = {
                                'Segment_Type': 'Recruiter',
                                'Cluster_Level': cluster_name,
                                'Recruiter': recruiter,
                                'Country': country,
                                'JobGroup': job_group,
                                'JobFamily': job_family,
                                'JobLevel': job_level,
                                'Grouping': ' + '.join(grouping_columns),
                                'Sample_Size': stats['count'],
                                'Mean': round(stats['mean'], 2),
                                'Std_Dev': round(stats['std'], 2),
                                'Min': stats['min'],
                                'Max': stats['max']
                            }
                            qualifying_segments.append(segment_info)
                            found_qualifying_cluster = True
                            break  # Take first qualifying cluster at this level
    
    return qualifying_segments

def generate_all_org_segments(df: pd.DataFrame, metric_column: str) -> List[Dict]:
    """
    Generate ALL qualifying organization segments using linear backoff fallback.
    Uses only the most recent 30 requisitions for each sample population.
    
    Args:
        df: Full dataset of filled requisitions  
        metric_column: The metric to calculate statistics for
        
    Returns:
        List of dictionaries containing all qualifying org segments
    """
    # Define org cluster hierarchy (no recruiter column)
    cluster_hierarchy = [
        (['Country', 'JobGroup', 'JobFamily', 'JobLevel'], 'Cluster_4'),
        (['Country', 'JobGroup', 'JobFamily'], 'Cluster_3'),  
        (['Country', 'JobGroup'], 'Cluster_2'),
        (['Country'], 'Cluster_1')
    ]
    
    qualifying_segments = []
    
    # For each cluster level, find all qualifying combinations
    for grouping_columns, cluster_name in cluster_hierarchy:
        
        # Group data by the current cluster columns
        grouped = df.groupby(grouping_columns)
        
        # Check each group to see if it qualifies
        for group_values, group_data in grouped:
            # Get most recent requisitions for this group
            recent_data = get_most_recent_reqs(group_data, RECENT_REQ_LIMIT)
            sample_size = len(recent_data)
            
            if sample_size >= MIN_SAMPLE_SIZE:
                # Calculate statistics for this qualifying group
                stats = calculate_statistics(recent_data, metric_column)
                
                if 'error' not in stats:
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
                        'Cluster_Level': cluster_name,
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
                        'Max': stats['max']
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
        
        st.subheader("Qualifying Sample Populations (n >= 30, Recent 30 Reqs)")
        
        # Add some debugging info first
        st.write("**Debug Info:**")
        st.write(f"- Total records: {len(df)}")
        st.write(f"- Unique recruiters: {df['Recruiter'].nunique()}")
        st.write(f"- Using most recent {RECENT_REQ_LIMIT} requisitions per sample")
        
        # Check if the metric column exists and has valid data
        if metric_column in df.columns:
            valid_metric_data = df[metric_column].dropna()
            st.write(f"- Valid {metric_column} records: {len(valid_metric_data)} / {len(df)}")
            if len(valid_metric_data) > 0:
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
            
        else:
            st.error("No qualifying segments found (all potential clusters have n < 30)")

# Run the application
if __name__ == "__main__":
    main()
