import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

# Set up the page configuration - this must be the first Streamlit command
st.set_page_config(page_title="RequisitionIQ", layout="wide")

# Title and description
st.title("RequisitionIQ - Requisition Intelligence Engine")
st.write("Statistical analysis of requisition performance using linear backoff fallback clustering")

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

def get_recruiter_clusters(df: pd.DataFrame, recruiter: str) -> Dict[str, pd.DataFrame]:
    """
    Create recruiter-based clusters using linear backoff fallback.
    
    Args:
        df: Full dataset of filled requisitions
        recruiter: The specific recruiter we're analyzing
        
    Returns:
        Dictionary with cluster names as keys and filtered DataFrames as values
    """
    # Filter data to only include requisitions from this recruiter
    recruiter_data = df[df['Recruiter'] == recruiter].copy()
    
    # Define cluster hierarchy from most specific to least specific
    clusters = {
        'Cluster_5': ['Recruiter', 'Country', 'JobGroup', 'JobFamily', 'JobLevel'],
        'Cluster_4': ['Recruiter', 'Country', 'JobGroup', 'JobFamily'], 
        'Cluster_3': ['Recruiter', 'Country', 'JobGroup'],
        'Cluster_2': ['Recruiter', 'Country'],
        'Cluster_1': ['Recruiter']
    }
    
    cluster_results = {}
    
    # Try each cluster level, starting from most specific
    for cluster_name, grouping_columns in clusters.items():
        # Group by the specified columns and count records in each group
        grouped = recruiter_data.groupby(grouping_columns).size()
        
        # For recruiter clusters, we're looking at the specific recruiter's data
        # So we just need to check if the total count meets our threshold
        total_count = len(recruiter_data)
        
        cluster_results[cluster_name] = {
            'data': recruiter_data,
            'count': total_count,
            'grouping': grouping_columns
        }
    
    return cluster_results

def get_org_clusters(df: pd.DataFrame, country: str, job_group: str, 
                    job_family: str, job_level: str) -> Dict[str, pd.DataFrame]:
    """
    Create organization-wide clusters (all recruiters) using linear backoff fallback.
    
    Args:
        df: Full dataset of filled requisitions
        country: Country to filter by
        job_group: Job group to filter by  
        job_family: Job family to filter by
        job_level: Job level to filter by
        
    Returns:
        Dictionary with cluster names as keys and filtered DataFrames as values
    """
    # Define org cluster hierarchy (no recruiter column)
    clusters = {
        'Cluster_4': ['Country', 'JobGroup', 'JobFamily', 'JobLevel'],
        'Cluster_3': ['Country', 'JobGroup', 'JobFamily'],
        'Cluster_2': ['Country', 'JobGroup'], 
        'Cluster_1': ['Country']
    }
    
    cluster_results = {}
    
    # Try each cluster level, starting from most specific
    for cluster_name, grouping_columns in clusters.items():
        # Build filter conditions based on cluster level
        filtered_df = df.copy()
        
        # Apply filters based on what columns are in this cluster
        if 'Country' in grouping_columns:
            filtered_df = filtered_df[filtered_df['Country'] == country]
        if 'JobGroup' in grouping_columns:
            filtered_df = filtered_df[filtered_df['JobGroup'] == job_group]
        if 'JobFamily' in grouping_columns:
            filtered_df = filtered_df[filtered_df['JobFamily'] == job_family]
        if 'JobLevel' in grouping_columns:
            filtered_df = filtered_df[filtered_df['JobLevel'] == job_level]
            
        cluster_results[cluster_name] = {
            'data': filtered_df,
            'count': len(filtered_df),
            'grouping': grouping_columns
        }
    
    return cluster_results

def find_best_cluster(cluster_results: Dict) -> Tuple[str, Dict]:
    """
    Find the most specific cluster that meets our minimum sample size requirement.
    
    Args:
        cluster_results: Dictionary of cluster results from get_recruiter_clusters or get_org_clusters
        
    Returns:
        Tuple of (cluster_name, cluster_data) or (None, None) if no cluster qualifies
    """
    # Go through clusters in order (most specific first)
    cluster_order = ['Cluster_5', 'Cluster_4', 'Cluster_3', 'Cluster_2', 'Cluster_1']
    
    for cluster_name in cluster_order:
        if cluster_name in cluster_results:
            cluster_info = cluster_results[cluster_name]
            if cluster_info['count'] >= MIN_SAMPLE_SIZE:
                return cluster_name, cluster_info
    
    return None, None

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
    
    # Input section for analysis parameters
    st.subheader("Analysis Parameters")
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique values for dropdown options
        recruiters = sorted(df['Recruiter'].unique())
        selected_recruiter = st.selectbox("Select Recruiter", recruiters)
        
        countries = sorted(df['Country'].unique()) 
        selected_country = st.selectbox("Select Country", countries)
    
    with col2:
        job_groups = sorted(df['JobGroup'].unique())
        selected_job_group = st.selectbox("Select Job Group", job_groups)
        
        job_families = sorted(df['JobFamily'].unique())
        selected_job_family = st.selectbox("Select Job Family", job_families)
    
    # Job level selection
    job_levels = sorted(df['JobLevel'].unique())
    selected_job_level = st.selectbox("Select Job Level", job_levels)
    
    # Metric selection (placeholder for future expansion to Core 4 metrics)
    st.subheader("Metric Selection")
    metric_column = st.selectbox("Select Metric", ["DaysToFirstScreen"])
    
    # Analysis button
    if st.button("Run Linear Backoff Analysis"):
        
        st.subheader("Linear Backoff Fallback Analysis")
        
        # Recruiter Segment Analysis
        st.write("### Recruiter Segment Analysis")
        recruiter_clusters = get_recruiter_clusters(df, selected_recruiter)
        best_recruiter_cluster, best_recruiter_data = find_best_cluster(recruiter_clusters)
        
        if best_recruiter_cluster:
            st.success(f"Found qualifying recruiter cluster: {best_recruiter_cluster}")
            
            # Calculate statistics
            stats = calculate_statistics(best_recruiter_data['data'], metric_column)
            
            # Display results in a nice table
            results_data = {
                'Segment': ['Recruiter'],
                'Cluster': [best_recruiter_cluster],
                'Grouping': [' + '.join(best_recruiter_data['grouping'])],
                'Sample Size (n)': [best_recruiter_data['count']],
                'Mean': [round(stats['mean'], 2)],
                'Std Dev': [round(stats['std'], 2)],
                'Min': [stats['min']],
                'Max': [stats['max']]
            }
            
            st.dataframe(pd.DataFrame(results_data))
            
            # Show backoff process
            with st.expander("View Backoff Process"):
                st.write("Sample sizes for each cluster level:")
                backoff_data = []
                for cluster_name, cluster_info in recruiter_clusters.items():
                    backoff_data.append({
                        'Cluster': cluster_name,
                        'Grouping': ' + '.join(cluster_info['grouping']),
                        'Sample Size': cluster_info['count'],
                        'Qualifies (n>=30)': 'Yes' if cluster_info['count'] >= MIN_SAMPLE_SIZE else 'No'
                    })
                st.dataframe(pd.DataFrame(backoff_data))
        else:
            st.error("No qualifying recruiter cluster found (all clusters have n < 30)")
        
        # Organization Segment Analysis  
        st.write("### Organization Segment Analysis")
        org_clusters = get_org_clusters(df, selected_country, selected_job_group, 
                                      selected_job_family, selected_job_level)
        best_org_cluster, best_org_data = find_best_cluster(org_clusters)
        
        if best_org_cluster:
            st.success(f"Found qualifying organization cluster: {best_org_cluster}")
            
            # Calculate statistics
            stats = calculate_statistics(best_org_data['data'], metric_column)
            
            # Display results
            results_data = {
                'Segment': ['Organization'],
                'Cluster': [best_org_cluster],
                'Grouping': [' + '.join(best_org_data['grouping'])],
                'Sample Size (n)': [best_org_data['count']],
                'Mean': [round(stats['mean'], 2)],
                'Std Dev': [round(stats['std'], 2)],
                'Min': [stats['min']],
                'Max': [stats['max']]
            }
            
            st.dataframe(pd.DataFrame(results_data))
            
            # Show backoff process
            with st.expander("View Backoff Process"):
                st.write("Sample sizes for each cluster level:")
                backoff_data = []
                for cluster_name, cluster_info in org_clusters.items():
                    backoff_data.append({
                        'Cluster': cluster_name,
                        'Grouping': ' + '.join(cluster_info['grouping']),
                        'Sample Size': cluster_info['count'],
                        'Qualifies (n>=30)': 'Yes' if cluster_info['count'] >= MIN_SAMPLE_SIZE else 'No'
                    })
                st.dataframe(pd.DataFrame(backoff_data))
        else:
            st.error("No qualifying organization cluster found (all clusters have n < 30)")

# Run the application
if __name__ == "__main__":
    main()
