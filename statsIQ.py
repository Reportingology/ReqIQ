"""
statsIQ.py - Core statistical functions for RequisitionIQ
Handles linear backoff fallback clustering, statistical calculations, and z-score generation
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import re

# Constants
MIN_SAMPLE_SIZE = 30
RECENT_REQ_LIMIT = 30

def extract_req_number(req_id: str) -> int:
    """Extract numeric part from RequisitionID for recency sorting."""
    try:
        numbers = re.findall(r'\d+', str(req_id))
        if numbers:
            return int(numbers[-1])
        return 0
    except:
        return 0

def get_most_recent_reqs(df: pd.DataFrame, limit: int = RECENT_REQ_LIMIT) -> pd.DataFrame:
    """Filter to most recent requisitions based on RequisitionID numbers."""
    df_copy = df.copy()
    df_copy['req_number'] = df_copy['RequisitionID'].apply(extract_req_number)
    df_sorted = df_copy.sort_values('req_number', ascending=False)
    recent_df = df_sorted.head(limit)
    return recent_df.drop('req_number', axis=1)

def calculate_statistics(df: pd.DataFrame, metric_column: str) -> Dict:
    """Calculate basic statistics for a metric."""
    if metric_column not in df.columns:
        return {'error': f'Column {metric_column} not found'}
    
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

def calculate_company_baseline(df: pd.DataFrame, metric_column: str) -> Dict:
    """Calculate overall company statistics as fallback baseline."""
    recent_data = get_most_recent_reqs(df, RECENT_REQ_LIMIT)
    return calculate_statistics(recent_data, metric_column)

def generate_recruiter_segments(df: pd.DataFrame, metric_column: str) -> List[Dict]:
    """Generate qualifying recruiter segments using linear backoff fallback."""
    cluster_hierarchy = [
        (['Recruiter', 'Country', 'JobGroup', 'JobFamily', 'JobLevel'], 'Cluster_5'),
        (['Recruiter', 'Country', 'JobGroup', 'JobFamily'], 'Cluster_4'),
        (['Recruiter', 'Country', 'JobGroup'], 'Cluster_3'),
        (['Recruiter', 'Country'], 'Cluster_2'),
        (['Recruiter'], 'Cluster_1')
    ]
    
    qualifying_segments = []
    recruiters = df['Recruiter'].unique()
    
    for recruiter in recruiters:
        recruiter_data = df[df['Recruiter'] == recruiter].copy()
        found_qualifying_cluster = False
        
        for grouping_columns, cluster_name in cluster_hierarchy:
            if found_qualifying_cluster:
                break
            
            if len(grouping_columns) == 1:  # Cluster_1: Just recruiter
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
                # Multi-column clusters
                grouped = recruiter_data.groupby(grouping_columns[1:])
                
                for group_values, group_data in grouped:
                    recent_data = get_most_recent_reqs(group_data, RECENT_REQ_LIMIT)
                    sample_size = len(recent_data)
                    
                    if sample_size >= MIN_SAMPLE_SIZE:
                        stats = calculate_statistics(recent_data, metric_column)
                        
                        if 'error' not in stats:
                            if not isinstance(group_values, tuple):
                                group_values = (group_values,)
                            
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
                            break
    
    return qualifying_segments

def generate_org_segments(df: pd.DataFrame, metric_column: str) -> List[Dict]:
    """Generate qualifying organization segments using linear backoff fallback."""
    cluster_hierarchy = [
        (['Country', 'JobGroup', 'JobFamily', 'JobLevel'], 'Cluster_4'),
        (['Country', 'JobGroup', 'JobFamily'], 'Cluster_3'),  
        (['Country', 'JobGroup'], 'Cluster_2'),
        (['Country'], 'Cluster_1')
    ]
    
    qualifying_segments = []
    
    for grouping_columns, cluster_name in cluster_hierarchy:
        grouped = df.groupby(grouping_columns)
        
        for group_values, group_data in grouped:
            recent_data = get_most_recent_reqs(group_data, RECENT_REQ_LIMIT)
            sample_size = len(recent_data)
            
            if sample_size >= MIN_SAMPLE_SIZE:
                stats = calculate_statistics(recent_data, metric_column)
                
                if 'error' not in stats:
                    if isinstance(group_values, tuple):
                        country = group_values[0]
                        job_group = group_values[1] if len(group_values) > 1 else 'All'
                        job_family = group_values[2] if len(group_values) > 2 else 'All'
                        job_level = group_values[3] if len(group_values) > 3 else 'All'
                    else:
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

def generate_qualifying_segments(df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """
    Main function to generate all qualifying sample populations.
    
    Args:
        df: DataFrame with filled requisitions
        metric_column: Column name for the metric to analyze
        
    Returns:
        DataFrame with all qualifying segments and their statistics
    """
    # Generate both types of segments
    recruiter_segments = generate_recruiter_segments(df, metric_column)
    org_segments = generate_org_segments(df, metric_column)
    
    # Combine and convert to DataFrame
    all_segments = recruiter_segments + org_segments
    
    if all_segments:
        results_df = pd.DataFrame(all_segments)
        # Sort by segment type, then by sample size descending
        results_df = results_df.sort_values(['Segment_Type', 'Sample_Size'], 
                                          ascending=[True, False])
        return results_df
    else:
        return pd.DataFrame()

def find_best_recruiter_match(segments_df: pd.DataFrame, recruiter: str, country: str, 
                             job_group: str, job_family: str, job_level: str) -> Optional[pd.Series]:
    """
    Find the most specific recruiter segment match using linear backoff.
    
    Args:
        segments_df: DataFrame with qualifying segments
        recruiter, country, job_group, job_family, job_level: Requisition attributes
        
    Returns:
        Series with the best matching segment, or None if no match
    """
    # Filter to recruiter segments only
    recruiter_segments = segments_df[segments_df['Segment_Type'] == 'Recruiter'].copy()
    recruiter_segments = recruiter_segments[recruiter_segments['Recruiter'] == recruiter]
    
    # Try clusters in order of specificity (5 is most specific)
    cluster_priority = ['Cluster_5', 'Cluster_4', 'Cluster_3', 'Cluster_2', 'Cluster_1']
    
    for cluster in cluster_priority:
        cluster_matches = recruiter_segments[recruiter_segments['Cluster_Level'] == cluster]
        
        for _, segment in cluster_matches.iterrows():
            # Check if this segment matches the requisition attributes
            if cluster == 'Cluster_5':
                if (segment['Country'] == country and segment['JobGroup'] == job_group and 
                    segment['JobFamily'] == job_family and segment['JobLevel'] == job_level):
                    return segment
            elif cluster == 'Cluster_4':
                if (segment['Country'] == country and segment['JobGroup'] == job_group and 
                    segment['JobFamily'] == job_family):
                    return segment
            elif cluster == 'Cluster_3':
                if (segment['Country'] == country and segment['JobGroup'] == job_group):
                    return segment
            elif cluster == 'Cluster_2':
                if segment['Country'] == country:
                    return segment
            elif cluster == 'Cluster_1':
                # Cluster_1 only depends on recruiter, which we already filtered for
                return segment
    
    return None

def find_best_org_match(segments_df: pd.DataFrame, country: str, job_group: str, 
                       job_family: str, job_level: str) -> Optional[pd.Series]:
    """
    Find the most specific organization segment match using linear backoff.
    
    Args:
        segments_df: DataFrame with qualifying segments
        country, job_group, job_family, job_level: Requisition attributes
        
    Returns:
        Series with the best matching segment, or None if no match
    """
    # Filter to organization segments only
    org_segments = segments_df[segments_df['Segment_Type'] == 'Organization'].copy()
    
    # Try clusters in order of specificity (4 is most specific for org)
    cluster_priority = ['Cluster_4', 'Cluster_3', 'Cluster_2', 'Cluster_1']
    
    for cluster in cluster_priority:
        cluster_matches = org_segments[org_segments['Cluster_Level'] == cluster]
        
        for _, segment in cluster_matches.iterrows():
            # Check if this segment matches the requisition attributes
            if cluster == 'Cluster_4':
                if (segment['Country'] == country and segment['JobGroup'] == job_group and 
                    segment['JobFamily'] == job_family and segment['JobLevel'] == job_level):
                    return segment
            elif cluster == 'Cluster_3':
                if (segment['Country'] == country and segment['JobGroup'] == job_group and 
                    segment['JobFamily'] == job_family):
                    return segment
            elif cluster == 'Cluster_2':
                if (segment['Country'] == country and segment['JobGroup'] == job_group):
                    return segment
            elif cluster == 'Cluster_1':
                if segment['Country'] == country:
                    return segment
    
    return None

def calculate_z_score(value: float, mean: float, std: float) -> float:
    """Calculate z-score: (value - mean) / std"""
    if pd.isna(value) or pd.isna(mean) or pd.isna(std) or std == 0:
        return np.nan
    return (value - mean) / std

def score_open_requisitions(open_reqs_df: pd.DataFrame, segments_df: pd.DataFrame, 
                           filled_reqs_df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """
    Score open requisitions using recruiter and org segment baselines.
    
    Args:
        open_reqs_df: DataFrame with open requisitions
        segments_df: DataFrame with qualifying segments (from generate_qualifying_segments)
        filled_reqs_df: DataFrame with filled requisitions (for company baseline)
        metric_column: Column name for the metric to score
        
    Returns:
        DataFrame with z-scores added
    """
    # Calculate company baseline for fallback
    company_stats = calculate_company_baseline(filled_reqs_df, metric_column)
    
    results = open_reqs_df.copy()
    
    # Initialize new columns
    results['Recruiter_Segment_Match'] = None
    results['Recruiter_Z_Score'] = np.nan
    results['Org_Segment_Match'] = None  
    results['Org_Z_Score'] = np.nan
    results['Company_Z_Score'] = np.nan
    
    for idx, req in open_reqs_df.iterrows():
        # Extract requisition attributes
        recruiter = req['Recruiter']
        country = req['Country']
        job_group = req['JobGroup']
        job_family = req['JobFamily']
        job_level = req['JobLevel']
        metric_value = req[metric_column]
        
        # Find best recruiter match
        recruiter_match = find_best_recruiter_match(segments_df, recruiter, country, 
                                                   job_group, job_family, job_level)
        
        if recruiter_match is not None:
            results.loc[idx, 'Recruiter_Segment_Match'] = recruiter_match['Cluster_Level']
            results.loc[idx, 'Recruiter_Z_Score'] = calculate_z_score(
                metric_value, recruiter_match['Mean'], recruiter_match['Std_Dev'])
        
        # Find best org match
        org_match = find_best_org_match(segments_df, country, job_group, job_family, job_level)
        
        if org_match is not None:
            results.loc[idx, 'Org_Segment_Match'] = org_match['Cluster_Level']
            results.loc[idx, 'Org_Z_Score'] = calculate_z_score(
                metric_value, org_match['Mean'], org_match['Std_Dev'])
        
        # Calculate company baseline z-score
        if 'error' not in company_stats:
            results.loc[idx, 'Company_Z_Score'] = calculate_z_score(
                metric_value, company_stats['mean'], company_stats['std'])
    
    return results
