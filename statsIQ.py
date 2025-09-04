"""
statsIQ.py - Core statistical functions for RequisitionIQ
Handles linear backoff fallback clustering and statistical calculations
"""

import pandas as pd
import numpy as np
from typing import List, Dict
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
