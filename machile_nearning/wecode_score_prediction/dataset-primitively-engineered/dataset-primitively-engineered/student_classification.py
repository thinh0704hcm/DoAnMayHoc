import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

def hybrid_classification(df, trait_columns=[], similarity_threshold=0.7, min_class_size=10):
    """
    Hybrid classification of students using both assignment patterns and student traits
    
    Parameters:
    df (DataFrame): Input data with student submissions
    trait_columns (list): Columns containing student traits (e.g., 'major', 'year')
    similarity_threshold (float): Jaccard similarity threshold for assignment patterns
    min_class_size (int): Minimum number of students per class
    """
    # Preprocess data
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })
    
    # Filter for final submissions
    final_submissions = df[df['is_final'] == 1].copy()
    
    # Step 1: Create student-assignment matrix
    assignment_matrix = (
        final_submissions.groupby(['student_id', 'assignment_id'])
        .size().unstack(fill_value=0)
        .clip(upper=1)  # Binary: 1 if submitted, 0 otherwise
    )
    
    # Step 2: Calculate Jaccard distance matrix
    jaccard_dist = pdist(assignment_matrix.values, metric='jaccard')
    jaccard_dist = squareform(jaccard_dist)
    
    # Step 3: Initial clustering based on assignment similarity
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    )
    initial_clusters = clustering.fit_predict(jaccard_dist)
    
    # Step 4: Merge small clusters
    cluster_sizes = pd.Series(initial_clusters).value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_class_size].index
    
    for sc in small_clusters:
        sc_indices = np.where(initial_clusters == sc)[0]
        if len(sc_indices) == 0: 
            continue
            
        # Find nearest cluster by average distance
        avg_distances = []
        for cluster_id in set(initial_clusters):
            if cluster_id == sc: 
                continue
            cluster_indices = np.where(initial_clusters == cluster_id)[0]
            avg_dist = jaccard_dist[np.ix_(sc_indices, cluster_indices)].mean()
            avg_distances.append((cluster_id, avg_dist))
        
        if avg_distances:
            best_match = min(avg_distances, key=lambda x: x[1])[0]
            initial_clusters[initial_clusters == sc] = best_match
    
    # Step 5: Create final output
    class_assignments = []
    for student, cluster_id in zip(assignment_matrix.index, initial_clusters):
        assignments_submitted = assignment_matrix.loc[student].sum()
        class_assignments.append({
            'student_id': student,
            'class_id': cluster_id,
            'assignments_submitted': assignments_submitted
        })
    
    return pd.DataFrame(class_assignments)

# Usage example
df = pd.read_csv('./annonimized.csv')
result = hybrid_classification(
    df,
    similarity_threshold=0.7,
    min_class_size=10
)
# After class assignments DataFrame (class_df) is created:
class_counts = result['class_id'].value_counts().reset_index()
class_counts.columns = ['class_id', 'student_count']

# Save both files
result.to_csv('student_class_assignments.csv', index=False)
class_counts.to_csv('class_student_counts.csv', index=False)

