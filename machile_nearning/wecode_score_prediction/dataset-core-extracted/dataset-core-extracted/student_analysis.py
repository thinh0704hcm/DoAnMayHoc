import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

# Configuration
SIMILARITY_THRESHOLD = 0.7
MIN_CLASS_SIZE = 10
HARD_PROBLEM_THRESHOLD = 0.3  # 30% AC rate

def hybrid_classification(df, similarity_threshold, min_class_size, skip_ids=None):
    """Classify students into sequential classes, skipping specified IDs."""
    if skip_ids is None:
        skip_ids = []

    # Preprocess data
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })

    # Filter for final submissions
    final_submissions = df[df['is_final'] == 1].copy()

    # EXCLUDE SKIP IDS from clustering
    filtered_submissions = final_submissions[~final_submissions['student_id'].isin(skip_ids)]

    # Create student-assignment matrix
    assignment_matrix = (
        filtered_submissions.groupby(['student_id', 'assignment_id'])
        .size().unstack(fill_value=0)
        .clip(upper=1)  # Binary matrix
    )

    # Calculate Jaccard distance matrix
    jaccard_dist = pdist(assignment_matrix.values, metric='jaccard')
    jaccard_dist = squareform(jaccard_dist)

    # Initial clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    )
    initial_clusters = clustering.fit_predict(jaccard_dist)

    # Merge small clusters
    cluster_sizes = pd.Series(initial_clusters).value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_class_size].index

    for sc in small_clusters:
        sc_indices = np.where(initial_clusters == sc)[0]
        if len(sc_indices) == 0: 
            continue
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

    # Reassign sequential class IDs
    unique_classes = sorted(set(initial_clusters))
    class_id_map = {old: new for new, old in enumerate(unique_classes)}
    sequential_class_ids = [class_id_map[c] for c in initial_clusters]

    # Create final output for clustered students
    class_assignments = []
    for student, cluster_id in zip(assignment_matrix.index, sequential_class_ids):
        class_assignments.append({
            'student_id': student,
            'class_id': cluster_id
        })

    # Optionally: Assign skipped IDs to a new class or exclude them
    # for student_id in skip_ids:
    #     class_assignments.append({'student_id': student_id, 'class_id': max(sequential_class_ids) + 1})

    return pd.DataFrame(class_assignments)

def calculate_metrics(df, class_assignments, qt_scores=None, th_scores=None, ck_scores=None):  # <-- Added th_scores, ck_scores
    """Calculate student performance metrics and class statistics"""
    # Preprocess data
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })
    
    # Merge class assignments
    df = df.merge(class_assignments, on='student_id')
    
    # Step 1: Get unionized problem sets per class
    class_problems = (
        df.groupby('class_id')['problem_id']
        .unique()
        .reset_index(name='class_problems')
    )
    class_problems['n_class_problems'] = class_problems['class_problems'].apply(len)
    
    # Step 2: Calculate student Wecode scores
    final_submissions = df[df['is_final'] == 1].copy()
    final_submissions['problem_score'] = (
        final_submissions['pre_score'] * final_submissions['coefficient'] / 100
    )
    
    # Merge class problems
    final_submissions = final_submissions.merge(class_problems, on='class_id')
    
    # Calculate Wecode score
    wecode_scores = (
        final_submissions.groupby(['student_id', 'class_id', 'n_class_problems'])
        .agg(total_score=('problem_score', 'sum'))
        .reset_index()
    )
    wecode_scores['wecode_score'] = (
        wecode_scores['total_score'] / wecode_scores['n_class_problems']
    )
    
    # Step 3: Identify hard problems per class
    # Calculate AC rate per problem per class
    ac_submissions = final_submissions[final_submissions['pre_score'] == 10000]
    problem_ac_counts = (
        ac_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='ac_count')
    )
    
    # Get student counts per class (as DataFrame)
    class_sizes = (
        class_assignments['class_id']
        .value_counts()
        .sort_index()
        .reset_index(name='class_size')
        .rename(columns={'index': 'class_id'})
    )
    problem_ac_counts = problem_ac_counts.merge(class_sizes, on='class_id')
    problem_ac_counts['ac_rate'] = (
        problem_ac_counts['ac_count'] / problem_ac_counts['class_size']
    )
    
    # Identify hard problems (AC rate < threshold or coefficient=0)
    hard_problems = problem_ac_counts[
        (problem_ac_counts['ac_rate'] < HARD_PROBLEM_THRESHOLD) |
        (problem_ac_counts['problem_id'].isin(
            df[df['coefficient'] == 0]['problem_id'].unique()
        ))
    ][['class_id', 'problem_id']].drop_duplicates()
    
    # Count hard problems solved per student
    hard_solved = ac_submissions.merge(
        hard_problems,
        on=['class_id', 'problem_id']
    )
    hard_solved_counts = (
        hard_solved.groupby(['student_id', 'class_id'])
        .size()
        .reset_index(name='hard_solved')
    )
    
    # Min-Max scaling per class
    scaler = MinMaxScaler()
    for class_id in hard_solved_counts['class_id'].unique():
        class_mask = hard_solved_counts['class_id'] == class_id
        if class_mask.sum() > 1:  # Requires multiple students to scale
            hard_solved_counts.loc[class_mask, 'hard_solved_scaled'] = scaler.fit_transform(
                hard_solved_counts.loc[class_mask, ['hard_solved']]
            )
        else:
            hard_solved_counts.loc[class_mask, 'hard_solved_scaled'] = 1.0
    
    # Step 4: Calculate submission metrics
    # Accept rate (AC submissions / total submissions)
    accept_count = (
        df[df['pre_score'] == 10000]
        .groupby('student_id')
        .size()
        .reset_index(name='accept_count')
    )
    
    # Error rate (Compilation errors / total submissions)
    error_count = (
        df[df['status'] == 'Compilation Error']
        .groupby('student_id')
        .size()
        .reset_index(name='error_count')
    )
    
    # Total submissions
    submit_count = (
        df.groupby('student_id')
        .size()
        .reset_index(name='submit_count')
    )
    
    # Merge metrics
    metrics = submit_count.merge(accept_count, on='student_id', how='left')
    metrics = metrics.merge(error_count, on='student_id', how='left')
    metrics = metrics.fillna(0)
    
    metrics['accept_rate'] = metrics['accept_count'] / metrics['submit_count']
    metrics['error_rate'] = metrics['error_count'] / metrics['submit_count']
    metrics['avg_attempt'] = metrics['submit_count'] / metrics['student_id'].map(
        df.groupby('student_id')['problem_id'].nunique()
    )
    
    # Step 5: Merge all metrics
    result = wecode_scores.merge(
        hard_solved_counts[['student_id', 'hard_solved', 'hard_solved_scaled']],
        on='student_id',
        how='left'
    ).fillna({'hard_solved': 0, 'hard_solved_scaled': 0})
    result['total_score_scaled'] = result.groupby('class_id')['total_score'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    result['wecode_score_scaled'] = result.groupby('class_id')['wecode_score'].transform(
        lambda x: x / x.max() if x.max() > 0 else 0
    )
    
    result = result.merge(metrics, on='student_id')
    
    # Add QT scores if available
    if qt_scores is not None:
        result = result.merge(
            qt_scores.rename(columns={'hash': 'student_id', 'diemqt': 'qt_score'}),
            on='student_id',
            how='left'
        )
        result['qt_score'] = pd.to_numeric(result['qt_score'], errors='coerce')
        class_avg_qt = (
            result[['class_id', 'qt_score']]
            .dropna()
            .groupby('class_id')['qt_score']
            .mean()
            .reset_index(name='class_avg_qt')
        )
        result = result.merge(class_avg_qt, on='class_id', how='left')
    
    # Add TH scores if available
    if th_scores is not None:
        result = result.merge(
            th_scores.rename(columns={'hash': 'student_id', 'TH': 'th_score'}),
            on='student_id',
            how='left'
        )
        result['th_score'] = pd.to_numeric(result['th_score'], errors='coerce')
        class_avg_th = (
            result[['class_id', 'th_score']]
            .dropna()
            .groupby('class_id')['th_score']
            .mean()
            .reset_index(name='class_avg_th')
        )
        result = result.merge(class_avg_th, on='class_id', how='left')
    
    # Add CK scores if available
    if ck_scores is not None:
        result = result.merge(
            ck_scores.rename(columns={'hash': 'student_id', 'CK': 'ck_score'}),
            on='student_id',
            how='left'
        )
        result['ck_score'] = pd.to_numeric(result['ck_score'], errors='coerce')
        class_avg_ck = (
            result[['class_id', 'ck_score']]
            .dropna()
            .groupby('class_id')['ck_score']
            .mean()
            .reset_index(name='class_avg_ck')
        )
        result = result.merge(class_avg_ck, on='class_id', how='left')
    
    return result, class_problems

def visualize_data(student_results, class_problems, class_assignments):
    """Create visualizations for student performance metrics"""
    plt.figure(figsize=(18, 18))
    plt.suptitle('Student Performance Analysis', fontsize=16)
    
    # Plot 1: Wecode Score Distribution
    plt.subplot(3, 2, 1)
    sns.histplot(data=student_results, x='wecode_score', bins=20, kde=True)
    plt.axvline(student_results['wecode_score'].mean(), color='r', linestyle='--')
    plt.title('Student Wecode Scores')
    plt.xlabel('Wecode Score')
    plt.ylabel('Student Count')
    
    # Plot 2: Hard Problems Solved
    plt.subplot(3, 2, 2)
    if 'hard_solved_scaled' in student_results:
        sns.histplot(data=student_results, x='hard_solved_scaled', bins=15, kde=True)
        plt.title('Scaled Hard Problems Solved')
        plt.xlabel('Scaled Count (Min-Max Normalized)')
        plt.ylabel('Student Count')
    else:
        plt.text(0.5, 0.5, 'No hard problems identified', 
                 ha='center', va='center', fontsize=12)
        plt.title('Hard Problems Solved')
    
    # Plot 3: Submission Metrics
    plt.subplot(3, 2, 3)
    if not student_results.empty:
        sns.scatterplot(
            data=student_results,
            x='accept_rate',
            y='error_rate',
            size='avg_attempt',
            alpha=0.7
        )
        plt.title('Accept Rate vs Error Rate')
        plt.xlabel('Accept Rate (AC/Total Submissions)')
        plt.ylabel('Error Rate (Errors/Total Submissions)')
        plt.legend(title='Avg Attempts')
    else:
        plt.text(0.5, 0.5, 'Insufficient submission data', 
                 ha='center', va='center', fontsize=12)
        plt.title('Submission Metrics')
    
    # Plot 4: Class Problem Distribution
    plt.subplot(3, 2, 4)
    if not class_problems.empty:
        class_sizes = class_assignments['class_id'].value_counts().sort_index()
        problem_counts = class_problems.set_index('class_id')['n_class_problems']
        class_data = pd.DataFrame({
            'class_id': class_sizes.index,
            'Class Size': class_sizes.values,
            'Problem Count': problem_counts.reindex(class_sizes.index).fillna(0).values
        })
        sns.barplot(
            data=class_data,
            x='class_id',
            y='Problem Count',
            color='skyblue',
            label='Problems per Class'
        )
        sns.lineplot(
            data=class_data,
            x='class_id',
            y='Class Size',
            color='red',
            marker='o',
            label='Class Size'
        )
        plt.title('Class Size vs Problem Count')
        plt.xlabel('Class ID')
        plt.ylabel('Count')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No class problem data', 
                ha='center', va='center', fontsize=12)
        plt.title('Class Problem Distribution')

    # Plot 5: Class Average QT, TH, and CK Scores
    plt.subplot(3, 2, 5)
    has_qt = 'class_avg_qt' in student_results
    has_th = 'class_avg_th' in student_results
    has_ck = 'class_avg_ck' in student_results
    plotted = False
    if has_qt:
        class_qt = student_results[['class_id', 'class_avg_qt']].drop_duplicates()
        if not class_qt.empty:
            sns.barplot(
                data=class_qt,
                x='class_id',
                y='class_avg_qt',
                color='green',
                alpha=0.7,
                label='QT'
            )
            plotted = True
    if has_th:
        class_th = student_results[['class_id', 'class_avg_th']].drop_duplicates()
        if not class_th.empty:
            sns.barplot(
                data=class_th,
                x='class_id',
                y='class_avg_th',
                color='blue',
                alpha=0.4,
                label='TH'
            )
            plotted = True
    if has_ck:
        class_ck = student_results[['class_id', 'class_avg_ck']].drop_duplicates()
        if not class_ck.empty:
            sns.barplot(
                data=class_ck,
                x='class_id',
                y='class_avg_ck',
                color='orange',
                alpha=0.4,
                label='CK'
            )
            plotted = True
    if plotted:
        plt.title('Class Average QT/TH/CK Scores')
        plt.xlabel('Class ID')
        plt.ylabel('Score')
        plt.ylim(0, 10)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No QT/TH/CK data available', 
                 ha='center', va='center', fontsize=12)
        plt.title('Class Average QT/TH/CK')

    # Plot 6: Empty for layout
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('student_performance_analysis.png', dpi=300)
    plt.show()

def main():
    # Load data
    df = pd.read_csv('./annonimized.csv')
    try:
        qt_scores = pd.read_csv('./qt-public.csv')
    except FileNotFoundError:
        qt_scores = None
        print("QT scores file not found, proceeding without QT data")
    # NEW: Load TH and CK scores
    try:
        th_scores = pd.read_csv('./th-public.csv')
    except FileNotFoundError:
        th_scores = None
        print("TH scores file not found, proceeding without TH data")
    try:
        ck_scores = pd.read_csv('./ck-public.csv')
    except FileNotFoundError:
        ck_scores = None
        print("CK scores file not found, proceeding without CK data")
    
    # Classify students
    class_assignments = hybrid_classification(
        df,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_class_size=MIN_CLASS_SIZE,
        skip_ids=['1bec7c0b6a9bd8f556a8554c5012dcb778460bac']  # Practice assignment
    )
    
    # Calculate metrics (pass TH and CK)
    student_results, class_problems = calculate_metrics(df, class_assignments, qt_scores, th_scores, ck_scores)
    
    # Save results
    student_results.to_csv('student_results.csv', index=False)
    class_problems.to_csv('class_problems.csv', index=False)
    
    # Visualization
    visualize_data(student_results, class_problems, class_assignments)

if __name__ == '__main__':
    main()
