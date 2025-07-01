import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

# Configuration
SIMILARITY_THRESHOLD = 0.3
MIN_CLASS_SIZE = 10
SCORE_THRESHOLD = 6800  # Minimum score to consider as success
STUDENT_PERCENTAGE_THRESHOLD = 0.50  # Less than 35% of students must succeed (increased from 5%)
MIN_ATTEMPT_RATE = 0.05  # At least 5% of students must attempt the problem to avoid extra credit

def hybrid_classification(df, similarity_threshold, min_class_size, skip_ids=None):
    """Classify students into sequential classes, skipping specified IDs."""
    if skip_ids is None:
        skip_ids = []

    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })

    final_submissions = df[df['is_final'] == 1].copy()
    filtered_submissions = final_submissions[~final_submissions['student_id'].isin(skip_ids)]

    assignment_matrix = (
        filtered_submissions.groupby(['student_id', 'assignment_id'])
        .size().unstack(fill_value=0)
        .clip(upper=1)
    )

    jaccard_dist = pdist(assignment_matrix.values, metric='jaccard')
    jaccard_dist = squareform(jaccard_dist)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=1 - similarity_threshold
    )
    initial_clusters = clustering.fit_predict(jaccard_dist)

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

    unique_classes = sorted(set(initial_clusters))
    class_id_map = {old: new for new, old in enumerate(unique_classes)}
    sequential_class_ids = [class_id_map[c] for c in initial_clusters]

    class_assignments = []
    for student, cluster_id in zip(assignment_matrix.index, sequential_class_ids):
        class_assignments.append({
            'student_id': student,
            'class_id': cluster_id
        })

    return pd.DataFrame(class_assignments)

def export_class_statistics(df, class_assignments):
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })

    # Get truly optional problems (max coefficient is 0)
    optional_problems = get_optional_problems(df)

    student_counts = (
        class_assignments['class_id']
        .value_counts()
        .sort_index()
        .reset_index(name='student_count')
        .rename(columns={'index': 'class_id'})
    )

    df_with_classes = df.merge(class_assignments, on='student_id', how='inner')
    final_submissions = df_with_classes[df_with_classes['is_final'] == 1].copy()

    problem_counts = (
        final_submissions.groupby('class_id')['problem_id']
        .nunique()
        .reset_index(name='problem_count')
    )
    assignment_counts = (
        final_submissions.groupby('class_id')['assignment_id']
        .nunique()
        .reset_index(name='assignment_count')
    )
    problems_per_student = (
        final_submissions[final_submissions['pre_score'] >= SCORE_THRESHOLD]
        .groupby(['class_id', 'student_id'])['problem_id']
        .nunique()
        .reset_index(name='problems_solved')
        .groupby('class_id')['problems_solved']
        .agg(['mean', 'std', 'min', 'max'])
        .round(2)
        .reset_index()
    )
    problems_per_student.columns = ['class_id', 'avg_problems_solved_per_student', 'std_problems_solved', 'min_problems_solved', 'max_problems_solved']

    # Count students who achieved score >= SCORE_THRESHOLD for each problem
    success_submissions = final_submissions[final_submissions['pre_score'] >= SCORE_THRESHOLD]
    problem_success_counts = (
        success_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='success_count')
    )
    
    # Count all students who attempted each problem
    all_problems_per_class = (
        final_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='attempt_count')
    )
    
    # Merge success and attempt counts
    problem_success_counts = all_problems_per_class.merge(
        problem_success_counts, 
        on=['class_id', 'problem_id'], 
        how='left'
    ).fillna({'success_count': 0})
    
    problem_success_counts = problem_success_counts.merge(student_counts, on='class_id')
    problem_success_counts['success_rate'] = problem_success_counts['success_count'] / problem_success_counts['student_count']
    problem_success_counts['attempt_rate'] = problem_success_counts['attempt_count'] / problem_success_counts['student_count']
    
    problem_metadata = (
        final_submissions[['problem_id', 'coefficient']]
        .drop_duplicates()
    )
    problem_success_counts = problem_success_counts.merge(problem_metadata, on='problem_id', how='left')
    hard_problems_per_class = (
        problem_success_counts[
            ((problem_success_counts['success_rate'] < STUDENT_PERCENTAGE_THRESHOLD) |
             (problem_success_counts['problem_id'].isin(optional_problems))) &
            (problem_success_counts['attempt_rate'] >= MIN_ATTEMPT_RATE)
        ]
        .groupby('class_id')
        .size()
        .reset_index(name='hard_problems_count')
    )
    total_submissions = (
        df_with_classes.groupby('class_id')
        .size()
        .reset_index(name='total_submissions')
    )
    submissions_per_student = (
        df_with_classes.groupby(['class_id', 'student_id'])
        .size()
        .reset_index(name='submissions')
        .groupby('class_id')['submissions']
        .mean()
        .round(2)
        .reset_index(name='avg_submissions_per_student')
    )

    class_stats = student_counts.merge(problem_counts, on='class_id')
    class_stats = class_stats.merge(assignment_counts, on='class_id')
    class_stats = class_stats.merge(problems_per_student, on='class_id', how='left')
    class_stats = class_stats.merge(hard_problems_per_class, on='class_id', how='left')
    class_stats = class_stats.merge(total_submissions, on='class_id')
    class_stats = class_stats.merge(submissions_per_student, on='class_id')

    class_stats = class_stats.fillna({
        'hard_problems_count': 0,
        'avg_problems_solved_per_student': 0,
        'std_problems_solved': 0,
        'min_problems_solved': 0,
        'max_problems_solved': 0
    })

    class_stats['hard_problems_percentage'] = (
        (class_stats['hard_problems_count'] / class_stats['problem_count'] * 100)
        .round(2)
    )
    class_stats['class_completion_rate'] = (
        (class_stats['avg_problems_solved_per_student'] / class_stats['problem_count'] * 100)
        .round(2)
    )

    column_order = [
        'class_id',
        'student_count',
        'problem_count',
        'assignment_count',
        'hard_problems_count',
        'hard_problems_percentage',
        'avg_problems_solved_per_student',
        'class_completion_rate',
        'std_problems_solved',
        'min_problems_solved',
        'max_problems_solved',
        'total_submissions',
        'avg_submissions_per_student'
    ]
    class_stats = class_stats[column_order]
    return class_stats

def calculate_problem_completion_rates(df, class_assignments):
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })

    # Get truly optional problems (max coefficient is 0)
    optional_problems = get_optional_problems(df)

    df_with_classes = df.merge(class_assignments, on='student_id', how='inner')
    final_submissions = df_with_classes[df_with_classes['is_final'] == 1].copy()
    class_sizes = (
        class_assignments['class_id']
        .value_counts()
        .sort_index()
        .reset_index(name='class_size')
        .rename(columns={'index': 'class_id'})
    )
    # Count students who achieved score >= SCORE_THRESHOLD for each problem
    success_submissions = final_submissions[final_submissions['pre_score'] >= SCORE_THRESHOLD]
    problem_success_counts = (
        success_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='success_count')
    )
    all_problems_per_class = (
        final_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='attempt_count')
    )
    problem_stats = all_problems_per_class.merge(
        problem_success_counts, 
        on=['class_id', 'problem_id'], 
        how='left'
    ).fillna({'success_count': 0})
    problem_stats = problem_stats.merge(class_sizes, on='class_id')
    problem_stats['success_rate'] = problem_stats['success_count'] / problem_stats['class_size']
    problem_stats['attempt_rate'] = problem_stats['attempt_count'] / problem_stats['class_size']
    problem_metadata = (
        final_submissions[['problem_id', 'assignment_id', 'coefficient']]
        .drop_duplicates()
    )
    problem_stats = problem_stats.merge(problem_metadata, on='problem_id', how='left')
    problem_stats['is_hard'] = (
        ((problem_stats['success_rate'] < STUDENT_PERCENTAGE_THRESHOLD) |
         (problem_stats['problem_id'].isin(optional_problems))) &
        (problem_stats['attempt_rate'] >= MIN_ATTEMPT_RATE)
    )
    problem_stats = problem_stats.sort_values(['class_id', 'success_rate', 'problem_id'])
    return problem_stats

def export_hard_problems(problem_completion_rates, class_assignments):
    hard_problems = problem_completion_rates[problem_completion_rates['is_hard'] == True].copy()
    if hard_problems.empty:
        print("No hard problems found to export.")
        return pd.DataFrame()
    class_sizes = (
        class_assignments['class_id']
        .value_counts()
        .sort_index()
        .reset_index(name='class_size')
        .rename(columns={'index': 'class_id'})
    )
    if 'class_size' not in hard_problems.columns:
        hard_problems = hard_problems.merge(class_sizes, on='class_id')
    hard_problems['students_succeeded'] = hard_problems['success_count'].astype(int)
    hard_problems['students_attempted'] = hard_problems['attempt_count'].astype(int)
    hard_problems['success_percentage'] = (hard_problems['success_rate'] * 100).round(2)
    hard_problems['attempt_percentage'] = (hard_problems['attempt_rate'] * 100).round(2)
    
    hard_problems_sorted = hard_problems.sort_values([
        'class_id', 
        'success_rate', 
        'assignment_id', 
        'problem_id'
    ])
    export_columns = [
        'class_id',
        'assignment_id', 
        'problem_id',
        'success_rate',
        'success_percentage',
        'students_succeeded',
        'students_attempted',
        'class_size',
        'attempt_rate',
        'attempt_percentage',
        'coefficient'
    ]
    hard_problems_export = hard_problems_sorted[export_columns].copy()
    return hard_problems_export

def calculate_metrics(df, class_assignments, qt_scores=None, th_scores=None, ck_scores=None):
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })
    
    # Get truly optional problems (max coefficient is 0)
    optional_problems = get_optional_problems(df)
    
    df = df.merge(class_assignments, on='student_id')
    class_problems = (
        df.groupby('class_id')['problem_id']
        .unique()
        .reset_index(name='class_problems')
    )
    class_problems['n_class_problems'] = class_problems['class_problems'].apply(len)
    final_submissions = df[df['is_final'] == 1].copy()
    final_submissions['problem_score'] = (
        final_submissions['pre_score'] * final_submissions['coefficient'] / 100
    )
    final_submissions = final_submissions.merge(class_problems, on='class_id')
    wecode_scores = (
        final_submissions.groupby(['student_id', 'class_id', 'n_class_problems'])
        .agg(total_score=('problem_score', 'sum'))
        .reset_index()
    )
    wecode_scores['wecode_score'] = (
        wecode_scores['total_score'] / wecode_scores['n_class_problems']
    )
    # Count students who achieved score >= SCORE_THRESHOLD for each problem
    success_submissions = final_submissions[final_submissions['pre_score'] >= SCORE_THRESHOLD]
    problem_success_counts = (
        success_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='success_count')
    )
    
    # Count all students who attempted each problem
    all_problems_per_class = (
        final_submissions.groupby(['class_id', 'problem_id'])
        .size()
        .reset_index(name='attempt_count')
    )
    
    # Merge success and attempt counts
    problem_success_counts = all_problems_per_class.merge(
        problem_success_counts, 
        on=['class_id', 'problem_id'], 
        how='left'
    ).fillna({'success_count': 0})
    
    class_sizes = (
        class_assignments['class_id']
        .value_counts()
        .sort_index()
        .reset_index(name='class_size')
        .rename(columns={'index': 'class_id'})
    )
    problem_success_counts = problem_success_counts.merge(class_sizes, on='class_id')
    problem_success_counts['success_rate'] = (
        problem_success_counts['success_count'] / problem_success_counts['class_size']
    )
    problem_success_counts['attempt_rate'] = (
        problem_success_counts['attempt_count'] / problem_success_counts['class_size']
    )
    hard_problems = problem_success_counts[
        ((problem_success_counts['success_rate'] < STUDENT_PERCENTAGE_THRESHOLD) |
         (problem_success_counts['problem_id'].isin(optional_problems))) &
        (problem_success_counts['attempt_rate'] >= MIN_ATTEMPT_RATE)
    ][['class_id', 'problem_id']].drop_duplicates()
    hard_solved = success_submissions.merge(
        hard_problems,
        on=['class_id', 'problem_id']
    )
    hard_solved_counts = (
        hard_solved.groupby(['student_id', 'class_id'])
        .size()
        .reset_index(name='hard_solved')
    )
    scaler = MinMaxScaler()
    for class_id in hard_solved_counts['class_id'].unique():
        class_mask = hard_solved_counts['class_id'] == class_id
        if class_mask.sum() > 1:
            hard_solved_counts.loc[class_mask, 'hard_solved_scaled'] = scaler.fit_transform(
                hard_solved_counts.loc[class_mask, ['hard_solved']]
            )
        else:
            hard_solved_counts.loc[class_mask, 'hard_solved_scaled'] = 1.0
    success_count = (
        df[df['pre_score'] >= SCORE_THRESHOLD]
        .groupby('student_id')
        .size()
        .reset_index(name='success_count')
    )
    error_count = (
        df[df['status'] == 'Compilation Error']
        .groupby('student_id')
        .size()
        .reset_index(name='error_count')
    )
    submit_count = (
        df.groupby('student_id')
        .size()
        .reset_index(name='submit_count')
    )
    metrics = submit_count.merge(success_count, on='student_id', how='left')
    metrics = metrics.merge(error_count, on='student_id', how='left')
    metrics = metrics.fillna(0)
    metrics['success_rate'] = metrics['success_count'] / metrics['submit_count']
    metrics['error_rate'] = metrics['error_count'] / metrics['submit_count']
    metrics['avg_attempt'] = metrics['submit_count'] / metrics['student_id'].map(
        df.groupby('student_id')['problem_id'].nunique()
    )
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

def generate_full_report(student_results, class_problems, class_assignments, 
                         problem_completion_rates, class_statistics, 
                         hard_problems_export, unique_hard_problems):
    report = []
    report.append("="*50)
    report.append("COMPREHENSIVE ANALYSIS REPORT".center(50))
    report.append("="*50)
    report.append(f"\nTotal students: {len(student_results)}")
    report.append(f"Total classes: {class_statistics['class_id'].nunique()}")
    report.append(f"Total problems: {problem_completion_rates['problem_id'].nunique()}")
    report.append("\n" + "="*50)
    report.append("CLASS STATISTICS".center(50))
    report.append("="*50)
    report.append(class_statistics.to_string(index=False))
    report.append("\n" + "="*50)
    report.append("STUDENT PERFORMANCE SUMMARY".center(50))
    report.append("="*50)
    report.append(student_results.describe().to_string())
    report.append("\n" + "="*50)
    report.append("PROBLEM COMPLETION RATES".center(50))
    report.append("="*50)
    problem_summary = problem_completion_rates.groupby('class_id').agg({
        'success_rate': ['mean', 'median', 'std'],
        'is_hard': 'sum'
    })
    report.append(problem_summary.to_string())
    if not hard_problems_export.empty:
        report.append("\n" + "="*50)
        report.append("HARD PROBLEM ANALYSIS".center(50))
        report.append("="*50)
        report.append(f"Total hard problems: {len(hard_problems_export)}")
        if not unique_hard_problems.empty:
            report.append(f"Unique hard problems: {len(unique_hard_problems)}")
        report.append("\nBy class:")
        report.append(hard_problems_export['class_id'].value_counts().sort_index().to_string())
    return "\n".join(report)

def visualize_data(student_results, class_problems, class_assignments):
    plt.figure(figsize=(18, 18))
    plt.suptitle('Student Performance Analysis', fontsize=16)
    plt.subplot(3, 2, 1)
    sns.histplot(data=student_results, x='wecode_score', bins=20, kde=True)
    plt.axvline(student_results['wecode_score'].mean(), color='r', linestyle='--')
    plt.title('Student Wecode Scores')
    plt.xlabel('Wecode Score')
    plt.ylabel('Student Count')
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
    plt.subplot(3, 2, 3)
    if not student_results.empty:
        sns.scatterplot(
            data=student_results,
            x='success_rate',
            y='error_rate',
            size='avg_attempt',
            alpha=0.7
        )
        plt.title('Success Rate vs Error Rate')
        plt.xlabel(f'Success Rate (Score >= {SCORE_THRESHOLD}/Total Submissions)')
        plt.ylabel('Error Rate (Errors/Total Submissions)')
        plt.legend(title='Avg Attempts')
    else:
        plt.text(0.5, 0.5, 'Insufficient submission data', 
                 ha='center', va='center', fontsize=12)
        plt.title('Submission Metrics')
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
    plt.subplot(3, 2, 6)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('student_performance_analysis.png', dpi=300)
    plt.show()

def get_unique_hard_problems(problem_completion_rates, df):
    """
    Get unique hard problems across all classes.
    
    Args:
        problem_completion_rates: DataFrame with problem completion rates and is_hard flag
        df: Original DataFrame to determine optional problems
        
    Returns:
        DataFrame with unique hard problems and their overall statistics
    """
    hard_problems = problem_completion_rates[problem_completion_rates['is_hard'] == True].copy()
    if hard_problems.empty:
        print("No hard problems found.")
        return pd.DataFrame()
    
    # Get truly optional problems (max coefficient is 0)
    optional_problems = get_optional_problems(df)
    
    # Get unique hard problems (deduplicated by problem_id only) with their overall statistics
    unique_hard_problems = (
        hard_problems.groupby('problem_id')
        .agg({
            'assignment_id': 'first',    # Take first assignment_id (should be same for all instances)
            'coefficient': 'first',      # Take first coefficient (should be same for all instances)
            'success_count': 'sum',      # Total students who succeeded across all classes
            'attempt_count': 'sum',      # Total attempts across all classes
            'class_size': 'sum',         # Total students across all classes
            'success_rate': 'mean'       # Average success rate across classes
        })
        .reset_index()
    )
    
    # Add information about whether the problem is truly optional
    unique_hard_problems['is_optional'] = unique_hard_problems['problem_id'].isin(optional_problems)
    
    # Recalculate overall success rate
    unique_hard_problems['overall_success_rate'] = (
        unique_hard_problems['success_count'] / unique_hard_problems['class_size']
    ).round(4)
    
    unique_hard_problems['overall_success_percentage'] = (
        unique_hard_problems['overall_success_rate'] * 100
    ).round(2)
    
    # Count how many classes each problem appears in as hard
    classes_per_problem = (
        hard_problems.groupby('problem_id')['class_id']
        .nunique()
        .reset_index(name='classes_count')
    )
    unique_hard_problems = unique_hard_problems.merge(classes_per_problem, on='problem_id')
    
    # Sort by overall success rate (ascending) to show hardest problems first
    unique_hard_problems = unique_hard_problems.sort_values([
        'overall_success_rate',
        'assignment_id',
        'problem_id'
    ])
    
    # Add difficulty rank
    unique_hard_problems['difficulty_rank'] = range(1, len(unique_hard_problems) + 1)
    
    # Reorder columns
    column_order = [
        'difficulty_rank',
        'problem_id',
        'assignment_id',
        'overall_success_rate',
        'overall_success_percentage',
        'success_count',
        'attempt_count',
        'class_size',
        'classes_count',
        'coefficient',
        'is_optional'
    ]
    
    return unique_hard_problems[column_order]

def get_optional_problems(df):
    """
    Identify problems that are truly optional (max coefficient is 0) vs overdue (max coefficient > 0).
    
    Args:
        df: DataFrame with problem submissions
        
    Returns:
        set: Set of problem_ids that are truly optional (max coefficient across all submissions is 0)
    """
    df = df.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id'
    })
    
    # Get the maximum coefficient for each problem across all submissions
    max_coefficients = (
        df.groupby(['assignment_id', 'problem_id'])['coefficient']
        .max()
        .reset_index()
    )
    
    # Problems are truly optional if their max coefficient is 0
    optional_problems = max_coefficients[
        max_coefficients['coefficient'] == 0
    ]['problem_id'].unique()
    
    return set(optional_problems)

def main():
    df = pd.read_csv('./annonimized.csv')
    try:
        qt_scores = pd.read_csv('./qt-public.csv')
    except FileNotFoundError:
        qt_scores = None
        print("QT scores file not found, proceeding without QT data")
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

    class_assignments = hybrid_classification(
        df,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_class_size=MIN_CLASS_SIZE,
        skip_ids=['1bec7c0b6a9bd8f556a8554c5012dcb778460bac']
    )
    student_results, class_problems = calculate_metrics(df, class_assignments, qt_scores, th_scores, ck_scores)
    problem_completion_rates = calculate_problem_completion_rates(df, class_assignments)
    hard_problems_export = export_hard_problems(problem_completion_rates, class_assignments)
    unique_hard_problems = get_unique_hard_problems(problem_completion_rates, df)
    class_statistics = export_class_statistics(df, class_assignments)

    # CSV exports
    student_results.to_csv('student_results.csv', index=False)
    class_problems.to_csv('class_problems.csv', index=False)
    problem_completion_rates.to_csv('problem_completion_rates.csv', index=False)
    class_statistics.to_csv('class_statistics.csv', index=False)
    if not hard_problems_export.empty:
        hard_problems_export.to_csv('hard_problems_by_class.csv', index=False)
    if not unique_hard_problems.empty:
        unique_hard_problems.to_csv('unique_hard_problems.csv', index=False)

    # Text report export
    report = generate_full_report(
        student_results,
        class_problems,
        class_assignments,
        problem_completion_rates,
        class_statistics,
        hard_problems_export,
        unique_hard_problems
    )
    with open('analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print("\n=== FILES EXPORTED ===")
    print("- analysis_report.txt: Full statistics report")
    print("- student_results.csv: Student performance metrics")
    print("- class_problems.csv: Problems per class summary")
    print("- problem_completion_rates.csv: Detailed problem success rates per class (using score >= 8000)")
    print("- class_statistics.csv: Class statistics with student count, problem count, and additional metrics")
    if not hard_problems_export.empty:
        print("- hard_problems_by_class.csv: Hard problems sorted by class with detailed metrics (using 15% success threshold, 5% attempt filter, and optional problems)")
    if not unique_hard_problems.empty:
        print(f"- unique_hard_problems.csv: {len(unique_hard_problems)} unique hard problems across all classes")
    visualize_data(student_results, class_problems, class_assignments)

if __name__ == '__main__':
    main()
