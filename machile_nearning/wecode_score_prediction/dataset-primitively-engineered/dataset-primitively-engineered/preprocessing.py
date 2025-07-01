import pandas as pd
import numpy as np

# Nạp dữ liệu và đổi tên các cột TOÀN CỤC
def load_and_rename_data():
    submissions = pd.read_csv('./annonimized.csv')
    submissions = submissions.rename(columns={
        "concat('it001',`assignment_id`)": 'assignment_id',
        "concat('it001',`problem_id`)": 'problem_id',
        "concat('it001', username)": 'student_id',
        "concat('it001',`language_id`)": 'language_id'
    })
    
    # Sửa các cột ngày giờ bằng cách thêm năm 2024
    def add_year(dt_series):
        return pd.to_datetime("2024-" + dt_series, format='%Y-%m-%d %H:%M:%S')
    
    submissions['created_at'] = add_year(submissions['created_at'])
    submissions['updated_at'] = add_year(submissions['updated_at'])
    
    return submissions

# 1. Tính các đặc trưng về hiệu suất của sinh viên
def calculate_student_features(df):
    # Lọc các bài nộp cuối cùng
    final_subs = df[df['is_final'] == 1].copy()
    
    # Chuyển đổi điểm về thang 0-100
    final_subs['normalized_score'] = final_subs['pre_score'] / 100
    
    # Tổng hợp theo từng sinh viên
    student_features = final_subs.groupby('student_id').agg(
        avg_score=('normalized_score', 'mean'),
        completed_assignments=('assignment_id', 'nunique'),
        completion_rate=('assignment_id', lambda x: x.nunique() / x.count())
    ).reset_index()
    
    # Thêm tỷ lệ lỗi
    error_subs = df[df['status'].str.contains('Error')]
    error_rate = error_subs.groupby('student_id').size() / df.groupby('student_id').size()
    student_features['error_rate'] = student_features['student_id'].map(error_rate).fillna(0)
    
    return student_features

# 2. Tính độ khó của bài tập
def calculate_assignment_difficulty(df):
    final_subs = df[df['is_final'] == 1].copy()
    final_subs['normalized_score'] = final_subs['pre_score'] / 100
    
    assignment_difficulty = final_subs.groupby('assignment_id').agg(
        assignment_avg_score=('normalized_score', 'mean'),
        assignment_completion=('student_id', 'count')
    ).reset_index()
    
    return assignment_difficulty

# 3. Tính các đặc trưng về hành vi theo thời gian
def calculate_behavioral_features(df):
    # Tính thời gian thực hiện (đã có định dạng datetime phù hợp)
    df['attempt_duration'] = (df['updated_at'] - df['created_at']).dt.total_seconds()
    
    behavioral_features = df.groupby(['student_id', 'problem_id']).agg(
        attempts=('problem_id', 'count'),
        total_duration=('attempt_duration', 'sum')
    ).groupby('student_id').agg(
        avg_attempts=('attempts', 'mean'),
        avg_time_per_problem=('total_duration', 'mean')
    ).reset_index()
    
    return behavioral_features

# 4. Kết hợp với các đặc trưng của lớp học
def create_final_dataset(submissions, class_assignments):
    # Tính các đặc trưng
    student_feats = calculate_student_features(submissions)
    assignment_feats = calculate_assignment_difficulty(submissions)
    behavioral_feats = calculate_behavioral_features(submissions)
    
    # Gộp các đặc trưng
    features = student_feats.merge(behavioral_feats, on='student_id', how='left')
    features = features.merge(class_assignments, on='student_id', how='left')
    
    # Thêm các tổng hợp ở cấp lớp học
    class_stats = features.groupby('class_id').agg(
        class_avg_score=('avg_score', 'mean'),
        class_size=('student_id', 'count')
    ).reset_index()
    
    final_features = features.merge(class_stats, on='class_id', how='left')
    
    # Thêm hiệu suất tương đối
    final_features['relative_performance'] = (
        (final_features['avg_score'] - final_features['class_avg_score']) 
        / final_features['class_avg_score'].std()
    )
    
    return final_features

# Thực thi chính
if __name__ == "__main__":
    # Nạp dữ liệu và đổi tên cột ĐẦU TIÊN
    submissions = load_and_rename_data()
    class_assignments = pd.read_csv('../dataset-full-raw/student_class_assignments.csv')
    
    # Chạy pipeline
    final_dataset = create_final_dataset(submissions, class_assignments)
    final_dataset.to_csv('score_prediction_features.csv', index=False)
    print("Hoàn thành trích xuất đặc trưng thành công!")
