import pandas as pd
import os

# Đường dẫn file gốc
LOCAL_PATH = "../"
qt_true_path = os.path.join(LOCAL_PATH, "dataset-primitively-engineered", "qt-public.csv")
th_true_path = os.path.join(LOCAL_PATH, "dataset-primitively-engineered", "th-public.csv")
ck_true_path = os.path.join(LOCAL_PATH, "dataset-primitively-engineered", "ck-public.csv")

# Hàm đọc điểm gốc
def load_true_scores(path, score_col_name):
    df = pd.read_csv(path)
    df = df.rename(columns={'hash': 'student_id', score_col_name: 'exam_score'})
    df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
    df = df[['student_id', 'exam_score']].dropna(subset=['exam_score'])
    df = df.drop_duplicates('student_id')
    return df

# Load dữ liệu điểm gốc
qt_true = load_true_scores(qt_true_path, 'diemqt')
th_true = load_true_scores(th_true_path, 'TH')
ck_true = load_true_scores(ck_true_path, 'CK')

# Ghép dữ liệu điểm gốc
final_scores = qt_true.merge(th_true, on='student_id', how='outer')
final_scores = final_scores.merge(ck_true, on='student_id', how='outer')

# Đổi tên cột điểm cho dễ xử lý
final_scores = final_scores.rename(columns={'exam_score_x': 'qt_score', 'exam_score_y': 'th_score', 'exam_score': 'ck_score'})

# Tính điểm tích lũy
final_scores['accumulated_score'] = (
    final_scores['qt_score'].fillna(0) * 0.4 +
    final_scores['th_score'].fillna(0) * 0.2 +
    final_scores['ck_score'].fillna(0) * 0.4
)

# Chỉ xuất ra accumulated_score và student_id
result_df = final_scores[['student_id', 'accumulated_score']]
print(result_df.head())

# Lưu kết quả
output_path = os.path.join(LOCAL_PATH, "dataset-primitively-engineered", "tbtl-public.csv")
result_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")
