import pandas as pd
from sklearn.preprocessing import minmax_scale

# Đọc dữ liệu bài nộp và điểm TH
df = pd.read_csv('annonimized.csv')
th = pd.read_csv('th-public.csv').rename(columns={'hash': 'student_id', 'TH': 'th_score'})

# Đổi tên cột cho đồng nhất
df = df.rename(columns={
    "concat('it001',`assignment_id`)": 'assignment_id',
    "concat('it001',`problem_id`)": 'problem_id',
    "concat('it001', username)": 'student_id'
})

# Lấy điểm trung bình của mỗi sinh viên trên các bài nộp cuối cùng
final_subs = df[df['is_final'] == 1].copy()
student_avg = final_subs.groupby('student_id')['pre_score'].mean().reset_index(name='avg_score')

# Chuẩn hóa min-max cột avg_score
student_avg['avg_score_scaled'] = minmax_scale(student_avg['avg_score'])

# Gộp với điểm TH
result = student_avg.merge(th[['student_id', 'th_score']], on='student_id', how='left')

# Chỉ giữ 3 cột cần thiết
result = result[['student_id', 'avg_score_scaled', 'th_score']]

# Xuất ra file kết quả
result.to_csv('student_simple_results.csv', index=False)
print("Đã lưu student_simple_results.csv với 3 cột: student_id, avg_score_scaled, th_score.")
