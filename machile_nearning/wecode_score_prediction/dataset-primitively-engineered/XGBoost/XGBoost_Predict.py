# Submission ID :       273208
import pandas as pd
import joblib
import xgboost as xgb
import os

# Load preprocessed features for ALL students
all_features = pd.read_csv('../dataset-primitively-engineered/score_prediction_features.csv')

# Load exam scores (TH) - only students with known scores
th_df = pd.read_csv('../dataset-primitively-engineered/th-public.csv')
th_df = th_df.rename(columns={'hash': 'student_id', 'TH': 'exam_score'})
th_df['exam_score'] = pd.to_numeric(th_df['exam_score'], errors='coerce')

# Identify students without scores
scored_students = set(th_df['student_id'].dropna())
all_students = set(all_features['student_id'])
unscored_students = list(all_students - scored_students)

print(f"Total students: {len(all_students)}")
print(f"Students with scores: {len(scored_students)}")
print(f"Students to predict: {len(unscored_students)}")

# Prepare data for prediction
unscored_data = all_features[all_features['student_id'].isin(unscored_students)]
X_predict = unscored_data.drop(columns=['student_id', 'class_id', 'exam_score'], errors='ignore')

# Load trained model
MODEL_PATH = "student_score_predictor_optuna_xgb.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = joblib.load(MODEL_PATH)

# Convert to DMatrix for XGBoost
d_predict = xgb.DMatrix(X_predict)

# Make predictions
predictions = model.predict(d_predict)

# Create results dataframe
results = pd.DataFrame({
    'student_id': unscored_data['student_id'],
    'predicted_exam_score': predictions
})

# Save predictions
results.to_csv('unscored_students_predictions.csv', index=False)
print(f"Predictions saved for {len(results)} students")
print("Sample predictions:")
print(results.head())
