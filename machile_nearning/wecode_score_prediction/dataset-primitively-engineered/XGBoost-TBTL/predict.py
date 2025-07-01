# Submission ID :       273499
import os
import pandas as pd
import joblib
import xgboost as xgb

# File paths
features_path = "../dataset-primitively-engineered/score_prediction_features.csv"
tbtl_path = "../dataset-primitively-engineered/tbtl-public.csv"
model_path = "tbtl_student_score_predictor_optuna_xgb.pkl"
output_path = "tbtl_unscored_predictions.csv"

def predict_tbtl_scores():
    # Load data
    all_features = pd.read_csv(features_path)
    tbtl_df = pd.read_csv(tbtl_path).rename(columns={'hash': 'student_id', 'accumulated_score': 'exam_score'})
    tbtl_df['exam_score'] = pd.to_numeric(tbtl_df['exam_score'], errors='coerce')
    
    # Identify unscored students
    scored_students = set(tbtl_df['student_id'].dropna())
    all_students = set(all_features['student_id'])
    unscored_students = list(all_students - scored_students)
    
    print(f"Students to predict: {len(unscored_students)}")
    
    if not unscored_students:
        print("No students to predict. Exiting.")
        return None
    
    # Prepare prediction data
    unscored_data = all_features[all_features['student_id'].isin(unscored_students)]
    X_predict = unscored_data.drop(columns=['student_id', 'class_id', 'exam_score'], errors='ignore')
    
    # Load trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    model = joblib.load(model_path)
    
    # Make predictions
    d_predict = xgb.DMatrix(X_predict)
    predictions = model.predict(d_predict)
    
    # Save results
    results = pd.DataFrame({
        'student_id': unscored_data['student_id'],
        'predicted_exam_score': predictions
    })
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    return results.head()

if __name__ == "__main__":
    # Install dependencies if needed (uncomment if necessary)
    # os.system("pip install xgboost scikit-learn pandas numpy joblib")
    
    sample_results = predict_tbtl_scores()
    if sample_results is not None:
        print("\nSample results:")
        print(sample_results)
