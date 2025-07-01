# Submission ID :           274275
import os
import pandas as pd
import joblib
import xgboost as xgb

# Configuration
FEATURES_FILE = "../dataset-student-performance/student_results.csv"
MODEL_PATH = "ck_student_score_predictor_optuna_xgb.pkl"
PREDICTION_OUTPUT = "ck_score_predictions.csv"

def predict_ck_scores():
    # Load student results
    student_results = pd.read_csv(FEATURES_FILE)
    
    # Identify students without CK scores
    unscored_students = student_results[student_results['ck_score'].isna()]
    print(f"Found {len(unscored_students)} students without CK scores")
    
    # Select relevant features
    features = [
        'n_class_problems',
        'hard_solved',
        'total_score_scaled',
        'wecode_score_scaled',
        'submit_count',
        'accept_rate',
        'error_rate',
        'avg_attempt',
        'class_avg_ck',
        'ai_trust_factor',
        'ai_high_accept_low_attempt',
        'ai_rapid_iteration',
        'ai_score_variance',
        'ai_code_complexity'
    ]
    
    # Prepare prediction data
    X_predict = unscored_students[features]
    
    # Load trained model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    model = joblib.load(MODEL_PATH)
    
    # Convert to DMatrix for prediction
    d_predict = xgb.DMatrix(X_predict)
    
    # Make predictions
    predictions = model.predict(d_predict)
    
    # Create results dataframe
    results = unscored_students[['student_id']].copy()
    results['predicted_ck_score'] = predictions
    
    # Save predictions
    results.to_csv(PREDICTION_OUTPUT, index=False)
    print(f"Predictions saved to {PREDICTION_OUTPUT}")
    print(f"Sample predictions:\n{results.head()}")
    
    return results

if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Please train the model first.")
    else:
        predictions = predict_ck_scores()