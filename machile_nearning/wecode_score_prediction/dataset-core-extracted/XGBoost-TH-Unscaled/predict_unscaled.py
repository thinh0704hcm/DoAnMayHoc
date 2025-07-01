# Submission ID :           273649
import os
import pandas as pd
import joblib
import xgboost as xgb

# Configuration
FEATURES_FILE = "../dataset-core-extracted/student_results.csv"
MODEL_PATH = "th_student_score_predictor_optuna_xgb_cuda.pkl"
PREDICTION_OUTPUT = "th_score_predictions_unscaled.csv"

def predict_qt_scores():
    # Load student results
    student_results = pd.read_csv(FEATURES_FILE)
    
    # Identify students without QT scores
    unscored_students = student_results[student_results['th_score'].isna()]
    print(f"Found {len(unscored_students)} students without TH scores")
    
    # Select relevant features
    features = [
        'wecode_score_scaled',           # Student Wecode score
        'class_avg_th',           # Class average TH
        'hard_solved',     # Hard problems solved (scaled)
        'n_class_problems',
        'submit_count',
        'error_rate',
        'avg_attempt'
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
    results['predicted_th_score'] = predictions
    
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
        predictions = predict_qt_scores()