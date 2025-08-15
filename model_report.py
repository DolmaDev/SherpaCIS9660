# model_report.py
import pandas as pd

def get_sales_model_evaluation():
    """Returns the evaluation metrics from your sales prediction model"""
    
    results_data = [
        {"Model": "XGBoost", "MAE": 9751.661140, "RMSE": 13451.634403, "R2": 0.619792},
        {"Model": "Random Forest", "MAE": 9018.936588, "RMSE": 14309.815048, "R2": 0.569732},
        {"Model": "Gradient Boosting", "MAE": 9957.269577, "RMSE": 14616.195663, "R2": 0.551110},
        {"Model": "Linear Regression", "MAE": 11684.297226, "RMSE": 15946.692287, "R2": 0.465667},
        {"Model": "Lasso Regression", "MAE": 11684.580960, "RMSE": 15947.122444, "R2": 0.465638},
        {"Model": "Ridge Regression", "MAE": 11691.135338, "RMSE": 15961.575244, "R2": 0.464669}
    ]
    
    return {
        "model_comparison": results_data,
        "best_model": {
            "name": "XGBoost",
            "hyperparameters": {
                "learning_rate": 0.05, 
                "max_depth": 3, 
                "n_estimators": 50
            },
            "performance": {
                "mae": 9751.66, 
                "rmse": 13451.63, 
                "r2": 0.6198
            }
        },
        "training_details": {
            "total_samples": 360,
            "train_samples": 288, 
            "test_samples": 72,
            "features": [
                'day_of_year_sin', 'day_of_year_cos', 'day_of_week', 
                'month', 'prev_day_rev', 'avg_rev_last_7_days', 'prev_day_rev_14'
            ]
        }
    }

def get_sales_feature_importance():
    """Returns feature importance from sales prediction XGBoost model"""
    return {
        'prev_day_rev': 0.35,
        'avg_rev_last_7_days': 0.25,
        'prev_day_rev_14': 0.20,
        'day_of_year_sin': 0.08,
        'day_of_year_cos': 0.06,
        'month': 0.04,
        'day_of_week': 0.02
    }

def get_nut_classification_evaluation():
    """Returns the evaluation metrics from your nut classification model"""
    
    return {
        "model_details": {
            "name": "MobileNetV2",
            "architecture": "Transfer Learning (Keras)",
            "format": "Keras (.keras)",
            "deployment": "Direct Keras model (no TFLite conversion)",
            "parameters": {
                "total_params": 2422339,
                "trainable_params": 164355,
                "frozen_params": 2257984
            },
            "training": {
                "epochs_trained": 11,
                "early_stopping": True,
                "optimizer": "Adam",
                "learning_rate": 2e-04,
                "batch_size": 32
            }
        },
        "performance": {
            "test_accuracy": 1.0,
            "test_samples": 63,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0
        },
        "class_performance": [
            {"Class": "Almonds", "Samples": 22, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0},
            {"Class": "Cashews", "Samples": 20, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0},
            {"Class": "Walnuts", "Samples": 21, "Precision": 1.0, "Recall": 1.0, "F1-Score": 1.0}
        ],
        "dataset_info": {
            "total_images": 309,
            "train_images": 246,
            "test_images": 63,
            "classes": 3,
            "class_names": ["Almonds", "Cashews", "Walnuts"]
        }
    }

def get_classification_report_text():
    """Returns the classification report as formatted text"""
    return """              precision    recall  f1-score   support

     Almonds       1.00      1.00      1.00        22
     Cashews       1.00      1.00      1.00        20
     Walnuts       1.00      1.00      1.00        21

    accuracy                           1.00        63
   macro avg       1.00      1.00      1.00        63
weighted avg       1.00      1.00      1.00        63"""

def get_training_history():
    """Returns training history data for nut classification"""
    return {
        "epochs": list(range(1, 12)),
        "accuracy": [0.9963, 1.0, 1.0, 0.9688, 0.9891, 1.0, 0.9950, 1.0, 0.9886, 1.0, 0.9963],
        "val_accuracy": [1.0] * 11,
        "loss": [0.0217, 0.0047, 0.0117, 0.1405, 0.0487, 0.0070, 0.0279, 0.0032, 0.0425, 0.0120, 0.0152],
        "val_loss": [0.0036, 0.0036, 0.0034, 0.0033, 0.0039, 0.0040, 0.0045, 0.0045, 0.0045, 0.0045, 0.0043]
    }
