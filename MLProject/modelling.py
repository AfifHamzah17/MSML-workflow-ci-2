"""
Modelling.py - MLflow Implementation for CI/CD
Kriteria 3: Workflow CI dengan MLflow Project
Enhanced with better exception handling and debugging features
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import argparse
import sys
import os
import traceback
from datetime import datetime

warnings.filterwarnings('ignore')

class MLflowModelTrainer:
    """Enhanced MLflow Model Trainer with better error handling"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start_time = datetime.now()
        self.data_loaded = False
        self.models_trained = {}
        
    def log_message(self, message, level="INFO"):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
        if self.verbose:
            print(f"[DEBUG] Process running for {datetime.now() - self.start_time}")

    def load_data(self):
        """Load preprocessed data with enhanced error handling"""
        self.log_message("Loading preprocessed data...")
        
        try:
            # Check if file exists
            data_file = 'automobile_clean.csv'
            if not os.path.exists(data_file):
                self.log_message(f"File {data_file} not found in current directory", "ERROR")
                self.log_message(f"Current directory: {os.getcwd()}", "DEBUG")
                self.log_message(f"Files in directory: {os.listdir('.')}", "DEBUG")
                
                # Try alternative file names
                alternative_files = ['automobile_clean.csv', 'automobile.csv', 'auto_clean.csv', 'data.csv']
                for alt_file in alternative_files:
                    if os.path.exists(alt_file):
                        self.log_message(f"Found alternative file: {alt_file}", "INFO")
                        data_file = alt_file
                        break
                else:
                    raise FileNotFoundError(f"No suitable data file found. Tried: {alternative_files}")
            
            # Load the dataset
            df = pd.read_csv(data_file)
            self.log_message(f"✅ Loaded {data_file}! Shape: {df.shape}", "SUCCESS")
            
            # Check if dataset is empty
            if df.empty:
                raise ValueError("Dataset is empty!")
            
            # Log basic info about the dataset
            if self.verbose:
                self.log_message(f"Dataset columns: {list(df.columns)}", "DEBUG")
                self.log_message(f"Dataset dtypes:\n{df.dtypes}", "DEBUG")
                self.log_message(f"Missing values:\n{df.isnull().sum()}", "DEBUG")
            
            self.data_loaded = True
            return df
            
        except Exception as e:
            self.log_message(f"Error loading data: {str(e)}", "ERROR")
            self.log_message(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            raise

    def clean_data(self, df):
        """Clean the data for modeling with better error handling"""
        self.log_message("Preparing data for modeling...")
        
        try:
            original_shape = df.shape
            
            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                raise ValueError("No numeric columns found in the dataset!")
            
            df_clean = df[numeric_cols].copy()
            
            # Remove columns with all NaN values
            df_clean = df_clean.dropna(axis=1, how='all')
            
            # Remove rows with any NaN values
            df_clean = df_clean.dropna(axis=0, how='any')
            
            if df_clean.empty:
                raise ValueError("Dataset is empty after cleaning!")
            
            self.log_message(f"✅ Data cleaned - Original: {original_shape}, Final: {df_clean.shape}", "SUCCESS")
            
            if self.verbose:
                self.log_message(f"Numeric columns: {numeric_cols}", "DEBUG")
                self.log_message(f"Cleaned columns: {list(df_clean.columns)}", "DEBUG")
            
            return df_clean
            
        except Exception as e:
            self.log_message(f"Error cleaning data: {str(e)}", "ERROR")
            raise

    def prepare_features(self, df):
        """Prepare features and target variable with enhanced error handling"""
        self.log_message("Preparing features and target...")
        
        try:
            # Possible target columns in order of preference
            possible_targets = ['mpg', 'price', 'highway-mpg', 'city-mpg', 'target']
            target_col = None
            
            # Find the best target column
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    self.log_message(f"✅ Using '{col}' as target variable", "SUCCESS")
                    break
            
            if target_col is None:
                available_cols = list(df.columns)
                self.log_message(f"Available columns: {available_cols}", "ERROR")
                
                # If no predefined target, use the last column as target
                if len(available_cols) > 1:
                    target_col = available_cols[-1]
                    self.log_message(f"⚠️ Using last column '{target_col}' as target", "WARNING")
                else:
                    raise ValueError("Insufficient columns for modeling!")
            
            # Separate features and target
            X = df.drop(target_col, axis=1)
            y = df[target_col]
            
            # Validate features and target
            if X.empty:
                raise ValueError("No features available after target separation!")
            
            if y.empty or y.isnull().all():
                raise ValueError("Target variable is empty or all NaN!")
            
            self.log_message(f"📈 Features shape: {X.shape}", "SUCCESS")
            self.log_message(f"🎯 Target shape: {y.shape}", "SUCCESS")
            
            if self.verbose:
                self.log_message(f"Feature columns: {list(X.columns)}", "DEBUG")
                self.log_message(f"Target statistics:\n{y.describe()}", "DEBUG")
            
            # Final validation - ensure all features are numeric
            non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                self.log_message(f"⚠️ Removing non-numeric columns: {non_numeric}", "WARNING")
                X = X.select_dtypes(include=[np.number])
                
                if X.empty:
                    raise ValueError("No numeric features available!")
            
            return X, y, target_col
            
        except Exception as e:
            self.log_message(f"Error preparing features: {str(e)}", "ERROR")
            raise

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets with validation"""
        self.log_message("Splitting data...")
        
        try:
            if len(X) < 10:
                raise ValueError(f"Dataset too small for splitting: {len(X)} samples")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                shuffle=True, stratify=None
            )
            
            self.log_message(f"🚂 Training set: {X_train.shape[0]} samples", "SUCCESS")
            self.log_message(f"🧪 Testing set: {X_test.shape[0]} samples", "SUCCESS")
            
            if self.verbose:
                self.log_message(f"Training target distribution:\n{pd.Series(y_train).describe()}", "DEBUG")
                self.log_message(f"Testing target distribution:\n{pd.Series(y_test).describe()}", "DEBUG")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.log_message(f"Error splitting data: {str(e)}", "ERROR")
            raise

    def evaluate_model(self, model, X_test, y_test, model_name="Unknown"):
        """Evaluate model performance with enhanced error handling"""
        try:
            self.log_message(f"Evaluating {model_name} model...")
            
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Validate metrics
            if np.isnan(mse) or np.isnan(mae) or np.isnan(r2):
                raise ValueError(f"Invalid metrics calculated for {model_name}")
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            }
            
            self.log_message(f"✅ {model_name} evaluation completed", "SUCCESS")
            
            return metrics, y_pred
            
        except Exception as e:
            self.log_message(f"Error evaluating {model_name}: {str(e)}", "ERROR")
            raise

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model with MLflow tracking and error handling"""
        self.log_message("Training Random Forest Model...")
        
        try:
            with mlflow.start_run(run_name="RandomForest_CI", nested=mlflow.active_run() is not None):
                # Enable MLflow autolog
                mlflow.sklearn.autolog()
                
                # Create and train model
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=10,
                    n_jobs=-1  # Use all available cores
                )
                
                # Train model
                self.log_message("Training Random Forest...")
                rf_model.fit(X_train, y_train)
                
                # Evaluate model
                metrics, y_pred = self.evaluate_model(rf_model, X_test, y_test, "Random Forest")
                
                # Log additional parameters
                mlflow.log_param("model_type", "RandomForestRegressor")
                mlflow.log_param("dataset_name", "automobile_dataset")
                mlflow.log_param("training_environment", "GitHub_Actions_CI")
                mlflow.log_param("features_count", X_train.shape[1])
                mlflow.log_param("training_samples", X_train.shape[0])
                
                # Print results
                self.log_message("📊 Random Forest Results:", "SUCCESS")
                for metric_name, metric_value in metrics.items():
                    self.log_message(f"   {metric_name.upper()}: {metric_value:.4f}")
                
                self.models_trained['RandomForest'] = {
                    'model': rf_model,
                    'metrics': metrics
                }
                
                return rf_model, metrics
                
        except Exception as e:
            self.log_message(f"Error training Random Forest: {str(e)}", "ERROR")
            self.log_message(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            raise

    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model with MLflow tracking and error handling"""
        self.log_message("Training Linear Regression Model...")
        
        try:
            with mlflow.start_run(run_name="LinearRegression_CI", nested=mlflow.active_run() is not None):
                # Enable MLflow autolog
                mlflow.sklearn.autolog()
                
                # Create and train model
                lr_model = LinearRegression()
                
                # Train model
                self.log_message("Training Linear Regression...")
                lr_model.fit(X_train, y_train)
                
                # Evaluate model
                metrics, y_pred = self.evaluate_model(lr_model, X_test, y_test, "Linear Regression")
                
                # Log additional parameters
                mlflow.log_param("model_type", "LinearRegression")
                mlflow.log_param("dataset_name", "automobile_dataset")
                mlflow.log_param("training_environment", "GitHub_Actions_CI")
                mlflow.log_param("features_count", X_train.shape[1])
                mlflow.log_param("training_samples", X_train.shape[0])
                
                # Print results
                self.log_message("📊 Linear Regression Results:", "SUCCESS")
                for metric_name, metric_value in metrics.items():
                    self.log_message(f"   {metric_name.upper()}: {metric_value:.4f}")
                
                self.models_trained['LinearRegression'] = {
                    'model': lr_model,
                    'metrics': metrics
                }
                
                return lr_model, metrics
                
        except Exception as e:
            self.log_message(f"Error training Linear Regression: {str(e)}", "ERROR")
            self.log_message(f"Full traceback:\n{traceback.format_exc()}", "ERROR")
            raise

    def save_training_summary(self, rf_metrics, lr_metrics, target_col):
        """Save comprehensive training summary"""
        try:
            best_model = "Random Forest" if rf_metrics['r2'] > lr_metrics['r2'] else "Linear Regression"
            
            summary_content = f"""MLflow Model Training Summary
{'=' * 40}
Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Training Duration: {datetime.now() - self.start_time}
Target Variable: {target_col}

Model Performance:
{'-' * 20}
Random Forest:
  - R² Score: {rf_metrics['r2']:.4f}
  - RMSE: {rf_metrics['rmse']:.4f}
  - MAE: {rf_metrics['mae']:.4f}
  - MSE: {rf_metrics['mse']:.4f}

Linear Regression:
  - R² Score: {lr_metrics['r2']:.4f}
  - RMSE: {lr_metrics['rmse']:.4f}
  - MAE: {lr_metrics['mae']:.4f}
  - MSE: {lr_metrics['mse']:.4f}

Best Model: {best_model}
Training Environment: GitHub Actions CI
MLflow Tracking: Enabled
"""
            
            with open('training_summary.txt', 'w') as f:
                f.write(summary_content)
            
            self.log_message("📄 Training summary saved successfully", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"Error saving training summary: {str(e)}", "ERROR")

def main():
    """Main function to run the modeling pipeline"""
    parser = argparse.ArgumentParser(description='Enhanced MLflow Model Training Pipeline')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state (default: 42)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Create trainer instance
    trainer = MLflowModelTrainer(verbose=args.verbose)
    
    trainer.log_message("🚀 Starting Enhanced MLflow Model Training Pipeline (CI/CD)")
    trainer.log_message("=" * 60)
    trainer.log_message(f"⚙️ Parameters: test_size={args.test_size}, random_state={args.random_state}, verbose={args.verbose}")
    
    try:
        # Set MLflow configuration
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Auto_Prediction_CI")
        trainer.log_message("✅ MLflow configuration set", "SUCCESS")
        
        # Load and prepare data
        df = trainer.load_data()
        df_clean = trainer.clean_data(df)
        X, y, target_col = trainer.prepare_features(df_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, args.test_size, args.random_state)
        
        # Train models
        trainer.log_message("🤖 Training Models with MLflow Tracking...")
        
        # Random Forest
        rf_model, rf_metrics = trainer.train_random_forest(X_train, X_test, y_train, y_test)
        
        # Linear Regression
        lr_model, lr_metrics = trainer.train_linear_regression(X_train, X_test, y_train, y_test)
        
        # Compare models
        trainer.log_message("🏆 Model Comparison:")
        trainer.log_message("=" * 30)
        trainer.log_message(f"Random Forest R²: {rf_metrics['r2']:.4f}")
        trainer.log_message(f"Linear Regression R²: {lr_metrics['r2']:.4f}")
        
        best_model = "Random Forest" if rf_metrics['r2'] > lr_metrics['r2'] else "Linear Regression"
        trainer.log_message(f"🥇 Best Model: {best_model}", "SUCCESS")
        
        # Save training summary
        trainer.save_training_summary(rf_metrics, lr_metrics, target_col)
        
        trainer.log_message("✅ Training completed successfully!", "SUCCESS")
        trainer.log_message("📊 MLflow artifacts saved to ./mlruns/")
        trainer.log_message("🤖 CI/CD Pipeline executed successfully!")
        
        # Final status
        total_time = datetime.now() - trainer.start_time
        trainer.log_message(f"⏱️ Total execution time: {total_time}", "SUCCESS")
        
    except Exception as e:
        trainer.log_message(f"❌ Pipeline failed: {str(e)}", "ERROR")
        trainer.log_message(f"Full error traceback:\n{traceback.format_exc()}", "ERROR")
        
        # Save error report
        try:
            with open('error_report.txt', 'w') as f:
                f.write(f"MLflow Pipeline Error Report\n")
                f.write(f"{'=' * 30}\n")
                f.write(f"Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Error: {str(e)}\n")
                f.write(f"Full Traceback:\n{traceback.format_exc()}\n")
            trainer.log_message("📝 Error report saved to error_report.txt", "INFO")
        except:
            pass
            
        sys.exit(1)

if __name__ == "__main__":
    main()