import sys
sys.path.append('src')

from data_processing import DataProcessor
from model_training import ModelTrainer
from utils import load_params

def main():
    # 1. Load parameters
    params = load_params("params.yaml")
    print("=" * 50)
    print("STARTING MLOPS PIPELINE")
    print("=" * 50)
    
    # 2. Process data
    print("\n[1/3] Processing data...")
    processor = DataProcessor(
        data_path="data/telco_churn.csv",
        test_size=params['data_processing']['test_size'],
        random_state=params['data_processing']['random_state']
    )
    
    processor.load_data()
    X_train, X_test, y_train, y_test = processor.preprocess()
    
    print(f"✓ Data processed: {X_train.shape[0]} training, {X_test.shape[0]} testing samples")
    print(f"✓ Features: {len(processor.get_feature_names())}")
    
    # 3. Train model with MLflow tracking
    print("\n[2/3] Training model with MLflow...")
    trainer = ModelTrainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        params=params['model_training']
    )
    
    model = trainer.train_and_log()
    
    # 4. Summary
    print("\n[3/3] Pipeline complete!")
    print("=" * 50)
    print("✓ Data processed and split")
    print("✓ Model trained and evaluated")
    print("✓ Experiment logged to MLflow")
    print("=" * 50)
    print("\nTo view your experiment:")
    print("1. Run: mlflow ui")
    print("2. Open: http://localhost:5000")
    print("=" * 50)

if __name__ == "__main__":
    main()