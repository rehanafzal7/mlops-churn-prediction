import sys
sys.path.append('src')

from data_processing import DataProcessor
from utils import load_params

# Load parameters
params = load_params("params.yaml")
print("Loaded parameters:", params)

# Initialize data processor
processor = DataProcessor(
    data_path="data/telco_churn.csv",
    test_size=params['data_processing']['test_size'],
    random_state=params['data_processing']['random_state']
)

# Load data
processor.load_data()

# Preprocess data
X_train, X_test, y_train, y_test = processor.preprocess()

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Features: {processor.get_feature_names()}")