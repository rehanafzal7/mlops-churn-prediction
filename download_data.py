import pandas as pd
import os

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Download dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
df.to_csv('data/telco_churn.csv', index=False)
print(f"Dataset downloaded successfully! Shape: {df.shape}")
print(f"Saved to: {os.path.abspath('data/telco_churn.csv')}")