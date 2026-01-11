import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class DataProcessor:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.label_encoders = {}
    
    def load_data(self):
        """Loads the raw data from the specified path."""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully from {self.data_path} with {len(self.df)} rows.")
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.data_path}")
            self.df = pd.DataFrame()  # Return empty DataFrame on error
    
    def preprocess(self):
        """Performs feature engineering and basic cleaning, and splits data."""
        if self.df.empty:
            print("No data to preprocess. Please load data first.")
            return None, None, None, None
        
        # Drop customerID column if it exists, as it's not a feature
        if 'customerID' in self.df.columns:
            self.df = self.df.drop('customerID', axis=1)
        
        # Convert 'TotalCharges' to numeric, coercing errors to NaN
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        # Drop rows with NaN values that resulted from coercion or original NaNs
        self.df.dropna(inplace=True)
        
        # Convert 'No' to 0 and 'Yes' to 1 for 'Churn' target variable
        self.df['Churn'] = self.df['Churn'].map({'No': 0, 'Yes': 1})
        
        # Handle categorical features using Label Encoding
        for column in self.df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.label_encoders[column] = le  # Store encoder for inverse transform if needed
        
        # Define features (X) and target (y)
        X = self.df.drop('Churn', axis=1)
        y = self.df['Churn']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Data preprocessed and split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self):
        """Returns the list of feature names after preprocessing."""
        if self.df is not None and not self.df.empty:
            return self.df.drop('Churn', axis=1).columns.tolist()
        return []