#Feature Reduction implementation using PCA on Random Forest Classifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv('/Heart_attack_risk_data.csv')

# Separate the target variable (y) and features (X)
X = data.drop(columns=['Heart Attack Risk'])
y = data['Heart Attack Risk']

# Identify categorical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']

# Apply one-hot encoding to categorical columns
categorical_transformer = OneHotEncoder(drop='first')
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Define a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Fit and transform the data
X = preprocessor.fit_transform(X)

# Creating the Random Forest Model
rf_model = RandomForestClassifier()

scaler = StandardScaler(with_mean=False)
X = scaler.fit_transform(X)

# Create the PCA model
svd = TruncatedSVD(n_components=100)

# Fit PCA on the standardized data
X_reduced = svd.fit_transform(X)
explained_variance = svd.explained_variance_ratio_.cumsum()
desired_variance = 0.833
num_components = next((i for i, explained_var in enumerate(explained_variance) if explained_var >= desired_variance), len(explained_variance))
print(num_components)
svd = TruncatedSVD(n_components=num_components)
X_reduced = svd.fit_transform(X)

# Training the model on the reduced dataset
X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(X_reduced, y, test_size=0.25, random_state=42)
rf_model.fit(X_train_reduced, y_train)
y_pred = rf_model.predict(X_test_reduced)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
explained_variance = svd.explained_variance_ratio_
print(f"Variance: {explained_variance}")