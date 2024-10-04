#Feature Selection implementation using RFE on Random Forest Classifier
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('/Heart_attack_risk_data.csv')

# Separate the target variable (y) and features (X)
X = data.drop(columns=['Heart Attack Risk'])  # Replace 'target_variable' with your target variable name.
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create the Random Forest model
rf_model = RandomForestClassifier()

# Create the RFE object
num_features_to_select = 10  # Adjust the number of features to select as needed.
rfe = RFE(estimator=rf_model, n_features_to_select=num_features_to_select)

# Fit RFE on the training data
rfe = rfe.fit(X_train, y_train)

# Get the selected features
selected_features = X_train[:, rfe.support_]

# Train the Random Forest model on the selected features
rf_model.fit(selected_features, y_train)

# Evaluate the model on the testing data
accuracy = rf_model.score(X_test[:, rfe.support_], y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Get the selected feature names or indices
selected_features_mask = rfe.support_

# Access the feature names or indices based on the mask
selected_feature_indices = np.where(selected_features_mask)[0]

print("\nSelected Feature Indices:")
print(selected_feature_indices)