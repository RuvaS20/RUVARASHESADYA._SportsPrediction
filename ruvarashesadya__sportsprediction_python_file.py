# -*- coding: utf-8 -*-
"""RUVARASHESADYA._SportsPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ySlZOyQ6--SekHqzVsFP-KTI49dtskez
"""

import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/My Drive/male_players (legacy).csv') #df with Columns: 110 entries

df.info()

"""## **Data preprocessing**

### **Dropping Columns with too many Null Values**
"""

def preprocess_data(df):
    # Dropping columns with more than 1/3 NA's
    for column in df.columns:
        na_count = df[column].isna().sum()
        if na_count > (len(df) / 3):
            df.drop(column, axis='columns', inplace=True)

    # Dropping the first 7 columns
    df = df.iloc[:, 7:]

    # Dropping specific features
    columns_to_drop = [
        'league_name', 'league_level', 'club_team_id', 'club_name',
        'club_position', 'club_jersey_number', 'club_joined_date',
        'club_contract_valid_until_year', 'nationality_id', 'nationality_name',
        'real_face', 'player_positions', 'dob', 'work_rate', 'body_type'
    ]
    df.drop(columns=columns_to_drop, axis='columns', inplace=True)

    # Dropping the last 28 columns
    df = df.iloc[:, :-28]

    # Handling 'preferred_foot'
    preferred_foot = df['preferred_foot']
    df.drop('preferred_foot', axis=1, inplace=True)

    # Dropping columns with less than 0.5 correlation with overall score
    corr_matrix = df.corr()
    sorted_correlations = corr_matrix['overall'].sort_values(ascending=False)
    columns_to_drop = sorted_correlations[sorted_correlations < 0.5].index.tolist()
    df.drop(columns=columns_to_drop, inplace=True)

    # Adding 'preferred_foot' back
    df['preferred_foot'] = preferred_foot

    return df

df = preprocess_data(df)

df.info()

"""### **Imputing the data**"""

from sklearn.impute import SimpleImputer
import pandas as pd

def impute_missing_values(df):
    # Splitting numeric and non-numeric data
    numeric_cols = df.select_dtypes(include=['number']).columns
    nonnumeric_cols = df.select_dtypes(exclude=['number']).columns

    # Impute na's that are numeric
    imputer = SimpleImputer(strategy="median")
    imputed_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)

    # Impute na's that are non-numeric
    imputer_str = SimpleImputer(strategy="most_frequent")
    imputed_nonnumeric = pd.DataFrame(imputer_str.fit_transform(df[nonnumeric_cols]), columns=nonnumeric_cols)

    # Concatenate the imputed numeric and nonnumeric
    imputed_df = pd.concat([imputed_numeric, imputed_nonnumeric], axis=1)

    # Return the imputed DataFrame
    return pd.DataFrame(imputed_df, columns=df.columns)

df = impute_missing_values(df)

"""### **Label encoding the data**"""

from numpy import array
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle as pkl

def encode_preferred_foot(df):
    # Create a LabelEncoder object
    label_encoder = LabelEncoder()

    # Extract 'preferred_foot' values and encode them
    values = np.array(df['preferred_foot'])
    integer_encoded = label_encoder.fit_transform(values)

    # Drop the original 'preferred_foot' column
    df.drop('preferred_foot', axis=1, inplace=True)

    # Add the integer encoded 'preferred_foot' column back to the DataFrame
    df['preferred_foot'] = integer_encoded

    with open('preferred_foot_encoder.pkl', 'wb') as f:
        pkl.dump(label_encoder, f)

    return df

df = encode_preferred_foot(df)

"""### **Standard Scaling**"""

from sklearn.preprocessing import StandardScaler

def prepare_features_and_target(df):
    # Selecting features and target
    X = df.drop(['overall'], axis=1)
    y = df['overall']

    # Standard Scaling
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)

    return X_scaled, y

X, y = prepare_features_and_target(df)

"""## **Training models**


"""

from sklearn.model_selection import train_test_split

# Splitting data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_and_evaluate_models(Xtrain, Ytrain, Xtest, Ytest):
    models = {
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10,  # Limit tree depth
            min_samples_split=5,  # Require more samples to split a node
            min_samples_leaf=2  # Require more samples in leaf nodes
        ),
        'XGBRegressor': XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            max_depth=6,  # Limit tree depth
            learning_rate=0.01,  # Lower learning rate
            n_estimators=1000,  # Increase number of trees
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1,  # L2 regularization
            subsample=0.8,  # Use 80% of data per tree
            colsample_bytree=0.8  # Use 80% of features per tree
        ),
        'GradientBoostingRegressor': GradientBoostingRegressor(
            random_state=42,
            n_estimators=100,
            max_depth=5,  # Limit tree depth
            learning_rate=0.1,
            subsample=0.8,  # Use 80% of data per tree
            min_samples_split=5,  # Require more samples to split a node
            min_samples_leaf=2  # Require more samples in leaf nodes
        )
    }

    results = {}

    for name, model in models.items():
        # Train the model
        model.fit(Xtrain, Ytrain)

        # Make predictions
        y_pred = model.predict(Xtest)

        # Calculate metrics
        mse = mean_squared_error(Ytest, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(Ytest, y_pred)
        r2 = r2_score(Ytest, y_pred)

        # Store results
        results[name] = {
            'model': model,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        # Print results
        print(f"\nResults for {name}:")
        print(f"Root Mean Square Error: {rmse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"R2 Score: {r2}")

    return results

results = train_and_evaluate_models(Xtrain, Ytrain, Xtest, Ytest)

"""## **Grid Search with Cross Validation**

### **Gradient Boosting Regressor**
"""

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle as pkl
import os
import numpy as np

# Setting up cross-validation and model parameters
cv = KFold(n_splits=3, shuffle=True, random_state=42)
gbr = GradientBoostingRegressor(random_state=42)
PARAMETERS_gb = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "n_estimators": [100, 500, 1000],
    "max_features": ['sqrt', 'log2', None]
}

# Performing grid search with cross-validation
model_gs = GridSearchCV(gbr, param_grid=PARAMETERS_gb, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
model_gs.fit(Xtrain, Ytrain)

# Saving the model
model_filename = 'GradientBoostingRegressor_GridSearch.pkl'
try:
    with open(model_filename, 'wb') as file:
        pkl.dump(model_gs, file)
    print(f"Model saved successfully as {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")

# Making predictions
y_pred = model_gs.predict(Xtest)

# Printing regression metrics
print(model_gs.best_estimator_.__class__.__name__)
print("Best parameters found: ", model_gs.best_params_)
print("Mean Squared Error:", mean_squared_error(Ytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Ytest, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(Ytest, y_pred))
print("R^2 Score:", r2_score(Ytest, y_pred))

# Verifying if the file was created
if os.path.exists(model_filename):
    print(f"File {model_filename} exists.")
    print(f"File size: {os.path.getsize(model_filename)} bytes")
else:
    print(f"File {model_filename} does not exist.")

"""### **XGB Regressor**"""

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle as pkl
import os
import numpy as np

# Setting up cross-validation and model parameters
cv = KFold(n_splits=3, shuffle=True, random_state=42)
xgbr = XGBRegressor(random_state=42)
PARAMETERS_xgb = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
    "min_child_weight": [1, 3, 5],
    "n_estimators": [100, 500, 1000],
    "gamma": [0, 0.1, 0.2],
}

# Performing grid search with cross-validation
model_gs = GridSearchCV(xgbr, param_grid=PARAMETERS_xgb, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
model_gs.fit(Xtrain, Ytrain)

# Saving the model
model_filename = 'XGBRegressor_GridSearch.pkl'
try:
    with open(model_filename, 'wb') as file:
        pkl.dump(model_gs, file)
    print(f"Model saved successfully as {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")

# Making predictions
y_pred = model_gs.predict(Xtest)

# Printing regression metrics
print(model_gs.best_estimator_.__class__.__name__)
print("Best parameters found: ", model_gs.best_params_)
print("Mean Squared Error:", mean_squared_error(Ytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Ytest, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(Ytest, y_pred))
print("R^2 Score:", r2_score(Ytest, y_pred))

# Verifying if the file was created
if os.path.exists(model_filename):
    print(f"File {model_filename} exists.")
    print(f"File size: {os.path.getsize(model_filename)} bytes")
else:
    print(f"File {model_filename} does not exist.")

"""### **RandomForest Regressor**"""

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle as pkl
import os
import numpy as np

# Setting up cross-validation and model parameters
cv = KFold(n_splits=3, shuffle=True, random_state=42)
rfr = RandomForestRegressor(random_state=42)
PARAMETERS_rf = {
    "max_depth": [5, 10, None],
    "n_estimators": [100, 300, 500],
    "max_features": ["sqrt", "log2", None],
}

# Performing grid search with cross-validation
model_gs = GridSearchCV(rfr, param_grid=PARAMETERS_rf, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
model_gs.fit(Xtrain, Ytrain)

# Saving the model
model_filename = 'RandomForestRegressor_GridSearch.pkl'
try:
    with open(model_filename, 'wb') as file:
        pkl.dump(model_gs, file)
    print(f"Model saved successfully as {model_filename}")
except Exception as e:
    print(f"Error saving the model: {e}")

# Making predictions
y_pred = model_gs.predict(Xtest)

# Printing regression metrics
print(model_gs.best_estimator_.__class__.__name__)
print("Best parameters found: ", model_gs.best_params_)
print("Mean Squared Error:", mean_squared_error(Ytest, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(Ytest, y_pred)))
print("Mean Absolute Error:", mean_absolute_error(Ytest, y_pred))
print("R^2 Score:", r2_score(Ytest, y_pred))

# Verifying if the file was created
if os.path.exists(model_filename):
    print(f"File {model_filename} exists.")
    print(f"File size: {os.path.getsize(model_filename)} bytes")
else:
    print(f"File {model_filename} does not exist.")

"""### **Training again on the best model**"""

from xgboost import XGBRegressor
import pickle as pkl
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Assuming Xtrain and Ytrain are already defined

# Create and fit the scaler
scaler = StandardScaler()
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xtest_scaled = scaler.transform(Xtest)

# Save the scaler
scaler_filename = 'scaler.pkl'
try:
    with open(scaler_filename, 'wb') as file:
        pkl.dump(scaler, file)
    print(f"Scaler saved successfully as {scaler_filename}")
except Exception as e:
    print(f"Error saving the scaler: {e}")

# Create the best XGBoost model with the optimal parameters
best_xgb_model = XGBRegressor(
    learning_rate=0.03,
    max_depth=6,
    n_estimators=1000,
    min_child_weight=1,
    random_state=42  # for reproducibility
)

# Fit the model on the scaled training data
best_xgb_model.fit(Xtrain_scaled, Ytrain)

# Save the model
best_model_filename = 'Best_XGBRegressor.pkl'
try:
    with open(best_model_filename, 'wb') as file:
        pkl.dump(best_xgb_model, file)
    print(f"Best model saved successfully as {best_model_filename}")
except Exception as e:
    print(f"Error saving the best model: {e}")

# Verify if the files were created
for filename in [scaler_filename, best_model_filename]:
    if os.path.exists(filename):
        print(f"File {filename} exists.")
        print(f"File size: {os.path.getsize(filename)} bytes")
    else:
        print(f"File {filename} does not exist.")

# Evaluate the best model on the scaled test set
y_pred_best = best_xgb_model.predict(Xtest_scaled)

# Calculate and print metrics
mse = mean_squared_error(Ytest, y_pred_best)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Ytest, y_pred_best)
r2 = r2_score(Ytest, y_pred_best)

print("\nBest XGBoost Model Performance:")
print(f"Root Mean Square Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

"""## **Testing with players_22 data**"""

import pandas as pd

# Load the new data
players = pd.read_csv('/content/drive/My Drive/players_22.csv') #df with Columns: 110 entries)

# Get the columns that are in df but not in new_data
columns_to_keep = [col for col in df.columns if col in players.columns]

# Keep only these columns in new_data
players = players[columns_to_keep]

players = impute_missing_values(players)

# Create a LabelEncoder object
label_encoder = LabelEncoder()

# Extract 'preferred_foot' values and encode them
values = np.array(players['preferred_foot'])
integer_encoded = label_encoder.fit_transform(values)

# Drop the original 'preferred_foot' column
players.drop('preferred_foot', axis=1, inplace=True)

# Add the integer encoded 'preferred_foot' column back to the DataFrame
players['preferred_foot'] = integer_encoded

# Prepare features (X) and target (y) for the new data
X = players.drop(['overall'], axis=1)
y = players['overall']

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

best_xgb_model.fit(Xtrain, Ytrain)

xgb_predictions = best_xgb_model.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, xgb_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, xgb_predictions)
r2 = r2_score(y_test, xgb_predictions)

print(f"\nResults for XGBoost:")
print(f"Root Mean Square Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"R2 Score: {r2}")