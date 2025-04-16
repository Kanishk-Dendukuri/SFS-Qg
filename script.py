import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from catboost import CatBoostRegressor
data = pd.read_csv('Combined.csv')

# Separate features and outputs
X = data[[f'Feature_{i+1}' for i in range(50)]]
y = data[[f'Output_{i+1}' for i in range(5)]]  # Assuming 5 outputs

# Perform the 80-10-10 split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Confirm split sizes
print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)

model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred, multioutput='raw_values')
print("Validation MSE for each output:", val_mse)


xgb_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
xgb_model.fit(X_train, y_train)
xgb_val_pred = xgb_model.predict(X_val)
xgb_val_mse = mean_squared_error(y_val, xgb_val_pred, multioutput='raw_values')

print("XGBoost Validation MSE:", xgb_val_mse)

# CatBoost model
catboost_model = MultiOutputRegressor(CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='MultiRMSE', random_seed=42, verbose=0))
catboost_model.fit(X_train, y_train)
catboost_val_pred = catboost_model.predict(X_val)
catboost_val_mse = mean_squared_error(y_val, catboost_val_pred, multioutput='raw_values')

print("CatBoost Validation MSE:", catboost_val_mse)
