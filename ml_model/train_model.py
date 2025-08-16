import pandas as pd
#from sklearn.base import r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import joblib
import os

from sklearn.datasets import fetch_california_housing

#Load the data set
# df = pd.read_csv('C:/Users/obame/OneDrive/Desktop/ml_model/data/mock_data.csv')

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df)

#previewing data examples
# print(df.head(2))

#seleceting relevent features
features = ['MedInc',  'HouseAge',  'AveRooms',  'AveBedrms', 'Population',  'AveOccup']
target = ['MedHouseVal']

#Drop values with missing values
df = df.dropna(subset = features + target)

X = df[features]
y = df[target]

#Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Train XGBOOST Regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators = 100, learning_rate = 0.1)
model.fit(X_train, y_train)

#Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#Save Model and Scaler 
os.makedirs("ml_model", exist_ok=True)
model.save_model("ml_model/model.json")
joblib.dump(scaler, "ml_model/scaler.plk")

print("Model and Scaler saved!")