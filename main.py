from src.data_preprocessing import *
from src.train import *
from src.evaluate import *

DATA_PATH = "data/raw/House_Price.csv"
MODEL_PATH = "models/linear_regression.pkl"

df = load_data(DATA_PATH)
df = clean_data(df)

X, y = split_features_target(df, "price")
X_scaled, scaler = scale_features(X)

model, X_test, y_test = train_model(X_scaled, y)
rmse, r2 = evaluate_model(model, X_test, y_test)

save_model(model, MODEL_PATH)

print("Model Trained Successfully")
print("RMSE:", rmse)
print("R2 Score:", r2)