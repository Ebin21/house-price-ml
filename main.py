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
def interpret_results(rmse, r2):
    r2_percent = round(r2 * 100, 2)
    rmse_rounded = round(rmse, 2)

    interpretation = (
        f"The model explains approximately {r2_percent}% of the variation in house prices, "
        f"indicating a reasonably strong predictive performance.\n"
        f"On average, the model’s predictions deviate from actual prices by about {rmse_rounded} units "
        f"(based on the dataset’s scale)."
    )

    return interpretation



print("Model Trained Successfully")
print(interpret_results(rmse, r2))
# print("RMSE:", rmse)
# print("R2 Score:", r2)