from src.data_preprocessing import *
from src.train import *
from src.evaluate import *
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = "data/raw/House_Price.csv"
MODEL_PATH = "models/linear_regression.pkl"

df = load_data(DATA_PATH)
df = clean_data(df)

X, y = split_features_target(df, "price")
X_scaled, scaler = scale_features(X)

model, X_test, y_test = train_model(X_scaled, y)
rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
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


# ==============================
# 1. Actual vs Predicted Plot
# ==============================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.savefig("actual_vs_predicted.png")
plt.show()

# ==============================
# 2. Residual Plot
# ==============================
residuals = y_test - y_pred

plt.figure()
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0)
plt.savefig("residual_plot.png")
plt.show()

# ==============================
# 3. Error Distribution
# ==============================
plt.figure()
plt.hist(residuals, bins=30)
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.savefig("error_distribution.png")
plt.show()

# print("RMSE:", rmse)
# print("R2 Score:", r2)