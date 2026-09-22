from src.data_preprocessing import *
from src.train import *
from src.evaluate import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "data/raw/House_Price.csv"
#DATA_PATH = "data/raw/Housing.csv"

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

plt.figure(figsize=(8,6))
plt.style.use('ggplot')
plt.scatter(y_test, y_pred)

# Perfect prediction line (IMPORTANT)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.savefig("actual_vs_predicted.png")
plt.show()
# ==============================
# 2. Residual Plot
# ==============================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))
plt.style.use('ggplot')
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red')  # zero error line
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.savefig("residual_plot.png")
plt.show()
# ==============================
# 3. Error Distribution
# ==============================
plt.figure(figsize=(8,6))
plt.style.use('ggplot')
plt.hist(residuals, bins=30)
plt.axvline(x=0, color='red')  # zero error center
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.savefig("error_distribution.png")
plt.show()
# ==============================
# 4. Room Number vs Price
# ==============================
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['room_num'], y=df['price'])
plt.title("Number of Rooms vs House Price")
plt.xlabel("Number of Rooms")
plt.ylabel("House Price")
plt.show()
# # ==============================
# # 5. Air Quality vs Price
# # ==============================
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['air_qual'], y=df['price'])
plt.title("Air Quality vs House Price")
plt.xlabel("Air Quality")
plt.ylabel("House Price")
plt.show()
# # # ==============================
# # # 6. Price Distribution Histogram
# # # ==============================
# plt.figure(figsize=(8,5))
# sns.histplot(df['price'], bins=30, kde=True)
# plt.title("Price Distribution")
# plt.xlabel("House Price")
# plt.ylabel("Frequency")
# plt.show()
# # # ==============================
# # # 7. Predicted vs Actual Prices (with Residuals)      
# # # ==============================
# plt.figure(figsize=(8,6))           
# sns.scatterplot(x=y_test, y=y_pred, hue=residuals, palette='coolwarm', edgecolor='k')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
# plt.xlabel("Actual Prices") 
# plt.ylabel("Predicted Prices")
# plt.title("Predicted vs Actual Prices (Colored by Residuals)")
# plt.legend(title='Residuals', loc='upper left')
# plt.show()
###
# # ==============================
# # 6. Feature Correlation Heatmap
# # ==============================
# plt.figure(figsize=(12,8))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.title("Feature Correlation Heatmap")
# plt.show()

# coefficients = model.coef_
# feature_names = X.columns   
# coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
# coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
# plt.figure(figsize=(10,6))
# sns.barplot(x='Coefficient', y='Feature', data=coef_df, palette='viridis')
# plt.title("Feature Importance (Coefficients)")  
# plt.xlabel("Coefficient Value")
# plt.ylabel("Feature")   
# plt.show()