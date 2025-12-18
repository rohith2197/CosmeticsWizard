import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import re
import matplotlib.pyplot as plt

# -Loads hte products CSV and the ingredienst CSM
products = pd.read_csv("/Users/rohithpallamreddy/Documents/CosmeticsClassification/cosmetics.csv")
ingredients = pd.read_csv("ingredient_stats_enriched.csv")

# Remove products with actual rating 0
products = products[products["Rank"] > 0].reset_index(drop=True)

# Remove unwanted ingredients
bad_ingredients = {"no info", "visit the dior boutique", "visit the sephora collection boutique"}

# Cleans the ingredients
def clean_ingredients(ingredient_str):
    if pd.isna(ingredient_str):
        return []
    ingredient_str = re.sub(r"\(.*?\)", "", ingredient_str)
    ingredient_str = ingredient_str.lower()
    ingredient_str = re.sub(r"[^a-z,\s]", " ", ingredient_str)
    ingredient_str = re.sub(r"\s+", " ", ingredient_str).strip()
    ingredients_list = [i.strip() for i in ingredient_str.split(",") if i.strip()]
    # remove unwanted ingredients
    return [i for i in ingredients_list if i not in bad_ingredients]

products["Ingredient_List"] = products["Ingredients"].apply(clean_ingredients)

# Builds feature vectors to train the model on
feature_rows = []

for _, row in products.iterrows():
    ing_df = ingredients[
        (ingredients["Ingredient"].isin(row["Ingredient_List"])) &
        (ingredients["Label"] == row["Label"])
        ]

    if ing_df.empty:
        continue

    feature_rows.append({
        "Avg_Ingredient_Mean_Rank": ing_df["Mean_Rank"].mean(),
        "Avg_Ingredient_Std_Rank": ing_df["Std_Rank"].mean(),
        "Max_Ingredient_Rank": ing_df["Mean_Rank"].max(),
        "Min_Ingredient_Rank": ing_df["Mean_Rank"].min(),
        "Ingredient_Count": len(ing_df),
        "Dry": row["Dry"],
        "Oily": row["Oily"],
        "Normal": row["Normal"],
        "Combination": row["Combination"],
        "Target_Rank": row["Rank"]
    })

model_df = pd.DataFrame(feature_rows)

X = model_df.drop(columns=["Target_Rank"])
y = model_df["Target_Rank"]

# Trains the model and splits up the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    loss="quantile",
    alpha=0.5,
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

weights = np.abs(y_train - y_train.mean()) + 0.5

model.fit(X_train, y_train, sample_weight=weights)

# Predicts the test dataset
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds) ** 0.5

joblib.dump(model, "rating_model_no_price.joblib")
joblib.dump(list(X.columns), "model_features_no_price.joblib")

print("Model saved.")
print("RMSE:", rmse)

# Creates visualizations of product accuracy versus actual information
plt.figure()
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([0, 5], [0, 5], color='red')
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Predicted vs Actual Ratings")
plt.tight_layout()
plt.show()

errors = preds - y_test

plt.figure()
plt.hist(errors, bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.show()
