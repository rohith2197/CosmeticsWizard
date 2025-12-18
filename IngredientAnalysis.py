import pandas as pd
import numpy as np
import joblib
import shap
import re

# Loads the previously trained models to analyze ingredients
model = joblib.load("rating_model_no_price.joblib")
feature_cols = joblib.load("model_features_no_price.joblib")

# Loads the dataset of all the ingreidents and their staitstics
products = pd.read_csv("/Users/rohithpallamreddy/Documents/CosmeticsClassification/cosmetics.csv")
ingredients = pd.read_csv("ingredient_stats_enriched.csv")

# Removes unwanted and uncessary information from the dataset when loading it
bad_ingredients = {"no info", "visit the dior boutique", "visit the sephora collection boutique"}

def clean_ingredients(ingredient_str):
    if pd.isna(ingredient_str):
        return []
    ingredient_str = re.sub(r"\(.*?\)", "", ingredient_str)
    ingredient_str = ingredient_str.lower()
    ingredient_str = re.sub(r"[^a-z,\s]", " ", ingredient_str)
    ingredient_str = re.sub(r"\s+", " ", ingredient_str).strip()
    ing_list = [i.strip() for i in ingredient_str.split(",") if i.strip()]
    # remove unwanted ingredients
    return [i for i in ing_list if i not in bad_ingredients]

products["Ingredient_List"] = products["Ingredients"].apply(clean_ingredients)

# Creates a feature matrix of each ingredients
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
        "Target_Rank": row["Rank"],
        "Ingredient_List": row["Ingredient_List"]
    })

model_df = pd.DataFrame(feature_rows)
X = model_df[feature_cols]

# Uses shap to analyze which ingredients have hte best and worst impacts
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Maps shap to the original ingredients
ingredient_features = ["Avg_Ingredient_Mean_Rank","Avg_Ingredient_Std_Rank"]

for feat_idx, feat_name in enumerate(feature_cols):
    if feat_name not in ingredient_features:
        continue

    contrib_dict = {}
    for i, ing_list in enumerate(model_df["Ingredient_List"]):
        shap_val = shap_values[i, feat_idx]
        if len(ing_list) == 0:
            continue
        per_ing_contrib = shap_val / len(ing_list)
        for ing in ing_list:
            contrib_dict[ing] = contrib_dict.get(ing, []) + [per_ing_contrib]

    # Compute average contribution per ingredient
    avg_contrib = {k: np.mean(v) for k,v in contrib_dict.items()}
    # Top 5 positive and negative
    top_pos = sorted(avg_contrib.items(), key=lambda x: x[1], reverse=True)[:5]
    top_neg = sorted(avg_contrib.items(), key=lambda x: x[1])[:5]

    print(f"\n===== Feature: {feat_name} =====")
    print("Top 5 Ingredients that Increase this feature → Rating:")
    for ing, val in top_pos:
        print(f"{ing}: {val:.4f}")
    print("Top 5 Ingredients that Decrease this feature → Rating:")
    for ing, val in top_neg:
        print(f"{ing}: {val:.4f}")
