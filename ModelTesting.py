import pandas as pd
import numpy as np
import joblib
import shap
import re

# Loads the models
model = joblib.load("rating_model_no_price.joblib")
feature_cols = joblib.load("model_features_no_price.joblib")
ingredients_df = pd.read_csv("ingredient_stats_enriched.csv")

bad_ingredients = {
    "no info",
    "visit the dior boutique",
    "visit the sephora collection boutique"
}

# Removes the unwanted ingredeints from the feature vectors
def clean_ingredients(ingredient_str):
    ingredient_str = ingredient_str.lower()
    ingredient_str = re.sub(r"\(.*?\)", "", ingredient_str)
    ingredient_str = re.sub(r"[^a-z,\s]", " ", ingredient_str)
    ingredient_str = re.sub(r"\s+", " ", ingredient_str).strip()
    return [
        i.strip() for i in ingredient_str.split(",")
        if i.strip() and i not in bad_ingredients
    ]

# Builds the feature vectors
def build_features(ingredient_list, label, dry=0, oily=0, normal=0, combination=0):
    ing_df = ingredients_df[
        (ingredients_df["Ingredient"].isin(ingredient_list)) &
        (ingredients_df["Label"] == label)
        ]

    if ing_df.empty:
        return None, None

    feature_row = {
        "Avg_Ingredient_Mean_Rank": ing_df["Mean_Rank"].mean(),
        "Avg_Ingredient_Std_Rank": ing_df["Std_Rank"].mean(),
        "Max_Ingredient_Rank": ing_df["Mean_Rank"].max(),
        "Min_Ingredient_Rank": ing_df["Mean_Rank"].min(),
        "Ingredient_Count": len(ing_df),
        "Dry": dry,
        "Oily": oily,
        "Normal": normal,
        "Combination": combination
    }

    return pd.DataFrame([feature_row])[feature_cols], ing_df

# Predicts the score and best/worst ingredients using shap
def predict_and_explain(
        ingredients_str,
        label,
        dry=0, oily=0, normal=0, combination=0,
        top_k=5
):
    # returns a dictionary
    ingredient_list = clean_ingredients(ingredients_str)
    X, ing_df = build_features(
        ingredient_list, label, dry, oily, normal, combination
    )

    if X is None:
        # Return empty result if no known ingredients
        return {"rating": None, "best": [], "worst": []}

    # Prediction
    pred = float(model.predict(X)[0])

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]

    # Ingredient-based features only
    ingredient_features = [
        "Avg_Ingredient_Mean_Rank",
        "Avg_Ingredient_Std_Rank",
        "Max_Ingredient_Rank",
        "Min_Ingredient_Rank",
        "Ingredient_Count"
    ]

    ingredient_scores = {}

    for feat in ingredient_features:
        idx = feature_cols.index(feat)
        shap_val = shap_vals[idx]

        for ing in ing_df["Ingredient"]:
            ingredient_scores[ing] = ingredient_scores.get(ing, 0) + shap_val / len(ing_df)

    # Rank ingredients
    ranked = sorted(ingredient_scores.items(), key=lambda x: x[1], reverse=True)

    best_ings = [ing for ing, val in ranked[:top_k]]
    worst_ings = [ing for ing, val in ranked[-top_k:]]

    # Return as dictionary for GUI
    return {
        "rating": round(pred, 2),
        "best": best_ings,
        "worst": worst_ings
    }


