# Cosmetics Wizard: Version 2

Cosmetics Wizard is a data science project designed to analyze and predict the effectiveness of skincare and cosmetic products based on their ingredient lists. By using machine learning, it identifies which specific ingredients contribute to a high rating and which ones might be dragging a formula down.

## What's new in Version 2

This version represents a significant shift from our initial prototype:
* **Framework Change**: We moved away from Kivy and rebuilt the interface using **CustomTkinter**. This allowed for a more modern, "Apple-style" UI that is easier to navigate on desktop.
* **Refined Analysis**: Instead of simple keyword matching, this version uses a **Gradient Boosting Regressor** to predict product ratings on a scale of 0.0 to 5.0.
* **Explainable AI**: We integrated **SHAP (SHapley Additive exPlanations)**. This means the app doesn't just give you a number; it mathematically determines which ingredients are the "Best" and "Worst" contributors to that specific prediction.

## The Data Science Process

### Data Cleaning
Cosmetic ingredient lists are notoriously messy. The project includes a custom cleaning pipeline that:
* Removes marketing jargon and "noise" (e.g., text inside parentheses).
* Standardizes ingredient names to lowercase.
* Filters out "bad data" like placeholder text (e.g., "visit the dior boutique").
* Cross-references input ingredients against an enriched dataset of known cosmetic components and their historical performance.

### Feature Engineering
The model doesn't just look at individual ingredients. It calculates:
* **Mean and Standard Deviation of Ingredient Ranks**: How well do these ingredients usually perform across other products?
* **Max/Min Rank**: Are there any "superstar" or "red-flag" ingredients in the mix?
* **Skin Type Compatibility**: Features are weighted differently depending on whether the user specifies Dry, Oily, Normal, or Combination skin.



## How to Run

This project is straightforward to set up for anyone interested in the intersection of beauty and data science.

### Prerequisites
You will need Python installed along with the following libraries:
```bash
pip install customtkinter pandas numpy scikit-learn joblib shap matplotlib
