import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load data
df = pd.read_csv("Housing.csv")

# --- Feature Engineering ---
df['price_per_sqft'] = df['price'] / df['area']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['bath_bed_ratio'] = df['bathrooms'] / df['bedrooms']
df['area_squared'] = df['area'] ** 2
df['area_bedrooms'] = df['area'] * df['bedrooms']
# New Feature Ideas
df['bed_bath_diff'] = df['bedrooms'] - df['bathrooms']
df['bed_area_ratio'] = df['bedrooms'] / df['area']
df['log_area'] = np.log(df['area'])


# Encode categorical variables
df_model = df.copy()
label_encoders = {}
categorical_columns = df_model.select_dtypes(include='object').columns

for col in categorical_columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Define target and features
y = df_model['price']
X = df_model.drop(columns=['price'])

# Add constant
X = sm.add_constant(X)

# --- Step 1: Drop high p-value features ---
def drop_high_p(X, y, threshold=0.05):
    while True:
        model = sm.OLS(y, X).fit()
        pvals = model.pvalues.drop('const', errors='ignore')
        max_p = pvals.max()
        if max_p > threshold:
            drop_col = pvals.idxmax()
            print(f"Dropping '{drop_col}' due to high p-value: {max_p:.4f}")
            X = X.drop(columns=[drop_col])
        else:
            break
    return X

X = drop_high_p(X, y)

# --- Step 2: Drop high VIF features ---
def drop_high_vif(X, threshold=3):
    while True:
        vif = pd.DataFrame()
        vif["Feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        max_vif = vif.loc[vif["Feature"] != 'const', "VIF"].max()
        if max_vif > threshold:
            drop_col = vif.sort_values("VIF", ascending=False).iloc[0]["Feature"]
            print(f"Dropping '{drop_col}' due to high VIF: {max_vif:.2f}")
            X = X.drop(columns=[drop_col])
        else:
            break
    return X

X = drop_high_vif(X)

# --- Final Model ---
final_model = sm.OLS(y, X).fit()
print("\nFinal Model Summary:")
print(final_model.summary())

# --- Calculate VIFs for final model ---
vif_df = pd.DataFrame()
vif_df["Feature"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factors (VIF):")
print(vif_df)