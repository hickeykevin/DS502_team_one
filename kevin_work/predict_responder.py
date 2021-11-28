#%%
from pathlib import Path
from typing_extensions import final
import pandas as pd
import numpy as np
from scipy.sparse.construct import rand
import wandb
import seaborn as sns
sns.set_theme()

from sklearn.model_selection import cross_validate, cross_val_score
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier, RUSBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

#wandb.init(name='DS502_Final', 
           #project='DS502 Supervised Learning')

#%%
def read_data(p: str):
    data_file = Path(p)
    data = pd.read_csv(data_file)
    return data

data = read_data("/mnt/c/Users/kevin/Documents/wpi_course_materials/DS502/DS502_team_one/Data/data_almost_analysis_ready.csv")

# %%
#a.	Model: Group = clinical assessments at baseline + demographics + sELSTOT + ERP (Exclude Responder and _6 clinical assessment variables)
clinical_measures = [
    "sBCog45S",
    "sBNeg45S",
    "sBSoc45S",
    "sERQreap",
    "sERQsupp",
    "sAnx42",
    "sDepr42",
    "sStres42",
    "HDRS17",
    "sQIDS_tot",
    "SOFAS_RAT"
]

demographics = [
    "sex",
    "Age",
    "Years_Education",
    "sBMI",
    "MDD_DUR",
    "TREATMENT",
]

sELSTOT = ["sELSTOT"]

erp_measures = [
    "std_N1_amp_min_pub_Fz",
    "std_N1_amp_min_pub_Cz",
    "trg_N1_amp_min_pub_Fz",
    "trg_N1_amp_min_pub_Cz",
    "std_P300_amp_max_Fz",
    "std_P300_amp_max_Cz",
    "std_P300_amp_max_Pz",
    "trg_P300_amp_max_Cz",
    "trg_P300_amp_max_Fz",
    "trg_P300_amp_max_Pz",
]

#%%
target = "Responder"
groups_to_use = [ 
    #sELSTOT,
    #demographics, 
    #erp_measures, 
    clinical_measures
     ]
final_columns = [target]
for c in groups_to_use:
    final_columns.extend(c)

X = data.loc[:, final_columns]
X = X.loc[X[target].notna()]

y = X.loc[:, target].replace({"N":0, "Y":1})
X = X.drop(columns=[target])
print(f"Using these features: \n{[a for a in X.columns]}")
print(f"Target feature: {target}")
print(f"X size: {X.shape}")
print(f"y size: {y.shape}")

clf = RUSBoostClassifier(random_state=42)
#clf = LogisticRegression()
encoder = OneHotEncoder(handle_unknown="ignore")
imputer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean", 
    #add_indicator=True
    )
scoring = "f1"
sampler = RandomUnderSampler(random_state=42)
selector = SequentialFeatureSelector(clf, n_features_to_select=1, cv=3)

#a.	Model: Group = clinical assessments at baseline + demographics + sELSTOT + ERP (Exclude Responder and _6 clinical assessment variables)


#%%

# determine categorical and numerical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'bool']).columns

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()) 
        ]
)

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
full_pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor), 
        #("sampler", sampler),
        #("selector", selector),
        ("clf", clf),
    ]
)

# Cross validation scores
results = cross_validate(full_pipe, X, y, cv=5, scoring=scoring, return_estimator=True)
print(f"{len(results['test_score'])}-fold {scoring} scores: {results['test_score']}")
print(f"Mean of {len(results['test_score'])}-fold {scoring} scores: {np.mean(results['test_score'])}")


#%%
feature_importance_df = pd.DataFrame(index=X.columns)

for i, c in enumerate(results["estimator"]):
    feature_importance_df[f"fold-{str(i)}"] = c["clf"].feature_importances_
feature_importance_df.plot(kind="bar", ylabel="Feature Importance")

# %%
