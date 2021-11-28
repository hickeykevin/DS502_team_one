#%%
from pathlib import Path
from typing_extensions import final
import pandas as pd
import numpy as np
from scipy.sparse.construct import rand
import wandb
import seaborn as sns
sns.set_theme()

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, LeaveOneOut
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier
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
    #"MDD_DUR",
    #"TREATMENT",
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

target = "Group"
groups_to_use = [[clinical_measures[10]]]
final_columns = ["Group"]
for c in groups_to_use:
    final_columns.extend(c)

X = data.loc[:, final_columns]
y = X.loc[:, target].replace({"Control":0, "MDD":1})
X = X.drop(columns=["Group"])
print(f"Using these features: \n{X.columns}")

#clf = BalancedRandomForestClassifier(random_state=42, max_depth=2)
clf = LogisticRegression(random_state=42)
encoder = OneHotEncoder(handle_unknown="ignore", )
imputer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean", 
    #add_indicator=True
    )
scoring = "f1"
sampler = RandomOverSampler(random_state=42)

#wandb.log(np.mean(scores))
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
        ("sampler", sampler),
        ("clf", clf)]
)

# Cross validation scores
results = cross_validate(full_pipe, X, y, cv=3, scoring=scoring, return_estimator=True)
print(f"{len(results['test_score'])}-fold {scoring} scores: {results['test_score']}")


#%%
feature_importance_df = pd.DataFrame(index=X.columns)

for i, c in enumerate(results["estimator"]):
    feature_importance_df[f"fold-{str(i)}"] = c["clf"].feature_importances_
feature_importance_df.plot(kind="bar", ylabel="Feature Importance")

# %%
