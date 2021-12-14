#%%
from itertools import combinations, chain
from functools import reduce
import itertools
from pathlib import Path
from typing_extensions import final
from imblearn.base import SamplerMixin
from numpy.core.numeric import full
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
import wandb
import seaborn as sns
from tqdm import tqdm
sns.set_theme()
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

#%%
def read_data(p: str):
    data_file = Path(p)
    data = pd.read_csv(data_file)
    return data

data = read_data("/mnt/c/Users/kevin/Documents/wpi_course_materials/DS502/DS502_team_one/Data/data_almost_analysis_ready.csv")

# %%
target = "ATYPICAL_TYPE"
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

snips = [x for x in data.columns if x[0:2] == "rs"]

# %%
all_features = [sELSTOT, demographics, erp_measures, clinical_measures, snips]
all_features = list(chain(*all_features))

X = data.loc[data[target].notna()].sample(frac=1)
X = X[X[target] != "2"]
y = X.loc[:, target].replace({"Y": 1, "N": 0})
X = X.loc[:, all_features]
#%%

info_df = pd.DataFrame(columns=["pipeline", "n_pca", "num_features", "cv_scores", "mean_score", "std_score"])
sampler = RandomUnderSampler(random_state=42)
SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                       self.feature_names_in_)

for npca in tqdm([0,3,5,7,9]):
    
    # Classifier
    clf = GradientBoostingClassifier(random_state=42)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'bool']).columns
    

    # define steps to be done on numeric features
    numeric_preprocessing = SimpleImputer(strategy="mean")
    categorical_preprocessing = OneHotEncoder(handle_unknown="ignore")
    
    if npca == 0:
        preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_preprocessing, numeric_features),
                        ("cat", categorical_preprocessing, categorical_features),
                ]
        )        
    elif npca != 0:
        snips_pca = PCA(n_components=npca)
        snips_transformer = Pipeline(
            steps = [
                ("snips_imputer", SimpleImputer(strategy="mean")),
                ("snips_scaler", StandardScaler()),
                ("snips_pca", snips_pca)
            ]
        )
        
        snips_features = [x for x in X.columns if x[:2] == "rs"]
        preprocessor = ColumnTransformer(
                    transformers=[
                        ("num", numeric_preprocessing, numeric_features),
                        ("cat", categorical_preprocessing, categorical_features),
                        ("snips", snips_transformer, snips_features)
            ]
        )
        
    

    
    min_features_to_select = 2  # Minimum number of features to consider
    
    #cv = StratifiedKFold(3)
    rfecv = RFECV(
        estimator=clf,
        step=2,
        cv=3,
        scoring="f1",
        min_features_to_select=min_features_to_select,
    )

    
    full_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            #("sampler", sampler), 
            ("rfe", rfecv),
            ("clf", clf),
        ]
    )
    
    results = cross_validate(full_pipe, X, y, cv=5, scoring="f1", return_estimator=True)
    
    if "rfe" in full_pipe.named_steps.keys():
        plt.figure()
        plt.title(f"RFECV Random Forest PCA={npca}")
        plt.xlabel(f"Number of features selected \nOptimal number of features: { results['estimator'][0][1].n_features_}")
        plt.ylabel("Cross validation score (f1)")
        plt.plot(
            np.array([x*2 for x in range(1, results['estimator'][0][1].grid_scores_.shape[0]+1)]),
            results['estimator'][0][1].grid_scores_,
            label=["cv1", "cv2", "cv3"])
        plt.legend()
        n_features = results['estimator'][0].named_steps['rfe'].n_features_ 
    else:
        n_features = "all"
    
    
    info = [
            full_pipe,
            npca,
            n_features,
            results['test_score'],
            np.mean(results['test_score']),
            np.std(results['test_score']),
            ]
    info = pd.Series(info, index=info_df.columns)
    info_df = info_df.append(info, ignore_index=True)
    
    
#display(info_df.sort_values(by="mean_score", ascending=False))
print(info_df.sort_values(by="mean_scores", ascending=False))   

# %%
