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
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer 
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, ConfusionMatrixDisplay, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE

#%%
data = pd.read_csv("Data/analysis_ready.csv")

target = "ANXIOUS_TYPE"
columns_to_drop = [
    target,
    "Responder", 
    "ATYPICAL_TYPE", 
    "MELANCHOLIC_TYPE",
    "subjID", 
    "Group", 
    "sBCog45S_6",
    "sBNeg45S_6",
    "sBSoc45S_6",
    "sERQreap_6",
    "sERQsupp_6",
    "sAnx42_6",
    "sDepr42_6",
    "sStres42_6",
    "HDRS17_6",
    "sQIDS_tot_6",
    "SOFAS_RAT_6" 
    ]
X = data.loc[data[target].notna()].sample(frac=1)
X = X[X[target] != "2"]
y = X.loc[:, target].replace({"Y": 1, "N": 0})
X = X.drop(columns=columns_to_drop)
#%%

info_df = pd.DataFrame(columns=["pipeline", "n_pca", "num_features", "cv_scores", "mean_score", "std_score"])
sampler = RandomUnderSampler(random_state=42)
SimpleImputer.get_feature_names_out = (lambda self, names=None:
                                       self.feature_names_in_)

for npca in tqdm([0,3,5,7,9]):
    for n_features in [2,4,6,8,10]:
    
        # Classifier
        clf = KNeighborsClassifier()
        
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool']).columns
        

        # define steps to be done on numeric features
        numeric_preprocessing = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()) 
                ]
        )
        categorical_preprocessing = OneHotEncoder(handle_unknown="ignore")
        
        if npca == 0:
            preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_preprocessing, numeric_features),
                            ("cat", categorical_preprocessing, categorical_features),
                    ]
            )        
        else:
            snips_features = [x for x in X.columns if x[:2] == "rs"]
            snips_pca = PCA(n_components=npca)
            snips_transformer = Pipeline(
                steps = [
                    ("snips_imputer", SimpleImputer(strategy="mean")),
                    ("snips_scaler", StandardScaler()),
                    ("snips_pca", snips_pca)
                ]
            )
            
            preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_preprocessing, numeric_features),
                            ("cat", categorical_preprocessing, categorical_features),
                            ("snips", snips_transformer, snips_features)
                ]
            )
            
        sfs = SequentialFeatureSelector(
            clf,
            n_features_to_select=n_features,
            direction="forward",
            scoring="f1",
            cv=3,
            n_jobs=2
        )

        
        full_pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                #("sampler", sampler), 
                ("sfs", sfs),
                ("clf", clf),
            ]
        )
        
        results = cross_validate(full_pipe, X, y, cv=5, scoring="f1", return_estimator=True)        
     
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
        
    
print(info_df.sort_values(by="mean_score", ascending=False))         



# %%
