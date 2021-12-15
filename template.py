###########################################################################################
# Template/Explantion of files in directory
# This file serves as a walkthrough of the code that is found in the four directories
# Each file carries a similar logic and structure of experiments
# We will explain the logic with a working example
###########################################################################################

#%%
# Imports 
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
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_selection import SequentialFeatureSelector
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

# Read data in to pandas dataframe
data = pd.read_csv("Data/analysis_ready.csv")

# Define the target feature 
target = "Responder"

# Sepearte the data from the target;
# Drop a instances where the target is missing
# If predicting an MDD subtype, will also drop instances == 2;
# This indicates the subject was part of the control 
X = data.loc[data[target].notna()].sample(frac=1)  #Shuffle the instances randomly
y = X.loc[:, target].replace({"N":0, "Y":1})
X = X.drop(columns=[target])

# Instantiate an empty dataframe to house results from experiments below
info_df = pd.DataFrame(columns=["pipeline", "n_pca", "penalty", "num_features", "cv_scores", "mean_score", "std_score"])

# Loop through parameters to experiment with
for penalty in tqdm(["l1", "l2"]):
    
    # Number of pca to use on SNIPS features
    for npca in [0,3,5,7,9]:
        
        # Classifier
        clf = LogisticRegression(penalty=penalty, solver="liblinear", random_state=42, class_weight="balanced")
        
        # Define the numeric features and categorical features
        # Will have custom transformations done to each
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
        
        # For experiment with no pca on SNIPS features
        if npca == 0:
            preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_preprocessing, numeric_features),
                            ("cat", categorical_preprocessing, categorical_features),
                    ]
            )
        # For experiment using pca on SNIPS features        
        else:
            # Define the SNIPS features
            snips_features = [x for x in X.columns if x[:2] == "rs"]
            snips_pca = PCA(n_components=npca)
            # Create custom pipeline for those pca on SNIPS features
            snips_transformer = Pipeline(
                steps = [
                    ("snips_imputer", SimpleImputer(strategy="mean")),
                    ("snips_scaler", StandardScaler()),
                    ("snips_pca", snips_pca)
                ]
            )
            
            # Combine all preprocessing steps into one sklearn ColumnTransformer object
            # Will correctly apply transformations in order
            preprocessor = ColumnTransformer(
                        transformers=[
                            ("num", numeric_preprocessing, numeric_features),
                            ("cat", categorical_preprocessing, categorical_features),
                            ("snips", snips_transformer, snips_features)
                ]
            )
        
        # When using Recursive Feature Elimination Selection method,
        # define the minimum number of features to use
        min_features_to_select = 2  # Minimum number of features to consider
        
        # Instantiate the RFE method, which will choose best number of features 
        # By cross validation
        rfecv = RFECV(
            estimator=clf,
            step=2,
            cv=3,
            scoring="f1",
            min_features_to_select=min_features_to_select,
        )

        # Instatiate the full pipeline of column transformers, RFE, and final classifier
        full_pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                #("sampler", sampler), 
                ("rfe", rfecv),
                ("clf", clf),
            ]
        )
        
        # Use 5 fold cross validation to estimate model's performance
        results = cross_validate(full_pipe, X, y, cv=5, scoring="f1", return_estimator=True)
        
        # Utility to plot the performance of number of features selected by RFECV
        if "rfe" in full_pipe.named_steps.keys():
            plt.figure()
            plt.title(f"RFECV Logistic Regression Penalty={penalty.upper()} PCA={npca}")
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
        
        # Store all metrics from the cross validation run to a series 
        # Which will be appended to the dataframe we created 
        info = [
                full_pipe,
                npca,
                penalty,
                n_features,
                results['test_score'],
                np.mean(results['test_score']),
                np.std(results['test_score']),
                ]
        info = pd.Series(info, index=info_df.columns)
        info_df = info_df.append(info, ignore_index=True)

# Print out the sorted results, with best configurations being the first row
print(info_df.sort_values(by="mean_scores", ascending=False))       




