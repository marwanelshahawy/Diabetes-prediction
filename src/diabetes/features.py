import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from diabetes import config


def add_bmi_missing(df):
    new_df = df.copy()
    new_df["bmi_missing"] = (np.isclose(new_df["bmi"] , config.BMI_IMPUTED_VALUE)).astype(int)
    return new_df

def build_preprocessor():
    preprocessor = ColumnTransformer([
        ("num",StandardScaler(),config.NUM_COLS),
        ("cat",OneHotEncoder(handle_unknown='ignore'),config.CAT_COLS),
        ("bin","passthrough",config.BIN_COLS)
    ])
    return preprocessor

def build_pipeline():
    bmi_missing = FunctionTransformer(add_bmi_missing,validate=False)
    pipe = Pipeline([
        ("add_bmi_missing",bmi_missing),
        ("prep",build_preprocessor()),
        ("clf",XGBClassifier(eval_metric="logloss",n_jobs=-1,random_state=config.RANDOM_STATE,**config.XGB_PARAMS))
    ])
    return pipe

