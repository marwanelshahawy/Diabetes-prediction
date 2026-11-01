import logging

import pandas as pd
from joblib import dump
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from diabetes import config, features

logger = logging.getLogger(__name__)


def load_and_clean():
    df = pd.read_csv(config.RAW_DATA)
    logger.info(f"Data Read Successfully shape of Data >> {df.shape}")
    df = df.drop_duplicates().reset_index(drop= True)
    logger.info(f"Duplicated Deleted shape of data >> {df.shape}")
    df = df.drop(df[df["gender"]=="Other"].index)
    logger.info(f"gender = \"other\" deleted shape of data >> {df.shape}")

    return df

def split_data(df):
    X = df.drop(columns = config.TARGET)
    y = df[config.TARGET]
    X_dev,X_test,y_dev,y_test = train_test_split(X,y,test_size = .15 ,stratify = y, random_state = config.RANDOM_STATE)

    logger.info(f"Shape of 1)X_dev >> {X_dev.shape} 2)y_dev > {y_dev.shape} 3)X_test >> {X_test.shape} 4)y_test >> {y_test.shape}")

    return X_dev,y_dev,X_test,y_test

def evaluate(pipe,X_test,y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= config.THRESHOLD).astype(int)

    metrics = {
        "pr_auc": average_precision_score(y_test, proba),
        "roc_auc": roc_auc_score(y_test, proba),
        "recall": recall_score(y_test, pred),
        "precision": precision_score(y_test, pred),
    }

    for name, value in metrics.items():
        logger.info("Test %s: %.4f", name, value)

    return metrics

def main():
    df = load_and_clean()
    X_dev,y_dev,X_test,y_test = split_data(df)
    pipe = features.build_pipeline()
    pipe.fit(X_dev,y_dev)
    evaluate(pipe,X_test,y_test)
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    dump({"pipeline": pipe, "threshold": config.THRESHOLD}, config.MODEL_PATH)
    logger.info(f"Model saved to >> {config.MODEL_PATH}")



if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    main()




