# import comet_ml in the top of your file
from comet_ml import Experiment
import logging


def get_logger():
    """
        credits to: https://www.kaggle.com/ogrellier/user-level-lightgbm-lb-1-4480
    """
    FORMAT = "[%(levelname)s]%(asctime)s:%(name)s:%(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger()

# create an experiment with your api key
logger.info("Experiment setup started...")
exp = Experiment(
    api_key="SBq1c69Q9lQJOF99NmGXKnzEd",
    project_name="sklearn-demos-commet",
    workspace="sachinkohli373",
)


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

random_state = 42


logger.info("Load data...")
cancer = load_breast_cancer()
print("cancer.keys(): {}".format(cancer.keys()))
print("Shape of cancer data: {}\n".format(cancer.data.shape))
print(
    "Sample counts per class:\n{}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    )
)
print("\nFeature names:\n{}".format(cancer.feature_names))

logger.info("Create train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=random_state
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression()

param_grid = {"C": [0.002, 0.02, 0.2, 2, 5, 10, 20, 50, 100]}

clf = GridSearchCV(logreg, param_grid=param_grid, cv=10, n_jobs=-1)

logger.info("Fit log reg with grid search...")
clf.fit(X_train_scaled, y_train)

logger.info("Get prediction...")
y_pred = clf.predict(X_test_scaled)

print("\nResults\nConfusion matrix \n {}".format(confusion_matrix(y_test, y_pred)))

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("F1 score is {:6.3f}".format(f1))
print("Precision score is {:6.3f}".format(precision))
print("Recall score is {:6.3f}".format(recall))

# these will be logged to your sklearn-demos project on Comet.ml
params = {
    "random_state": random_state,
    "model_type": "logreg",
    "scaler": "standard scaler",
    "param_grid": str(param_grid),
    "stratify": True,
}

metrics = {"f1": f1, "recall": recall, "precision": precision}

logger.info("Logging params, meta info to commet.ml ...")
exp.log_dataset_hash(X_train_scaled)
exp.log_parameters(params)
exp.log_metrics(metrics)

logger.info("Done...")
