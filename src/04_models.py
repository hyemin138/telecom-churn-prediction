"""모델 비교: RF, DRF, XGBoost, CatBoost, CatBoost(튜닝 파라미터)

SMOTE-ENN 적용한 학습 데이터로 학습, 평가 데이터(1,409행)로 비교
실행: python src/04_models.py
"""
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier

from data import FINAL_PARAMS, RESULTS, SEED, load_train_test, smote_enn

OUT = RESULTS / "04_models"
MODEL_PATH = OUT / "catboost_final.cbm"  # 06_shap.py에서 사용


class DoubleRandomForest:
    """트리별 부트스트랩 + 분기별 변수 무작위 선택 앙상블 (팀 구현 버전)"""

    def __init__(self, n_estimators=500, max_features="sqrt", max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for b in range(self.n_estimators):
            Xb, yb = resample(X, y)
            tree = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth,
                                          random_state=self.random_state + b if self.random_state else None)
            self.trees.append(tree.fit(Xb, yb))
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def build_models():
    """비교 모델 5개"""
    return {
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Double Random Forest": DoubleRandomForest(n_estimators=500, max_depth=10, random_state=SEED),
        "XGBoost": XGBClassifier(random_state=SEED, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(random_state=SEED, eval_metric="Logloss", verbose=0),
        "CatBoost (Tuned)": CatBoostClassifier(**FINAL_PARAMS, random_state=SEED,
                                               eval_metric="Logloss", verbose=0),
    }


def evaluate(model, X_test, y_test):
    """정확도, F1, Recall, ROC-AUC"""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {"Accuracy": accuracy_score(y_test, pred), "F1": f1_score(y_test, pred),
            "Recall": recall_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, proba)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_train_test()
    X_res, y_res = smote_enn(X_train, y_train)

    models = build_models()
    rows = []
    for name, model in models.items():
        model.fit(X_res, y_res)
        rows.append({"Model": name, **evaluate(model, X_test, y_test)})

    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "model_comparison.csv", index=False)
    print(result.to_string(index=False))

    # 최종 모델 저장
    models["CatBoost (Tuned)"].save_model(str(MODEL_PATH))
    final = result.iloc[-1].drop("Model").to_dict()
    (OUT / "metrics.json").write_text(json.dumps({"params": FINAL_PARAMS, "test": final}, indent=2))


if __name__ == "__main__":
    main()
