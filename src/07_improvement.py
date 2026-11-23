"""개선 실험: 범주형 변수 처리 방식 비교 (대회 이후 추가)

한계 1, 5 보완. 같은 분할·같은 CatBoost 파라미터로 비교
- 대회 방식: 원-핫 + SMOTE-ENN + CatBoost
- 개선 1: 원-핫 없이 SMOTE-NC(범주형 전용) + CatBoost 범주형 직접 입력
- 개선 2: 샘플링 없이 CatBoost 범주형 직접 입력 + 클래스 가중치
실행: python src/07_improvement.py
"""
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from data import FINAL_PARAMS, RESULTS, SEED, categorical_cols, load_train_test, smote_enn

OUT = RESULTS / "07_improvement"


def catboost(**extra):
    """최종 파라미터 CatBoost"""
    return CatBoostClassifier(**FINAL_PARAMS, random_state=SEED, eval_metric="Logloss", verbose=0, **extra)


def evaluate(model, X_test, y_test):
    """정확도, F1, Recall, ROC-AUC"""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {"Accuracy": accuracy_score(y_test, pred), "F1": f1_score(y_test, pred),
            "Recall": recall_score(y_test, pred), "ROC-AUC": roc_auc_score(y_test, proba)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    # 대회 방식
    X_train, X_test, y_train, y_test = load_train_test()
    X_res, y_res = smote_enn(X_train, y_train)
    model = catboost().fit(X_res, y_res)
    rows.append({"Method": "원-핫 + SMOTE-ENN (대회)", **evaluate(model, X_test, y_test)})

    # 범주형 그대로 사용
    X_train, X_test, y_train, y_test = load_train_test(onehot=False)
    cats = categorical_cols(X_train)

    # 개선 1
    sampler = SMOTENC(categorical_features=[X_train.columns.get_loc(c) for c in cats],
                      sampling_strategy=0.7, k_neighbors=3, random_state=SEED)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    model = catboost(cat_features=cats).fit(X_res, y_res)
    rows.append({"Method": "SMOTE-NC + CatBoost 범주형", **evaluate(model, X_test, y_test)})

    # 개선 2
    model = catboost(cat_features=cats, auto_class_weights="Balanced").fit(X_train, y_train)
    rows.append({"Method": "클래스 가중치 + CatBoost 범주형", **evaluate(model, X_test, y_test)})

    result = pd.DataFrame(rows).round(4)
    result.to_csv(OUT / "improvement_comparison.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
