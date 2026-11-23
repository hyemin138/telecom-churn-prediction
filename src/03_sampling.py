"""불균형 처리 비교: 원본 / SMOTE / SMOTE-ENN (RF 기준, 학습 데이터 3-fold CV)

샘플링은 Pipeline 안에서 학습 폴드에만 적용
실행: python src/03_sampling.py
"""
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from data import RESULTS, SEED, load_train_test, smote_enn

OUT = RESULTS / "03_sampling"

# 기법 자체 비교라 기본값 사용, 최종 설정은 data.make_sampler()
SAMPLERS = {
    "None": "passthrough",
    "SMOTE": SMOTE(random_state=SEED),
    "SMOTE-ENN": SMOTEENN(random_state=SEED),
}
SCORING = ["accuracy", "f1", "recall", "roc_auc"]


def compare_samplers(X_train, y_train):
    """샘플링 기법별 CV 평균 지표"""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for name, sampler in SAMPLERS.items():
        pipe = Pipeline([("sampler", sampler), ("model", RandomForestClassifier(random_state=SEED))])
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=SCORING, n_jobs=-1)
        rows.append({"sampler": name, **{m: round(scores[f"test_{m}"].mean(), 4) for m in SCORING}})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X_train, _, y_train, _ = load_train_test()

    result = compare_samplers(X_train, y_train)
    result.to_csv(OUT / "sampler_comparison.csv", index=False)
    print(result.to_string(index=False))

    # 최종 설정의 클래스 비율 변화
    _, y_res = smote_enn(X_train, y_train)
    ratio = pd.DataFrame({"before": y_train.value_counts(), "after": y_res.value_counts()})
    ratio.to_csv(OUT / "class_ratio.csv")
    print(ratio)


if __name__ == "__main__":
    main()
