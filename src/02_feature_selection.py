"""2차 변수 선택: CatBoost 변수 중요도 순위 + 변수 개수별 F1

1) 1차 제거 후 33개 변수로 CatBoost 중요도 계산 (원본 노트북 설정)
2) 중요도 낮은 변수부터 하나씩 제거하며 RF 3-fold CV F1 비교
   (보고서 Fig.9 재구성, 원본 코드 미보관)
실행: python src/02_feature_selection.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data import RESULTS, SEED, categorical_cols, load, preprocess, split

OUT = RESULTS / "02_feature_selection"
MIN_FEATURES = 10


def catboost_importance(X_train, y_train):
    """범주형 그대로 넣은 CatBoost 변수 중요도 (내림차순)"""
    pool = Pool(X_train, y_train, cat_features=categorical_cols(X_train))
    model = CatBoostClassifier(iterations=800, depth=10, learning_rate=0.05, l2_leaf_reg=3,
                               random_state=SEED, verbose=0)
    model.fit(pool)
    imp = model.get_feature_importance(pool)
    return pd.Series(imp, index=X_train.columns).sort_values(ascending=False)


def f1_by_n_features(X_train, y_train, ranking):
    """하위 변수부터 제거하며 변수 개수별 CV F1 계산"""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    rows = []
    for n in range(len(ranking), MIN_FEATURES - 1, -1):
        X = pd.get_dummies(X_train[ranking.index[:n]], dtype=int)
        f1 = cross_val_score(RandomForestClassifier(random_state=SEED), X, y_train,
                             cv=cv, scoring="f1").mean()
        rows.append({"n_features": n, "f1": round(f1, 4)})
    return pd.DataFrame(rows)


def plot_curve(curve):
    """변수 개수별 F1 그래프"""
    best = curve.loc[curve["f1"].idxmax()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["n_features"], curve["f1"], marker="o", ms=3)
    ax.scatter(best["n_features"], best["f1"], s=80, facecolors="none", edgecolors="red",
               label=f"best: {int(best['n_features'])}")
    ax.set(title="F1 by Number of Features", xlabel="Number of features", ylabel="CV F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "f1_by_n_features.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = preprocess(load(), drop_low_importance=False)
    X_train, _, y_train, _ = split(df, onehot=False)

    ranking = catboost_importance(X_train, y_train)
    ranking.round(3).to_csv(OUT / "catboost_importance.csv", header=["importance"])

    curve = f1_by_n_features(X_train, y_train, ranking)
    curve.to_csv(OUT / "f1_by_n_features.csv", index=False)
    plot_curve(curve)

    print("중요도 하위 4개:", ranking.index[-4:].tolist())
    print(curve.sort_values("f1", ascending=False).head(3).to_string(index=False))


if __name__ == "__main__":
    main()
