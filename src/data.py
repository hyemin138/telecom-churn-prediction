"""공통: 데이터 로드, 전처리, 학습/평가 분할, SMOTE-ENN"""
from pathlib import Path

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco.csv"
COUNTY_PATH = ROOT / "data" / "zip_county.csv"
RESULTS = ROOT / "results"

TARGET = "Churn Label"
SEED = 42

# 1차 변수 제거
LEAKAGE = ["Churn Score", "Churn Category", "Churn Reason", "Customer Status"]  # 이탈 이후 정보
POST_RESULT = ["Satisfaction Score"]                                            # 사후 결과
ID_DUPLICATE = ["Customer ID", "Under 30", "Senior Citizen", "Dependents", "Referred a Friend"]
CONSTANT = ["Country", "State", "Quarter"]
HIGH_CARDINALITY = ["Zip Code", "Latitude", "Longitude", "City"]  # County로 대체
FIRST_DROP = LEAKAGE + POST_RESULT + ID_DUPLICATE + CONSTANT + HIGH_CARDINALITY

# 2차 변수 제거: 변수 중요도 하위 4개 (02_feature_selection.py)
LOW_IMPORTANCE = ["Unlimited Data", "Device Protection Plan", "Internet Service", "Phone Service"]

# 최종 CatBoost 파라미터 (04_models.py, 06_shap.py 공통)
FINAL_PARAMS = {"depth": 8, "iterations": 800, "l2_leaf_reg": 3, "learning_rate": 0.03}


def load(path=DATA_PATH):
    """IBM Telco 원본 불러오기 (7,043행 x 50열)"""
    return pd.read_csv(path)


def zip_to_county(zips):
    """Zip Code별 County 매핑 (첫 실행 시 pgeocode로 만들어 data/zip_county.csv에 저장)"""
    if COUNTY_PATH.exists():
        table = pd.read_csv(COUNTY_PATH)
        return dict(zip(table["Zip Code"], table["County"]))

    import pgeocode
    nomi = pgeocode.Nominatim("us")
    uniq = sorted(set(zips))
    county = nomi.query_postal_code([str(z) for z in uniq])["county_name"].tolist()
    table = pd.DataFrame({"Zip Code": uniq, "County": county})
    table.loc[table["Zip Code"] == 95250, "County"] = "Calaveras County"  # pgeocode 누락 (Sheep Ranch)
    table.to_csv(COUNTY_PATH, index=False)
    return dict(zip(table["Zip Code"], table["County"]))


def preprocess(df, drop_low_importance=True):
    """County 생성, 변수 제거, 결측값 처리"""
    df = df.copy()
    df["County"] = df["Zip Code"].map(zip_to_county(df["Zip Code"]))

    drop = FIRST_DROP + (LOW_IMPORTANCE if drop_low_importance else [])
    df = df.drop(columns=drop)

    # 미가입·미이용 고객의 결측은 별도 범주로 처리
    df["Offer"] = df["Offer"].fillna("-")
    df["Internet Type"] = df["Internet Type"].fillna("-")
    return df


def categorical_cols(X):
    """문자형(범주형) 변수 목록"""
    return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]


def split(df, onehot=True):
    """타깃 분리, 원-핫 인코딩, 8:2 층화 분할"""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    if onehot:
        X = pd.get_dummies(X, columns=categorical_cols(X), dtype=int)
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)


def load_train_test(drop_low_importance=True, onehot=True):
    """로드, 전처리 후 학습/평가 분할 (학습 5,634행 / 평가 1,409행)"""
    return split(preprocess(load(), drop_low_importance), onehot)


def make_sampler():
    """SMOTE-ENN 설정 (소수:다수 = 0.7까지 증강 후 경계 노이즈 제거)"""
    return SMOTEENN(
        smote=SMOTE(sampling_strategy=0.7, k_neighbors=3, random_state=SEED),
        enn=EditedNearestNeighbours(n_neighbors=3, kind_sel="mode"),
        random_state=SEED,
    )


def smote_enn(X_train, y_train):
    """학습 데이터에만 SMOTE-ENN 적용"""
    return make_sampler().fit_resample(X_train, y_train)
