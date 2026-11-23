"""데이터·전처리 확인. 실행: python -m pytest -q"""
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import COUNTY_PATH, FIRST_DROP, LEAKAGE, LOW_IMPORTANCE, TARGET, load, load_train_test  # noqa: E402

needs_county = pytest.mark.skipif(
    not COUNTY_PATH.exists() and importlib.util.find_spec("pgeocode") is None,
    reason="data/zip_county.csv 또는 pgeocode 필요",
)


def test_raw_data():
    """7,043행 x 50열, 이탈 비율 26.5%"""
    df = load()
    assert df.shape == (7043, 50)
    assert round(df[TARGET].eq("Yes").mean(), 3) == 0.265


def test_drop_columns_exist():
    """제거 대상 변수가 모두 원본에 존재"""
    cols = set(load().columns)
    assert set(FIRST_DROP + LOW_IMPORTANCE) <= cols
    assert len(cols) - len(FIRST_DROP) - len(LOW_IMPORTANCE) + 1 == 30  # County 추가, 타깃 포함


@needs_county
def test_split():
    """누수 변수 제거, 결측 없음, 학습 5,634행 / 평가 1,409행"""
    X_train, X_test, y_train, y_test = load_train_test()
    assert (len(X_train), len(X_test)) == (5634, 1409)
    assert not any(c.startswith(tuple(LEAKAGE)) for c in X_train.columns)
    assert not X_train.isna().any().any()
    assert abs(y_train.mean() - y_test.mean()) < 0.01
