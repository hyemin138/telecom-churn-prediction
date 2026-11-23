# archive

팀 Colab 원본 노트북 (참고용, 정리본은 `src/`)

| 파일 | 원본 | 내용 |
|---|---|---|
| `original_notebooks/competition_pipeline.py` | `1111.ipynb` | 대회 제출 버전, 33개 변수(하위 4개 미제거). 대회 제출 수치(F1 0.7271)의 출처 |
| `original_notebooks/pipeline_29_features.py` | `통계자료분석 전처리 & eda` | 하위 4개 제거한 29개 변수 버전 (F1 0.7182) |
| `original_notebooks/feature_importance.py` | `통자분 제귀제곱 결과` | CatBoost 변수 중요도 순위 |

## 정리하면서 바뀐 점

- 경로: Colab(`/content/...`) -> 레포 기준 상대 경로
- County: 행마다 pgeocode 조회 -> 고유 Zip Code만 조회 후 `data/zip_county.csv`로 저장 (결과 동일)
- 샘플링 비교: 1111의 RF 3-fold CV 방식 그대로, 29개 변수 기준으로 실행
- Grid Search: SMOTE-ENN을 Pipeline 안으로 옮겨 검증 폴드에 합성 데이터가 섞이지 않도록 수정
- 변수 개수별 F1 곡선: 원본 코드 미보관, 같은 방식으로 재구성
- 기본 파라미터 CatBoost: 보고서 비교표에 있으나 원본 코드 미보관, 기본값으로 추가

## 제외한 노트북

| 노트북 | 제외 이유 |
|---|---|
| `통자분 모델링` | 샘플링 기법 Grid Search 실험 (SMOTE·ENN·SMOTE-ENN 파라미터 탐색), 최종 흐름과 별개 |
| 파생변수 실험 노트북 | Family_Size 파생변수 검토, 최종 모델 미사용 |
| `통계자료분석 EDA` | EDA 그래프 원본 (`src/01_eda.py`로 정리) |
| `통계자료분석의 사본` | 다른 버전 데이터(Kaggle 기본판 21열) |
