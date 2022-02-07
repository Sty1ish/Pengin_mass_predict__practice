# Pengin_mass_predict__practice
데이콘에 존재하는 펭귄 몸무게 예측 모델링 연습 결과입니다.
원문 : https://dacon.io/competitions/official/235862/overview/description

***
## 전처리 과정

데이터는 특별한 형태를 갖지 않았으며, 한쪽으로 치우친 결과값을 가지지 않고 고르게 분포되어있었습니다.
결측치는 명목형변수(성별), 수치형변수(delat 13, 15)변수에 존재하였으며 이를 아래와 같이 처리하였습니다.

수치형 변수 --> 평균값 처리
명목형 변수 --> KNN 결측값 처리

***
## 사용한 scaler

OneHotEncoder와, standardScaler를 통하여 스케일링 작업을 진행하였습니다.

***
## 사용한 모델

- 선형회귀 + 확률적 경사하강법(선형회귀)

- XGBoost

- Catboost

- randomforest

GridSearch가 가능한 항목들에 대해, 그리드서치 또한 실시하였습니다.

***
## 주석
실행 결과는 "펭귄 분석 프로젝트.ipynb"에 존재하며, EDA결과는 dataset 폴더안 R마크다운 형태로 존재합니다.
