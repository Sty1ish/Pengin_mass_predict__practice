print('이거 작업 코드 아님.')
Pipeline ==> https://stickie.tistory.com/77
파이프라인은 여러 변환 단계를 정확한 순서대로 실행할 수 있도록 하는 것입니다. 
사이킷런은 연속된 변환을 순서대로 처리할 수 있도록 도와주는 Pipeline 클래스가 있습니다.

Pipeline은 연속된 단계를 나타내는 이름/추정기 쌍의 목록을 입력으로 받습니다. 
마지막 단계에는 변환기와 추정기를 모두 사용할 수 있고 그 외에는 모두 변환기여야 합니다
(즉 fit_transform() 메서드를 가지고 있어야 합니다). 이름은 무엇이든 상관없지만,
 이중 밑줄 문자(__)는 포함하지 않아야 합니다.

파이프라인의 fit() 메서드를 호출하면 모든 변환기의 fit_transform() 메서드를 
순서대로 호출하면서 한 단계의 출력을 다음 단계의 입력으로 전달합니다.
 마지막 단계에서는 fit()메서드만 호출합니다.

파이프라인 객체는 마지막 추정기와 동일한 메서드를 제공합니다. 
이 예에서는 마지막 추정기가 변환기 StandardScaler이므로 파이프라인이 데이터에 대해 모든 변환을 
순서대로 적용하는 transform() 메서드를 가지고 있습니다(또한 fit_transform() 메서드도 가지고 있습니다).

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.preprocessing import OnehotEncoder, CategoricalEncoder

# 숫자형 변수를 전처리하는 Pipeline
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attr)),
    ('imputer', Imputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
])

# 범주형 변수를 전처리하는 Pipeline
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attr)),
    ('cat_encoder', CategoricalEncoder(encoding = 'onehot-dense'))
])

# 유니온이 이해 안가면 요것도 읽어보자. https://givitallugot.github.io/articles/2020-06/Python-preprocessing-3-pipe3
Feature Union
위와 같이 하나의 데이터셋에서 수치형 변수, 범주형 변수에 대한 파이프라인을 각각 만들었습니다. 
어떻게 두 파이프라인을 하나로 합칠 수 있을까요? 정답은 사이킷런의 FeatureUnion입니다
. 변환기 목록을 전달하고 transform() 메서드를 호출하면 각 변환기의 transform() 메서드를
 병렬로 실행합니다. 그런 다음 각 변환기의 결과를 합쳐 반환합니다. 
 숫자형과 범주형 특성을 모두 다루는 전체 파이프라인은 다음과 같습니다.

# num_pipeline과 cat_pipeline을 합치는 FeatureUnion
full_pipeline = FeatureUnion(transformer_list = [
  ('num_pipeline', num_pipeline),
  ('cat_pipeline', cat_pipeline),
  ])
  
# 전체 파이프라인 실행  
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.preprocessing import OnehotEncoder, CategoricalEncoder


num_attr = list(housing_num)
cat_attr = ["ocean_proximity"]

# 숫자형 변수를 전처리하는 Pipeline
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attr)),
    ('imputer', Imputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
])

# 범주형 변수를 전처리하는 Pipeline
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attr)),
    ('cat_encoder', CategoricalEncoder(encoding = 'onehot-dense'))
])

# num_pipeline과 cat_pipeline을 합치는 FeatureUnion
full_pipeline = FeatureUnion(transformer_list = [
  ('num_pipeline', num_pipeline),
  ('cat_pipeline', cat_pipeline),
  ])

# 전체 파이프라인 실행  
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)

#%%
파이프라인 관련글 2 https://hhhh88.tistory.com/6

파이프라인에 대해서 알아봅시다! 사용할 데이터셋은 다음과 같습니다.

load_breast_cancer
사이킷런에서 제공하는 이진분류데이터셋으로 y는 0과 1입니다

Classes	2
Samples per class	212(M),357(B)
Samples total	569
Dimensionality	30
Features	real, positive
최솟값, 최댓값을 찾아 데이터의 스케일을 바꾸고 SVM을 훈련시켜 평가해보겠습니다

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# 1. 데이터 수집, 로드
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
​
# 표준화
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
​
# 학습
svc = SVC(gamma="auto")
svc.fit(X_train, y_train)
​
# 예측 및 평가
pred = svc.predict(X_test)
print('테스트점수 :{:.2f}'.format(svc.score(X_test, y_test)))
(426, 30) (143, 30) (426,) (143,)
테스트점수 :0.94
sklearn.pipeline.Pipeline(steps, memory=None)
이번에는 같은 모델을 파이프라인으로 재구성해보겠습니다.

파이프라인의 목적
to assemble several steps that can be cross-validated together while setting different parameters
데이터변환(전처리)와 모델을 연결하여 코드를 줄이고 재사용성을 높이기위함
위에서는 스케일러를 불러오고, fit, transform 한 후 모델학습을 하는 일련의 작업이 있었지만 파이프라인을 사용하면 단순히 어떤 스케일러를 쓰고 모델을 쓸 것인지만 쓰면됩니다. 훨씬 편하죠! 여기서 테스트점수는 실행할때마다 바뀝니다.

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC(gamma='auto')) ])
pipline.fit(X_train, y_train)
print('테스트점수 :{:.2f}'.format(pipline.score(X_test, y_test)))
pipline.get_params
테스트점수 :0.91
 

 

<bound method Pipeline.get_params of Pipeline(memory=None,
     steps=[('scaler', MinMaxScaler(copy=True, feature_range=(0, 1))), ('svm', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False))])>
 

레이블 정리
갯수	타입	내용
단일열	이진	분류, 단 하나의 feature는 2개의 Class중 반드시 하나를 가진다.
0 or 1
단일열	실수	회귀, 단 하나의 값으로 예측하면 된다.
예) 텍스트를 보고 나이를 판단
여러열	이진	분류, 각 label 들은 2개의 Class에서 하나를 가진다.
여러 팀들에 대한 선호도 : 00001000
여러열	실수	회귀, 각 feature들은 연속적인 값들을 가지며 여러개를 예측해야한다.
건강검진을 수행한후, 여러가지 검사결과에 대한 예측
여러열	.	분류, 각 데이터들이 여러가지의 카테고리에 속할 수 있다.
모형 최적화
모형최적화를 위해 데이터 전처리와 매개변수 선택하겠습니다.
절차상, 머신러닝 모델을 만든 뒤 최적화 과정을 진행합니다.
이런 모델을 제공해주는 sklearn의 하이퍼 파라미터 튜닝 도구는 아래와 같습니다.
validation_curve : 단일 하이퍼 파라미터 최적화

GridSearchCV : 그리드를 이용한 복수 하이퍼 파라미터 최적화
ParameterGrid : 복수 하이퍼 파라미터 최적화

sklearn.model_selection.GridSearchCV(estimator, param_grid)
param_grid : dict or list of dictionaries
이번에는 그리드 서치로 해보겠습니다.
그리드 서치는 매개변수 탐색 방법으로 지정된 매개변수의 모든 조합을 고려하여 최적의 매개변수를 택합니다.
fit 과 predict 시 쓰인 X_train과 X_test는 위에서 미리 스케일링시킨 데이터입니다.
 

from sklearn.model_selection import GridSearchCV
​
# 1. 매개변수 세팅
params = {
            'C'     : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            'gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] 
         }
grid = GridSearchCV(SVC(), param_grid = params, cv = 5)
grid.fit(X_train, y_train)
print('최상의 교차검증 정확도 {:.2f}'.format(grid.best_score_))
print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
print('최적의 매개변수 : {}'.format(grid.best_params_))
최상의 교차검증 정확도 0.93
테스트 점수 0.94
최적의 매개변수 : {'C': 100, 'gamma': 0.0001}
그러나 여기에는 큰 문제점이 있습니다. 그리드 서치는 훈련폴드와 검증폴드를 나누는데 ★스케일링은 훈련폴드에만★ 적용되어야합니다. 그러나 그리드 서치 수행 전 이미 스케일링을 시켰기때문에 검증폴드에도 전처리과정이 적용되었습니다. 그럼 이러한 오류를 일으키지 않으려면 어떻게 해야할까요?

그리드 서치에 파이프라인 적용하기
그리드 서치를 이후에 전처리를 하면됩니다. 아래코드를 보면 그리드 서치 수행시(grid.fit) 파이프라인을 불러오면서 그제서야 전처리를 하죠. 교차 검증의 각 분할에 스케일러가 훈련폴드에만 매번 적용되고 검증 폴드에는 영향을 미치지않습니다!

+) 파이프라인용 매개변수 그리드는 단계 이름과 매겨변수 이름을 __로 연결합니다

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC()) ])
pipline.fit(X_train, y_train)
print('테스트점수 :{:.2f}'.format(pipline.score(X_test, y_test)))
​
​
params = {
            'svm__C'     : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            'svm__gamma' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000] 
         }
grid = GridSearchCV(pipline, param_grid = params, cv = 5)
grid.fit(X_train, y_train)
print('최상의 교차검증 정확도 {:.2f}'.format(grid.best_score_))
print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
print('최적의 매개변수 : {}'.format(grid.best_params_))
테스트점수 :0.95
최상의 교차검증 정확도 0.98
테스트 점수 0.96
최적의 매개변수 : {'svm__C': 10, 'svm__gamma': 0.1}
그냥 파이프라인으로 실행했을때와 최적의 매개변수를 찾은 뒤 실행했을 때의 차이가 꽤 있네요.

sklearn.pipeline.make_pipeline(*steps, **kwargs)
이번에는 조금 더 간편하게 파이프라인을 만들어봅시다. make_pipeline은 따로 튜플로 단계를 쓸 필요가 없습니다. 그냥 모델만 써줘도 자동으로 class 이름을 소문자로하여 생성해줍니다.

# 표준 방법 
pipline = Pipeline([('scaler',MinMaxScaler()), ('svm', SVC()) ])
# 이름을 개발자가 임의로 지정이 가능하다.
pipline_short = make_pipeline(MinMaxScaler(), SVC())
​
# 파이프 라인 확인
# 해당함수의 이름을 자동적으로 지정된다.
for pip in pipline_short.steps:
    print(pip)
('minmaxscaler', MinMaxScaler(copy=True, feature_range=(0, 1)))
('svc', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False))
 

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
​
pipeline = make_pipeline(StandardScaler(), PCA(n_components = 2), RobustScaler())
pipeline.fit(X_train, y_train)
​
# pca를 통해 주성분 2개를 획득
pipeline.named_steps['pca'].components_
 

array([[ 0.21352408,  0.11169776,  0.22240769,  0.21579146,  0.13936009,
         0.23766175,  0.25409034,  0.2570735 ,  0.14170691,  0.0704947 ,
         0.2104832 ,  0.0289034 ,  0.21876275,  0.21908038,  0.00795456,
         0.16912333,  0.15919868,  0.18094992,  0.04922474,  0.10647299,
         0.22370447,  0.11179843,  0.23290378,  0.22022722,  0.12653686,
         0.20977434,  0.22350897,  0.24818213,  0.13265346,  0.13542811],
       [-0.23894516, -0.06616963, -0.21939265, -0.2330771 ,  0.18988721,
         0.15078108,  0.05540106, -0.0354878 ,  0.18730805,  0.36231021,
        -0.09487267,  0.08368153, -0.07195857, -0.14933674,  0.21908426,
         0.22921029,  0.1871889 ,  0.1177492 ,  0.19660776,  0.27372235,
        -0.22557509, -0.05406291, -0.20400483, -0.22302458,  0.17250359,
         0.14654965,  0.09427313, -0.01196401,  0.14869047,  0.26936063]])
 

파이프 라인(스케일링, 로지스틱 회귀) + 하이퍼 파라미터
from sklearn.linear_model import LogisticRegression
pipline = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=100000))
​
# step은 (이름, 객체)의 형태를 원소로 가지는 list이다.
lr_name = pipline.steps[1][0]
​
# 파라미터 대상 모델을 지정 : 키 = "알고리즘별칭__파라미터명"
params = {
            '{}__C'.format(lr_name): [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
         }
​
# 데이터 가공, 전처리(훈련/테스트 데이터 분류)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
​
​
grid = GridSearchCV(pipline, param_grid = params, cv = 5)
grid.fit(X_train, y_train)
print('최상의 모델 {}'.format(grid.best_estimator_))
print('로지스틱 회귀 단계 {}'.format(grid.best_estimator_.named_steps[lr_name]))
print('로지스틱 회귀 계수 {}'.format(grid.best_estimator_.named_steps[lr_name].coef_))
print('최상의 교차검증 정확도 {:.2f}'.format(grid.best_score_))
print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
print('최적의 매개변수 : {}'.format(grid.best_params_))
최상의 모델 Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('logisticregression', LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100000, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False))])
로지스틱 회귀 단계 LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100000, multi_class='warn',
          n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
          tol=0.0001, verbose=0, warm_start=False)
로지스틱 회귀 계수 [[-0.3929256  -0.39952884 -0.38017917 -0.37679157 -0.17405702 -0.0033258
  -0.35412644 -0.40761829 -0.06515424  0.22631314 -0.46723371  0.09060721
  -0.33356918 -0.34084096 -0.013305    0.19513636 -0.00068799 -0.12766301
   0.17380405  0.21329793 -0.49869097 -0.49841221 -0.44623736 -0.44166956
  -0.381662   -0.1725893  -0.46347628 -0.4668935  -0.3633781  -0.09375148]]
최상의 교차검증 정확도 0.97
테스트 점수 0.98
최적의 매개변수 : {'logisticregression__C': 0.1}
전처리기 2개에 모델 1개를 가지는 파이프라인
스케일조정, 다항식 특성 선택, 릿지 회귀의 세 단계로 구성된 파이프라인을 구현했습니다.

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso
data2 = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data2.data, data2.target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
​
from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
​
# 파라미터 -> alpha가 커질수록 계수가 작아지면서 복잡도가 감소 -> 단순화
# 다른 작업에 대한 파라미터이지만 한꺼번에 수행이 가능하다 점이 바로 파이프라인의 장점
param = {
    # 전처리기에 대한 파라미터
    'polynomialfeatures__degree': [1,2,3],
    # 알고리즘에 파라미터
    'ridge__alpha' : [0.001, 0.01, 0.1, 1, 10, 100]
}
​
#그리드 서치
# iid : 각 테스트 세트의 샘플 수, 가중치가 적용된 폴드(cv에서 세트 규정)에 평균점수 반환
grid = GridSearchCV(pipe, param_grid = param, cv = 5, n_jobs = -1, iid = True)
​
#훈련
grid.fit(X_train, y_train)
(379, 13) (127, 13) (379,) (127,)
 

 

GridSearchCV(cv=5, error_score='raise-deprecating',
       estimator=Pipeline(memory=None,
     steps=[('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('polynomialfeatures', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), ('ridge', Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, random_state=None, solver='auto', tol=0.001))]),
       fit_params=None, iid=True, n_jobs=-1,
       param_grid={'polynomialfeatures__degree': [1, 2, 3], 'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
 

시각화
데이터프레임으로 변환후 seaborn사용
mglearn 사용
# 시각화 자료
import pandas as pd
import seaborn
​
df = pd.DataFrame(grid.cv_results_['mean_test_score'].reshape([3,-1]))
​
# -1를 사용하여 다른 하나의 기준으로 나머지를 연산하여준다.
grid.cv_results_['mean_test_score'].reshape([3,-1])
df.columns = param['ridge__alpha']
df.index = param['polynomialfeatures__degree']
df.index.name = 'polynomialfeatures__degree'
df.columns.name = 'ridge__alpha'
df
 

.dataframe tbody tr th {
    vertical-align: top;
}
​
.dataframe thead th {
    text-align: right;
}
ridge__alpha	0.001	0.01	0.1	1.0	10.0	100.0
polynomialfeatures__degree						
1	0.722189	0.722192	0.722216	0.722410	0.721702	0.692022
2	0.756721	0.762969	0.784703	0.826684	0.855482	0.826077
3	-29.591460	-12.134137	-2.686031	-0.253970	0.692706	0.839140
 

#히트맵으로 처리
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
​
plt.figure(figsize=(8,4))
sns.heatmap( df, annot = True, fmt = '.2f')
plt.show()


 

import mglearn
mglearn.tools.heatmap(grid.cv_results_['mean_test_score'].reshape([3,-1]), 
                      xlabel = 'ridge__alpha', ylabel='polynomialfeatures__degree',
                      xticklabels = param['ridge__alpha'], 
                      yticklabels = param['polynomialfeatures__degree'], vmin=0)
 

<matplotlib.collections.PolyCollection at 0xe53f5f8>
 



아래는 다항식 특성이 없는 그리드 서치입니다. 다항 특성이 없는 경우 성능이 낮아진 것을 볼 수 있습니다.

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso
data2 = load_boston()
X_train, X_test, y_train, y_test = train_test_split(data2.data, data2.target)
​
from sklearn.preprocessing import PolynomialFeatures
pipe = make_pipeline(StandardScaler(), Ridge())
​
param = {
        'ridge__alpha' : [0.001, 0.01, 0.1, 1, 10, 100]
        }
​
grid = GridSearchCV(pipe, param_grid = param, cv = 5, n_jobs = -1, iid = True)
grid.fit(X_train, y_train)
print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
테스트 점수 0.69
알고리즘 선택을 위한 그리드 서치
여러가지 알고리즘에 따른 매개변수값을 지정하며, 알고리즘 선택까지 가능하게 한다.

from sklearn.ensemble import RandomForestClassifier
​
# 순서는 파이프라인 객체 생성시에 결정.
pipe = Pipeline( [('preprocessing', StandardScaler() ), ('classifier', SVC())] )
​
# naming을 통해서 모델에 대한 변경도 가능하다. 
param = [
        {
            'classifier' : [SVC()],
            'classifier__C' : [0.001, 0.01, 0.1, 1, 10],
            'classifier__gamma'  : [0.001, 0.01, 1, 10, 100],
            'preprocessing' : [StandardScaler()]
        },
        {
            'classifier' : [RandomForestClassifier(n_estimators=100)],
            'classifier__max_features' : [1, 2, 3],
            'preprocessing' : [None]  # 전처리 안함
        }
        ]
​
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
​
grid = GridSearchCV(pipe, param, cv = 5, n_jobs = -1)
grid.fit(X_train, y_train)
​
print('최적의 매개변수 : {}'.format(grid.best_params_))
print('테스트 점수 {:.2f}'.format(grid.score(X_test, y_test)))
최적의 매개변수 : {'classifier': SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False), 'classifier__C': 10, 'classifier__gamma': 0.01, 'preprocessing': StandardScaler(copy=True, with_mean=True, with_std=True)}
테스트 점수 1.00
 

from IPython.core.display import display, HTML
display(HTML("undefined<style>.container { width:90% !important; }</style>"))
undefined