###########################
###EDA / 데이터 파악 파트###
###########################
#%%
# 처음부터 다시작업.
import os
import pandas as pd
import numpy as np
import matplotlib as plt
#%%
# 데이터셋 임포트
os.chdir(r'C:\Users\styli\Desktop\데이콘 프로젝트\동아리 pj-펭귄\dataset')
train = pd.read_csv('train.csv')

#%%
# 결측처리 방안.
# 명목형 = 최빈값 대체
# 수치형 = 평균대체.

train.info()
# 결측 있는 행, sex, delta 15, delta 13에서 결측발생.
# id라는 무의미한 값이 있음.
train = train.drop("id",axis=1)

#%%
# 결측값 있는 열 값 확인
nacell = train[train.isna().sum(axis=1) > 0]

# 특이 패턴 없음.

#%% 
# 테스트셋 미리 확인.
test = pd.read_csv('test.csv')
test = test.drop("id",axis=1)

test.info()
# test에서도, sex, delta15, delta 13에서만 결측 발생중. id행 무쓸모 함.

# 즉 결측처리는 이 3행에 대해 실시하여야함.


# 산점도를 통한 상관관계 파악 / 상관계수를 통해서도 파악되지.
# EDA시 분산분석을 통해 차이가 있다 없다 가능함.



###########################
###   결측값 처리 파트   ###
###########################
#%%

train.loc[:,'Sex'].describe()
train.loc[:,'Sex'].value_counts()
# Sex의 최빈은 MAlE

train.loc[:,'Delta 15 N (o/oo)'].describe()
train.loc[:,'Delta 15 N (o/oo)'].mean()
# delta 15의 평균은 8.73.... 해당연산값

train.loc[:,'Delta 13 C (o/oo)'].describe()
train.loc[:,'Delta 13 C (o/oo)'].mean()
# delta 13의 평균은 -25.72....

#%%
# 결측값 처리

train.loc[:,'Delta 15 N (o/oo)'] = train.loc[:,'Delta 15 N (o/oo)'].fillna(train.loc[:,'Delta 15 N (o/oo)'].mean())
train.loc[:,'Delta 13 C (o/oo)'] = train.loc[:,'Delta 13 C (o/oo)'].fillna(train.loc[:,'Delta 13 C (o/oo)'].mean())
# train[train.isna().sum(axis=1) > 0]
train.info() # 결측값 더이상 없음 확인 -> rangeidx 114 = 모든 요소 114 non-null으로 확인됨.

#%%
# 결측값 처리-test셋은 train셋 값으로 동일하게 실시

test.loc[:,'Delta 15 N (o/oo)'] = test.loc[:,'Delta 15 N (o/oo)'].fillna(train.loc[:,'Delta 15 N (o/oo)'].mean())
test.loc[:,'Delta 13 C (o/oo)'] = test.loc[:,'Delta 13 C (o/oo)'].fillna(train.loc[:,'Delta 13 C (o/oo)'].mean())


#test[test.isna().sum(axis=1) > 0]
test.info() # 결측값 더이상 없음 확인 -> rangeidx 228 = 모든 요소 228 non-null으로 확인됨.



# 최빈값으로 Sex의 결측을 처리 가능하다
# train.loc[:,'Sex'] = train.loc[:,'Sex'].fillna('MALE')
# test.loc[:,'Sex'] = test.loc[:,'Sex'].fillna('MALE')
# 하지만 이 방법 사용하지 않는다.

########################
## 결측값 KNN 처리방안 ##
########################


# 명목형 변수는 수치형 라벨로 코딩필요. 이 경우 mapping을 이용한 값의 처리, NA값은 NA로만 표기 되었으므로, 변경할 필요가 없음.
# 라벨 코딩을 위해 아래 파악은 필수.
train['Species'].value_counts()
train['Island'].value_counts()
train['Clutch Completion'].value_counts()
train['Sex'].value_counts()

# train셋과 test셋의 가지는 명목형 변수의 범주는 아래와 같이 동일하다.
idx_species = {'Gentoo penguin (Pygoscelis papua)' : 0, 'Adelie Penguin (Pygoscelis adeliae)' : 1,  'Chinstrap penguin (Pygoscelis antarctica)' : 2}
idx_island = {'Biscoe' : 0, 'Dream' : 1, 'Torgersen' : 2}
idx_clutch = {'Yes' : 0, 'No' : 1}
idx_sex = {'MALE' : 0, 'FEMALE' : 1}

# 라벨링과정 실시, NA를 살려야하기 때문에 labelencoder를 사용하지 않았다.
train['Species'] = train['Species'].map(idx_species)
train['Island'] = train['Island'].map(idx_island)
train['Clutch Completion'] = train['Clutch Completion'].map(idx_clutch)
train['Sex'] = train['Sex'].map(idx_sex)

# test셋에 대해서도 동등하게 매핑 실시.
test['Species'] = test['Species'].map(idx_species)
test['Island'] = test['Island'].map(idx_island)
test['Clutch Completion'] = test['Clutch Completion'].map(idx_clutch)
test['Sex'] = test['Sex'].map(idx_sex)


# KNN을 Sex행에 대해 적용해보자.
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors = 1) 
# KNN 기준 n_neighbors = 이웃 파라미터, 디폴트값이 5임. 
# 예를들어 5를 쓰는데 Na값의 이웃으로 해당값과 유사한, 0,0,0,1,1 이라는 Sex의 이웃 KNN에 포함될경우 이들의 평균인 0.4로 예측하게 된다.
# 따라서 우리는 가장 비슷한 예제로 채우고 싶기 때문에 n_neighbor = 1을 사용하였다.
# 만약 큰 숫자를 사용한다면 시그모이드 같은 함수를 써서 분류를 해야하는 상황이 올지 확신이 서지는 않는다.

# 지금 조심해야 하는게, train셋은 Bodymass(예측해야 할 값) 포함되어 있음. 따라서 bodymass는 KNN과정에 포함되면 안됨.
# 따로 적용하자.
temp = train.drop('Body Mass (g)',axis=1)
temp_y = train.loc[:,'Body Mass (g)']

# KNN은 temp에 대해 실행하고, 아래 전처리 작업에 일관성을 위해 다시 train셋에 합처주겠다.
temp = pd.DataFrame(imputer.fit_transform(temp), columns = temp.columns)
train = pd.concat([temp, temp_y], axis=1)

# test에는 따로 body mass가 존재하지 않으니 빼면 좋지.
test = pd.DataFrame(imputer.transform(test), columns = test.columns)

# temp는 지워준다.
del temp; del temp_y

###########################
###    Scaling 파트     ###
###########################

#%%
'''
# 명목형 변수 존재.
# 라벨 인코더로 처리 가능하다. 그렇지만 실제 분석단계에서 라벨 인코더는 성능이 적은경우가 다수.
# 명목형 변수는 Species, Island, Clutch completion, Sex 변수이다.
# 원핫 인코딩이 유용할것. = 그냥 라벨링 하면 결과 안좋음.

train_transform = pd.concat([train, pd.get_dummies(train['Species'], drop_first=True)], axis=1)
train_transform = pd.concat([train_transform, pd.get_dummies(train['Island'], drop_first=True)], axis=1)
train_transform = pd.concat([train_transform, pd.get_dummies(train['Clutch Completion'], drop_first=True)], axis=1)
train_transform = pd.concat([train_transform, pd.get_dummies(train['Sex'], drop_first=True)], axis=1)

# 변환한 명목현 변수 4개 지울것.
# 변환된 원핫 인코딩에서 하나씩은 자유도 명목으로 삭제 가능함. 이는 drop_first 옵션으로 지워졌다.
train_transform = train_transform.drop(['Species','Island','Clutch Completion','Sex'],axis=1)


# test셋에도 동등한 처리.

test_transform = pd.concat([test, pd.get_dummies(test['Species'], drop_first=True)], axis=1)
test_transform = pd.concat([test_transform, pd.get_dummies(test['Island'], drop_first=True)], axis=1)
test_transform = pd.concat([test_transform, pd.get_dummies(test['Clutch Completion'], drop_first=True)], axis=1)
test_transform = pd.concat([test_transform, pd.get_dummies(test['Sex'], drop_first=True)], axis=1)

# 변환한 명목현 변수 4개 지울것.
# 변환된 원핫 인코딩에서 하나씩은 자유도 명목으로 삭제 가능함.
test_transform = test_transform.drop(['Species','Island','Clutch Completion','Sex'],axis=1)

# pd.get_dummies(df[], drop_first=True)를 통해 자유도 확보 가능하다
# 자유도 확보 이유 -> 차원 저주 해소나, 다중공선성의 개선이 일어남.
# pd.get_dummies(demo_df, columns = ['숫자 특성', '범주형 특성'])
# 이런식으로 여러열에 대해서 동시에 처리도 가능하다.

# 혹은 OneHotEncoder()를 통해 변경도 됨. pd로 만들때랑 다르게 매개변수로 열 삭제는 못하지만, 
# OneHotEncoder().fit_transform(X).toarray()[:,1:]이런 방식으로 첫 열을 지울 수 있다. => drop 매개변수 찾아볼것. 된다는 소리가 있음.

# OneHotEncoder()는 0~최대값에서 빈수가 없는 정수 값을 가진 열을 명목형 변수로 기본 인식한다. = 큰 문제 발생 자주함.
# 고유한 값을 선택하기 위해선 categories 매개변수 이용 필요. 공부 해볼것.
'''


#%%
# ColumnTransformer()라는 좋은 수단이 있음.
# 범주형과 수치형의 작업을 동시에 할 수 있는 메소드. 
# OneHot + 표준화.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


OneHotScaleing = OneHotEncoder(drop='first') # drop = 'first'로 첫행 버림으로서, 자유도? 제공함.
StandardScaleing = StandardScaler()

ColumnScaleing = ColumnTransformer([
    ("onehot", OneHotScaleing, ['Species', 'Island', 'Clutch Completion', 'Sex']),
    ("scaling", StandardScaleing, ['Culmen Length (mm)', 'Culmen Depth (mm)','Flipper Length (mm)','Delta 15 N (o/oo)','Delta 13 C (o/oo)'])
])
# 스케일링 결과는 행 불러온 순서대로 정해진다.

# 파이프라인 이용하면 되니, 데이터셋의 형태만 올바르게 해주자.
train_X = train.drop('Body Mass (g)',axis=1)

# y값은  따로 스케일링을 진행해준다 => 원상복구할때 이 형태가 편함.
y_scaling = StandardScaler()
train_y = np.array(train.loc[:,'Body Mass (g)']).reshape(-1,1)
train_y_transform = y_scaling.fit_transform(train_y)

# test셋에 대해서도 동등한 처리를 하여주자.


#%%
# OneHot + 정규화.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

OneHotScaleing2 = OneHotEncoder(drop='first') # drop = 'first'로 첫행 버림으로서, 자유도? 제공함.
MinMaxScaleing2 = MinMaxScaler()

ColumnScaleing2 = ColumnTransformer([
    ("onehot", OneHotScaleing2, ['Species', 'Island', 'Clutch Completion', 'Sex']),
    ("scaling", MinMaxScaleing2, ['Culmen Length (mm)', 'Culmen Depth (mm)','Flipper Length (mm)','Delta 15 N (o/oo)','Delta 13 C (o/oo)'])
])
# 스케일링 결과는 행 불러온 순서대로 정해진다.

# 파이프라인 이용하면 되니, 데이터셋의 형태만 올바르게 해주자.
train_X = train.drop('Body Mass (g)',axis=1)

# y값은  따로 스케일링을 진행해준다 => 원상복구할때 이 형태가 편함.
y_scaling_minmax = MinMaxScaler()
train_y = np.array(train.loc[:,'Body Mass (g)']).reshape(-1,1)
train_y_transform_minmax = y_scaling_minmax.fit_transform(train_y)

# test셋에 대해서도 동등한 처리를 하여주자.


###########################
###    데이터셋 분리     ###
###########################
#%%

# 상당히 데이터 세트가 적다.(114개의 훈련 레이블 -> 0.8, 0.2비율로 나누는게 적절하지 않을까? 큰 샘플이면 더 작아져아하지만...)
# train_test_split에서 따로 층화표집 필요성이 존재하지 않는다. 회귀니까. => 전처리 끝.
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y_transform, test_size=0.2, random_state = 404324)

'''
# 대충 생각나는거. -> 랜덤포레스트 회귀, XGBOOST 회귀, 결정나무 회귀 등.
# 선형회귀, 라소릿지. XGBOOST만 해보자.

# 자료 특징 - 라벨 인코딩 되어있음. 원핫 인코딩은 안됨. 명목형 4개, 수치형 5개, 예측은 수치형(회귀)
# k-mean 회귀는 변수 적을때 유용(2-3변수) 제외. -> 적을땐 선형회귀보다 유용. 변수개수 증가시 MSE가 선형회귀보다 크게 증가.
'''





###########################
###    선형 회귀 분석    ###
###########################
#%%
from sklearn.pipeline import Pipeline, make_pipeline # pipeline = 튜플(제목+모듈)로 입력, make_pipeline은 튜플 쓸 필요가 없는 파이프라인.
from sklearn.linear_model import LinearRegression # 선형 회귀 모델


linear_model = Pipeline([
    ('scaleing', ColumnScaleing),
    ('linearReg', LinearRegression())
])
# 스케일러에 의해서 자동으로 정리됨.

# 모델 훈련
linear_model.fit(X_train, y_train)

# 모델 예측
linear_model.predict(X_valid)

# 모델 score(회귀-결정계수)
linear_model.score(X_train, y_train)
linear_model.score(X_valid, y_valid)

# 결정계수 맞는지 따로 확인.
from sklearn.metrics import r2_score, mean_squared_error
y_train_pred = linear_model.predict(X_train)
y_valid_pred = linear_model.predict(X_valid).reshape(-1,1)

r2_score(y_train, y_train_pred)
r2_score(y_valid, y_valid_pred)

# mse가 궁금한가?
mean_squared_error(y_valid, y_valid_pred) # 표준화된 mse지. 원래 식에 대해선 이거 아닐까?
mean_squared_error(y_scaling.inverse_transform(y_valid), y_scaling.inverse_transform(y_valid_pred)) # 실제 MSE

# RMSE 구해보기.
import math
math.sqrt(mean_squared_error(y_scaling.inverse_transform(y_valid), y_scaling.inverse_transform(y_valid_pred)))

# 테스트셋에 대해서 예측한다면 이렇게 된다.
test_pred = linear_model.predict(test)

# 다시 원래대로 돌려준다면? 이렇게 된다.
linear_test_pred = y_scaling.inverse_transform(test_pred)


# 선형회귀에 그리드 서치는 불가능 => 목표함수가 정해져 있기 때문에
# 맞는지는 모르겠지만, 회귀분석에서. 식 자체에서 지정할 수 있는 모수 없음 MLE에 의해 b = (t(X)X)^-1 * t(x)y로 구해진다.
# t()는 전치행렬, -1승은 역행렬.

#%%
# sgdRegressor = 회귀분석과 동일하지만, 경사하강법을 이용하는것이 특징. 하이퍼파라미터가 존재함.
from sklearn.linear_model import SGDRegressor

SGD_model = Pipeline([
    ('scaleing', ColumnScaleing),
    ('SGD_reg', SGDRegressor())
])

# 모델 훈련
SGD_model.fit(X_train, y_train)

# 모델 예측
SGD_model.predict(X_valid)

# 모델 score(회귀-결정계수)
SGD_model.score(X_train, y_train)
SGD_model.score(X_valid, y_valid)

from sklearn.metrics import r2_score, mean_squared_error
y_train_pred = SGD_model.predict(X_train)
y_valid_pred = SGD_model.predict(X_valid).reshape(-1,1)

# mse 체크
mean_squared_error(y_valid, y_valid_pred) # 표준화된 mse
mean_squared_error(y_scaling.inverse_transform(y_valid), y_scaling.inverse_transform(y_valid_pred)) # 실제 MSE

# RMSE 구해보기.
import math
math.sqrt(mean_squared_error(y_scaling.inverse_transform(y_valid), y_scaling.inverse_transform(y_valid_pred)))

# 교차검증. 방법.
from sklearn.model_selection import cross_val_score
cross_val_score(SGD_model, X_train, y_train, cv=5, scoring='r2')


# GRID_SEARCH (SGD-선형회귀는 파라미터 지정 가능.)
from sklearn.model_selection import KFold, GridSearchCV

kfold = KFold(n_splits=10, shuffle = True) # shuffle=True로 매 k-fold의 편향을 없애준다. 층화가 필요하면 StratifiedKFold 사용할것.
params = {'SGD_reg__loss' : ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
          'SGD_reg__learning_rate' : ['optimal','invscaling','adaptive'],
          'SGD_reg__eta0' : [0.05, 0.01, 0.001],
          'SGD_reg__epsilon' : [0.01, 0.05, 0.1, 0.2]
}
# max_iter나 tol(=eps)는 최초 파이프라인에서 선정하면 더 좋을것으로 생각.
# panelty항과 여러 변수를 조정하면, SGDRegressor는 lasso 혹은 elasticnet도 연산 가능합니다.
# fit_interceptbool = 절편이 있는지 없는지.
# 'learning_ratestring' = default값은 'invscaling': eta = eta0 / pow(t, power_t)
# epsilonfloat, default=0.1 'huber', 'epsilon_insensitive'또는 'squared_epsilon_insensitive'인 경우에만 해당됩니다 
# 'huber'의 경우 예측을 정확하게 얻는 것이 덜 중요 해지는 임계 값을 결정합니다
# 내 임의대로 배치한거임. 훈련결과 확인후 지속적인 조정 필요.

grid = GridSearchCV(estimator= SGD_model, param_grid = params, scoring= 'neg_root_mean_squared_error', cv = kfold, n_jobs=-1)
# grid.get_params().keys() 통해 키 이름 파악할것. Pipeline이라서 기본 변수명과 다름.

grid.fit(X_train, y_train)

# 최적의 Grid Search 파라미터.
print(f'최적의 하이퍼 파라미터 : {grid.best_params_}'); print(f'최적의 모델 평균 성능 : {grid.best_score_}')

# 그리드 서치 된 결과, 결정계수.
npred = grid.predict(X_valid).reshape(-1,1)
r2_score(y_valid, npred)



#%%
# 추가 요소.
# 쉽게 설명하면 lasso, ridge를 가중평균을 통해 구한값이 ElasticNet이 된다. -> 실제로는 그냥 penalty항이 다른거다.
# 현재로선 선형회귀 이상 진행하지 않았습니다.
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso



###########################
###    XGBoost 회귀     ###
###########################
#%%
# Xgboost 회귀
import xgboost
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score

# xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75, colsample_bytree=1, max_depth=7)
xgb_model = Pipeline([
    ('scaleing', ColumnScaleing),
    ('XGB_Reg', XGBRegressor())
])


xgb_model.fit(X_train,y_train)

# xgb모델의 특징인, xgb 모델의 변수 중요성 그래프.
xgboost.plot_importance(xgb_model[1])
# xgboost.plot_importance(xgb_model)
# 원래는 xgb 모델에 바로 돌리면 되지만, 파이프라인으로 들가서, xgb_model위치를 지정받아야 그릴 수 있다.

# 결정계수 확인.
xgb_model.score(X_train, y_train)
xgb_model.score(X_valid, y_valid)

# 결정계수 맞는지 따로 확인. mse, 설명분산비율 확인.
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
y_valid_pred = xgb_model.predict(X_valid)
r2_score(y_valid, y_valid_pred)
mean_squared_error(y_valid, y_valid_pred)
explained_variance_score(y_valid_pred,y_valid) # R^2과 같다고 알려져 있지만, 다름. 왜?

# 테스트셋에 대해서 예측한다면 이렇게 된다.
test_pred = xgb_model.predict(test).reshape(-1,1)

# 다시 원래대로 돌려준다면? 이렇게 된다.
y_scaling.inverse_transform(test_pred)



# k-fold 검정을 할수 있을까?

from sklearn.model_selection import cross_val_score
cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')


# GridSearch 해보자.
from sklearn.model_selection import KFold, GridSearchCV

kfold = KFold(n_splits=10, shuffle = True) # shuffle=True로 매 k-fold의 편향을 없애준다. 층화가 필요하면 StratifiedKFold 사용할것.
params = {
         'XGB_Reg__learning_rate': [0.5, 0.2, 0.1, 0.05], #so called `eta` value  
         'XGB_Reg__max_depth': [2, 3, 4],
         'XGB_Reg__gamma': [0.1, 0.5, 1],
         'XGB_Reg__subsample': [0.5, 0.6, 0.7],
         'XGB_Reg__colsample_bytree': [0.6, 0.8, 1.0],
         'XGB_Reg__n_estimators': [10,50,100,200]
}
# 내 임의대로 배치한거임. 아래 기준보고 계속 조정 필요.

grid = GridSearchCV(estimator= xgb_model, param_grid = params, scoring= 'neg_root_mean_squared_error', cv = kfold, n_jobs=4)
# n_jobs = 작업 cpu개수-> 4코어 노트북에서 작업함. verbose - 작업 상황 표시
# cv=k-fold 개수. 매개변수 이용위해 따로 지정하였음. scoring = 여러 변수 참조. 사실 잘 모르겠음. mse, mae, acc, r2 등등 있는데 종류 잘 모름.

grid.fit(X_train, y_train)
# 훈련이 안되는 에러를 발견했는데 =>  Check the list of available parameters with `estimator.get_params().keys()`. 에러.
# grid.get_params().keys()해서 나오는 변수가 params가 된다. pipeline을 사용하면서 변화한 변수명이 문제임.
# estimator__XGB_Reg__n_estimators로 나왔다면 'estimator__' 부분은 떼고 'XGB_Reg__n_estimators'만 사용.

print(f'최적의 하이퍼 파라미터 : {grid.best_params_}'); print(f'최적의 모델 평균 성능 : {grid.best_score_}')

# 그리드 서치 된 결과, 결정계수.
npred = grid.predict(X_valid)
r2_score(y_valid, npred)


'''
"\n최적의 하이퍼 파라미터 : {'XGB_Reg__colsample_bytree': 0.6, 'XGB_Reg__gamma': 0.5, 'XGB_Reg__learning_rate': 0.2,
'XGB_Reg__max_depth': 3, 'XGB_Reg__n_estimators': 100, 'XGB_Reg__subsample': 0.6}\n
뭐 이런 짓 해도 결정계수 63%, defalut로 돌려서 높은 값을 갖는\n최초 모델선정이 제일 중요한것으로 여겨짐.
\n"뭐 이런 짓 해도 결정계수 63%, defalut로 돌려서 높은 값을 갖는
최초 모델선정이 제일 중요한것으로 여겨짐.
'''

#%%
# 부스팅 모델
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor


# 시간이 오래걸리는데 RandomizedSearchCV를 쓰는건 어떤가. 그리드서치와 유사하다.



#%% 
# 그리드 서치 parameter는 아래 예제를 이용함.

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


rforest_model = Pipeline([
    ('scaleing', ColumnScaleing),
    ('rforest_Reg', RandomForestRegressor())
])


rforest_model.fit(X_train,y_train)

# 결정계수 확인.
rforest_model.score(X_train, y_train)
rforest_model.score(X_valid, y_valid)

# 결정계수 맞는지 따로 확인. mse, 설명분산비율 확인.
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
y_valid_pred = rforest_model.predict(X_valid)
r2_score(y_valid, y_valid_pred)
mean_squared_error(y_valid, y_valid_pred)
explained_variance_score(y_valid_pred,y_valid) # R^2과 같다고 알려져 있지만, 다름. 왜?

# 테스트셋에 대해서 예측한다면 이렇게 된다.
test_pred = rforest_model.predict(test).reshape(-1, 1)

# 다시 원래대로 돌려준다면? 이렇게 된다.
y_scaling.inverse_transform(test_pred)


# grid_search

kfold = KFold(n_splits=10, shuffle = True) # shuffle=True로 매 k-fold의 편향을 없애준다. 층화가 필요하면 StratifiedKFold 사용할것.
params = { 'rforest_Reg__n_estimators' : [10, 50,100],
           'rforest_Reg__max_depth' : [6, 12, 18, 24],
           'rforest_Reg__min_samples_leaf' : [1, 6, 12, 18],
           'rforest_Reg__min_samples_split' : [2, 8, 16, 20]
            }
grid2 = GridSearchCV(estimator= rforest_model, param_grid = params, scoring= 'neg_root_mean_squared_error', cv = kfold, n_jobs=4)

grid2.fit(X_train, np.ravel(y_train))
# 너는 왜 np.ravel 써야하는지 잘 모르겠음. => 1차원 변환 함수. reshape(-1,1)는 통하지 않음.
# XGboost에서는 np (변수수,1)형으로 넣으면 따로 요구하지도 않았는데? 얘는 따로 평탄화도 해야함.
# 안쓰면 A column-vector y was passed when a 1d array was expected 에러 등장함.

print(f'최적의 하이퍼 파라미터 : {grid.best_params_}'); print(f'최적의 모델 평균 성능 : {grid.best_score_}')

# 그리드 서치 된 결과, 결정계수.
npred = grid2.predict(X_valid)
r2_score(y_valid, npred)


#%%




# 뭔가 더 해보고 싶은 모듈들.

# 앙상블 모델.(나무 기반.)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor



