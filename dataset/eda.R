setwd('C:/Users/styli/Desktop/데이콘 프로젝트/펭귄/dataset')
library(data.table)
library(dplyr)
library(ggplot2)

train = fread('train.csv')

# 변수 형태 탐색 - 명목형
table(train$Species)
table(train$Island)
table(train$`Clutch Completion`)
table(train$Sex) # 결측 3

library(gridExtra)
a = ggplot(train, aes(Species))+geom_bar()
b = ggplot(train, aes(Island))+geom_bar()
c = ggplot(train, aes(`Clutch Completion`))+geom_bar()
d = ggplot(train, aes(Sex))+geom_bar()
grid.arrange(a,b,c,d, nrow=2, ncol=2)

# 수치형
train %>% select(c('Culmen Depth (mm)','Culmen Length (mm)','Flipper Length (mm)','Delta 15 N (o/oo)','Delta 13 C (o/oo)','Body Mass (g)')) %>% summary()

par(mfrow=c(2,3))
hist(train$`Culmen Depth (mm)`)
hist(train$`Culmen Length (mm)`)
hist(train$`Flipper Length (mm)`)
hist(train$`Delta 15 N (o/oo)`)
hist(train$`Delta 13 C (o/oo)`)
hist(train$`Body Mass (g)`)

a = ggplot(train, aes(`Culmen Depth (mm)`)) + geom_boxplot()
b = ggplot(train, aes(`Culmen Length (mm)`)) + geom_boxplot()
c = ggplot(train, aes(`Flipper Length (mm)`)) + geom_boxplot()
d = ggplot(train, aes(`Delta 15 N (o/oo)`)) + geom_boxplot()
e = ggplot(train, aes(`Delta 13 C (o/oo)`)) + geom_boxplot()
f = ggplot(train, aes(`Body Mass (g)`)) + geom_boxplot()
grid.arrange(a,b,c,d,e,f, nrow=2, ncol=3)

rm(a,b,c,d,e,f)


# 상관관게 분석(피어슨 상관계수-수치형 자료 한정)

train %>% select(c('Culmen Depth (mm)','Culmen Length (mm)','Flipper Length (mm)','Delta 15 N (o/oo)','Delta 13 C (o/oo)','Body Mass (g)')) %>% cor()
# 에러발생

# 결측 탐색.
train %>% is.na() %>% colSums()

# 결측처리 - 행 버리기

train %>% na.omit() %>% 
  select(c('Culmen Depth (mm)','Culmen Length (mm)','Flipper Length (mm)','Delta 15 N (o/oo)','Delta 13 C (o/oo)','Body Mass (g)')) %>% 
  cor() %>% print()

# 산점도 탐색

train %>% na.omit() %>% 
  select(c('Culmen Depth (mm)','Culmen Length (mm)','Flipper Length (mm)','Delta 15 N (o/oo)','Delta 13 C (o/oo)','Body Mass (g)')) %>% plot()


# 명목형 변수 변경

train[, c('Species', 'Island', 'Clutch Completion', 'Sex')] = lapply(train[ , c('Species', 'Island', 'Clutch Completion', 'Sex')], factor)
str(train)

# 회귀분석
reg = lm(`Body Mass (g)`~.,data = train)
summary(reg)

test = fread('test.csv')
test[, c('Species', 'Island', 'Clutch Completion', 'Sex')] = lapply(test[ , c('Species', 'Island', 'Clutch Completion', 'Sex')], factor)

# 대충 이렇게 예측하면 되겠지.
head(predict(reg, test))

# 베이스라인 회귀 끝.
# 잔차의 4가정, IID에 특이값. 선형성 정규성, 등분산성가정.
par(mfrow=c(2,2))
plot(reg)

# 변수선택법 사용. both

step.reg = step(reg, direction = 'both')
summary(step.reg)
