# VKR
Импортируем библиотеки
```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn
import itertools
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from pandas import DataFrame
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras import models
from keras.layers.core import Dense
from sklearn.model_selection import GridSearchCV
from scipy import stats
from tensorflow.keras.optimizers import Adam
```
Загружаем данные и конвертируем в csv
```
x_bp_df = pd.read_excel('C:/Users/HOME/Documents/Бауманка/Итоговый проект/X_bp.xlsx', sheet_name='X_bp.csv')
x_nup_df = pd.read_excel('C:/Users/HOME/Documents/Бауманка/Итоговый проект/X_nup.xlsx', sheet_name='X_nup.csv')
```
Объединяем по типу INNER
```
npbp_df = x_bp_df.merge(x_nup_df, left_index=True, right_index=True, how='inner')
```
Убираем ненужные колонки
```
npbp_df.drop(columns =['Unnamed: 0_x', 'Unnamed: 0_y'],axis=1,inplace=True)
```
## Описателььная статистика первоначального набора
```
npbp_df.describe()
```
Построение гистограмм первоначальных данных
```
for col in npbp_df.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = npbp_df[col], kde=True)
    plt.show
```
Графики взаимосвязей первоначального набора данных
```
seaborn.pairplot(npbp_df, height=2.5)
```
Графики корреляции первоначального набора данных
```
plt.figure(figsize = (10,3))
seaborn.heatmap(npbp_df.corr(), cmap= 'rainbow', annot = True, linewidths=1, linecolor='black' )
```
### Статистические параметры
```
npbp_df.mean()
npbp_df.median()
npbp_df.describe().transpose()[['mean','std']]
```
Ищем выбросы при помощи boxplot
```
npbp_df.boxplot(rot=90)
```
Выявление выбросов с помощью метода квартилей и удадение
```
for x in ['Соотношение матрица-наполнитель']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Плотность, кг/м3']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['модуль упругости, ГПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Количество отвердителя, м.%']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Содержание эпоксидных групп,%_2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Температура вспышки, С_2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Поверхностная плотность, г/м2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Модуль упругости при растяжении, ГПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Прочность при растяжении, МПа']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Потребление смолы, г/м2']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Угол нашивки, град']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Шаг нашивки']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
    for x in ['Плотность нашивки']:
    q75,q25 = np.percentile(npbp_df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    npbp_df.loc[npbp_df[x] < min,x] = np.nan
    npbp_df.loc[npbp_df[x] > max,x] = np.nan
    
npbp_df.isnull().sum()

npbp_df.duplicated().sum()
```
### Статистика, гистограммы и графики после удаления выбросов.
```
npbp_df.info()
npbp_df.describe()
for col in npbp_df.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = npbp_df[col], kde=True)
    plt.show
    
    seaborn.pairplot(npbp_df, height=2.5)
    
    plt.figure(figsize = (14,6))
seaborn.heatmap(npbp_df.corr(), cmap= 'rainbow', annot = True, linewidths=1, linecolor='black' )

npbp_df.mean()

npbp_df.median()

npbp_df.describe().transpose()[['mean','std']]
```
## Нормализация
```
minmax_scaler = MinMaxScaler ()
df_norm = minmax_scaler.fit_transform(np.array(npbp_df[['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']]))
df_norm = pd.DataFrame(data = df_norm, columns = ['Соотношение матрица-наполнитель','Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки'])
```
Статистика и гистраграммы после нормализации
```
df_norm.head()
df_norm.describe()
for col in df_norm.columns:
    plt.figure(figsize=(6,2))
    plt.title('Гистограмма'+ ' ' + col)
    plt.ylabel('Количество элементов')
    seaborn.histplot(data = df_norm[col], kde=True)
    plt.savefig('C:/Users/Alexandra/Desktop/вкр/ГистограммаНорм.pdf')
    plt.show
boxplot = df_norm.boxplot(rot=90)
```
Разметка данных
```
import splitfolders 
from sklearn.model_selection import train_test_split
x_upr=df_norm.drop(['Модуль упругости при растяжении, ГПа'], axis=1)
x_pr=df_norm.drop(['Прочность при растяжении, МПа'], axis=1)
y_upr=df_norm['Модуль упругости при растяжении, ГПа']
y_pr=df_norm['Прочность при растяжении, МПа']

x_train_upr, x_test_upr, y_train_upr, y_test_upr=train_test_split(x_upr, y_upr, test_size=0.3, random_state=1)
x_train_pr, x_test_pr, y_train_pr, y_test_pr=train_test_split(x_pr, y_pr, test_size=0.3, random_state=1)
print('Размер тренировочного датасета на входе:', x_train_upr.shape)
print('Размер тестового датасета на входе:', x_test_upr.shape)
print('Размер тренировочного датасета на выходе:', y_train_upr.shape)
print('Размер тестового датасета на выходе:', y_test_upr.shape)

print('Размер тренировочного датасета на входе:', x_train_pr.shape)
print('Размер тестового датасета на входе:', x_test_pr.shape)
print('Размер тренировочного датасета на выходе:', y_train_pr.shape)
print('Размер тестового датасета на выходе:', y_test_pr.shape)
```

### Линейная регрессия
```
Ir = LinearRegression()
Ir_params = { 'fit_intercept' : ['True','False']}
GSCV_ir_upr = GridSearchCV(Ir, Ir_params, n_jobs=-1, cv=10)
GSCV_ir_upr.fit(x_train_upr, y_train_upr)
GSCV_ir_upr.best_params_
{'fit_intercept': 'True'}
Ir_upr = GSCV_ir_upr.best_estimator_
print(f'R2-score LR для модуля упругости при растяжении: {Ir_upr.score(x_test_upr, y_test_upr).round(3)}')

Ir_upr_result = pd.DataFrame({
    'Model': 'LinearRegression_upr',
    'MAE': mean_absolute_error(y_test_upr, Ir_upr.predict(x_test_upr)),
    'R2 score':Ir_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])

Ir_upr_result
```
### Решающее дерево
```
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt_params = { 'criterion': ['mse', 'friedman_mse', 'mae']}
GSCV_dt_upr = GridSearchCV(dt, dt_params, n_jobs=-1, cv=10)
GSCV_dt_upr.fit(x_train_upr, y_train_upr)
GSCV_dt_upr.best_params_

dt_upr = GSCV_dt_upr.best_estimator_

dt_upr_result = pd.DataFrame({
    'Model': 'DecisionTreeRegressor',
    'MAE': mean_absolute_error(y_test_upr, dt_upr.predict(x_test_upr)),
    'R2 score':dt_upr.score(x_test_upr, y_test_upr).round(3)
}, index=['Прочность при растяжении, МПа'])

dt_upr_result
```
### Поиск гиперпараметров по разным регрессиям
```
Ir = LinearRegression()
Ir.fit(x_train_pr, y_train_pr)
y_pred_ir = Ir.predict(x_test_pr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_ir, 'g', label='prediction')
pl.plot(y_test_pr.values, label='actual')
pl.grid=True;

Ir = LinearRegression()
Ir.fit(x_train_upr, y_train_upr)
y_pred_ir = Ir.predict(x_test_upr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_ir, 'g', label='prediction')
pl.plot(y_test_upr.values, label='actual')
pl.grid=True;

param_grid = {'n_neighbors': range(1,50)}
gs = GridSearchCV(knr, param_grid, cv=10, verbose = 1, n_jobs=-1)
gs.fit(x_train_pr,y_train_pr)
knn_3 = gs.best_estimator_
gs.best_params_

dt = DecisionTreeRegressor()
dt.fit(x_train_pr, y_train_pr)
y_pred_dt = dt.predict(x_test_pr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_dt, 'g', label='prediction')
pl.plot(y_test_pr.values, label='actual')
pl.grid=True;

param_grid = {'criterion': ['friedman_mse']}
gs = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=10, verbose = 1, n_jobs=-1)
gs.fit(x_train_upr,y_train_upr)
dt_3 = gs.best_estimator_
gs.best_params_

dt = DecisionTreeRegressor()
dt.fit(x_train_upr, y_train_upr)
y_pred_dt = dt.predict(x_test_upr)
pl.figure(figsize=(12,10))
pl.plot(y_pred_dt, 'g', label='prediction')
pl.plot(y_test_upr.values, label='actual')
pl.grid=True;
```

# Обучение и создание нейросети
Разметка данных
```
input_columns_names = ['Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']
output_columns_names = ['Соотношение матрица-наполнитель']
x = df_norm[input_columns_names]
y = df_norm[output_columns_names]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
```
Обучение
```
knn = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [1,2,5,10,20]}
GSCV = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10, verbose=2)
GSCV.fit(x_train, y_train)
GSCV.best_params_

knn.fit(x_train, y_train)
prediction=knn.predict(x_test)
np.mean((y_test - prediction)*(y_test - prediction))

input_columns_names = ['Плотность, кг/м3','модуль упругости, ГПа','Количество отвердителя, м.%','Содержание эпоксидных групп,%_2','Температура вспышки, С_2','Поверхностная плотность, г/м2','Модуль упругости при растяжении, ГПа','Прочность при растяжении, МПа','Потребление смолы, г/м2','Угол нашивки, град','Шаг нашивки','Плотность нашивки']
output_columns_names = ['Соотношение матрица-наполнитель']
x = df_norm[input_columns_names]
y = df_norm[output_columns_names]
```
Просмотр статистики набора данных
```
print(x.shape, y.shape)

seaborn.pairplot(pd.DataFrame(np.column_stack([x, y])), diag_kind='kde')

pd.DataFrame(np.column_stack(([x, y])))
```
## Создание нейросети
```
from keras.models import Sequential
from keras import models

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(64, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs))
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return model
    
model = get_model(12,1)
model.summary()
hist = model.fit(x, y, verbose=0, epochs=2000, validation_data = (x_train, y_train))
Просмотр графика уменьшения ошибки

df = pd.DataFrame(hist.history)

import matplotlib.pyplot as plt

plt.plot(df)
Просмотр точности и потерь на тесте

score = model.evaluate(x_test, y_test, verbose=1)
print('Потери на тесте:', score[0])
print('Точность на тесте:', score[1])
```
Просмотр рекомендованных данных Соотношения матрица-наполнитель
```
x

prediction = model.predict(x)

prediction
Статистика

np.mean((y-prediction)*(y-prediction), axis=0)
```
## Сохранение модели

model_path = 'C:/Users/Alexandra/Desktop/вкр/models/my_model_2'
model.save(model_path)
