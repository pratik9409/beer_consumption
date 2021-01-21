# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:06:21 2020

@author: ADMIN
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import plotly as py
#import plotly.graph_objs as go
import warnings
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split




data = pd.read_csv(r'C:\Users\ADMIN\Desktop\data_sci\regressions\Consumo_cerveja.csv')

data.info()
describe_data=data.describe()

data = data.dropna()

#data.Temperatura_Media_C

data.columns= data.columns.str.replace(' ','_')

data.columns= data.columns.str.replace('[()]','')

data['Temperatura_Media_C']=data['Temperatura_Media_C'].str.replace(',','.').astype(float)

data['Temperatura_Minima_C']=data['Temperatura_Minima_C'].str.replace(',','.').astype(float)

data['Temperatura_Maxima_C']=data['Temperatura_Maxima_C'].str.replace(',','.').astype(float)

data['Precipitacao_mm']=data['Precipitacao_mm'].str.replace(',','.').astype(float)

data['Final_de_Semana']=data['Final_de_Semana'].astype(int)

data_corr=data.corr()

data.info()
plt.figure(figsize=(10,7))

sns.violinplot(x='Final_de_Semana',y='Consumo_de_cerveja_litros',data=data)

plt.title("Beer Consumption by day per week")

plt.xlabel("Weekdays vs Weekends")

plt.ylabel("beer consumption")

plt.show()

#####We can infer that consumption of alcohol increases during weekends

weekdays=sum(data.Consumo_de_cerveja_litros[data.Final_de_Semana==0])/5

weekend = sum(data.Consumo_de_cerveja_litros[data.Final_de_Semana==1])/2

labels = ['Beer Consumption by weekdays','Beer Consumption by Weekend']

values =[weekdays,weekend]


df = pd.DataFrame({'Consumption':[1252.7071999999998,1503.9815000000006]},
                  index=['Weekdays','Weekend'])

plot = df.plot.pie(y='Consumption',autopct='%1.1f%%',figsize=(7,7))

weekdays1 = sum(data.Consumo_de_cerveja_litros[data.Final_de_Semana==0])/261

weekend1 = sum(data.Consumo_de_cerveja_litros[data.Final_de_Semana==1])/104


days = ['Beer Consumption by weekdays in a year','Beer Consumption by Weekend in a year']

values =[weekdays1,weekend1]



plt.figure(figsize = (10,10))
sns.heatmap(data_corr,square=True,annot=True,linewidths=4,linecolor='k')


plt.figure(figsize = (12,6))
sns.distplot(data['Consumo_de_cerveja_litros'],kde=False,bins=20)
plt.xlabel("Consumption of Beer")
plt.title("Destribuicio de frequencia")
plt.grid(linestyle='-.',linewidth = .5)


plt.figure(figsize = (12,6))
sns.boxplot(data['Consumo_de_cerveja_litros'])
plt.xlabel("Consumption of Beer")
plt.xlim(10,40)
plt.grid(linestyle='-.',linewidth = .5)

g = sns.FacetGrid(data, col= 'Final_de_Semana',hue='Final_de_Semana',height=4,aspect = 1.5)
g.map(sns.distplot,'Consumo_de_cerveja_litros',kde=False,bins=20)
g.set_xlabels("Consumption of beer")

group=data.groupby(['Final_de_Semana']).agg({'Consumo_de_cerveja_litros':['count','sum','mean','std','min',\
            'median','max']}).round(3)



sns.lmplot(x='Temperatura_Media_C', y = 'Consumo_de_cerveja_litros',hue= 'Final_de_Semana',data = data,\
           aspect = 1.5,height =6)

plt.xlabel("Temperature Median in C")
plt.ylabel("Consumption of beer in litres")
plt.grid(linestyle='-.',linewidth = .5)

data['Data']= pd.to_datetime(data['Data'])

data['month']= pd.to_datetime(data['Data']).dt.month

data['day'] = pd.to_datetime(data['Data']).dt.dayofweek

data['day_number'] = pd.to_datetime(data['Data']).dt.day

#####Monthly Analysis##########################################################

plt.figure(figsize=(12,7))

sns.boxplot(x='month', y = 'Consumo_de_cerveja_litros', data = data)

plt.xlabel('Month')

plt.ylabel('Consumo_de_cerveja_litros')

plt.title('Consumption of beer on monthly basis')

plt.grid(linestyle='-.',linewidth = .2)

####Putting all the conclusion in a dataframe

group_by_month_rain_weekend=data.groupby(['month','Final_de_Semana']).\
    agg({'Consumo_de_cerveja_litros':['count','sum','mean','std','min','median','max']}).round(3)


f, ax = plt.subplots(1,2,figsize=(16,6))

a = sns.boxplot(x='month',y='Consumo_de_cerveja_litros',hue='Final_de_Semana',data=data,ax=ax[0])

a.set_ylabel('Consumption of beer')

a.set_xlabel('Month')

a.set_title("Consumption of beer/month")

a.grid(linestyle='-.',linewidth = .5)

b= sns.boxplot(x='month',y='Temperatura_Media_C',hue='Final_de_Semana',data=data,ax=ax[1])

b.set_ylabel('Median Temperature')

b.set_xlabel('Month')

b.set_title('Temperature Median/month')

b.grid(linestyle='-.',linewidth = .5)


sns.lmplot(x='Precipitacao_mm',y ='Consumo_de_cerveja_litros', hue ='Final_de_Semana',\
           data= data, aspect = 1.5 , height = 6)
plt.xlabel("Rainfall in mm")
plt.ylabel('Consumption of beer')
plt.xlim(-8,100)
plt.grid(linestyle='-.',linewidth = .5)

plt.figure(figsize=(16,6))

sns.lineplot(x='Data',y='Consumo_de_cerveja_litros',data=data,alpha=0.5)

sns.lineplot(x='Data',y=data['Consumo_de_cerveja_litros'].rolling(15).mean(),data=data,alpha=0.5)

plt.ylabel('Consumption of beer')
plt.grid(linestyle='-.',linewidth = .5)


plt.figure(figsize=(16,6))

sns.heatmap(data.pivot_table(values='Consumo_de_cerveja_litros',index='month',columns='day'),annot=True)
plt.yticks(rotation = 0)


#########################Fitting a regression function######################################

X= data.drop(['Data','Temperatura_Media_C','Consumo_de_cerveja_litros'], axis =1)
Y = data['Consumo_de_cerveja_litros'].values

X.head()

oneH = OneHotEncoder(categorical_features=[3,4,5,6])
sSC = StandardScaler()

X= oneH.fit_transform(X).toarray()
X= sSC.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.25,random_state=42, shuffle = True)


def info(yTest_, pred_,sg = 'X'):
    ''' Return resultados '''
   
def graphics(y, pred):
   
    # scatter
    plt.figure(figsize=(12, 6))

    plt.plot(y,y)
    plt.scatter(pred,y, c='r', marker='o')
    plt.legend(['Actual','Predicted'])
    plt.grid(ls='-.', lw=0.2, c='k');
   
    # distplot
    plt.figure(figsize=(12, 6))    
    sns.distplot(y)
    sns.distplot(pred)
    plt.legend(['Actual','Predicted'])
    plt.grid(ls='-.', lw=0.2, c='k')
   
from sklearn.linear_model import LinearRegression

lR = LinearRegression()

lR.fit(X_train,y_train)

pred_lR = lR.predict(X_test)

info(y_test, pred_lR, 'LinearRegression')

graphics(y_test, pred_lR)
