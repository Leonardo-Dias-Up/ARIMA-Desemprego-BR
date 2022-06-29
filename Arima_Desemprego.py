import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = (15,6) 
import math
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
#from pmdarima.arima import auto_arima => Não Usar!
from ThymeBoost import ThymeBoost as tb
from statsmodels.tsa.api import VAR
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf
import datetime

TxPopDesemp = pd.read_excel(r"D:\07. UFOP\Monetaria I\Trabalho Final\01. PopDesocupada.xlsx")

TxPopDesemp = TxPopDesemp.set_index(pd.to_datetime(TxPopDesemp['Data']))
TxPopDesemp = TxPopDesemp.drop(columns='Data')

TxPopDesemp.plot()
#Tratando o indice DESEMPREGO 2012.03 a 2022.04
lista_index11 = TxPopDesemp.index
type(lista_index11)
var11=[]
for i in lista_index11:
    #print(i)
    var11.append(i)
type(var11)

#Derivada 1ª
var11.remove(var11[0])
#var.remove(var[0])
lista_index11 = pd.DataFrame(var11)
lista_index11 = lista_index11.rename(columns={0: "Time"})
lista_index11 = lista_index11.set_index(pd.to_datetime(lista_index11['Time']))
type(lista_index11.index)

#Derivada 2ª
var12 = var11
var12.remove(var12[0])
#var.remove(var[0])
lista_index12 = pd.DataFrame(var12)
lista_index12 = lista_index12.rename(columns={0: "Time"})
lista_index12 = lista_index12.set_index(pd.to_datetime(lista_index12['Time']))
type(lista_index12.index)

#TxPopDesemp normal e com 1 derivada
test_TxPopDesemp = TxPopDesemp.iloc[:,1]
#Criando as Derivadas do PIB
diff_TxPopDesemp = np.diff(test_TxPopDesemp)
#1ª derivada
diff_TxPopDesemp = pd.DataFrame(diff_TxPopDesemp)
diff_TxPopDesemp.columns = ['diff_TxPopDesemp']
diff_TxPopDesemp.insert(1,"Time",lista_index11,True)
diff_TxPopDesemp = diff_TxPopDesemp.set_index('Time')

#2ª derivada
diff_TxPopDesemp2 = np.diff(diff_TxPopDesemp['diff_TxPopDesemp'])
diff_TxPopDesemp2 = pd.DataFrame(diff_TxPopDesemp2)
diff_TxPopDesemp2.columns = ['diff_TxPopDesemp2']
diff_TxPopDesemp2.insert(1,"Time",lista_index12,True)
diff_TxPopDesemp2 = diff_TxPopDesemp2.set_index('Time')

#Autocorrelação Diferença 1ª Selic
plot_acf(diff_TxPopDesemp)
#Partial Autoorrelação Diferença 1ª Selic
plot_pacf(diff_TxPopDesemp)

#Autocorrelação Diferença 2ª Selic
plot_acf(diff_TxPopDesemp2)
#Partial Autoorrelação Diferença 2ª Selic
plot_pacf(diff_TxPopDesemp2)

#Construindo o modelo
model = ARIMA(test_TxPopDesemp, order=(2,2,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

resid = DataFrame(model_fit.resid)
resid.plot()

resid.plot(kind='kde')
plt.show()
print(resid.describe())

# =============================================================================
# #Construindo o modelo ARIMA
# =============================================================================
test_TxPopDesemp = test_TxPopDesemp.iloc[::-1]
X = test_TxPopDesemp.values
size = int(len(X)*0.75)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2,2,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('Previsto=%f, Esperado=%f', yhat, obs)

plt.plot(test, color='blue')
plt.plot(predictions, color='purple')
plt.show()

test_TxPopDesemp.plot()

# =============================================================================
# #Construindo a previsão Bai-Parron
# =============================================================================

Inver_TxPopDesemp = test_TxPopDesemp.iloc[:-1]
y = Inver_TxPopDesemp
y = y.to_numpy()
y
#seasonality 
#Otimização 1
boosted_model = tb.ThymeBoost(
                            approximate_splits=True,
                            verbose=1,
                            cost_penalty=.001,
                            n_rounds=2
                            )

output = boosted_model.fit(y,
                           trend_estimator='arima',
                           arima_order=(2, 2, 2),
                           seasonal_estimator='fourier',
                           seasonal_period=25,
                           split_cost='mae',
                           global_cost='maicc',
                           fit_type='global',
                           seasonality_lr=.1
                           )
predicted_output = boosted_model.predict(output, 100)
boosted_model.plot_results(output, predicted_output)

#Otimização 2
boosted_model = tb.ThymeBoost(
                           approximate_splits=True,
                           verbose=0,
                           cost_penalty=.001,
                           )

output = boosted_model.optimize(y, 
                                verbose=1,
                                lag=20,
                                optimization_steps=1,
                                trend_estimator=['mean', 'linear', ['mean', 'linear']],
                                seasonal_period=[0, 25],
                                fit_type=['local', 'global'])

predicted_output = boosted_model.predict(output, 100)
boosted_model.plot_results(output, predicted_output)

#Otimização 3
boosted_model = tb.ThymeBoost(
                           approximate_splits=True,
                           verbose=0,
                           cost_penalty=.001,
                           )

output = boosted_model.optimize(y, 
                                lag=10,
                                optimization_steps=1,
                                trend_estimator=['mean', boosted_model.combine(['ses', 'des', 'damped_des'])],
                                seasonal_period=[0, 25],
                                fit_type=['global'])

predicted_output = boosted_model.predict(output, 100)
boosted_model.plot_results(output, predicted_output)

output.plot()
predicted_output.plot()
predicted_output.head()

backcasting = predicted_output.head(26)
backcasting.plot()
backcasting.columns
backcasting = backcasting.drop(columns=['predicted_exogenous'])
backcasting.head()

backcasting1 = predicted_output.head(100)
backcasting1.plot()
# =============================================================================
# #Criando o DataFrame Previsto        
# =============================================================================
date_list = pd.period_range(start='2010-01-01', end='2012-02-01', freq='M')
date_list = list(date_list)            

var = []
for i in date_list:
    i = i.to_timestamp()
    var.append(i)

data = pd.DataFrame(var)
data = data.rename(columns={0:'Data'})
data.head()
data.tail()

backcasting = backcasting.set_index(data['Data'])
backcasting.tail()

# =============================================================================
# Análisando as Previsões Geradas
# =============================================================================
test_TxPopDesemp.tail()
backcasting.head()

#Correlação
corr_df = backcasting.corr(method='pearson')
plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.title('Backcasting')
plt.show()

# plot BoxPlot
ax = sns.boxplot(data=backcasting, orient='h', width=0.5)
ax.figure.set_size_inches(12, 6)
ax.set_title('Box plot', fontsize=20)
ax.set_xlabel('Values', fontsize=16)
ax

# =============================================================================
# # Nova Série Temporal - Taxa de Desemprego
# =============================================================================
backcasting = backcasting['predictions']
b = test_TxPopDesemp.append(backcasting)
df = b
df = pd.DataFrame(df,columns=['Tx Desemprego'])
df.plot()

#df.to_excel(r'D:\07. UFOP\Monetaria I\Trabalho Final\Base de Dados\01. Tx Desemprego Forecasting.xlsx')

#Analisando Estatística Descritiva Novos Dados
df.describe()
ax = sns.boxplot(data=df, orient='h', width=0.5)
ax.figure.set_size_inches(12, 6)
ax.set_title('Box plot', fontsize=20)
ax.set_xlabel('Values', fontsize=16)
ax

#Analisando Estatística Descritiva Antigos Dados
test_TxPopDesemp.describe()
ax = sns.boxplot(data=test_TxPopDesemp, orient='h', width=0.5)
ax.figure.set_size_inches(12, 6)
ax.set_title('Box plot', fontsize=20)
ax.set_xlabel('Values', fontsize=16)
ax


#Plotando novas Boxplot Junto com Antiga
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10,5))
fig.suptitle('Boxplot Desemprego')
# Backcasting
sns.boxplot(ax=axes[0],y=df.values)
axes[0].set_title('Backcasting')
# test_TxPopDesemp
sns.boxplot(ax=axes[1], y=test_TxPopDesemp.values)
axes[1].set_title('Série Original')
















