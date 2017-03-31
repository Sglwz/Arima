from pandas import read_csv
from pandas import datetime
from pandas import DataFrame
from datetime import timedelta
import numpy as np
from pandas import Series
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from matplotlib import pyplot
from pandas.tools.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error


series = pd.read_csv(r'C:\Users\siddiqus\Desktop\Analytics Project\Python\memutil.csv', parse_dates=['Date'], index_col='Date')




rolling = series.rolling(window=3)
rolling_mean = rolling.mean()

rolling_mean.values[0] = series.values[0]
rolling_mean.values[1] = series.values[1]

print(rolling_mean)
series.plot()
rolling_mean.plot(color='red')
pyplot.show()


(smt,p,q,d,p1,q1,d1) = (10000,0,0,0,0,0,0)
for p in range(5):
    for d in range(2):
        for q in range(1):
            try:
                arima_mod=ARIMA(series, order=(p,d,q)).fit(transparams=True)
                aic = arima_mod.aic
                print (aic,p,q,d)
                if (aic<smt):
                    smt =  aic
                    p1=p
                    q1=q
                    d1=d
            except:
                pass

print("The final vlaue of p, q and d are \n")
print(p1,q1,d1)
                



rolling_mean.plot()
pyplot.show();
# fit model
model = ARIMA(rolling_mean, order=(p1,d1,q1)).fit()
print(model.summary())

# plot residual errors
residuals = DataFrame(model.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


me = np.mean(series.head(15))
ta = np.mean(series.tail(15))
print("Hello")
mean = abs(ta-me)
print(mean)
X = series.values
train = X[0:len(X)]
history = [x for x in train]
model = ARIMA(history, order=(p1,d1,q1))
model_fit = model.fit(disp=0)
model_fit.plot_predict(start=0, end=(len(X-1)), exog=None, dynamic=False, alpha=0.05,plot_insample=True, ax=None)
output=model_fit.predict()
datetime = series.index[0]
date=datetime.date()
print(output)
for i in range(len(output)):
    output[i]= output[i] + mean

print(output)

date_index = pd.date_range(date, periods=len(series.values), freq='D')
series = series.reindex(date_index, method=None,copy=True)

Ldatetime = series.index[len(X)-1]
Ldate = Ldatetime.date()
modified_date = Ldate + timedelta(days=1)

date_index = pd.date_range(modified_date, periods=len(output), freq='D')
s2 = pd.Series(output,name = "Pred",index=date_index)



print(s2)
predictions=pd.concat([series, s2], axis=0 )
predictions.columns = ['Original Sales', 'Predictions']
predictions.index_col='Date'

print(predictions)


predictions.plot()
pyplot.show()
df = pd.DataFrame(predictions)
writer = pd.ExcelWriter(r'C:\Users\siddiqus\Desktop\Analytics Project\Python\test1.xlsx', engine='xlsxwriter')
df.to_excel(writer,'Sheet1')
writer.save()