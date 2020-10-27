from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
register_matplotlib_converters()
from time import time

def parser(s):
    return datetime.strptime(s, '%d/%m/%Y')

df = pd.read_csv("./archive/datos.csv",delimiter=";", parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

print(df.head())
#df.info()
df = df.asfreq(pd.infer_freq(df.index))

start_date = datetime(2017,1,1)
end_date = datetime(2020,1,1)
lim_catfish_sales = df[start_date:end_date]

first_diff = lim_catfish_sales.diff()[1:]


acf_vals = acf(first_diff)
num_lags = 20


pacf_vals = pacf(first_diff)
num_lags = 20


train_end = datetime(2019,7,1)
test_end = datetime(2020,1,1)

train_data = lim_catfish_sales[:train_end]
test_data = lim_catfish_sales[train_end + timedelta(days=1):test_end]
print(train_data)

my_order = (0,1,0)
my_seasonal_order = (1, 0, 1, 7)
# define model
model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)

#fit the model
start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)

## summary of the model
print(model_fit.summary())

#get the predictions and residuals
predictions = model_fit.forecast(len(test_data))
predictions = pd.Series(predictions, index=test_data.index)
residuals = test_data - predictions


print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data)),4))
print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))

print(predictions)





