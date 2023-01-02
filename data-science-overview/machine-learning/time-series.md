# Time Series

# What?

- weather, music, stock market, covid
- trends in repeated observations over a regular (equal) time interval
    - linear, exponential
- **Understand**: decompose, explain behavior
- **Forecast**: predict future values based on past/cuurent observations

### Structure

- semi-structured: order doesn’t matter
- **highly-structured**: order matters
    - using date as an index

# Classic Machine Learning Approach

### (X, y)

- y = df.values
- X = past y values
- row/observation :

    ```python
    X[t] = [y[t-1], y[t-2],...]
    # t-1 = yesterday, t-2 = day before
    ```


### train/test split

- test data has to have happened after training data (chronological)
    - can’t use holdout, must use contiguous

```python
# keep the last 40% of the values out for testing
train_size = 0.6
index = round(train_size*df.shape[0])

df_train = df.iloc[:index]
df_test = df.iloc[index:]
```

### Predict

- baseline: X[t] = y[t-1] (no feature)

```python
y_pred = df_test.shift(1)
y_pred
```

```python
from sklearn.metrics import r2_score

y_pred = df_test.shift(1).dropna()
y_true = df_test[1:]

print(f"R2: {r2_score(y_true, y_pred)}")
#>> R2: 0.5069517261286796
```

### Linear model with 12 autoreg featues (t - n)

```python
# build dataset
df2 = df.copy(); df2_train = df_train.copy(); df2_test = df_test.copy()

for i in range(1, 13):
    df2_train[f't - {i}'] = df_train['value'].shift(i)
    df2_test[f't - {i}'] = df_test['value'].shift(i)

df2_train.dropna(inplace=True)
df2_test.dropna(inplace=True)

df2_train.head()
```

```python
# Train Test Split
X2_train = df2_train.drop(columns = ['value'])
y2_train = df2_train['value']
X2_test = df2_test.drop(columns = ['value'])
y2_test = df2_test['value']

print(X2_train.shape,y2_train.shape, X2_test.shape,y2_test.shape)
#>> (110, 12) (110,) (70, 12) (70,)
```

```python
# Predict and measure R2
model = LinearRegression()
model = model.fit(X2_train, y2_train)

print('R2: ', r2_score(y2_test, model.predict(X2_test)))
pd.Series(model.coef_).plot(kind='bar')
plt.title('partial regression coefficients');
#>> R2:  0.8580874548863759
```

- Only the 12th coefficient conveys information to the model (a Linear Regression based on these 12 autoregressive features)
- Except for the 12th coefficient, most of the prediction is done by
the intercept (basically predicting the mean of the last 12 values)
- All the other features are statistically insignificant (high p-values)

### Predicting a longer time horizon

- We need to train **one model per forecast horizon!**
- model performance drops quickly as the horizon increases
- the more features, the longer it will take to drop off

# Decomposition

1. Trend
    1. i.e. mean of “cycle” over time
2. Seasonal (calendar) or Periodic (non-calendar)
    1. i.e. cyclical trend
3. Irregularities
    1. i.e. outliers
- multiplicative model has “less notion of time” (better)
    - Models work best when forecasting TS that **do not exhibit meaningful statistical changes over time** (so that we can capture these statistical properties and project/extrapolate them into the future)
- forecasting methods are designed to work on stationary TS
    1. convert non-stationary → stationary
    2. forecast by extrapolating stationary properties
    3. reintroduce seasonality etc.

### Code

```python
# Additive Decomposition (y = Trend + Seasonal + Residuals)
result_add = seasonal_decompose(df['value'], model='additive')
result_add.plot();
```

```python
# Multiplicative Decomposition (y = Trend * Seasonal * Residuals)
result_mul = seasonal_decompose(df['value'], model='multiplicative')
result_mul.plot();
```

```python
# Plot the residuals with "result_add.resid" to decide
f, (ax1, ax2) = plt.subplots(1,2, figsize=(13,3))
ax1.plot(result_add.resid); ax1.set_title("Additive Model Residuals")
ax2.plot(result_mul.resid); ax2.set_title("Multiplicative Model Residuals");
```

# Stationarity

- when time does not influence the stat properties of a dataset
    - mean
    - variance
    - auto-correlation (covariance with lagged terms)
- distribution is affected only by the size of the time window, **not** the location of a time window

```python
# Stationary TS with small autocorrelation but stronger "variance"
plot_stationary_ts(ar1=0.4, sigma=3);
```

```python
# White noise has no information to extract!
plt.figure(figsize=(20,5));
plt.plot(np.arange(500), [scipy.stats.uniform().rvs() for i in np.arange(500)]);
```

### Test

1. visually
2. calculate (mean, variance, auto-correlation) in various intervals
3. Augmented Dickey Fuller ADF test (p-values)
    1. null hypothesis: $H_0$ : the series in snot stationary
    2. A `p-value` close to 0 (e.g. p < 0.05) indicates stationarity

    ```python
    from statsmodels.tsa.stattools import adfuller

    adfuller(df.value)[1]  # p-value
    #>> 1.0
    ```

    ```python
    print('additive resid: ', adfuller(result_add.resid.dropna())[1])
    print('multiplicative resid: ', adfuller(result_mul.resid.dropna())[1])
    #>> additive resid:  0.00028522210547377003
    #>> multiplicative resid:  1.747259579533223e-07
    ```


### **Different ways to achieve stationarity**

- decomposition
- differencing
- transformations (log, exp)

    ```python
    df.value.diff(12) # can remove seasonality
    # (one year in DF with 1 measurement per month)
    ```


# Autocorrelation

- correlation between TS Y(t) and a lagged version of itself Y(t-i)

```python
from scipy.stats import pearsonr
pearsonr(sf.value[1:],df.value.shift(5).dropna())
pearsonr(sf.value[10:],df.value.shift(5).dropna())
```

### Autocorrelation

- (**Auto Regression**) AR
    - linear combo of p lags
        - values are direct linear combo of past values
        - one time “shock” will propagate far into the future (sensitive to outliers)
        - not necessarily stationary
- ACF (autocorrelation graph)
- blue cone represents a confidence interval (the default is 95%)
    - Peak inside of cone ➔ not statistically significant
- Peaks every 12 months indicate the seasonal nature of weather data
- general decline in correlation between a specific time and a day further in the past; the further back in time you look, the less likely it is that that shift will have an influence
- autocorrelation doesn't just measure the **direct** effect of a lagged point in time... it measures the **indirect** effect **as well**!
    - so we have to analyze each lag in isolation:

### Partial Autocorrelation

- **Partial Autocorrelation Function** (PACF)
    - much sharper decline with PACF lag values as opposed to ACF
- to influence a specific time lag:
    - LINREG, p = total number of lags
- **Compute lagged coeff $(ß_i)$**
    - **Weight of the lin reg**
    - if it’s larger than 1, it’s because the weight includes a positive trend, which doesn’t work to make a series stationary
    - BUT butterfly effect can make it go negative if there is a fluctuation
        - BECAUSE it’s very sensitive to outliers
- ACF is simply the correlation of the series with itself
    - slow exponential decrease
    - if X(t) is always correlated with X(t−1), then it is also correlated with X(t−2)
- PACF is even more informative
    - it removes intermediary correlations

### Moving Averages

- Predicting the next value based on the average value of a specific window size
    - linearly model the **errors** between our moving average predictions and the actual values
        - linear combo of q residuals
        - values are a direct linear combination of its past changes
        - **any "shock" will have a limited time effect (not sensitive to outliers)**
        - always stationary

# Auto Regressive Moving Average (ARMA)

- combine auto regressive (AR and moving average models into one
    - linear combo of p lags of Y + linear combo of q lagged prediction errors
    - p: lags in AR
    - q: lags in MA

```python
ar_params = np.array([1.35, -.35]) # beta
ma_params = np.array([.65, .95]) # phi
```

```python
from statsmodels.tsa.arima_process import ArmaProcess

ar = np.r_[1, -ar_params] # add zero-lag and negate (this is how ArmaProcess needs to be coded)
ma = np.r_[1, ma_params] # add zero-lag
arma_process = ArmaProcess(ar, ma)
```

```python
np.random.seed(1)
y = arma_process.generate_sample(200)
```

```python
from statsmodels.tsa.stattools import adfuller
adfuller(y)[1]
# want to be close to zero
```

- Count the number of lags before the values drop below the confidence levels
    - Note that the first lag (0) is ignored, as it represents auto- and partial autocorrelation between Y(t) and itself
    - PACF plot to see Pearson correlation value for p above cone

        ```python
        from statsmodels.graphics.tsaplots import plot_pacf

        plot_pacf(df.value, lags=50, c='r');
        ```

    - ACF plot to see Pearson correlation value for q above cone

        ```python
        from statsmodels.graphics.tsaplots import plot_acf

        plot_acf(df.value, lags=50) # lag - steps in the past
        ```


# Auto Reg. Integrated Moving Avg. (ARIMA)

- We can apply **differencing** to our dataset to turn non-stationary data into stationary data
    - plot the **difference** between each lag rather than the value for each lag
    - **predict the trend in differences instead of the trend of the values**

```python
non_differenced_data = pd.Series([7, 4, 4, 6, 8, 6])
differenced_1 = non_differenced_data.diff()
print(differenced_1)
```

```python
print(differenced_1.diff())
```

- first order diff:
    - predicting *growth* instead of the values itself
    - an advance in terms of *growth* may not propagate as far into the future as values do
    - *differencing* tends to turn AR processes into MA ones
    - usually 1-diff is enough, if not, you might have exponential behavior
        - use log transformation or de-trend
        - OR add negative ar2

### Make values stationary via differencing

- using seasonal data with exponential trend

```python
zero_diff = df.value
first_order_diff = df.value.diff(1)
second_order_diff = df.value.diff(1).diff(1)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))
ax1.plot(zero_diff); ax1.set_title('Original Series')
ax2.plot(first_order_diff); ax2.set_title('1st Order Differencing')
ax3.plot(second_order_diff); ax3.set_title('2nd Order Differencing');
```

- seasonality is obviously present
- **Cannot apply ARIMA** directly (we will see SARIMA later)
- *de-seasonalize* our Time Series first, using our *decomposition* tool

```python
# Let's remove seasons
df['deseasonalized'] = df.value.values/result_mul.seasonal

plt.figure(figsize=(15,4)); plt.subplot(1,2,1); plt.plot(df.deseasonalized);
plt.title('Drug Sales Deseasonalized', fontsize=16);

# Also remove exponential trend
df['linearized'] = np.log(df['deseasonalized'])

plt.subplot(1,2,2); plt.plot(df['linearized'])
plt.title('Drug Sales Linearized', fontsize=16);
```

```python
# Let's difference this and look at the ACFs
fig, axes = plt.subplots(1, 3,figsize=(15,4))

axes[0].plot(df['linearized']); axes[0].set_title('Linearized Series')

# 1st Differencing
y_diff = df['linearized'].diff().dropna()
axes[1].plot(y_diff); axes[1].set_title('1st Order Differencing')

# 2nd Differencing
y_diff_diff = df['linearized'].diff().diff().dropna()
axes[2].plot(y_diff_diff); axes[2].set_title('2nd Order Differencing');
```

```python
# check with ADF Test for stationarity
print('p-value zero-diff: ', adfuller(df['linearized'])[1])
print('p-value first-diff: ', adfuller(df['linearized'].diff().dropna())[1])
print('p-value second-diff: ', adfuller(df['linearized'].diff().diff().dropna())[1])
#>> p-value zero-diff:  0.7134623265852265
#>> p-value first-diff:  1.0092820652730429e-09
#>> p-value second-diff:  1.3181782398644668e-12
```

- don’t over-difference: d=1

```python
# automatically estimate differencing term
from pmdarima.arima.utils import ndiffs
ndiffs(df['linearized'])
#>> 1
```

### Hyperparams of p&q

```python
# ACF / PACF analysis of y_diff linearized
fig, axes = plt.subplots(1,3, figsize=(16,2.5))
axes[0].plot(y_diff); axes[0].set_title('1st Order Differencing')
plot_acf(y_diff, ax=axes[1]);
plot_pacf(y_diff, ax=axes[2], c='r');
```

Use the **Box-Jenkins method**

- IMPORTANT: more than one model might explain your data.

```python
# from statsmodels.tsa.arima_model import ARIMA #statsmodels 0.11
from statsmodels.tsa.arima.model import ARIMA  #statsmodels 0.12+

arima = ARIMA(df['linearized'], order=(2, 1, 1), trend='t')
arima = arima.fit()
```

```python
arima.summary()
# Covariance matrix calculated using the outer product of gradients (complex-step)
```

### Performance Metrics - Akaike information Criterion (AIC)

- lower = better

```python
import pmdarima as pm
smodel = pm.auto_arima(df['linearized'],
                       start_p=1, max_p=2,
                       start_q=1, max_q=2,
                       trend='t',
                       seasonal=False,
                       trace=True)
```

### **Evaluate Performance**

```python
from statsmodels.graphics.tsaplots import plot_predict

fig, axs = plt.subplots(1, 1, figsize=(12, 5))
axs.plot(df['linearized'], label='linearized')
plot_predict(arima, start=1, end=250, ax=axs);
```

```python
# Create a correct train_test_split to predict the last 50 points
train = df['linearized'][0:150]
test = df['linearized'][150:]

# Build model
arima = ARIMA(train, order=(2, 1, 0), trend='t')
arima = arima.fit()

## Forecast
# Forecast values
forecast = arima.forecast(len(test), alpha=0.05)  # 95% confidence

# Forecast values and confidence intervals
forecast_results = arima.get_forecast(len(test), alpha=0.05)
forecast = forecast_results.predicted_mean
confidence_int = forecast_results.conf_int().values
```

```python
# We define here a "Plot forecast vs. real", which also shows historical training set

def plot_forecast(fc, train, test, upper=None, lower=None):
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    # Prepare plot series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    # Plot
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(train, label='training', color='black')
    plt.plot(test, label='actual', color='black', ls='--')
    plt.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8);
```

```python
plot_forecast(forecast, train, test, confidence_int[:,0], confidence_int[:,1])
```

(this wasn't our original Time Series)

```python
# Re-compose back to initial TS

forecast_recons = np.exp(forecast) * result_mul.seasonal[150:]
train_recons = np.exp(train) * result_mul.seasonal[0:150]
test_recons = np.exp(test) * result_mul.seasonal[150:]
lower_recons = np.exp(confidence_int)[:, 0] * result_mul.seasonal[150:]
upper_recons = np.exp(confidence_int)[:, 1] * result_mul.seasonal[150:]

# Plot
plot_forecast(forecast_recons, train_recons, test_recons, lower_recons.values, upper_recons.values)
```

### **Check residuals for inference validity**

```python
residuals = pd.DataFrame(arima.resid)

fig, ax = plt.subplots(1,2, figsize=(16,3))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1]);
```

- we can trust out confidence interval if:
    - equal variance over time (not homoscedastic)
    - approx normal distribution

### Box Jenkins Method (details)

1. Apply transformations to make Time Series stationary (*de-trending*, transformations, differencing)
2. If differencing, keep track of order `d` of differencing; don't over-difference!
3. Confirm stationarity (visually, ACF, ADF test)
4. Plot ACF/PACF, identify likely AR and MA orders `p`, `q`.
5. Fit **original** (non-differenced) data with ARIMA model of order `p`, `d`, `q`
6. Try a few other values around these orders
7. If all models with similarly low AIC, pick the least complex one
8. Inspect residuals: if ACF and PACF show white noise (no signal left) ➔ you are done.
9. Otherwise ➔ iterate (try other transformations, change order of differencing, add/remove MA/AR terms)
10. (Optional) Compare with Auto-ARIMA output trace

# Seasonal ARIMA (SARIMA)

- Removes the need to *de-seasonalize* our data
- window size (12 for yearly)
- hyperparams: `SARIMA(p,d,q)(P,D,Q)[S]` (7)

    *p* ~ AR hyperparameter for individual lag level

    *d* ~ Integration hyperparameter for individual lag level

    *q* ~ MA hyperparameter for individual lag level

    *P* ~ AR hyperparameter for seasonal lag level

    *D* ~ Integration hyperparameter for seasonal lag level

    *Q* ~ MA hyperparameter for seasonal lag level

    S - choose manually (S = 12, for annual seasonality)


```python
fig, axs = plt.subplots(2, 2, figsize=(18,8))
# keeping just log transform to stay ~ linear
df['log'] = np.log(df.value)

# linearized series
axs[0,0].plot(df.log); axs[0,0].set_title('linearized Series')

# Normal differencing
axs[0,1].plot(df.log.diff(1)); axs[0,1].set_title('1st Order Differencing')

# Seasonal differencing
axs[1,0].plot(df.log.diff(12))
axs[1,0].set_title('Seasonal differencing of period 12')

# Sesonal + Normal differencing
axs[1,1].plot(df.log.diff(12).diff(1))
axs[1,1].set_title('First order diff of seasonal differencing 12');
```

```python
# Create a correct Training/Test split to predict the last 50 points
train = df.log[0:150]
test = df.log[150:]

smodel = pm.auto_arima(train, seasonal=True, m=12,
                       start_p=0, max_p=1, max_d=1, start_q=0, max_q=1,
                       start_P=0, max_P=2, max_D=1, start_Q=0, max_Q=2,
                       trace=True, error_action='ignore', suppress_warnings=True)
#>> Best model:  ARIMA(0,1,1)(1,0,2)[12]
```

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Build Model
sarima = SARIMAX(train, order=(0, 1, 1), seasonal_order=(2, 0, 2, 12))
sarima = sarima.fit(maxiter=75)

# Forecast
results = sarima.get_forecast(len(test), alpha=0.05)
forecast = results.predicted_mean
confidence_int = results.conf_int()
```

```python
# Reconstruct by taking exponential
forecast_recons = pd.Series(np.exp(forecast), index=test.index)
lower_recons = np.exp(confidence_int['lower log']).values
upper_recons = np.exp(confidence_int['upper log']).values

plot_forecast(forecast_recons, np.exp(train), np.exp(test), upper = upper_recons, lower=lower_recons)
```

### **SARIMA(X) for eXogenous features**

```python
# Let's download a 'dummy exogenous variable'
df_exog = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')
# Compute Seasonal Index
from dateutil.parser import parse

# multiplicative seasonal component
result_mul = seasonal_decompose(df_exog['value'][-36:],   # 3 years
                                model='multiplicative',
                                extrapolate_trend='freq')

seasonal_index = result_mul.seasonal[-12:].to_frame()
seasonal_index['month'] = pd.to_datetime(seasonal_index.index).month

# merge with the base data
df['month'] = df.index.month
df_augmented = pd.merge(df, seasonal_index, how='left', on='month')
# df_augmented.columns = ['value', 'month', 'seasonal_index']
df_augmented.index = df.index  # reassign the index.
df_augmented.drop(columns='month', inplace=True)
```

```python
# Auto-fit the best SARIMAX with help from this exogenous time series
import pmdarima as pm
sarimax = pm.auto_arima(df[['value']], exogenous=df_augmented[['seasonal']],
                           start_p=0, start_q=0,
                           test='adf',
                           max_p=2, max_q=2, m=12,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           suppress_warnings=True,
                           stepwise=True)
```

### Other mods:

- Exponential Smoothing (ETS): moving average with exponentially
decaying memory of past values; great for non-linear trends with
changing mean over time
- Holt's Trend-Corrected Exponential Smoothing: series has a linear trend with a slope that changes over time
- Holt-Winter's method: adds seasonality

# Challenges

### $ß_1$

- 0 → white noise
- [0-1] → sticky time series regressing to the mean
- 1 → linear trend
- >1 → exponential trend
- <0 (negative) → oscillations
- >-1 → exponential oscillations

### 1

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
```

```python
def plot_ar_process(ar1, ar2=0, diff=0, n=500, mu=0, sigma=1, show_plot=True, return_y=False):
    '''
    Plot an auto-regressive time series, as well as its ACF/PACF plots
    '''
    X=np.arange(n)
    y_list = []
    y0 = 0
    y1 = 0
    for i in range(n):
        # build an AR process of params (beta_1, beta_2) = (ar1, ar2)
        # With noise epsilon(t) = Normal(mu, sigma)
        y_new = ar1 * y1 + ar2* y0 + scipy.stats.norm.rvs(mu,sigma)
        y0 = y1
        y1 = y_new.copy()
        y_list.append(y0)
    if diff > 0:
        for i in range(diff):
            y_list = list(pd.Series(y_list).diff())
        y_list = y_list[diff:]
        X = X[diff:]
    if show_plot:
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(2,1,1)
        ax1.set_title(f'Auto-regressive stationary TS with lagged coefs = ({ar1}, {ar2})')
        ax2 = fig.add_subplot(2,2,3)
        ax3 = fig.add_subplot(2,2,4)
        ax1.plot(X,y_list)
        plot_acf(y_list, lags=50, auto_ylims=True,ax=ax2);
        plot_pacf(y_list, lags=50, method='ywm', auto_ylims=True, ax=ax3, color='r');
        plt.show()
    if return_y:
        return y_list
```

```python
np.random.seed(1)
plot_ar_process(ar1=0.5, ar2=0.5, n=1000, diff=1)
```

---

```python
def ma_process(coef_list, n=200, show_plot=True, return_y=False):
    '''
    Generates an MA process from prediction error normally distributed
    '''

    X = np.arange(n)

    coef_list = [1]+coef_list
    coefs = np.asarray(coef_list)
    n_coef = len(coefs)

    noise_size = n + len(coefs)
    noise = np.random.normal(size=noise_size)

    # correlating random values with its immediate neighbors
    y_list = np.convolve(noise, coefs[::-1])[(n_coef-1):(n+n_coef-1)]

    if show_plot:
        fig = plt.figure(figsize=(20,10))

        ax1 = fig.add_subplot(2,1,1)
        ax1.set_title(f'MA TS with lagged coefs = {coef_list}')

        ax2 = fig.add_subplot(2,2,3)
        ax3 = fig.add_subplot(2,2,4)

        ax1.plot(X,y_list)

        plot_acf(y_list, lags=20, auto_ylims=True, ax=ax2);
        plot_pacf(y_list, lags=20, method='ywm', auto_ylims=True,ax=ax3, color='r');

        plt.show()

    if return_y:
        return y_list
```

```python
ma_process(coef_list=[0.8, 0.2, 0.7], n=200)
```

---

```python
ar_params = np.array([1.35, -.35]) # beta
ma_params = np.array([.65, .95]) # phi

ar = np.r_[1, -ar_params] # add zero-lag and negate (this is how ArmaProcess needs to be coded)
ma = np.r_[1, ma_params] # add zero-lag
arma_process = ArmaProcess(ar, ma)
y = arma_process.generate_sample(200)
```

```python
adfuller(y)[1]
```

```python
X = np.arange(200)
fig = plt.figure(figsize=(10,4))
plt.plot(X,y)
```

```python
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
plot_acf(y, lags=20, auto_ylims=True, ax=ax1);
plot_pacf(y, lags=20, method='ywm', auto_ylims=True,ax=ax2, color='r');
```

```python
arima = ARIMA(y, order=(2, 0, 1))
arima = arima.fit()
arima1.summary()
```

---

```python
model = pm.auto_arima(
    y,
    start_p=0, start_q=0,
    max_p=3, max_q=3, # maximum p and q
    d = None,          # no diff
    seasonal=False
)   # no seasonality
```

```python
model.summary()
#>> model: SARIMAX(3,1,2)
```

### 2

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.tsaplots import plot_predict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate
```

```python
!curl https://wagon-public-datasets.s3.amazonaws.com/05-Machine-Learning/09-Time-Series/www_usage.csv > data/www_usage.csv
```

```python
df = pd.read_csv('data/www_usage.csv', names=['value'], header=0)
y = df.value

df.plot();
```

```python
adfuller(y)[1]
```

```python
zero_diff = y
first_order_diff = y.diff(1).dropna()
second_order_diff = first_order_diff.diff(1).dropna()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,4))
ax1.plot(zero_diff); ax1.set_title('Original Series')
ax2.plot(first_order_diff); ax2.set_title('1st Order Differencing')
ax3.plot(second_order_diff); ax3.set_title('2nd Order Differencing');
print(adfuller(first_order_diff)[1])
my_diff = 2
```

```python
fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
plot_acf(first_order_diff, lags=50, auto_ylims=True, ax=ax1)
plot_acf(second_order_diff, lags=50, auto_ylims=True, ax=ax2)
plt.show()
```

```python
y_diff = y.diff(1).dropna()
d = 1
```

```python
plot_acf(y_diff)
plt.show()
q = 2
```

```python
plot_pacf(y_diff)
plt.show()
p = 1
```

```python
arima = ARIMA(y, order=(1, 1, 1), trend = 't')
arima = arima.fit()
arima.summary()
```

```python
fig, axs = plt.subplots(1, 1, figsize=(12, 5))
axs.plot(y, label='y')
plot_predict(arima, ax=axs, dynamic=False);
plt.ylim((-200,300))
```

```python
fig, axs = plt.subplots(1, 1, figsize=(12, 5))
axs.plot(y, label='y')
plot_predict(arima, start=85, ax=axs, dynamic=True);
plt.show()
```

```python
train = y[:85]
test = y[85:]

arima = ARIMA(train, order=(1,1,1), trend='t')
arima = arima.fit()
```

```python
forecast_results = arima.get_forecast(len(test), alpha=0.05)
forecast = forecast_results.predicted_mean
confidence_int = forecast_results.conf_int().values
```

```python
def plot_forecast(fc, train, test, upper=None, lower=None):
    is_confidence_int = isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray)
    # Prepare plot series
    fc_series = pd.Series(fc, index=test.index)
    lower_series = pd.Series(upper, index=test.index) if is_confidence_int else None
    upper_series = pd.Series(lower, index=test.index) if is_confidence_int else None

    # Plot
    plt.figure(figsize=(10,4), dpi=100)
    plt.plot(train, label='training', color='black')
    plt.plot(test, label='actual', color='black', ls='--')
    plt.plot(fc_series, label='forecast', color='orange')
    if is_confidence_int:
        plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
    plt.title('Forecast vs Actuals')
    plt.legend(loc='upper left', fontsize=8);
```

```python
plot_forecast(forecast, train, test, confidence_int[:,0], confidence_int[:,1])
```

```python
residuals = pd.DataFrame(arima.resid)

fig, ax = plt.subplots(1,2, figsize=(16,3))
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1]);
```

```python
import numpy as np
from statsmodels.tsa.stattools import acf

def forecast_accuracy(y_pred: pd.Series, y_true: pd.Series) -> float:

    mape = np.mean(np.abs(y_pred - y_true)/np.abs(y_true))  # Mean Absolute Percentage Error
    me = np.mean(y_pred - y_true)             # ME
    mae = np.mean(np.abs(y_pred - y_true))    # MAE
    mpe = np.mean((y_pred - y_true)/y_true)   # MPE
    rmse = np.mean((y_pred - y_true)**2)**.5  # RMSE
    corr = np.corrcoef(y_pred, y_true)[0,1]   # Correlation between the Actual and the Forecast
    mins = np.amin(np.hstack([y_pred.values.reshape(-1,1), y_true.values.reshape(-1,1)]), axis=1)
    maxs = np.amax(np.hstack([y_pred.values.reshape(-1,1), y_true.values.reshape(-1,1)]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(y_pred-y_true, fft=False)[1]                      # Lag 1 Autocorrelation of Error

    forecast = ({
        'mape':mape,
        'me':me,
        'mae': mae,
        'mpe': mpe,
        'rmse':rmse,
        'acf1':acf1,
        'corr':corr,
        'minmax':minmax
    })

    return forecast
```

```python
forecast_accuracy(forecast, test)
```

```python
import pmdarima as pm

model = pm.auto_arima(
    train,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,           # let model determine 'd'
    test='adf',       # using adf test to find optimal 'd'
    trace=True, error_action='ignore',  suppress_warnings=True
)

print(model.summary())
```

### 3

- m=12
- d=0
- D=1
- q=0
- p=0

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
```

```python
!curl https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly_champagne_sales.csv > data/monthly_champagne_sales.csv
```

```python
df = pd.read_csv("data/monthly_champagne_sales.csv", parse_dates=['Month'], index_col='Month')
df.head()
```

```python
result_add = seasonal_decompose(df['Sales'], model='additive')
result_add.plot();
result_mul = seasonal_decompose(df['Sales'], model='multiplicative')
result_mul.plot();
```

```python
f, (ax1, ax2) = plt.subplots(1,2, figsize=(13,3))
ax1.plot(result_add.resid); ax1.set_title("Additive Model Residuals")
ax2.plot(result_mul.resid); ax2.set_title("Multiplicative Model Residuals");
```

```python
df_train = df.loc[:'1970-01-01']
df_test = df.loc['1970-01-01':]
```

```python
adfuller(df.Sales)[1]
```

```python
fig1 = plt.figure(figsize=(5,5))
plot_acf(df.Sales, lags=20, auto_ylims=True);
```

```python
diff12 = df.Sales.diff(12).dropna()

fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
plot_acf(diff12, lags=20, auto_ylims=True, ax=ax1);
plt.plot(diff12);
adfuller(diff12)[1]
```

```python
diff13 = diff12.diff(1).dropna()
plot_acf(diff13, lags=20, auto_ylims=True);
adfuller(diff13)[1]
```

```python
fig1 = plt.figure(figsize=(10,5))
ax1 = fig1.add_subplot(1,2,1)
ax2 = fig1.add_subplot(1,2,2)
plot_acf(diff12, lags=24, auto_ylims=True, ax=ax1);
plot_pacf(diff12, lags=24, method='ywm', auto_ylims=True,ax=ax2, color='r');
```

```python
import pmdarima as pm

model = pm.auto_arima(
    df_train,
    start_p=0, max_p=1,
    start_q=0, max_q=1,
    start_P=0, max_P=1,
    start_Q=0, max_Q=1,
    seasonal=True,
    m=12,
    d=0,
    D=1,
    nmodels=5,
    n_jobs=-1, trace=True, error_action='ignore',  suppress_warnings=True
)

print(model.summary())
```

```python
from statsmodels.tsa.arima.model import ARIMA

arima = ARIMA(df_train, order=(0,0,0), seasonal_order=(0,1,0,12))
arima = arima.fit(df_train)

predictions = arima.predict(n_periods=len(df_test))
plt.scatter(predictions, df_train)
```

# Recap

   /\__/\

  | ö  ö |

   \  Y  /

      T

    /    \

   |      |

    \/  \/
