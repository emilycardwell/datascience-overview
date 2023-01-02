# Model Tuning

## Recap from Under the hood..

### **problem setting**

- X = features
- y = target = h(X,β)+error
- h = hypothesis function (Linear, Logistic Regression, ...)

### **Parameters of the model: β**

- computed automatically during `.fit()`
- by minimizing L(β)

### **Hyperparameters of the model** (chosen manually)

- loss function L (MSE, Log-Loss,...)
- loss-parameters of the loss (learning_rate, eta0...)
- solver = method used to miminize L ('newton', 'sdg', ...)
- model-specificities ('n_neighbors', 'kernel' ...)

# Model complexity

- linreg: $Y = ß_0 + ß_1X_1 + ϵ$
- logreg: $Y = ... + ß_2X_1^2 + ... + ϵ$
- more complex: $X^3, X^{11}$

### Overfitting & Underfitting

- under: undersimp, not complex enough (high bias)
- over: models the noise, overcomplex (high variance)
- bias-variance tradeoff
    - lowest error is at the point of convergence
    - $TotalError = Bias^2 + Variance + IrreductibleError$

### Optimization

- minimize error for TEST sample to get best fit
- use a validation set with cross validation to generalize better
- overfitting?
    - more observations/samples
    - feature selection
    - dimensionality reduction
    - early stopping (Deep Learning)
    - regularization of your loss function

# Regularization

- **What**?
    - adding a **penalty term** to the Loss that **increases** with ß
- Regularization *tends to* **penalize** features that are **not statistically significant**
- **When**?
    - **Regularize** when you think you are **overfitting** (e.g. Learning Curves not converging)
    - **Ridge** when you believe all coefficients may have an impact
    - **Lasso** as a feature selection tool (much better for interpretability!)
    - Regularization is almost always appropriate.
        - Ridge often turned on by default in most Machine Learning Models.
        - You just have to tune the regularization parameter.

```python
X,y = datasets.load_diabetes(return_X_y=True, as_frame=True)
X.head()
y.head()
```

```python
from sklearn.linear_model import Ridge, Lasso, LinearRegression

linreg = LinearRegression().fit(X, y)
ridge = Ridge(alpha=0.2).fit(X, y)
lasso = Lasso(alpha=0.2).fit(X, y)

coefs = pd.DataFrame({
    "coef_linreg": pd.Series(linreg.coef_, index = X.columns),
    "coef_ridge": pd.Series(ridge.coef_, index = X.columns),
    "coef_lasso": pd.Series(lasso.coef_, index= X.columns)})

coefs\
    .applymap(lambda x: int(x))\
    .style.applymap(lambda x: 'color: red' \
		if x == 0 else 'color: black')
```

```python
from sklearn.linear_model import ElasticNet
model = ElasticNet(alpha=1, l1_ratio=0.2)
```

```python
# Let's check the p-values of our features before regularization

import statsmodels.api as sm
ols = sm.OLS(y, sm.add_constant(X)).fit()
ols.summary()
```

```python
coefs_with_p_value
```

# Grid Search & Random Search

- **FIT** = finding best **PARAMS** so as to minimize **LOSS**
- **FINETUNE** = finding best **HYPERPARAMS** so as to maximize **PERF METRIC**

### ElasticNet Regularization

- average of L1, L2

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.20, random_state=1)
```

```python
# Select hyperparam values to try

alphas = [0.01, 0.1, 1] # L1 + L2
l1_ratios = [0.2, 0.5, 0.8] # L1 / L2 ratio

# create all combinations [(0.01, 0.2), (0.01, 0.5), (...)]
import itertools
hyperparams = itertools.product(alphas, l1_ratios)
```

```python
# Train and CV-score model for each combination
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selectionl import cross_val_score

for hyperparam in hyperparams:
    alpha = hyperparam[0]
    l1_ratio = hyperparam[1]
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    r2 = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"alpha: {alpha}, l1_ratio: {l1_ratio},   r2: {r2}")
```

### Grid Search

1. Hold-out a *validation set* (never use test set for model tuning!)
2. Select which grid of values of hyper-parameters to try out
3. For each combinations of values, measure your performance on the *validation set*
4. Select hyperparams that produce the best performance
- **Grid Search CV**
    1. Randomy split your training set into `k` folds of same size
    2. Make fold `#1` a val_set, train model on other `k-1` folds & mesure val_score
    3. Make fold `#2` a val_set and repeat
    4. ...
    5. Compute average val_score over all folds
    - Repeat for each value of hyper-param to test
    - Save the test set for final evaluation only (AFTER hyper-params are chosen)
- limitations:
    - Computationally costly
    - The optimal hyperparameter value can be missed
    - Can overfit hyperparameters to the training set if too many combinations are tried out for too small a dataset

```python
from sklearn.model_selection import GridSearchCV

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# Instanciate model
model = ElasticNet()

# Hyperparameter Grid
grid = {'alpha': [0.01, 0.1, 1],
        'l1_ratio': [0.2, 0.5, 0.8]}

# Instanciate Grid Search
search = GridSearchCV(model, grid,
                           scoring = 'r2',
                           cv = 5,
                           n_jobs=-1 # paralellize computation
                          )

# Fit data to Grid Search
search.fit(X_train,y_train);
```

```python
# Best score
search.best_score_

# Best Params
search.best_params_

# Best estimator
search.best_estimator_
```

### Random Search

- pros:
    - Less typing, if you want to try many values
    - Control for the number of combinations to try / search time
    - Useful when some hyperparams are more important than others
- **`RandomizedSearchCV`**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats

# Instanciate model
model = ElasticNet()

# Hyperparameter Grid
grid = {'l1_ratio': stats.uniform(0, 1), 'alpha': [0.001, 0.01, 0.1, 1]}

# Instanciate Grid Search
search = RandomizedSearchCV(model, grid,
                            scoring='r2',
                            n_iter=100,  # number of draws
                            cv=5, n_jobs=-1)

# Fit data to Grid Search
search.fit(X_train, y_train)
search.best_estimator_
```

```python
# Choose hyperparameter probability distribution wisely
scipy.stats.distributions
```

```python
from scipy import stats

dist = stats.norm(10, 2) # if you have a best guess (say: 10)

dist = stats.randint(1,100) # if you have no idea
dist = stats.uniform(1, 100) # same

dist = stats.loguniform(0.01, 1) # Coarse grain search

r = dist.rvs(size=10000) # Random draws
plt.hist(r);
```

- loguniform great for coarse-grain search across several orders of magnitude

# Support Vector Machines (Margin Classifiers)

- new model
- finds the maximum margin hyperplane between two disparate classes
1. maximum margin classifier
    1. generalizes best to unseen data is the one that is furthest from all the point
    2. s
    3. overfits, sensitive to outliers
    4. classifying training data well
2. soft margin classifier
    1. Allows a few points to be misclassified but with a **penalty(ξ)**
    2. **Hinge Loss** is the penalty applied to each point on the wrong side
    3. generalizing to new data

### **Regulariation hyperparameter (C)**

- Strength of the penalty applied on points located on the wrong side of the margin
    - The higher `C`, the stricter the margin
    - The smaller `C`, the softer the margin, the more it is **regularized**

```python
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=10)

# equivalent but with SGD solver
from sklearn.linear_model import SGDClassifier
svc_bis = SGDClassifier(loss='hinge', penalty='l2', alpha=1/10)
```

### **SVM Regressors**

```python
from sklearn.svm import SVR
regressor = SVR(epsilon=0.1, C=1, kernel='linear')
```

### **SVM Kernels**

- finding the best vector w
    - whose **direction** uniquely determines the decision boundary hyperplane (orthogonal)
    - which minimizes the sum of **hinge losses** for outliers
- non-linearly separable:
    - add new feature that makes it a 3d space separable by a hyperplane

# Kernel Tricks

- **Kernel**: measure of similarity between points that can be used to classify points in SVM models (two points with large similarity would be classified similarly)

Instead of explicitly creating all the new features, smart people came up with a very clever "trick":

- Each time the loss function is calculated, it calculates a sort of **similarity** K(a,b) between all pairs of datapoints, called a **Kernel**
- Two points with large similarity would be classified similarly
- We can **simulate** feature mapping by replacing the kernel wisely in the loss function
- Much more computationally **efficient**

### SVM Kernels lists

`kernel` specifies the type of **feature mapping** to be used to make data **linearly separable** again

- **linear - 3D**
    - hypothesis function hw(X) of a Linear SVM
    - distance is also called the **cosine similarity** between the two vectors
- **polynomial** (of dimension `d`) - **Exponential**
    - allows to fit non-linear **regression** very easily
- **rbf** (of coef `gamma`) - **amoeba**
    - Similarity between two datapoints is "gaussian"
    - Two points far away from one another are exponentially more likely to be different.
- **sigmoid** (of coef `gamma`)

---

`C` is the **strength** of the cost associated with the **wrong classification**

# Challenges

### 1

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

st_scaler = StandardScaler()
X_test_sc = st_scaler.fit_transform(X_test)
X_train_sc = st_scaler.fit_transform(X_train)
```

```python
kn_model2 = KNeighborsRegressor(n_neighbors=2)

cv_results_baseline = cross_validate(kn_model2, X_train_sc, y_train, cv=5, scoring='r2')
cv_results_baseline = pd.DataFrame(cv_results_baseline)
baseline_r2 = cv_results_baseline.test_score.mean()
baseline_r2
```

```python
# Instantiate model
kn_model = KNeighborsRegressor()

# Hyperparameter Grid
neighbors_list = np.arange(5, 20, 1, dtype='int')
grid = {'n_neighbors': neighbors_list}

# Instantiate Grid Search
search = GridSearchCV(kn_model,
                      grid,
                      scoring = 'r2',
                      cv = 5,
                      n_jobs=-1)

# Fit data to Grid Search
search.fit(X_train_sc,y_train)

search.best_score_
search.best_estimator_
```

```python
looped_k = []
for k in range(1,51):
    kn_model_n = KNeighborsRegressor(n_neighbors=k)
    cv_results_looped = cross_validate(kn_model_n, X_train_sc, y_train, cv=5, scoring='r2')
    cv_results_looped = pd.DataFrame(cv_results_looped)
    lopped_r2 = cv_results_looped.test_score.mean()
    looped_k.append(lopped_r2)
plt.plot(looped_k)
```

```python
# Instantiate model
kn_model = KNeighborsRegressor()

# Hyperparameter Grid
neighbors_list = [1,5,10,20,50]
p_list = [1,2,3]
grid = {'n_neighbors': neighbors_list, 'p': p_list}

# Instantiate Grid Search
search = GridSearchCV(kn_model,
                      grid,
                      scoring = 'r2',
                      cv = 5,
                      n_jobs=-1)

# Fit data to Grid Search
search.fit(X_train_sc,y_train)
search.best_params_
```

```python
kn_model = KNeighborsRegressor()

grid = {'n_neighbors': randint(1,50),'p': [1,2,3]}

search = RandomizedSearchCV(kn_model,
                            grid,
                            scoring='r2',
                            n_iter=75,
                            cv=5,
                            n_jobs=-1)
search.fit(X_train_sc, y_train)
search.best_estimator_
search.best_score_
```

```python
kn_model_gen = KNeighborsRegressor(n_neighbors=2, p=1)
kn_model_gen.fit(X_train_sc, y_train)
r2_test = kn_model_gen.score(X_test_sc,y_test)
```

### 2

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
```

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X.shape
```

```python
log_mod = LogisticRegression(max_iter=80, penalty='none')

log_mod.fit(X_scaled, y)
log_odds = pd.DataFrame(log_mod.coef_.T, X.columns.values)
log_odds
```

```python
log_mod = LogisticRegression(max_iter=80, penalty='l2', C=10)

log_mod.fit(X_scaled, y)
log_odds = pd.DataFrame(log_mod.coef_.T, X.columns.values)
log_odds.sort_values(by=0)
```

```python
log_mod = LogisticRegression(penalty='l1', C=.5, solver='liblinear')

log_mod.fit(X_scaled, y)
log_odds = pd.DataFrame(log_mod.coef_.T, X.columns.values)
log_odds.sort_values(by=0)
```

# Flashcards

- How would you ensure that you are not overfitting when tuning the hyperparameters of a model?
    1. Holdout method on the entire dataset
    2. Cross-validated model tuning on the train set
    3. Final evaluation on the test set
- What are the Support Vectors of an SVM model?
    - The support vectors are the observations upon which the model relies in order to place its separating hyperplane. The support vectors are located at the border between classes.
- Probabilistic models like Logistic Regression are not adapted to multiclassification. What are two strategies to bypass this problem and perform multiclassification?
    - One vs One
        - splits a multi-class classification into one binary classification problem per each pair of classes
    - One vs All (Rest)
        - splits a multi-class classification into one binary classification problem per class.
        - Advantage: Far fewer classifiers to train than OvO.
        - Disadvantage: It may discard pair-specific relations, because “rest” is a mixture of classes.
        - Overall: It is used more often than OvO
- What are the two most common Regularization techniques and their associated penalty terms?
    - Ridge (L2) Regularization: its penalty is the **squared sum of each parameter**.
    - Lasso (L1) Regularization: its penalty is the **absolute sum of each parameter**.
- No regularization at all! If is zero, then the penalty term also equals zero.

# Recap

### Kernel Trick

- you can reformulate with a dot product
-
