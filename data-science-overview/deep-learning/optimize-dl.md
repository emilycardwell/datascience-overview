# Optimizing Deep Learning: Loss, Fit

### Flow:

1. start model from last layer
2. implement easiest architecture first
3. stick with same batch size (32)
4. don’t think about epochs

---

1. optimize:
    1. try to overfit or tune learning rate or change architecture
    2. if train loss was on steep downward trajectory when it hit early stopping, **regularize,** starting with last layers

# Reminders

- Neural network: stack of layers
- each layer is composed of a set of neurons
- a neuron is a linear combo + activation function
- tensorflow.keras

    ```python
    ###### Keras cheatsheet ######

    # STEP 1: ARCHITECTURE
    model = Sequential()
    model.add(layers.Dense(100, input_dim=128, activation='relu'))  # /!\ Must specify input size
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(5, activation='softmax')) # /!\ Must correspond to the task at hand

    # STEP 2: OPTIMIZATION METHODS
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # SETP 3: DATA AND FITTING METHODS
    model.fit(X, y, batch_size=32, epochs=100)
    ```

    ```python
    ### REGRESSION WITH 1 OUTPUT
    model.add(layers.Dense(1, activation='linear'))

    ### REGRESSION WITH 16 OUTPUTS
    model.add(layers.Dense(16, activation='linear'))

    ### CLASSIFICATION WITH 2 CLASSES
    model.add(layers.Dense(1, activation='sigmoid'))

    ### CLASSIFICATION WITH 14 CLASSES
    model.add(layers.Dense(14, activation='softmax')
    ```

    - when would you need a reg with 16 outputs?
        - if you want to regress multiple things at once (temp, wind-speed, raining)

### Tensor objects

- similar to np arrays

```python
import tensorflow as tf

X = tf.ones((3,3))
```

`<tf.Tensor: shape=(3, 3), dtype=float32, numpy=`

`array([[1., 1., 1.],`

       `[1., 1., 1.],`

       `[1., 1., 1.]], dtype=float32)>`

# Model.compile(…)

- model predicts: $ˆy = f_θ(X)$
- loss function measures smooth distance between ˆy and y to optimize θ: $L_x(θ)$
- metrics measure human distances between ˆy and y for performance eval.

```python
# REGRESSION
model.compile(loss='mse',
              optimizer='adam',
              metrics=['mae'])

# CLASSIFICATION WITH 2 CLASSES
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# CLASSIFICATION WITH N (let's say 14) CLASSES
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'precision'])
```

## Metrics

- human measures of how good predictions are, computed by forward propagation at each epoch
- classification: precision, recall, accuracy, f1, roc-auc
- regression: MSE, MAE, RMSE, $R^2$
- metrics and losses are based on two points in vector space (A, B)
    - distances (smaller the better): Euclidean, Manhattan, etc.
    - similarities (larger the better): Cosine, Jaccard

```python
# use strings for quick access
model.compile(metrics=['accuracy', 'precision'])

# use Keras metric objects for fine-tuning
metric = keras.metrics.AUC(
    num_thresholds = 200,
    curve='ROC', # or curve='PR'
)
model.compile(metric=metric)

# Custom metrics
def custom_mse(y_true, y_pred):
    squared_diff = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_diff)

model.compile(metrics=[custom_mse])
```

### Forward Propagation Ex

- x = p = 5 (features)
- hd = 1 (1 hidden layer)
- n = 4 (neurons)
- o = 1 (predictive layer)
    - $g(\sum_{j=1}^{4}w_j*a_j+b_1)$
    - b (bias) acts like an extra feature
    - params: 4(5 + 1) + 1(4+1) = 29
    - like one matrix multiplication per layer

## Loss Function

- function you choose to optimize algorithm
- one loss per model
- must be smooth (so we can compute its gradient)
    - continuous and sub-differentiable
- **accuracy** isn’t great (jumps from 0 → 1) as weight varies
- **cross-entropies** output probabilities which vary smoothly
    - **binary cross-entropy** = Log Loss

    ```python
    # use strings for quick access
    model.compile(loss = "binary_crossentropy")

    # use Keras metric objects for fine-tuning
    loss = keras.losses.BinaryCrossentropy(...)
    model.compile(loss = loss)

    # Custom losses
    def custom_mse(y_true, y_pred):
        squared_diff = tf.square(y_true - y_pred)
        return tf.reduce_mean(squared_diff)

    model.compile(loss=custom_mse)
    ```


# Optimization

### backpropagation (BP)

- takes loss and then adjusts weights on next epochs
- computes all **partial derivatives** together
    - network make of simple composite function (addition, mult, sig, ReLU)
    - derivatives obtained from individual contributions using **chain rule**
        - iterate backward so we can re-use many terms
- super fast (time for forward pass ≈ backpropagation)
    - 1 iteration on minibatch:
        - 1 forward pass:
            - compute outputs for each observation
            - compute loss
            - *store intermediary computations in RAM*
        - 1 backward pass:
            - *re-use intermediary values*
- vanishing gradient:
    - The weights of the first (deeper) layers are **harder** to move than from the last layer (outputs)

### Which optimizer?

- adam is your bestie
    - gradient +
    - momentum (inertia) +
    - AdaGrad (adapt. learning rate per feature - prioritize weak params) +
    - RMSProp (decay - only recent gradient matters)

### Hyper-parameters

```python
opt = tensorflow.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.99
)
model.compile(loss=..., optimizer=opt)
```

### Learning Rate

- start with default
- amount of change on the weights you want at each update
- smaller rates = more epochs

### Batch Size

- if p = 100 & batch_size = 10:
    - weights 0 → 1 on samples 1-10
    - weights 1 → 2 on samples 11-20
    - etc…
- The smaller the batch, the more stochastic the process is and the faster it may converge
- The larger the batch, the better it generalizes, but the more computationally intensive it becomes
- 16-32 for most things
- more only for VERY SMALL data

### Epochs

- as the batch size increases, so do the required epochs
- as many as possible as long as the neural network is able to generalize to unseen data
- use train/validate/test split
    - train test split → validation split in only train set
    - stop neural net when overfitting occurs on validation set
    - evaluate model on test set

```python
# Give validation set explicitly
history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          batch_size=16,
          epochs=100)
```

```python
history = model.fit(X_train, y_train,
          validation_split=0.3, # /!\ LAST 30% of train indexes are used as validation
          batch_size=16,
          epochs=100,)
          # shuffle=True) # Training data is shuffled at each epoch by default
```

### Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping()

model.fit(X_train, y_train,
          batch_size=16,
          epochs=1000,
          validation_split=0.3,
          callbacks=[es])

# "callback" means that the early stopping criterion
# will be called at the end of each epoch
```

- need to allow given number of iterations without improvement (because of noise)

```python
es = EarlyStopping(patience=20)

model.fit(X_train, y_train,
          batch_size=16,
          epochs=1000,
          validation_split=0.3,
          callbacks=[es])
```

```python
# restore weights that correspond to best validation loss
es = EarlyStopping(patience=20, restore_best_weights=True)

model.fit(X_train, y_train,
          batch_size=16,
          epochs=1000,
          validation_split=0.3,
          callbacks=[es])
```

# Regularization

1. build initial (somewhat arbitrary) architecture
2. once you see results, know what to change to improve them
    - regularization helps overfitting

### Regularizers (L1, L2)

- regularization layers act the same way as L1 and L2 regularization in "vanilla" Machine Learning
- Neural Network will optimize the loss you declared *plus* this regularization
- you can apply to: (only during training)
    - weights (kernel_regularizer)
    - biases (bias_regularizer)
    - output (activity_regularizer)

```python
from tensorflow.keras import regularizers, Sequential, layers

reg_l1 = regularizers.L1(0.01)
reg_l2 = regularizers.L2(0.01)
reg_l1_l2 = regularizers.l1_l2(l1=0.005, l2=0.0005)

model = Sequential()

model.add(layers.Dense(100, activation='relu', input_dim=13))
model.add(layers.Dense(50, activation='relu', kernel_regularizer=reg_l1))
model.add(layers.Dense(20, activation='relu', bias_regularizer=reg_l2))
model.add(layers.Dense(10, activation='relu', activity_regularizer=reg_l1_l2))
model.add(layers.Dense(1, activation='sigmoid'))
```

### Dropout Layer

- dropout randomly "kills" (=0) the activity from some neurons so their weights are not updated
- Prevents any neuron from updating its weights only according to a particular input
- Prevents neurons from over-specializing / being too specific to this input and unable to generalize

```python
model = Sequential()

model.add(layers.Dense(20, activation='relu', input_dim=56))
model.add(layers.Dropout(rate=0.2))  # The rate is the percentage of neurons that are "killed"

model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(rate=0.2))

model.add(layers.Dense(3, activation='softmax'))

# —— What is the number of parameters of the Dropout layer?
```

# Pipelines in Tensorflow

?

# Examples

```python
with tf.GradientTape() as tape:
		y_pred = tf.linspaec(2., 6, 100)
		tape.watch(y_pred)
		y_true = tf.Variable(4.)
		loss = mse(y_true, y_pred)

plt.plot(y_pred, loss.numpy())
ax = plt.gca()
ax.quiver(y_pred.numpy(), loss.numpy(), -grad.numpy(),
		np.zeros(lengrad.numpy())), color='r')
# error landscape
```

# Challenges

### 1

```python
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np

learning_rates = [0.0001, 0.001, 0.01, 0.1, 1, 10]
test_loss = [0.487, 0.233, 0.026, 0.004, 0.022, 0.463]

plt.plot(np.log(learning_rates), test_loss)
```

### 2

```python
# SK-Learn
from sklearn.datasets import make_blobs
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# DATA MANIPULATION
import numpy as np
import pandas as pd

# DATA VISUALIZATION
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# DEEP
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
```

```python
X, y = make_blobs(n_samples=2000, n_features=10, centers=8, cluster_std=7, random_state=42)
X_df = pd.DataFrame(X)
print(X.shape, y.shape)
```

```python
y_cat = to_categorical(y, num_classes=8)
y_cat.shape
```

```python
def initialize_model():
    model = models.Sequential()
    model.add(layers.Dense(25, activation='relu', input_dim=10))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
```

```python
%%time

kf = KFold(n_splits=10)
kf.get_n_splits(X)

results = []

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_cat[train_index], y_cat[test_index]

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_tr_sc = scaler.transform(X_train)
    X_tst_sc = scaler.transform(X_test)

    model = initialize_model()

    model.fit(X_tr_sc, y_train, batch_size=32, epochs=150)

   results.append(model.evaluate(X_tst_sc, y_test))

results
```

```python
results_df = pd.DataFrame(results)
average_accuracy = results_df[1].mean()
standard_deviation = np.std(results_df[1])
print(average_accuracy, standard_deviation)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.3)
X_train.shape
```

```python
%%time

scaler = StandardScaler()
scaler.fit(X_train)
X_tr_sc = scaler.transform(X_train)
X_tst_sc = scaler.transform(X_test)

model = initialize_model()

history = model.fit(X_tr_sc, y_train, validation_data=(X_tst_sc, y_test),
                    batch_size=32, epochs=500)
```

```python
deep_accuracy = model.evaluate(X_tst_sc, y_test)[1]
```

```python
def plot_loss_accuracy(history, title=None):
    fig, ax = plt.subplots(1,2, figsize=(20,7))

    # --- LOSS ---

    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('Model loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylim((0,3))
    ax[0].legend(['Train', 'Test'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- ACCURACY

    ax[1].plot(history.history['accuracy'])
    ax[1].plot(history.history['val_accuracy'])
    ax[1].set_title('Model Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Test'], loc='best')
    ax[1].set_ylim((0,1))
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    if title:
        fig.suptitle(title)

plot_loss_accuracy(history)
```

```python
# Regularize

reg_l2 = regularizers.L2(0.01)

# 1. Model Architecture
model = models.Sequential()
model.add(layers.Dense(25, activation='relu', input_dim=10))
model.add(layers.Dense(10, activation='relu', kernel_regularizer=reg_l2))
model.add(layers.Dense(8, activation='softmax', bias_regularizer=reg_l2))

# 2. Model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 3. Training
history = model.fit(X_train, y_train,
                    validation_split = 0.3,
                    epochs = 300,           # Notice that we are not using any Early Stopping Criterion
                    batch_size = 16,
                    verbose=0)

# 4. Evaluation
results_train = model.evaluate(X_train, y_train, verbose = 0)
results_test = model.evaluate(X_test, y_test, verbose = 0)

# 5. Looking back at what happened during the training phase
print(f'The accuracy on the test set is {results_test[1]:.2f}...')
print(f'...whereas the accuracy on the training set is {results_train[1]:.2f}!')
plot_loss_accuracy(history)
```

```python
# Dropout

# 1. Model Architecture
model = models.Sequential()
model.add(layers.Dense(25, activation='relu', input_dim=10))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(8, activation='softmax'))

# 2. Model compilation
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 3. Training
history = model.fit(X_train, y_train,
                    validation_split = 0.3,
                    epochs = 300,           # Notice that we are not using any Early Stopping Criterion
                    batch_size = 16,
                    verbose=0)

# 4. Evaluation
results_train = model.evaluate(X_train, y_train, verbose = 0)
results_test = model.evaluate(X_test, y_test, verbose = 0)

# 5. Looking back at what happened during the training phase
print(f'The accuracy on the test set is {results_test[1]:.2f}...')
print(f'...whereas the accuracy on the training set is {results_train[1]:.2f}!')
plot_loss_accuracy(history)
```

### Kaggle

```python
%load_ext autoreload
%autoreload 2

# DATA MANIPULATION
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers
from sklearn.preprocessing import MinMaxScaler

# DATA VISUALISATION
import matplotlib.pyplot as plt
import seaborn as sns

# VIEWING OPTIONS IN THE NOTEBOOK
from sklearn import set_config; set_config(display='diagram')
```

```python
X_train, X_val, y_train, y_val = train_test_split(X, y,  test_size=0.3, random_state=42)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
```

```python
from utils.preprocessor import create_preproc

preproc = create_preproc(X_train)
preproc
```

```python
scaler = MinMaxScaler()
scaler.fit(pd.DataFrame(y_train))
y_train = scaler.transform(pd.DataFrame(y_train)).reshape(1022,)
y_val = scaler.transform(pd.DataFrame(y_val)).reshape(438,)

preproc.fit(X_train, y_train)
X_train = preproc.transform(X_train)
X_val = preproc.transform(X_val)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
```

```python
def initialize_model():

    #############################
    #  1 - Model architecture   #
    #############################
    model = Sequential()
    model.add(layers.Dense(10, activation='relu', input_dim=159))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dense(3, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    #############################
    #  2 - Optimization Method  #
    #############################
    model.compile(loss='msle',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
```

```python
model = initialize_model()
history = model.fit(X_train,
                    y_train,
                    validation_data = (X_val, y_val),
                    epochs = 50,
                    batch_size = 16,
                    verbose = 0)
```

```python
def plot_history(history):
    plt.plot(np.sqrt(history.history['loss']))
    plt.plot(np.sqrt(history.history['val_loss']))
    plt.title('Model Loss')
    plt.ylabel('RMSLE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='best')
    plt.show()

plot_history(history)
```

```python
base_score = np.sqrt(model.evaluate(X_val, y_val)[0])
```

```python
y_pred = model.predict(preproc.transform(X_test))
```

```python
y_unscaled = scaler.inverse_transform(y_pred)
```

```python
id_col = X_test.Id
results = pd.DataFrame(y_unscaled, columns=['SalePrice'])
results.insert(loc=0, column='Id', value=id_col)

results.to_csv("submission_final.csv", header = True, index = False)
```

# Recap

```jsx
preproc = make_column_transformer(
							(StandardScaler(), make_column_selector(dtype_exclude='object')),
							(OneHotEncoder(handle_unknown='ignore', sparse=False),
										make_column_selector(dtype_include='object'))
					)
```
