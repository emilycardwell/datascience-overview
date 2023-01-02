# Unsupervised Learning

- Dataset = X, y
- feature matrix (X: n, p)
- targets vector (y: n, 1)

### Supervised Learning

- find hypothesis for new X as close to y as possible

### Unsupervised Learning

- find patterns in feature matrix (X) without supervision from target (y)
- reduce dimensions
    - feature engineering (saves time)
    - compress (saves space)
- cluster data (group by points based on similarities)
    - understand data (explore, visualize)
    - find anomalies/outliers
    - recommendations
    - semi-supervised classifications

# Principal Component Analysis (PCA)

- Squashes our high-dimensional dataset down into a lower dimension
- Aims to find the best linear combination of features (= columns) that best represents the underlying structure of the data
- PCA = finding the **best linear combination** of features
- cancelling all multi-collinearity
- ranking new Zs from most to least important (principal components - PCs)
- projection of the data
    - oriented toward specific directions, defined by PCs
    - orthonormal to each other (0 multi-col.)
    - ranked by decreasing “explaining power” measured by variance

### Intuition

- If we had to keep **only one direction** to describe our data, this direction should:
    - **preserve** most of the **variance** in the data when projected onto it (see spread of red dots)
    - minimize "reconstruction errors" - residuals
- 3D
- PCA helps reduce dimensions

### Pros

- We use PCA to deal with high-dimensional datasets; some pros are:
    - Better visualization of the data
    - Reduction of the effects of the curse of dimensionality
    - Reduction of file size
- PCA compresses the datasets into a lower-dimensional state by projecting observations onto a new space
- More variation, more information, easier to distinguish between observations

### Limitations

- When we use PCA we lose data interpretability
- manifolds:
    - n-dimensional shape that can be bent/twisted into a higher dimensional shape
    - i.e. - spiral
- other techniques to reduce dimensions:
    - **t-Distributed Stochastic Neighbor Embedding (t-SNE)** — Aims to reduce dimensionality while keeping similar observations
    close together and dissimilar ones apart. This is a great technique for
    visualizing clusters of higher dimensions
    - **Kernel PCA** — Captures non-linear patterns (similar principle to SVM kernels)

### Code

```python
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
X = wine.data
y = wine.target
wine_features = X.columns

#⚠️ Data must be centered around its mean before applying PCA ⚠️
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X), columns=wine_features)
X
```

```python
sns.heatmap(pd.DataFrame(X).corr(), cmap='coolwarm')
```

- compute PCs (linear combo of initial wine features)

```python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
```

```python
# Access our 13 PCs
W = pca.components_

# Print PCs as COLUMNS
W = pd.DataFrame(W.T,
                 index=wine_features,
                 columns=[f'PC{i}' for i in range(1, 14)])
W
```

- project dataset into new space of PCs

```python
X_proj = pca.transform(X)
X_proj = pd.DataFrame(X_proj, columns=[f'PC{i}' for i in range(1, 14)])
X_proj
```

```python
sns.heatmap(X_proj.corr(), cmap='coolwarm');
```

```python
# 2D-slice
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.title('X1 vs. X0 before PCA (initial space)'); plt.xlabel('X0'); plt.ylabel('X1')
plt.scatter(X.iloc[:,0], X.iloc[:,1])

plt.subplot(1,2,2)
plt.title('PC1 vs PC2 (new space)'); plt.xlabel('PC 1'); plt.ylabel('PC 2')
plt.scatter(X_proj.iloc[:,0], X_proj.iloc[:,1]);
```

```python
# Computational proof
W = pca.components_.T
print("Shape of W: ", W.shape)
print("Shape of X", X.shape)

np.allclose(
    pca.transform(X),
    np.dot(X,W)
)
```

### Mathematical Computation

```python
import numpy as np

# Compute PCs
eig_vals, eig_vecs = np.linalg.eig(np.dot(X.T,X)) #takes ages
```

```python
# Show all 13 principal components (unranked)
W = pd.DataFrame(eig_vecs,
                 index=wine_features,
                 columns=[f'PC{i}' for i in range(1, 14)])
W
```

- rank it

```python
# Let's compute it
X_proj.std()**2 / ((X.std()**2).sum())
```

```python
# Sklearn provides it automatically
pca.explained_variance_ratio_

plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Principal Component'); plt.ylabel('% explained variance');
```

## PCA for Dimensionality Reduction

- keep only k most important PCs
    - compresses data
    - reduces model complexity & fit time
    - reduce overfitting

### choosing k

- trade off between compression and performance

    ```python
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.ylim(ymin=0)
    plt.title('cumulated share of explained variance')
    plt.xlabel('# of principal component used');
    ```

- Elbow method:
    - look at inflection point in above graph (elbow) and choose k (x-axis)
    - if k=3 for example:

    ```python
    # Fit a PCA with only 3 components
    pca3 = PCA(n_components=3).fit(X)

    # Project your data into 3 dimensions
    X_proj3 = pd.DataFrame(pca3.fit_transform(X), columns=['PC1', 'PC2', 'PC3'])

    # We have "compressed" our dataset in 3D
    X_proj3
    ```

    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # accuracy 3 PCs
    cross_val_score(LogisticRegression(), X_proj3, y, cv=5).mean()
    #>>0.9609523809523809

    # accuracy all 13 initial features
    cross_val_score(LogisticRegression(), X, y, cv=5).mean()
    #>>0.9888888888888889
    ```


### Decompress

- can you perfectly reconstruct X from X_proj3?
    - not if you kept k<13 dimensions (info lost)
    - approximate X by reconstructing it with inverse_transform()

    ```python
    X_reconstructed = pca3.inverse_transform(X_proj3)
    X_reconstructed.shape
    ```

    ```python
    plt.figure(figsize=(15,4))
    plt.subplot(1,2,1)
    sns.heatmap(X)
    plt.title("original data")
    plt.subplot(1,2,2)
    plt.title("reconstructed data")
    sns.heatmap(X_reconstructed);
    ```


# Clustering with K-means

- The process of organizing data points into groups whose members are similar in some way
- Find **categories** (classes, segments) of **unlabelled** data rather than just trying to reduce dimensionality
- Works better on data that is already clustered, geometrically speaking
- Use PCA for dimensionality reduction beforehand:
    - Euclidean distances work better in lower dimensions
- K-means is usually run a few times with different random initializations
- We can use a random mini-batch at each epoch instead of the full dataset
- The algorithm is quite fast

### Process of one epoch

1. Choose the number of clusters (k) to look for
2. Initialize k **centroids** at random
3. Compute the **mean square distance** between each data point and each centroid
4. Assign each data point to the closest centroid (a cluster is formed)
5. Compute the mean μj of each cluster, the result of which becomes your new centroid
6. Repeat, starting at 3, for further epochs

### Code

- `scikit.clustering.KMeans`
- `scikit.clustering.MiniBatchKMeans`

```python
X_proj
```

```python
from sklearn.cluster import KMeans

# Fit K-means
km = KMeans(n_clusters=3)
km.fit(X_proj)
```

```python
# The 3 centroids' coordinates (expressed in the space of PCs)
km.cluster_centers_.shape
```

```python
# The 177 observations are classified automatically
km.labels_
```

```python
plt.scatter(X_proj.iloc[:,0], X_proj.iloc[:,1], c=km.labels_)
plt.title('KMeans clustering'); plt.xlabel('PC 1'); plt.ylabel('PC 2');
```

- measure performance

    ```python
    # Visualization
    plt.figure(figsize=(13,5))

    plt.subplot(1,2,1)
    plt.scatter(X_proj.iloc[:,0], X_proj.iloc[:,1], c=km.labels_)
    plt.title('KMeans clustering'); plt.xlabel('PC 1'); plt.ylabel('PC 2')

    plt.subplot(1,2,2)
    plt.scatter(X_proj.iloc[:,0], X_proj.iloc[:,1], c=y)
    plt.title('True wine labels'); plt.xlabel('PC 1'); plt.ylabel('PC 2');
    ```

    ```python
    # Accuracy
    from sklearn.metrics import accuracy_score

    y_pred = pd.Series(km.labels_).map({0:0, 1:2, 2:1}) # WARNING: change this manually!
    accuracy_score(y_pred, y)
    ```

- predict wit unsupervised k-means alg. (to classify new X)

    ```python
    # Build DF with column names from X_proj and some random data
    new_X = pd.DataFrame(data = np.random.random((1,13)), columns = X_proj.columns)

    km.predict(new_X)
    ```


## K-means Loss Function

- `Kmeans().inertia_`
- loss function = inertia L(µ)
    - = **sum** of **squared distance** between each observation and their **closest centroid**
    - = sum of **within-cluster sum of squares** (WCSS)
    - = variance
- when?
    - Document classification (finding unlabeled categories or topics)
    - Delivery store optimization (find the optimal number of launch locations)
    - Customer segmentation (classify different types of customer based on their behavior)

### Choosing hyperparam K

- choose k such that loss function minimized
- use elbow method

    ```python
    nertias = []
    ks = range(1,10)

    for k in ks:
        km_test = KMeans(n_clusters=k).fit(X)
        inertias.append(km_test.inertia_)

    plt.plot(ks, inertias)
    plt.xlabel('k cluster number')
    ```


### **other clustering approaches**

- [https://scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)

# Challenges

### 1

```python
from sklearn.datasets import make_blobs
random_state=42

# Generate data
X, y = make_blobs(n_samples=500, centers=4, random_state=random_state)
```

```python
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y)
```

```python
X_new, y_new = make_blobs(n_samples=500, centers=4, random_state=random_state)
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, random_state=42)

km.fit(X)
y_pred = km.predict(X_new)
```

```python
KMeans(n_clusters=2, random_state=random_state).fit(X).inertia_
```

```python
# Apply the elbow method to find the optimal number of clusters.
wcss = []
clusters = list(range(1, 11))

for k in clusters:
    km_test = KMeans(n_clusters=k).fit(X)
    wcss.append(km_test.inertia_)

plt.plot(clusters, wcss)
plt.xlabel('k cluster number')
```

```python
km_opt = KMeans(n_clusters=4)
km_opt.fit(X)
y_pred = km_opt.predict(X_new)
sns.scatterplot(x=X_new[:,0], y=X_new[:,1], hue=y_pred)
```

---

```python
from sklearn.cluster import KMeans

# Let's create vector of 100*3 elements with a value between 0 and 1
image_c = np.random.uniform(low=0., high=1., size=100*3)

# Reshape it into a squared image of 10x10 pixels with 3 colors
image_c = image_c.reshape((10, 10, 3))

# Finally display the generated image
plt.imshow(image_c);
```

```python
# To get some intuition, let's plot each color layer
fig, axs = plt.subplots(1, 3, figsize=(8, 6))
colors = {0:'Reds', 1:'Greens', 2:'Blues'}

for i in colors:
    axs[i].imshow(image_c[:, :, i], cmap=colors[i])
    axs[i].set_title(colors[i])
```

```python
from skimage import data
img = data.astronaut()
plt.imshow(img);
```

```python
X = img.reshape((img_shape[0]*img_shape[1], 3))
color_count = len(np.unique(X, axis=0))
```

```python
km = KMeans(n_clusters=32, random_state=42)
kmeans = km.fit(X)
```

```python
X_compressed = kmeans.cluster_centers_[kmeans.labels_]
X_compressed = X_compressed.astype('uint8')
```

```python
X_uncomp = X_compressed.reshape((512, 512, 3))
plt.imshow(X_uncomp)
```

### 2

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a dataset with 100 observations and 2 correlated features.
seed = np.random.RandomState(42)
feature_1 = seed.normal(5, 1, 100)
feature_2 = .7 * feature_1 + seed.normal(0, .5, 100)
X = np.array([feature_1, feature_2]).T
X = pd.DataFrame(X)

X.corr().round(3)
```

```python
plt.scatter(X[0], X[1])
```

```python
from sklearn.decomposition import PCA

pca_mod = PCA()
pca = pca_mod.fit(X)
```

```python
plt.figure(figsize=(5,5))

plt.scatter(X[0], X[1])

for (length, vector) in zip(pca.explained_variance_, pca.components_):
    v = vector * np.sqrt(length) # Square root of their lenghts to compare same "units"
    plt.quiver(*X.mean(axis=0), *v, units='xy', scale=1, color='r')
```

```python
X_transformed = pca.transform(X)
plt.scatter(X_transformed[:,0], X_transformed[:,1])
```

---

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from time import time
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
```

```python
fig = plt.figure(figsize=(7,10))

for i in range(15):
    plt.subplot(5, 5, i + 1)
    plt.title(faces.target_names[faces.target[i]], size=11)
    plt.imshow(faces.images[i], cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

plt.tight_layout()
```

```python
pca = PCA(n_components=150)

pca.fit(faces.data)
data_projected = pca.transform(faces.data)
```

```python
reshaped = pca.inverse_transform(data_projected)[12].reshape((50,37))
plt.imshow(reshaped)
```

```python
fig = plt.figure(figsize=(7,10))
for i in range(15):
    plt.subplot(5, 5, i + 1)
    plt.title(faces.target_names[faces.target[i]], size=11)
    plt.imshow(pca.inverse_transform(data_projected)[i].reshape((50,37)), cmap=plt.cm.gray)
    plt.xticks(())
    plt.yticks(())

plt.tight_layout()
```

```python
n_rows, n_cols = 3, 5
fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 9))

for i in range(n_rows * n_cols):
    ax = axs[i // n_cols, i % n_cols]
    ax.set_title(f'PC {i * 10 + 1}', size=12)
    ax.set_xticks(()), ax.set_yticks(())
    ax.imshow(pca.components_[i * 10].reshape(50, 37), cmap='gray')

plt.tight_layout()
```

```python
plt.figure(figsize=(16, 9))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid()
plt.xlim((-5, 151))
plt.ylim((0, 1))
plt.hlines(y=[.72, .88], xmin=[-5, -5], xmax=[20, 70],
           linestyles='dotted', colors=['red', 'orange'], linewidth=2)
plt.vlines(x=[20, 70], ymin=[0, 0], ymax=[.72, .88],
           linestyles='dotted', colors=['red', 'orange'], linewidth=2);
```

```python
ratios = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_))
minimal_pc_count = ratios.index[ratios[0] >= .8][0] + 1
minimal_pc_count
```

---

```python
forest = RandomForestClassifier(max)
cv_forest = cross_validate(forest, X_train, y_train, cv=5, scoring='r2')['test_score'].mean()
cv_forest
```

```python
pca_pipe = PCA()
svc = SVC()
pipe = make_pipeline(pca_pipe, svc)
pipe.fit(X_train, y_train)
```

```python
grid = {'pca__n_components': [50,100,200,300]}
search = GridSearchCV(pipe, param_grid=grid, scoring='r2', cv=5, n_jobs=-1)
search.fit(X_test, y_test)
search.best_params_
```

```python
best_n_components = 300
pca_pipe = PCA(n_components=best_n_components)
svc = SVC()
pipe = make_pipeline(pca_pipe, svc)
pipe.fit(X_train, y_train)
y_pred = search.predict(X_test)
```

```python
scale = StandardScaler()
pca300 = PCA(n_components=best_n_components)
scv = SVC()

scaled_pipe = make_pipeline(scale, pca300, scv)
```

```python
cv_scaled = cross_validate(scaled_pipe, X, y, cv=3)
score_scaling = cv_scaled['test_score'].mean()
score_scaling
```

```python
scale = StandardScaler()
pca300 = PCA(n_components=best_n_components)
scv_unbal = SVC(class_weight='balanced')

balanced_pipe = make_pipeline(scale, pca300, scv_unbal)

cv_bal = cross_validate(balanced_pipe, X, y, cv=3)
score_balanced = cv_bal['test_score'].mean()
score_balanced
```

```python
balanced_pipe.get_params()
```

```python
grid = {'svc__kernel': ['rbf', 'poly', 'sigmoid'],
        'svc__gamma': [1e-4, 1e-3, 1e-2],
        'svc__C': [10, 1e2, 1e3]}
search = GridSearchCV(balanced_pipe, param_grid=grid, scoring='r2', cv=5, n_jobs=-1)
search.fit(X, y)
search.best_params_
```

```python
scale = StandardScaler()
pca300 = PCA(n_components=best_n_components, random_state=42)
scv_tuned = SVC(class_weight='balanced', kernel='rbf', gamma=0.0001, C=1000)

tuned_pipe = make_pipeline(scale, pca300, scv_tuned)
tuned_pipe.fit(X, y)
score_tuned = cross_val_score(tuned_pipe, X, y).mean()
score_tuned
#>>.79
```

# Recap

```python
spotify_num = spotify.select_dtypes(exclude=['object'])
```

```python
sns.heatmap(spotify_num.corr(), cmap='PuRd', annot=True, annot_kws={'fontsize': 8})
```

```python
daily_mixes = {}

for numero_cluster in np.unique(labelling):
    daily_mixes[numero_cluster] = spotify_labelled[spotify_labelled.label == numero_cluster]
```

```python
for key,value in daily_mixes.items():
    print("-"*50)
    print(f"Here are some songs for the playlist number {key}")
    print("-"*50)
    display(value.sample(20))
```

```python
pipe = make_pipeline(r_scaler, PCA(n_components=3), KMeans(n_clusters=6))
pipe.fit(spotify_num)
labels = pipe.predict(spotify_num)
fig = px.scatter_3d(spot_proj,
                           x = 0,
                           y = 1,
                           z = 2,
                           color = labels)
fig.show()
```

- [https://projector.tensorflow.org/](https://projector.tensorflow.org/)
