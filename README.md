# Supervised-ML-for-Music-Genre-Classification from Audio Data

The project aims to classify music genre from audio data using five different machine learning classifiers:
* Logistic Regression
* Decistion Tree
* K-nearest neighbors (KNN)
* Support Vector Machines (SVM)
* Random Forest

![Music Genre](https://github.com/kyadanarw/Supervised-ML-for-Music-Genre-Classification/blob/xgb/images/music_genre.jpg)\
Image from https://blog.byjus.com/knowledge-vine/music-genres-types/ </br>

## Software Requirements
Required libraries:
* Python 3.x
* Scikit-Learn
* Pandas
* Matplotlib

## Dataset
We use fma-rock-vs-hiphop.csv metadata about our tracks alongside the track metrics compiled by <b>The Echo Nest</b>.
* fma-rock-vs-hiphop.csv
* echonest-metrics.json

## File Struture
* To perform exploratory data analysis, run <b>EDA.ipynb</b>
* To evaluate each ML classifier, run <b>Classification.ipynb</b>


<h3>1. Data Preparation </h3>
<p> First prepare the data by importing and merging two data files</p>

```python
#first import pandas 
import pandas as pd

#read the metadata
tracks = pd.read_csv('datasets/fma-rock-vs-hiphop.csv')
echonest_metrics = pd.read_json('datasets/echonest-metrics.json', precise_float=True)

# Merge the medata ata
echo_tracks = pd.merge(echonest_metrics, tracks[['track_id' , 'genre_top']], how='inner', on='track_id')

# Inspect the resultant dataframe
echo_tracks.info()
```
|   Int64Index: 4802 entries, 0 to 4801   |                       |
|:---------------------------------------:|:---------------------:|
|     Data columns (total 10 columns):    |                       |
|               acousticness              | 4802 non-null float64 |
|               danceability              | 4802 non-null float64 |
|                  energy                 | 4802 non-null float64 |
|             instrumentalness            | 4802 non-null float64 |
|                 liveness                | 4802 non-null float64 |
|               speechiness               | 4802 non-null float64 |
|                  tempo                  | 4802 non-null float64 |
|                 track_id                | 4802 non-null int64   |
|                 valence                 | 4802 non-null float64 |
|                genre_top                | 4802 non-null object  |
| dtypes: float64(8), int64(1), object(1) |                       |
|         memory usage: 412.7+ KB         |                       |


<h3>2. Check the correlations between features </h3>
<p>In order to avoid using the features which have strong correlations with each other -- hence avoiding feature redundancy, we check the correlated features in our data using built-in functions in the <code>pandas</code> package <code>.corr()</code>. </p>

```python
# check the correlations between features
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()
```

![correlationImages](https://github.com/kyadanarw/Supervised-ML-for-Music-Genre-Classification/blob/xgb/images/correlation.png)


<h3>3. Data Splitting </h3>
<p>Since we didn't find any particular strong correlations between our features, we can now split our data into an array containing our features, and another containing the labels - the genre of the track.</p>
  
 ```python
  # Import train_test_split function and Decision tree classifier
from sklearn.model_selection import train_test_split

# Create features
features = echo_tracks.drop(["genre_top", "track_id"], axis=1).values

# Create labels
labels = echo_tracks["genre_top"].values

# Split our data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, 
                                                                            random_state=10)
```

<h3>4. Normalize the data </h3>
<p> Once the data is splitted, data preprocessing steps are performed to optimize our model development. To avoid bias, the data is scaled by normalizing with <code>StandardScaler</code> method.</p>

```python
#import StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler

features = echo_tracks.drop(['track_id', 'genre_top'], axis=1)
labels = echo_tracks.genre_top

#normalize the features using StandardScaler
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)
```

<h3>5. Principal Component Analysis for Dinmentinality Reduction</h3>
 <p align="justify">Since we didn't find any particular strong correlations between our features, the common feature reduction methods can be used to reduce the dimensionality of the features.After data preprocessing, PCA is here used to determine by how much we can reduce the dimensionality of our data. <b>Scree-plots</b> and <b>Cumulative explained ratio plots</b> can be used to find the number of components to use in further analyses.<br> First let's look at the scree plots. When using scree plots, an 'elbow' (a steep drop from one data point to the next) in the plot is typically used to decide on an appropriate cutoff.</p>
  
```python
# import PCA from sklean
from sklearn.decomposition import PCA

#inititae PCA and transform features into pricipal components
pca = PCA()
pca.fit(scaled_train_features)

exp_variance = pca.explained_variance_ratio_

fig, ax = plt.subplots()
ax.bar(range(pca.n_components_), exp_variance)
```

![pca](https://github.com/kyadanarw/Supervised-ML-for-Music-Genre-Classification/blob/xgb/images/pca.png)

<p>Unfortunately, there does not appear to be a clear elbow in this scree plot, which means it is not straightforward to find the number of intrinsic dimensions using this method.</p>

<h3>6. Further visualization of PCA</h3>
<p>Now, let's nownlook at the <b>cumulative explained variance plot</b> to determine how many features are required to explain, say, about 85% of the variance. (85% is just arbitary value here, it can be determined by rule of thumb.</p>

```python
#compute cumulative variance 
cum_exp_variance = np.cumsum(exp_variance)

#plot it
fig, ax = plt.subplots()
ax.plot(cum_exp_variance)
ax.axhline(y=0.85, linestyle='--')

# choose the n_components where about 85% of our variance can be explained
n_components = 6
pca = PCA(n_components=6, random_state=10)

# Fit and transform the scaled training and testing features using pca
train_pca = pca.fit_transform(scaled_train_features)
test_pca = pca.transform(scaled_test_features)
```
![cumulativeVariance](https://github.com/kyadanarw/Supervised-ML-for-Music-Genre-Classification/blob/xgb/images/cumu_variance.png)

## Classification
For each classifier, the .ipynb npotebooks started with <code>tain___Variants.ipynb</code> include variants of each classifier with data preprocessing, data scaling, with or without PCA and with or without hyperparameters tuning.
To simulate and test classifiers, use Classification.ipynb by calling the desired variant of each classifier.
* <code>train_LogisticRegression_Variants.ipynb</code> for Logistic Regression
* <code>train_DecisionTree_Variants.ipynb</code> for Decistion Tree
* <code>train_KNN_Variants.ipynb</code> for K-nearest neighbors (KNN)
* <code>train_SVM_Variants.ipynb</code> for Support Vector Machines (SVM)
* <code>train_ensemble_Variants.ipynb</code> for Random Forest

## Results
<table style="width:100%">
  <tr>
    <th>Classifier</th>
    <th>Accuracy</th> 
  </tr>
 
  <tr>
    <td>Logistic Regression</td>
    <td><strong>88%</strong></td>
  </tr>
  
  <tr>
    <td>Decision Tree</td>
    <td><strong>88%</strong></td>
  </tr>
  
  <tr>
    <td> Support Vector Machine</td>
    <td><strong>90%</strong></td>
  </tr>
  
  <tr>
    <td> K Nearest Neighbors</td>
    <td><strong>90%</strong></td>
  </tr>
  
  <tr>
    <td> Random Forest</td>
    <td><strong>90%</strong></td>
  </tr>

</table>
