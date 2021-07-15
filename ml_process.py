#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa, IPython
import librosa.display
import os
import glob
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVCfrom sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix


# In[ ]:


cwd = os.getcwd()


# In[ ]:


path_genres = "../input/gtzan-dataset-music-genre-classification"


# In[ ]:


temp_pd = pd.read_csv("../input/gtzan-dataset-music-genre-classification/Data/features_30_sec.csv")


# In[ ]:


features2 = temp_pd.loc[:,['filename',"chroma_stft_mean","chroma_stft_var","rms_mean","rms_var", 'spectral_centroid_mean', 'spectral_centroid_var', 'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
       'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
       'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo','mfcc10_mean' ]]

features =temp_pd.loc[:,['filename',"chroma_stft_mean","rms_mean", 'spectral_centroid_mean', 'spectral_bandwidth_mean', "rolloff_mean", 'zero_crossing_rate_mean',
       'harmony_mean', 'perceptr_mean', 'tempo','mfcc10_mean' ]]


# In[ ]:


features = temp_pd.iloc[:,:]


# In[ ]:


features.drop(["length"],axis=1).head()


# In[ ]:


pd.DataFrame(features.drop(["length"],axis=1).columns)


# In[ ]:


features["label"] = features["filename"].str.split(".",expand=True)[0]


# In[ ]:


X = features.iloc[:,2:-1]
y = features["label"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)


# In[ ]:


genres_list = ['disco',
 'metal',
 'reggae',
 'blues',
 'rock',
 'classical',
 'jazz',
 'hiphop',
 'country',
 'pop']

temp_label_train = []
counter = 0
for i in range(600):
    temp_label_train.append(genres_list[counter])
    if (i+1) % 60==0:
        counter += 1
temp_label_val = []
counter = 0
for i in range(200):
    temp_label_val.append(genres_list[counter])
    if (i+1) % 20==0:
        counter += 1
temp_label_test = []
counter = 0
for i in range(200):
    temp_label_test.append(genres_list[counter])
    if (i+1) % 20==0:
        counter += 1


# In[ ]:


train_pd = pd.read_csv("../input/features/train_feats.csv",header=None)#
test_pd=pd.read_csv("../input/features/test_feats.csv",header=None)
val_pd=pd.read_csv("../input/features/val_feats.csv",header=None)


# In[ ]:


train_pd["label"]=temp_label_train
train_pd=train_pd.sample(frac = 1)


# In[ ]:


lsvc=LinearSVC(max_iter=10000,dual=True)
clf = GridSearchCV(lsvc,
                      param_grid={"C":[0.01,0.1,0.5,1]}, 
                      cv=5,
                      )

lsvc.fit(train_pd.iloc[:,:-1], train_pd.iloc[:,-1])


# In[ ]:


preds = lsvc.predict(test_pd)
print(accuracy_score(preds,temp_label_test))
print(confusion_matrix(preds,temp_label_test))


# In[ ]:


random_forest = RandomForestClassifier()
"""rf = GridSearchCV(random_forest,
                             param_grid={"n_estimators":[8,16,20,32],
                            "max_depth":[10,16,32]}, 
                      cv=5                      ) """
random_forest.fit(train_pd.iloc[:,:-1], train_pd.iloc[:,-1])


# In[ ]:


preds = random_forest.predict(test_pd)
print(accuracy_score(preds,temp_label_test))
print(classification_report(preds,temp_label_test))
print(confusion_matrix(preds,temp_label_test))


# In[ ]:




lr_clf = LogisticRegression(max_iter=10000,dual=False)
"""lr_best = GridSearchCV(lr_clf,
                      param_grid={"dual":[False],"max_iter":[10000],
                                  "C":[0.01,0.1,0.5,1],"solver":["newton-cg", "lbfgs", "liblinear", "sag"]}
                       , cv=5) # Turn on cv train scores"""
lr_clf.fit(train_pd.iloc[:,:-1], train_pd.iloc[:,-1])


# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
ax.set_title('Confusion Matrix of Random Forest',fontdict={'size':'21',"color":"red"})
preds = lr_clf.predict(test_pd)
print(accuracy_score(preds,temp_label_test))
print(classification_report(preds,temp_label_test))
plot_confusion_matrix(lr_clf,test_pd,temp_label_test,xticks_rotation=45,ax=ax)


# In[ ]:



neigh = KNeighborsClassifier()
"""knn_best = GridSearchCV(neigh,
                      param_grid={"n_neighbors":[5,7,9,11,13,17],"weights":["uniform","distance"]}, 
                      cv=5,
                      return_train_score=True) # Turn on cv train scores"""
neigh.fit(train_pd.iloc[:,:-1], train_pd.iloc[:,-1])


# In[ ]:


preds = neigh.predict(test_pd)
print(accuracy_score(preds,temp_label_test))
print(classification_report(preds,temp_label_test))
print(confusion_matrix(preds,temp_label_test))


# In[ ]:


svm=SVC()
svm_best=GridSearchCV(svm,
                      param_grid={"C":[1,2,4],"kernel":["poly","sigmoid"],"degree":[2,3,4,5,6]
                    ,"coef0":[2,3,4,5,6]}, 
                      cv=5,
                      return_train_score=True)

svm_best.fit(train_pd.iloc[:,:-1], train_pd.iloc[:,-1])
preds = svm_best.predict(test_pd)
print(accuracy_score(preds,temp_label_test))
print(classification_report(preds,temp_label_test))
print(confusion_matrix(preds,temp_label_test))


# In[ ]:



transformer = Normalizer().fit(X_train)
scaledX = transformer.transform(X_train)
scaledX_test = transformer.transform(X_test)



neigh = KNeighborsClassifier()
neigh.fit(scaledX, y_train)
preds = neigh.predict(scaledX_test)
print(accuracy_score(y_test, preds))


# In[ ]:



knn_pipe = make_pipeline(StandardScaler(),
                    KNeighborsClassifier())
knn_pipe.fit(X_train,y_train)
preds = knn_pipe.predict(X_test)
print(accuracy_score(y_test, preds))


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(y)
transformed_yTrain = le.transform(y_train).reshape(-1,1)
transformed_yTest = le.transform(y_test).reshape(-1,1)


# In[ ]:


transformed_yTrain.shape


# In[ ]:


clf = RandomForestClassifier()
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
print(accuracy_score(y_test, preds))


# In[ ]:



svc_pipe = make_pipeline(StandardScaler(),
                    SVC())
fig, ax = plt.subplots(figsize=(10,10))
ax.set_title('Confusion Matrix of SVC',fontdict={'size':'21',"color":"red"})
svc_pipe.fit(X_train, y_train)
preds = svc_pipe.predict(X_test)
print(accuracy_score(y_test, preds))
plot_confusion_matrix(svc_pipe,X_test,y_test,xticks_rotation=45,ax=ax)


# In[ ]:


lsvc = make_pipeline(StandardScaler(),
                    LinearSVC(max_iter=10000))
lsvc.fit(X_train, y_train)
preds = lsvc.predict(X_test)
print(accuracy_score(y_test, preds))


# In[ ]:


lr_pipe = make_pipeline(StandardScaler(),
                    LogisticRegression(max_iter=10000))

#lr_clf = LogisticRegression(max_iter = 10000,dual=False)
lr_pipe.fit(X_train, y_train)
preds = lr_pipe.predict(X_test)
print(accuracy_score(y_test, preds))


# In[ ]:


pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
clf = GridSearchCV(estimator= pipe,
                      param_grid={"svc__C":[0.01,0.1,0.5,1]}, 
                      cv=5,
                      return_train_score=True) # Turn on cv train scores
clf.fit(X_train, y_train)


# In[ ]:


preds = clf.predict(X_train)
print(accuracy_score(y_train, preds))

preds = clf.predict(X_test)
print(accuracy_score(y_test, preds))


# In[ ]:


pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
knn_best = GridSearchCV(estimator= pipe,
                      param_grid={"knn__n_neighbors":[5,7,9,11,13,17],"knn__weights":["uniform","distance"]}, 
                      cv=5,
                      return_train_score=True) # Turn on cv train scores
knn_best.fit(X_train, y_train)


# In[ ]:


preds = knn_best.predict(X_train)
print(accuracy_score(y_train, preds))

preds = knn_best.predict(X_test)
print(accuracy_score(y_test, preds))
knn_best.best_params_


# In[ ]:



pipe = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
rf_best = GridSearchCV(estimator= pipe,
                      param_grid={"rf__n_estimators":[10,16,32,50,100],"rf__criterion":["gini", "entropy"]
                                  ,"rf__max_depth":[10,16,32]}, 
                      cv=5,
                      return_train_score=True) # Turn on cv train scores
rf_best.fit(X_train, y_train)


# In[ ]:


preds = rf_best.predict(X_train)
print(accuracy_score(y_train, preds))

preds = rf_best.predict(X_test)
print(accuracy_score(y_test, preds))
rf_best.best_params_


# In[ ]:


pipe = Pipeline([ ('rf', RandomForestClassifier())])
rf_best = GridSearchCV(estimator= pipe,
                      param_grid={"rf__n_estimators":[10,16,32,50,100],"rf__criterion":["gini", "entropy"]
                                  ,"rf__max_depth":[10,16,32]}, 
                      cv=5,
                      return_train_score=True) # Turn on cv train scores
rf_best.fit(X_train, y_train)


# In[ ]:


preds = rf_best.predict(X_train)
print(accuracy_score(y_train, preds))

preds = rf_best.predict(X_test)
print(accuracy_score(y_test, preds))
rf_best.best_params_


# In[ ]:


pipe = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
lr_best = GridSearchCV(estimator= pipe,
                      param_grid={"lr__dual":[False],"lr__max_iter":[10000],
                                  "lr__C":[10,16,32,50,100],"lr__solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"]}
                       ,
                      cv=5,
                      return_train_score=True) # Turn on cv train scores
lr_best.fit(X_train, y_train)


# In[ ]:


preds = lr_best.predict(X_train)
print(accuracy_score(y_train, preds))

preds = lr_best.predict(X_test)
print(accuracy_score(y_test, preds))

lr_best.best_params_


# In[ ]:


plot_confusion_matrix(rf_best, X_test, y_test, xticks_rotation=45)

