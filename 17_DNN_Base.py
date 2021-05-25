# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
import copy


# %%
with open("SavedFeatures/final_feature_vector.json", 'r') as f:
    XY = json.load(f)


# %%
# XY[0]


# %%
# XY[1]


# %%
step = 50000000
lo = 0
clss = 0
num_classes = 7

while(lo < step*(num_classes-1) + 1):
    hi = lo + step
    lo_txt = lo
    hi_txt = hi
    if(lo_txt == 0):
        lo_txt = '---'
    if(lo_txt == step*(num_classes-1)):
        hi_txt = '+++'
    print('{} to {} :: {}'.format(lo_txt, hi_txt, clss))

    clss += 1
    lo += step


# %%
len(XY)


# %%
def getClass(revenue):
    clss = revenue//step
    return min(clss, num_classes-1)


# %%
xrr = []
yrr = []
frq_dst = {}
thresh = 200

for i in range(num_classes):
    frq_dst[i] = 0

for xy in XY:
    clss = getClass(xy['target'])
    if(thresh != -1 and frq_dst[clss] >= thresh):
        continue
    xrr.append(xy['feature'])
    yrr.append(clss)
    frq_dst[clss] += 1
frq_dst


# %%
# xrr[0], xrr[1]

# %% [markdown]
# # Normalize

# %%
mxrr = np.zeros(len(xrr[1]))
len(mxrr)


# %%
for feature in xrr:
    for i in range(len(feature)):
        mxrr[i] = max(mxrr[i], feature[i])


# %%
mnrr = copy.deepcopy(mxrr)

for feature in xrr:
    for i in range(len(feature)):
        mnrr[i] = min(mnrr[i], feature[i])


# %%
# mxrr


# %%
# mnrr


# %%
(len(xrr[0]) , len(xrr))


# %%
nxrr = np.zeros( (len(xrr) , len(xrr[0])) )

for i in range(len(xrr)):
#     print(i)
    for j in range(len(xrr[i])):
        if(mxrr[j]-mnrr[j] != 0): 
            nxrr[i][j]= (xrr[i][j] - mnrr[j]) / (mxrr[j] - mnrr[j])
#             print(i, j, nxrr[i][j])
#     if(i == 3):
#         break


# %%
xrr[0][0],xrr[1][0]


# %%
nxrr[0][0], nxrr[1][0]


# %%
xrr = np.array(xrr)
nxrr = np.array(nxrr)
yrr = np.array(yrr)

xrr.shape, nxrr.shape, yrr.shape

# %% [markdown]
# # Train-test split

# %%
# def turnIntoOrdinalOnehot(clss):
#     one_hot = [0]*num_classes
#     for i in range(clss+1):
#         one_hot[i] = 1
#     return one_hot


# %%
# def turnIntoOrdinalOnehot__Vector(labels):
#     onehot_vec = []
#     for i in range(len(labels)):
#         onehot_vec.append(turnIntoOrdinalOnehot(labels[i]))
#     return np.array(onehot_vec)


# %%
X_train, X_test, y_train, y_test = train_test_split(nxrr, yrr, test_size=0.30, random_state=17)
y_train_trans = y_train.reshape(-1, 1)
y_test_trans = y_test.reshape(-1, 1)
# y_train_tarns = turnIntoOrdinalOnehot__Vector(y_train)


# %%
# turnIntoOrdinalOnehot(y_train[2])


# %%
# turnIntoOrdinalOnehot__Vector([0,1,2,3])


# %%
y_train_trans[10], y_train[10]


# %%
X_train.shape, y_train.shape, y_train_trans.shape


# %%
X_test.shape, y_test.shape, y_test_trans.shape


# %%
from sklearn import preprocessing

onehot = preprocessing.OneHotEncoder()
onehot.fit(y_train_trans)


# %%
y_train_ohe = onehot.transform(y_train_trans).toarray()
y_test_ohe = onehot.transform(y_test_trans).toarray()


# %%
y_train_ohe.shape, y_test_ohe.shape


# %%
idx = 100
y_train[idx], y_train_trans[idx]

# %% [markdown]
# # DNN

# %%
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


# %%
from tensorflow.keras import backend


# %%
def custom_loss(y_true, y_pred):
    true_val = tf.argmax(y_true)
    pred_val = tf.argmax(y_pred)
    # print(true_val.numpy(), pred_val.numpy(), tf.square(true_val - pred_val).numpy())
    # return tf.reduce_mean(tf.square(y_true-y_pred))
    loss = tf.square(true_val - pred_val)
    return tf.reduce_mean(tf.cast(loss, tf.float64))


# %%
a = 10
b = 131
y_train[a], y_train[b]


# %%
custom_loss(y_train_ohe[a], y_train_ohe[b]).numpy()


# %%
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model. kernel_initializer='he_normal'))
model.add(num_classes, activation='softmax'))


# %%
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')
# fit the model
model.fit(X_train, y_train_ohe, epochs=15, batch_size=32, verbose=1)


# %%
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)


# %%
row = np.array([X_test[10]])
yhat = model.predict(row)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)), y_test[10])


# %%
def getConfusionMatrix(target, predicted):
    classes = sorted(list(set(target)))
    matrix = {}
    for c in classes:
        matrix[c] = {}
        for cc in classes:
            matrix[c][cc] = 0
    for i in range(len(target)):
        t = target[i]
        p = predicted[i]

        matrix[t][p] += 1
    
    return matrix


# %%
def plotMatrix(dct2D):
    # print(dct2D)
    arr2D = []
    for ik in dct2D:
        arr1D = []
        for jk in dct2D[ik]:
            arr1D.append(dct2D[ik][jk])
        arr2D.append(arr1D)
    df_cm = pd.DataFrame(arr2D, index = list(dct2D.keys()),
                  columns = list(dct2D.keys()))
    plt.figure(figsize = (20,14))
    sn.heatmap(df_cm, annot=True)
    plt.show()


# %%
def getPrediction(model, X_test):
    pred = model.predict(X_test)
    y_hat = []
    for i in range(len(pred)):
        y_hat.append(np.argmax(pred[i]))
    return y_hat


# %%
matrix = getConfusionMatrix(y_test, getPrediction(model, X_test))


# %%
plotMatrix(matrix)

# %% [markdown]
# # Ordinal Regression (classification) resources
# 
# https://nbviewer.jupyter.org/github/fabianp/minirank/blob/master/notebooks/comparison_ordinal_logistic.ipynb
# https://stackoverflow.com/questions/57561189/multi-class-multi-label-ordinal-classification-with-sklearn
# https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/
# http://fa.bianp.net/blog/tag/ordinal-regression.html
# https://www.statsmodels.org/devel/examples/notebooks/generated/ordinal_regression.html
# https://pythonhosted.org/mord/
# https://pypi.org/project/coral-ordinal/
# https://github.com/sarvothaman/ordinal-classification
# 
# %% [markdown]
# 

