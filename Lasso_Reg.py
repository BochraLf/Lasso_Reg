
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import sklearn


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X= np.random.normal(1,1, size=600000).reshape(60000,10)
#X[:,5:10] = X[:,:5]
Y= np.sum(X, axis=1)
X_train, X_test = X[:50000,:5], X[50000:60000,0:5]
Y_train, Y_test = Y[:50000],Y[50000:60000]                                
np.var(Y)

model = LinearRegression(fit_intercept= True)
model.fit(X_train, Y_train)
print(model.coef_)
Y_predict = model.predict(X_test)
MSE = mean_squared_error(Y_predict, Y_test)


print(model.coef_)
print(model.intercept_)
print(np.mean(Y_predict))
print(MSE)


A= pd.read_csv('/Users/PC/Desktop/fraud/ds.dat', header = None)
B= pd.DataFrame(A)
A_x = B.iloc[:,:100]
A_y = B.iloc[:,100]

print(A.shape)


from sklearn.linear_model import Lasso
def lasso_regression(alpha):
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(A_x, A_y)
    y_pred = lassoreg.predict(A_x)
    rss = sum((y_pred-A_y)**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret

A.head(2)
#trying diff alpha
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,101)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]

coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Iterate for the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(alpha_lasso[i])

coef_matrix_lasso.apply(lambda x: sum(x.values==0),axis=1)

for i,j in enumerate(coef_matrix_lasso.iloc[5,2:]):
    if j!=0:
       print(i+1, j)

coef_matrix_lasso.iloc[5,]

for i in range(102):
    if coef_matrix_lasso.iloc[3,i]!=0:
       print(coef_matrix_lasso.iloc[4,i])

