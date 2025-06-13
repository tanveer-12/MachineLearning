import math
import numpy as np
from linear_regression import train, compute_L, compute_yhat
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''

#--------------------------

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.array(y).T
X = np.array(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]

#########################################
## INSERT YOUR CODE HERE

alpha = 0.0001
n_epoch = 93089

w = train(Xtrain, Ytrain, alpha=alpha, n_epoch=n_epoch)

ytrain_pred = compute_yhat(Xtrain, w)
ytest_pred = compute_yhat(Xtest, w)

train_loss = compute_L(ytrain_pred, Ytrain)
test_loss = compute_L(ytest_pred, Ytest)

print(f"Alpha: {alpha}, Epochs: {n_epoch}")
print(f"Training Loss: {train_loss:.8f}")
print(f"Testing Loss: {test_loss:.8f}")
print("Weights:", w)

#########################################

