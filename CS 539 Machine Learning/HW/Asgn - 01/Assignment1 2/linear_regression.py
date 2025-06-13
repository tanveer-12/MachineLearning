import math
import numpy as np
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 1: Linear Regression
    In this problem, you will implement the linear regression method based upon gradient descent.
    Xw  = y
    You could test the correctness of your code by typing `pytest -v test.py` in the terminal.
    Note: please don't use any existing package for linear regression problem, implement your own version.
'''

#--------------------------
def compute_Phi(x,p):
    '''
        Compute the feature matrix Phi of x. We will construct p polynoials, the p features of the data samples. 
        The features of each sample, is x^0, x^1, x^2 ... x^(p-1)
        Input:
            x : a vector of samples in one dimensional space, a numpy vector of shape (n,).
                Here n is the number of samples.
            p : the number of polynomials/features
        Output:
            Phi: the design/feature matrix of x, a numpy array of shape (n,p).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n = len(x)
    Phi = np.ones((n, p))
    for i in range(n):
        for j in range(1, p):
            Phi[i, j] = x[i] ** j
    ########################################
    return Phi

    
#print(compute_Phi(np.array([1, 2, 3, 4]), 4))

#--------------------------
def compute_yhat(Phi, w):
    '''
        Compute the linear logit value (predicted value) of all data instances. z = <w, x>
        Here <w, x> represents the dot product of the two vectors.
        Input:
            Phi: the feature matrix of all data instance, a float numpy array of shape (n,p). 
            w: the weights parameter of the linear model, a float numpy array of shape (p,). 
        Output:
            yhat: the logit value (predicted value) of all instances, a float numpy array of shape (n,)
        Hint: you could solve this problem using 1 line of code. Though using more lines of code is also okay.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    yhat = np.dot(Phi, w)

    #########################################
    return yhat

#print(compute_yhat(compute_Phi(np.array([1, 2, 3, 4]), 4), np.array([1, 2, 3, 4])))

    #--------------------------
def compute_L(yhat,y):
    '''
        Compute the loss function: mean squared error. In this function, divide the original mean squared error by 2 for making gradient computation simple. Remember our loss function in the slides.  
        Input:
            yhat: the predicted sample labels, a numpy vector of shape (n,).
            y:  the sample labels, a numpy vector of shape (n,).
        Output:
            L: the loss value of linear regression, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    L = 0
    n = yhat.shape[0]
    for i in range(n):
        diff = yhat[i] - y[i]
        squared_error = diff ** 2
        L += squared_error

    L = (1 / (2 * n)) * L
    #########################################
    return L
#print(compute_L(compute_yhat(compute_Phi(np.array([1, 2, 3, 4]), 4), np.array([1, 2, 3, 4])),np.array([1, 2, 3, 4])))



def compute_dL_dw(y, yhat, Phi):
    '''
        Compute the gradients of the loss function L with respect to (w.r.t.) the weights w. 
        Input:
            Phi: the feature matrix of all data instances, a float numpy array of shape (n,p). 
               Here p is the number of features/dimensions.
            y: the sample labels, a numpy vector of shape (n,).
            yhat: the predicted sample labels, a numpy vector of shape (n,).
        Output:
            dL_dw: the gradients of the loss function L with respect to the weights w, a numpy float array of shape (p,). 

    '''
    #########################################
    ## INSERT YOUR CODE HERE
    length = Phi.shape[0]
    p = Phi.shape[1]
    error = yhat - y
    dL_dw = np.zeros(p)

    for j in range(p):
        gradient_sum = 0
        for i in range(length):
            gradient_sum += error[i] * Phi[i, j]
        dL_dw[j] = gradient_sum / length

    #########################################
    return dL_dw

#print(compute_dL_dw(np.array([1, 2, 3, 4]), compute_yhat(compute_Phi(np.array([1, 2, 3, 4]), 4), np.array([1, 2, 3, 4])), compute_Phi(np.array([1, 2, 3, 4]), 4)))

#--------------------------
def update_w(w, dL_dw, alpha = 0.001):
    '''
       Given the instances in the training data, update the weights w using gradient descent.
        Input:
            w: the current value of the weight vector, a numpy float array of shape (p,).
            dL_dw: the gradient of the loss function w.r.t. the weight vector, a numpy float array of shape (p,). 
            alpha: the step-size parameter of gradient descent, a float scalar.
        Output:
            w: the updated weight vector, a numpy float array of shape (p,).
        Hint: you could solve this problem using 1 line of code
    '''
    
    #########################################
    ## INSERT YOUR CODE HERE
    w = w - alpha * dL_dw

    #########################################
    return w
#print(update_w(np.array([1, 2, 3, 4]), compute_dL_dw(np.array([1, 2, 3, 4]), compute_yhat(compute_Phi(np.array([1, 2, 3, 4]), 4), np.array([1, 2, 3, 4])), compute_Phi(np.array([1, 2, 3, 4]), 4)), alpha=0.001))

#--------------------------
def train(X, Y, alpha=0.001, n_epoch=100):
    '''
       Given a training dataset, train the linear regression model by iteratively updating the weights w using the gradient descent
        We repeat n_epoch passes over all the training instances.
        Input:
            X: the feature matrix of training instances, a float numpy array of shape (n, p). Here n is the number of data instance in the training set, p is the number of features/dimensions.
            Y: the labels of training instance, a numpy integer array of shape (n,). 
            alpha: the step-size parameter of gradient descent, a float scalar.
            n_epoch: the number of passes to go through the training set, an integer scalar.
        Output:
            w: the weight vector trained on the training set, a numpy float array of shape (p,). 
    '''

    # initialize weights as 0
    w = np.array(np.zeros(X.shape[1]))

    for _ in range(n_epoch):
        
    #########################################
    ## INSERT YOUR CODE HERE

    # Back propagation: compute local gradients 
        yhat = compute_yhat(X, w)
        dL_dw = compute_dL_dw(Y, yhat, X)
    # update the parameters w
        w = update_w(w, dL_dw, alpha)
        
     #########################################
    return w



