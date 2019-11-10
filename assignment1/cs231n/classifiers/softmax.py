from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # as we're only concerned about softmax loss
    # we can vectorize this step
    scores = X @ W
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
      # shift values to avoid numeric overflow
      a = scores[i] - np.max(scores[i])
      # exponantiate (make all positive)
      a = np.exp(a)
      # normalize
      softmax = a / np.sum(a)

      loss += -np.log(softmax[y[i]])

      # grad = ((i==j) - S_j) * X_i
      for j in range(num_classes):
        dW[:,j] -= ((j == y[i]) - softmax[j]) * X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = X @ W
    a = scores - scores.max(axis=1, keepdims=True)
    a = np.exp(a)
    softmax = a / np.sum(a, axis=1, keepdims=True)

    loss = np.sum(-np.log(softmax[range(num_train), y]))

    dscores = softmax

    # grad = (S_j - 1(i==j)) * X_i
    dscores[range(num_train), y] -= 1
    dW = X.T @ (dscores)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
