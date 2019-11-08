from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    # gradient of weights not satisfying the desired margin
    # remains zero. Rest are updated in the loop
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_contributor_classes = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                # for j != y_i
                dW[:,j] += X[i]
                # need for correct class gradient
                loss_contributor_classes += 1
        # correct class gradient
        dW[:,y[i]] += (-1) * loss_contributor_classes * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # add derivative of Regularization term R(w)
    # R(w) = lambda * sum(w * w)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_train = X.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X @ W

    # extract correct scores from each row
    correct_scores = scores[range(num_train), y]
    correct_scores = correct_scores.reshape((num_train, -1))

    # calculate margins with delta=1
    margins = scores - correct_scores + 1

    # make sure margin for j=y_i is set to 0
    margins[range(num_train), y] = 0

    # neglect negative margins (score < score[y_i])
    margins = np.fmax(margins, 0)

    # calculate loss by summing up rest of the margins
    loss = np.sum(margins) / num_train

    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # We need to create mask array to vectorize gradient operations
    X_mask = np.zeros(margins.shape)

    # We add X_i to dW for incorrect class, setting it to 1
    # as we are eventually performing matrix inner product
    # to get gradients
    X_mask[margins > 0] = 1

    # We multiply "no. of margins > 0" to X_i for correct class
    # as we've already set every incorrect class idx to 1
    # we can directly sum across X-axis to get the result
    X_mask[range(num_train), y] = -np.sum(X_mask, axis=1)

    # This operation will multiply X_i with correct coefficient
    # 1 for incorrect class
    # "no. of margins > 0" for correct class
    dW = X.T @ X_mask
    dW /= num_train

    # derivative of R(W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
