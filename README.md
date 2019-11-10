# CS231n
Assignment solutions for Stanford's CS231n: Convolutional Neural Networks for Visual Recognition. (Spring 2019)

## [Assignment 1](http://cs231n.github.io/assignments2019/assignment1/)
  - **K-Nearest Neighbour Classifier:** [(notebook)](../master/assignment1/knn.ipynb)  [(source)](../master/assignment1/cs231n/classifiers/k_nearest_neighbor.py): 
    
    Vectorized and non-vectorized implementation for L2 distance. K-fold cross validation for selecting 'k'.  Z-score normalization _(general and pixelwise)_

  - **SVM Loss:** [(notebook)](../master/assignment1/svm.ipynb) [(source)](../master/assignment1/cs231n/classifiers/linear_svm.py): 
    
    Vectorized and non-vectorized implementation of SVM loss and gradients. Visualizing learned weights. 

  - **Softmax Loss:** [(notebook)](../master/assignment1/softmax.ipynb) [(source)](../master/assignment1/cs231n/classifiers/softmax.py): 
    
    Vectorized and non-vectorized implementation of Softmax loss and gradients. Visualizing learned weights. [[reference]](https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)

  - **Two Layer Neural Network:** [(notebook)](../master/assignment1/two_layer_net.ipynb) [(source)](../master/assignment1/cs231n/classifiers/neural_net.py): 
    
    Forward and backward pass of Neural network with following architecture:
    
    input &rarr; dense &rarr; relu &rarr; dense &rarr; softmax

  - **Custom Features:** [(notebook)](../master/assignment1/features.ipynb) [(source)](../master/assignment1/cs231n/features.py): 
    
    Training SVM Classifier and Neural Network using provided feature extractors - HoG (Histogram of Gradients to extract textures) and HSV Color Histogram (to extract color information).

    

    