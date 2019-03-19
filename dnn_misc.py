"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only classes/functions you need to implement in this template is linear_layer, relu, and softmax.
"""

import numpy as np


### Modules ###

########################################################################################
#   The following three modules (class) are what you need to complete  (check TODO)    #
########################################################################################

class linear_layer:
    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D):
        ###########################################################################
        #          Please DO NOT change the __init__() function of the class      #
        ###########################################################################
        self.params = dict()
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        self.gradient = dict()
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):
        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (i.e., X[i], where i = 1,...,N).
                The mini-batch size is N.

            Operation:
            - You are going to generate a N-by-output_D numpy array named forward_output.
            - For each row x of X (say X[i]), perform X[i] self.params['W'] + self.params['b'], and store the output in forward_output[i].
            - Please use np.XX to call a numpy function XX.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """

        ################################################################################
        # TODO: Implement the linear forward pass. Store the result in forward_output  #
        ################################################################################

        forward_output = X @ self.params['W'] + self.params['b']

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'], and X.
            - You are going to generate a N-by-input_D numpy array named backward_output.
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. X in backward_output.
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'] in self.gradient['W'].
            - Store the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['b'] in self.gradient['b'].
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss
                 w.r.t. X[i].
        """

        ##########################################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following three terms)                                            #
        # self.gradient['W'] = ? (input_D-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['W'])  #
        # self.gradient['b'] = ? (1-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['b'])        #
        # backward_output = ? (N-by-input_D numpy array, the gradient of the mini-batch loss w.r.t. X)                           #
        # only return backward_output, but need to compute self.gradient['W'] and self.gradient['b']                             #
        ##########################################################################################################################

        backward_output = grad @ np.transpose(self.params['W'])
        self.gradient['W'] = np.transpose(X) @ grad
        self.gradient['b'] = np.ones((1, len(grad))) @ grad


        return backward_output


class relu:
    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):
        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Operation:
            - You are to generate a numpy array named forward_output of the same shape of X.
            - For each element x of X, perform max{0, x}, and store it in the corresponding element of forward_output.
            - Please use np.XX to call a numpy function XX if necessary.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask to store what you may need (except X) for the use in the backward pass.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the relu forward pass. Store the result in forward_output    #
        ################################################################################

        self.mask = X > 0
        forward_output = X * self.mask

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Operation:
            - You are to generate a numpy array named backward_output of the same shape of X.
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. X, and store it in backward_output.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.
            - You can use self.mask.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in  X.
        """

        ##########################################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following term)                                                   #
        # backward_output = ? (A numpy array of the shape of X, the gradient of the mini-batch loss w.r.t. X)                    #
        ##########################################################################################################################

        backward_output = self.mask * grad

        return backward_output


class softmax_cross_entropy:
    """
        Module that computes softmax cross entropy loss.
        It has no parameters to learn.
        self.prob is an N x K matrix that can be used to store the probabilites of N samples for each of the K classes(calculated in the forward pass).
        self.expand_Y is an N x K matrix that can be used to store the one-hot encoding of the true label Y for the N training samples.
    """

    def __init__(self):
        self.expand_Y = None
        self.prob = None

    def forward(self, X, Y):
        """
            The forward pass of the softmax_cross_entropy module.

            Input:
            - X: A numpy array of of shape N_by_K where N is the mini-batch size and K is the number of classes.
            - Y: A numpy array of shape N_by_1 which has true labels for the N training samples in the minibatch.

            Operation:
            - You need to compute the softmax cross entropy loss. 
            - First, compute softmax of X using euqation-5 from project description and store it in self.prob.
            - Next, compute the one-hot encoding of true labels Y and store it in self.expand_Y.
            - Next compute the cross entropy loss between the calculated self.prob and self.expand_Y using equation-10 from project description.
            - Refer to the project document to avoid underflow and overflow.
            - Please use np.XX to call a numpy function XX if necessary.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - forward_output: A single number that indicates the cross entropy loss between softmax(X) and Y
        """

        ################################################################################
        # TODO: Implement the forward pass. Store the result in forward_output         #
        ################################################################################

        #norm_X = X - np.max(X)
        norm_X = X
        for i in range(len(norm_X)):
            max = np.max(X[i])
            X[i] -= max

        exps = np.exp(norm_X)

        self.prob = exps / np.sum(exps, axis=1, keepdims=True)

        # log(z)
        Z = exps
        for i in range(len(Z)):
            row = np.sum(Z[i])
            Z[i] = np.log(Z[i]) - np.log(row)

        # one-hot encoding
        self.expand_Y = np.zeros((Y.shape[0], X.shape[1]))
        self.expand_Y[np.arange(Y.shape[0]), np.transpose(Y.astype(int))] = 1

        # output matrix whose diagonal gives each y_i * z_i
        forward_output = -self.expand_Y @ np.transpose(Z)
        forward_output = np.sum(np.diag(forward_output)) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        """
            The backward pass of the softmax_cross_entropy module.

            Input:
            - X: A numpy array of of shape N_by_K where N is the mini-batch size and K is the number of classes.
            - Y: A numpy array of shape N_by_1 which has true labels for the N training samples in the minibatch.

            Operation:
            - You need to compute the gradient of the softmanx cross entropy loss. 
            - Make use of self.prob and self.expand_Y computed in the forward pass to avoid duplicate computation.
            - Please use np.XX to call a numpy function XX if necessary.
            - You are encouraged to use matrix/element-wise operations to avoid using FOR loop.

            Return:
            - backward_output: A matrix of shape N_by_K where N is the size of the minibatch and K is the number of classes.
        """

        ################################################################################
        # TODO: Implement the backward pass. Store the result in backward_output       #
        ################################################################################

        backward_output = self.prob - self.expand_Y
        backward_output /= X.shape[0]

        return backward_output


###########################################################################
#          Please DO NOT change the following parts of the script         #
###########################################################################
class flatten_layer:

    def __init__(self):
        self.size = None

    def forward(self, X):
        self.size = X.shape
        out_forward = X.reshape(X.shape[0], -1)

        return out_forward

    def backward(self, X, grad):
        out_backward = grad.reshape(self.size)

        return out_backward


### Momentum ###
def add_momentum(model):
    momentum = dict()
    for module_name, module in model.items():
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                momentum[module_name + '_' + key] = np.zeros(module.gradient[key].shape)
    return momentum