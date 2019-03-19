"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.
"""
import json
import numpy as np
import sys
import dnn_misc
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape

    def get_example(self, idx):
        batchX = np.zeros((len(idx), self.d))
        batchY = np.zeros((len(idx), 1))
        for i in range(len(idx)):
            batchX[i] = self.X[idx[i]]
            batchY[i, :] = self.Y[idx[i]]
        return batchX, batchY


def data_loader_mnist(dataset):
    # This function reads the MNIST data and separate it into train, val, and test set
    with open(dataset, 'r') as f:
        data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    Ytrain = train_set[1]
    Xvalid = valid_set[0]
    Yvalid = valid_set[1]
    Xtest = test_set[0]
    Ytest = test_set[1]

    return np.array(Xtrain), np.array(Ytrain), np.array(Xvalid), \
           np.array(Yvalid), np.array(Xtest), np.array(Ytest)


def plot_metrics(num_epochs, train_accuracy, valid_accuracy, train_loss, valid_loss, name):
    epochs = list(range(1, num_epochs + 1))
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.plot(epochs, train_accuracy)
    ax1.plot(epochs, valid_accuracy)
    ax1.legend(['train acc', 'valid acc'], loc='upper left')
    ax2.plot(epochs, train_loss)
    ax2.plot(epochs, valid_loss)
    ax2.legend(['train loss', 'valid loss'], loc='upper left')
    fig1.savefig(name + ".png")


def plot_tSNE(X, Y, plot_name):
    model = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200, n_iter=1000)
    X_tsne = model.fit_transform(X)

    color_map = {0: 'blue', 1: 'orange', 2: 'green', 3: 'red', \
                 4: 'purple', 5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}
    fig1, ax1 = plt.subplots()
    ax1.scatter(X_tsne.T[0], X_tsne.T[1], c=list(map(lambda c: color_map[c], Y)), s=50, alpha=0.5)
    fig1.savefig(plot_name)


def predict_label(f):
    # This is a function to determine the predicted label given scores
    return np.argmax(f, axis=1).astype(float).reshape((f.shape[0], -1))


def main(main_params):
    ### set the random seed ###
    np.random.seed(int(main_params['random_seed']))

    ### data processing ###
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = data_loader_mnist(dataset='mnist_subset.json')
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape
    N_test, _ = Xtest.shape
    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)
    testSet = DataSplit(Xtest, Ytest)
    ### building/defining MLP ###
    """
    In this script, we are going to build an MLP for a 10-class classification problem on MNIST.
    The network structure is input --> linear --> relu --> dropout --> linear --> softmax_cross_entropy loss
    the hidden_layer size (num_L1) is 1000
    the output_layer size (num_L2) is 10
    """
    model = dict()
    num_L1 = 1000
    num_L2 = 10

    # experimental setup
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    # optimization setting: _alpha for momentum, _lambda for weight decay
    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = 0.0
    _lambda = float(main_params['lambda'])
    _optimizer = main_params['optim']
    _epsilon = main_params['epsilon']

    # create objects (modules) from the module classes
    model['L1'] = dnn_misc.linear_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = dnn_misc.relu()
    model['L2'] = dnn_misc.linear_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = dnn_misc.softmax_cross_entropy()

    # create variables for momentum
    if _optimizer == "Gradient_Descent_Momentum":
        # creates a dictionary that holds the value of momentum for learnable parameters
        momentum = dnn_misc.add_momentum(model)
        _alpha = 0.9
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []
    train_loss_record = []
    val_loss_record = []

    ### run training and validation ###
    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            # learning_rate decay
            _learning_rate = _learning_rate * 0.1

        # shuffle the train data
        idx_order = np.random.permutation(N_train)

        for i in range(int(np.floor(N_train / minibatch_size))):

            # get a mini-batch of data
            x, y = trainSet.get_example(idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            ### forward ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            a2 = model['L2'].forward(h1)
            loss = model['loss'].forward(a2, y)

            ### backward ###
            grad_a2 = model['loss'].backward(a2, y)
            grad_h1 = model['L2'].backward(h1, grad_a2)
            grad_a1 = model['nonlinear1'].backward(a1, grad_h1)
            grad_x = model['L1'].backward(x, grad_a1)

            ### gradient_update ###
            for module_name, module in model.items():
                # model is a dictionary with 'L1', 'L2', 'nonLinear1' and 'loss' as keys.
                # the values for these keys are the corresponding objects created in line 123-126 using classes 
                # defined in dnn_misc.py

                # check if the module has learnable parameters. not all modules have learnable parameters.
                # if it does, the module object will have an attribute called 'params'. See Linear Layer for more details.
                if hasattr(module, 'params'):
                    for key, _ in module.params.items():
                        # gradient computed during the backward pass + L2 regularization term
                        # _lambda is the regularization hyper parameter
                        g = module.gradient[key] + _lambda * module.params[key]

                        if _optimizer == "Minibatch_Gradient_Descent":
                            ################################################################################
                            # TODO: Write the gradient update for the module parameter.                    #
                            # module.params[key] has to be updated with the new value.                     #
                            # parameter update will be of the form: w = w - learning_rate * dl/dw          #
                            ################################################################################

                            #
                            module.params[key] -= _learning_rate * g


                        elif _optimizer == "Gradient_Descent_Momentum":
                            ################################################################################
                            # TODO: Understand how the update differs when we use momentum.                #
                            # module.params[key] has to be updated with the new value.                     #
                            # momentum(w) = _aplha * momemtum(w) at previous step + _learning_rate * g     #
                            # parameter update will be of the form: w = w - momentum(w)                    #
                            ################################################################################
                            parameter = module_name + '_' + key
                            momentum[parameter] = _alpha * momentum[parameter] + _learning_rate * g
                            module.params[key] -= momentum[parameter]

        ### Compute train accuracy ###
        train_acc = 0.0
        train_loss = 0.0
        train_count = 0
        for i in range(int(np.floor(N_train / minibatch_size))):
            x, y = trainSet.get_example(np.arange(minibatch_size * i, minibatch_size * (i + 1)))

            ### forward ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            a2 = model['L2'].forward(h1)
            loss = model['loss'].forward(a2, y)
            train_loss += len(y) * loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_loss = train_loss / train_count
        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)
        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' + str(t + 1) + ' is ' + str(train_acc))

        ### Compute validation accuracy ###
        val_acc = 0.0
        val_loss = 0.0
        val_count = 0
        for i in range(int(np.floor(N_val / minibatch_size))):
            x, y = valSet.get_example(np.arange(minibatch_size * i, minibatch_size * (i + 1)))

            ### forward ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            a2 = model['L2'].forward(h1)
            loss = model['loss'].forward(a2, y)
            val_loss += len(y) * loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)

        val_loss = val_loss / val_count
        val_acc = val_acc / val_count

        val_acc_record.append(val_acc)
        val_loss_record.append(val_loss)

        print('Validation loss at epoch ' + str(t + 1) + ' is ' + str(val_loss))
        print('Validation accuracy at epoch ' + str(t + 1) + ' is ' + str(val_acc))

    ### Compute test accuracy ###
    ################################################################################
    # TODO: Do a forward pass on test data and compute test accuracy and loss.     #
    # populate them in test_loss and test_acc.                                     #
    ################################################################################
    test_loss = 0.0
    test_acc = 0.0
    test_count = 0
    for i in range(int(np.floor(N_test / minibatch_size))):
        x, y = testSet.get_example(np.arange(minibatch_size * i, minibatch_size * (i + 1)))

        ### forward ###
        a1 = model['L1'].forward(x)
        h1 = model['nonlinear1'].forward(a1)
        a2 = model['L2'].forward(h1)
        loss = model['loss'].forward(a2, y)
        test_loss += len(y) * loss
        test_acc += np.sum(predict_label(a2) == y)
        test_count += len(y)

    test_loss = test_loss / test_count
    test_acc = test_acc / test_count

    print('Test loss at epoch ' + str(t + 1) + ' is ' + str(test_loss))
    print('Test accuracy at epoch ' + str(t + 1) + ' is ' + str(test_acc))

    plot_metrics(num_epoch, train_acc_record, val_acc_record, train_loss_record, val_loss_record, _optimizer)
    # save file
    json.dump({'train_accuracy': train_acc_record, 'train_loss': train_loss_record,
               'val_accuracy': val_acc_record, 'val_loss': val_loss_record,
               'test_accuracy': test_acc, 'test_loss': test_loss},
              open(_optimizer + '_lr' + str(main_params['learning_rate']) +
                   '_m' + str(_alpha) +
                   '_w' + str(main_params['lambda']) +
                   '.json', 'w'))

    # plotting to understand what the network is trying to do

    # plot raw mnist data
    plot_tSNE(Xtest, Ytest, 't-SNE raw MNIST.png')
    ################################################################################
    # TODO: Vizualize the ouput of the first and second layer of the neural network#
    # on test data using t-SNE. Populate arrays 'first_layer_out' and              #
    # and 'second_layer_out'.                                                      #
    ################################################################################

    # first_layer_out = output of the neural network first layer on test data
    first_layer_out = np.zeros((N_test, num_L1), dtype=float)
    # second_layer_out = output of the neural network second layer on test data
    second_layer_out = np.zeros((N_test, num_L2), dtype=float)

    # Add your code here
    first_layer_out = model['L1'].forward(testSet.X)
    h1 = model['nonlinear1'].forward(first_layer_out)
    second_layer_out = model['L2'].forward(h1)

    ###########################################################################
    #          Please DO NOT change the following parts of the script         #
    ###########################################################################
    plot_tSNE(first_layer_out, Ytest, _optimizer + '_t-SNE_1.png')
    plot_tSNE(second_layer_out, Ytest, _optimizer + '_t-SNE_2.png')
    print('Finish running!')


###########################################################################
#          Please DO NOT change the following parts of the script         #
###########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=2)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--lambda', default=0.001)
    parser.add_argument('--num_epoch', default=20)
    parser.add_argument('--minibatch_size', default=5)
    parser.add_argument('--optim', default='Minibatch_Gradient_Descent')
    parser.add_argument('--epsilon', default=0.001)
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)