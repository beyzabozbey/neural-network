"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.
"""

import numpy as np
import dnn_misc


np.random.seed(123)

# example data
X = np.random.normal(0, 1, (5, 3))


# example modules
check_linear = dnn_misc.linear_layer(input_D = 3, output_D = 2)
check_relu = dnn_misc.relu()


# check_linear.forward
hat_X = check_linear.forward(X)
ground_hat_X = np.array([[ 0.42525407, -0.2120611 ],
 [ 0.15174804, -0.36218431],
 [ 0.20957104, -0.57861084],
 [ 0.03460477, -0.35992763],
 [-0.07256568,  0.1385197 ]])

if (hat_X.shape[0] != 5) or (hat_X.shape[1] != 2):
    print('Wrong output dimension of linear.forward')
else:
    max_relative_diff = np.amax(np.abs((ground_hat_X - hat_X) / (ground_hat_X + 1e-8)))
    print('max_diff_output: ' + str(max_relative_diff))
    if max_relative_diff >= 1e-6:
        print('linear.forward might be wrong')
    else:
        print('linear.forward should be correct')
    # if np.allclose(ground_hat_X, hat_X, atol=1e-08, rtol=1e-08):
    #   print('linear.forward should be correct')
    # else:
    #   print('linear.forward might be wrong')
print('##########################')


# check_linear.backward
grad_hat_X = np.random.normal(0, 1, (5, 2))
grad_X = check_linear.backward(X, grad_hat_X)

ground_grad_X = np.array([[-0.32766959,  0.13123228, -0.0470483 ],
 [ 0.22780188, -0.04838436,  0.04225799],
 [ 0.03115675, -0.32648556, -0.06550193],
 [-0.01895741, -0.21411292, -0.05212837],
 [-0.26923074, -0.78986304, -0.23870499]])

ground_grad_W = np.array([[-0.27579345, -2.08570514],
 [ 4.52754775, -0.40995374],
 [-1.2049515,   1.77662551]])

ground_grad_b = np.array([[-4.55094716, -2.51399667]])

if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
    print('Wrong output dimension of linear.backward')
else:
    max_relative_diff_X = np.amax(np.abs((ground_grad_X - grad_X) / (ground_grad_X + 1e-8)))
    print('max_diff_grad_X: ' + str(max_relative_diff_X))
    max_relative_diff_W = np.amax(np.abs((ground_grad_W - check_linear.gradient['W']) / (ground_grad_W + 1e-8)))
    print('max_diff_grad_W: ' + str(max_relative_diff_W))
    max_relative_diff_b = np.amax(np.abs((ground_grad_b - check_linear.gradient['b']) / (ground_grad_b + 1e-8)))
    print('max_diff_grad_b: ' + str(max_relative_diff_b))
    if (max_relative_diff_X >= 1e-6) or (max_relative_diff_W >= 1e-6) or (max_relative_diff_b >= 1e-6):
        print('linear.backward might be wrong')
    else:
        print('linear.backward should be correct')
print('##########################')


# check_relu.forward
hat_X = check_relu.forward(X)
ground_hat_X = np.array([[ 0.,          0.99734545,  0.2829785 ],
 [ 0.,          0.,          1.65143654],
 [ 0.,          0.,          1.26593626],
 [ 0.,          0.,          0.        ],
 [ 1.49138963,  0.,          0.        ]])

if (hat_X.shape[0] != 5) or (hat_X.shape[1] != 3):
    print('Wrong output dimension of relu.forward')
else:
    max_relative_diff = np.amax(np.abs((ground_hat_X - hat_X) / (ground_hat_X + 1e-8)))
    print('max_diff_output: ' + str(max_relative_diff))
    if max_relative_diff >= 1e-6:
        print('relu.forward might be wrong')
    else:
        print('relu.forward should be correct')
print('##########################')

# check_relu.backward
grad_hat_X = np.random.normal(0, 1, (5, 3))
grad_X = check_relu.backward(X, grad_hat_X)
ground_grad_X = np.array([[-0.,          0.92746243, -0.17363568],
 [ 0.,          0.,         -0.87953634],
 [ 0.,         -0.,         -1.72766949],
 [-0.,          0.,          0.        ],
 [-0.01183049,  0.,          0.        ]])

if (grad_X.shape[0] != 5) or (grad_X.shape[1] != 3):
    print('Wrong output dimension of relu.backward')
else:
    max_relative_diff_X = np.amax(np.abs((ground_grad_X - grad_X) / (ground_grad_X + 1e-8)))
    print('max_diff_grad_X: ' + str(max_relative_diff_X))

    if (max_relative_diff_X >= 1e-6):
        print('relu.backward might be wrong')
    else:
        print('relu.backward should be correct')
print('##########################')


#check softmax forward
inp_x = np.array([[-0.43435128,  2.20593008,  2.18678609],
       [ 1.0040539 ,  0.3861864 ,  0.73736858],
       [ 1.49073203, -0.93583387,  1.17582904],
       [-1.25388067, -0.6377515 ,  0.9071052 ],
       [-1.4286807 , -0.14006872, -0.8617549 ]])

Y = np.array([2,0,1,2,0])
Y = np.reshape(Y, (5,1))

ground_loss = 1.34731143168356
softmax_layer = dnn_misc.softmax_cross_entropy()
loss_hat = softmax_layer.forward(inp_x, Y)
if loss_hat.shape != ():
    print('Wrong output dimension of softmax_cross_entropy.forward')
else:
    if abs(loss_hat - ground_loss) >= 1e-6:
        print('softmax_cross_entropy.forward might be wrong')
    else:
        print("softmax_cross_entropy.forward should be correct")
print('##########################')

#check softmax backward
l = np.array([[0., 0., 1.],
       [1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.],
       [1., 0., 0.]])
p = np.array([[0.03476024, 0.48723938, 0.47800039],
       [0.43383818, 0.23387905, 0.33228276],
       [0.54999469, 0.04858636, 0.40141895],
       [0.08671946, 0.16058245, 0.75269809],
       [0.15648015, 0.56767049, 0.27584937]])
grad_ground = np.array([[ 0.00695205, 0.09744787, -0.10439992],
 [-0.11323236, 0.04677581, 0.06645655],
 [ 0.10999894, -0.19028273,  0.08028379],
 [ 0.01734389 , 0.03211649, -0.04946038],
 [-0.16870397 , 0.11353410,   0.05516987]])

grad_hat = softmax_layer.backward(inp_x, Y)
if grad_hat.shape != grad_ground.shape:
    print('Wrong output dimension of softmax_cross_entroy.backward')
else:
    max_relative_diff = np.amax(np.abs((grad_ground - grad_hat)/(grad_ground + 1e-9)))
    if (max_relative_diff >= 1e-6):
        print('softmax_cross_entroy.backward might be wrong')
    else:
        print('softmax_cross_entroy.backward should be correct')

