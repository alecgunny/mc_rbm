from __future__ import print_function
import cPickle as pickle
import sys
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

np.random.seed = 0 # set the seed for keras for repeatability
import keras.backend as K
from keras.initializations import glorot_normal
from keras.optimizers import RMSprop
from keras.datasets import mnist

sys.dont_write_bytecode = True
from parser import parser
args, _ = parser.parse_known_args()

h_dim         = args.h_dim
n_monte_carlo = args.n_mc
learning_rate = args.lr
epochs        = args.epochs
batch_size    = args.batch_size
reg           = args.reg


def preprocess(X):
    # flattens and binarizes images
    batch_dim = X.shape[0]
    v_dim = X.shape[1] * X.shape[2]
    X = X.reshape(batch_dim, v_dim) > 0 # binarize
    return X.astype('float32')


def clip(x):
    # quick utility function for maintaining numeric stability
    return K.clip(x, K.epsilon(), 1-K.epsilon())


def sample_bernoulli(p_h):
    # samples an uncorrelated bernoulli distribution with p(h_i = 1) = p_h_i
    r      = K.random_normal((p_h.shape[0], h_dim))
    mu     = -K.sqrt(2)*K.T.erfinv(1 - 2*clip(p_h)) # clip p_h for stability
    sample = mu + r
    sample = K.switch(K.equal(sample, 0), K.epsilon(), sample)
    return (sample + K.abs(sample)) / (2*K.abs(sample))


def compute_epoch_loss(X, func):
    # performs one epoch of training or validation and returns the associated loss
    epoch_loss = 0
    np.random.shuffle(X)
    for batch_num in range((X.shape[0] - 1) // batch_size + 1):
        epoch_loss += func(get_batch(X, batch_num))
    epoch_loss = epoch_loss / (batch_num + 1)
    return epoch_loss


def predict_on_dataset(X):
    # returns reconstructions of the dataset X as computed by the model
    num_batches  = (X.shape[0] - 1) // batch_size + 1
    predictions  = np.zeros((num_batches * batch_size, v_dim))
    for batch_num in range(num_batches):
        predictions[batch_slice(batch_num)] = predict_func(get_batch(X, batch_num))
    return predictions


# load and preprocess data
(X_train, y_train), (X_valid, y_valid) = mnist.load_data()
X_train, X_valid = preprocess(X_train), preprocess(X_valid)
v_dim = X_train.shape[-1]

# build the parameters of the RBM
W = glorot_normal(shape=(v_dim, h_dim), name='W')
a = K.zeros(shape=(v_dim,), name='a')
b = K.zeros(shape=(h_dim,), name='b')
params = [W, a, b]

# now build the model
# first build visible input and map to hidden state probabilities
v   = K.placeholder(ndim=2)
p_h = K.sigmoid(K.dot(v, W) + b)

# now monte carlo sample a few hs from p_h and map back to p(v|h) then average
p_v = 0
for i in range(n_monte_carlo):
    h    = sample_bernoulli(p_h)
    p_v += K.sigmoid(K.dot(h, W.T) + a)
p_v = clip(p_v / n_monte_carlo)

# compute the error and add in l2 regularization
loss = K.binary_crossentropy(v, p_v).mean()
for param in params:
    loss += reg * K.sum(K.square(param))

# get weight updates
optimizer = RMSprop(lr=learning_rate)
updates = optimizer.get_updates(params, {}, loss)

# now compile the train, validation, and prediction functions
train_func   = K.function(inputs=[v], outputs=loss, updates=updates)
valid_func   = K.function(inputs=[v], outputs=loss)
predict_func = K.function(inputs=[v], outputs=p_v)

# training loop
best_val    = np.inf
batch_slice = lambda i: slice(i*batch_size, (i+1)*batch_size)
get_batch   = lambda x, i: [x[batch_slice(i)]]
loop_params = OrderedDict([('train', {'X': X_train, 'func': train_func, 'loss': []}),
                           ('valid', {'X': X_valid, 'func': valid_func, 'loss': []})])
for epoch in range(epochs):
    try:
        print("Epoch {}/{}".format(epoch + 1, epochs))
        for split, split_params in loop_params.items():
            epoch_loss = compute_epoch_loss(split_params['X'], split_params['func'])
            split_params['loss'].append(epoch_loss)
            print('\t{} loss: {}'.format(split, epoch_loss))

        # if we achieve the best valid loss so far, save the parameters
        if loop_params['valid']['loss'][-1] < best_val:
            best_val = loop_params['valid']['loss'][-1]
            with open('best_weights.pkl', 'w') as f:
                pickle.dump({param.name: param.get_value() for param in params}, f)

    except KeyboardInterrupt:
        break

print("Done training, loading in parameters from best validation epoch and predicting on validation set")

# load in the best parameters to compute predictions
with open('best_weights.pkl', 'r') as f:
    params_to_load = pickle.load(f)
    for param in params:
        param.set_value(params_to_load[param.name])

predictions = {}
fig, ax = plt.subplots() # plot loss vs epoch while we're at it
for split, split_params in loop_params.items():
    predictions[split] = predict_on_dataset(split_params['X'])
    ax.plot(split_params['loss'], label=split)
ax.legend()
fig.suptitle('Model loss by epoch')
fig.show(0)

# now let's plot our model weights and see what those look like
# have to do some ugly math to get the number of plots right
w = W.get_value().T
w_fig = plt.figure(figsize=(10, 10))
root_h_dim = np.sqrt(h_dim)
rounded = np.round(root_h_dim)
n_cols, n_rows = int(rounded), int(rounded + int(rounded < root_h_dim))
leftover = h_dim - n_cols*(n_rows - 1)

for i in range(n_rows):
    if i == n_rows - 1 and leftover != 0:
        row_length = leftover
    else:
        row_length = n_cols
    for j in range(row_length):
        ax = w_fig.add_subplot(n_rows, n_cols, i*n_rows + j + 1)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.imshow(w[i*n_rows + j].reshape(28, 28))
plt.tight_layout()
w_fig.show(0)

# quick function for looking at reconstructed images after training is done
def plot_sample(i, split='valid'):
    fig = plt.figure()
    ax_left = fig.add_subplot(121)
    ax_left.xaxis.set_visible(False)
    ax_left.yaxis.set_visible(False)

    ax_rite = fig.add_subplot(122)
    ax_rite.xaxis.set_visible(False)
    ax_rite.yaxis.set_visible(False)

    x, pred = loop_params[split]['X'], predictions[split]

    ax_left.imshow(x[i].reshape(28, 28))
    ax_left.set_title('Original')

    ax_rite.imshow(pred[i].reshape(28, 28))
    ax_rite.set_title('Reconstruction')

    plt.tight_layout()
    fig.show(0)

    return fig
