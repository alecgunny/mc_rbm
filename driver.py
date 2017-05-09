from __future__ import print_function
import cPickle as pickle
import sys
import numpy as np
from collections import OrderedDict
import subprocess
import os
import multiprocessing as mp
import Queue
import matplotlib.gridspec as gridspec
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
    return K.random_binomial(shape=p_h.shape, p=p_h)


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


def plot_and_save_weights(q):
    n_cols = int(np.round(np.sqrt(h_dim)))
    n_rows = n_cols if n_cols**2 >= h_dim else n_cols + 1

    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.015, hspace=0.015)
    axes = []
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i*n_rows + j
            if idx >= h_dim:
                break
            axes.append(fig.add_subplot(gs[idx]))

    output_dir = './imgs/weights/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    glyphs = []
    while True:
        try:
            W, epoch = q.get(timeout=0.5)
            if epoch is None:
                break
            if epoch == 0:
                for w, ax in zip(W, axes):
                    glyphs.append(ax.imshow(w.reshape(28,28)))
                    ax.xaxis.set_visible(False)
                    ax.yaxis.set_visible(False)
            else:
                for w, glyph, in zip(W, glyphs):
                    glyph.set_data(w.reshape(28,28))
                    glyph.set_clim(vmin=w.min(), vmax=w.max())
                    plt.draw()
            epoch = str(epoch).zfill(4)
            fig.savefig(os.path.join(output_dir, 'epoch{}.png'.format(epoch)), bbox_inches='tight', pad_inches=0)
        except Queue.Empty:
            continue

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

plt.ioff()
q = mp.Queue()
p = mp.Process(target=plot_and_save_weights, args=(q,))
p.start()
for epoch in range(epochs):
    q.put([W.get_value().T, epoch])
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

print("Done training, waiting for figures to generate...")
q.put([W.get_value().T, epoch + 1])
q.put([None, None])
p.join()
plt.ion()

print("Figures generated, writing gif of weight development")
ffmpeg_args = ['-nostats', '-loglevel', 'panic', '-hide_banner', '-y']
subprocess.call(['ffmpeg', '-i', './imgs/weights/epoch%04d.png', './imgs/weights.avi'] + ffmpeg_args)
subprocess.call(['ffmpeg', '-i', './imgs/weights.avi', '-pix_fmt', 'rgb8', '-t', '3', './imgs/weights.gif'] + ffmpeg_args)

print("loading in parameters from best validation epoch and predicting on validation set")
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
