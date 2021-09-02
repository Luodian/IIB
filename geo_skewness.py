import matplotlib as mpl
import tensorflow as tf

mpl.use('Agg')
import matplotlib.pyplot as plt

# import cvxpy as cp
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from sklearn import preprocessing

from keras.models import Model
import numpy as np
import cvxpy as cp

colors = ['black', '#377eb8', '#ff7f00', '#4daf4a',
          '#984ea3', '#a65628', '#f781bf',
          '#999999', '#e41a1c', '#dede00']
markers = ['h', '*', '<', 'o', 's', 'v', 'D']
local_dir = '/home/jupyter'

plot_dir = f'{local_dir}/plot_pdfs'
np.random.seed(seed=42)

# Load and shuffle data
data = np.loadtxt('datasets/ObesityDataSet.csv', skiprows=1, delimiter=',')
np.random.shuffle(data)

spur_ind = 18  # The 18th column inthe data corresponds to the "spurious feature"
label_ind = 20  # The 20th column corresponds to the label

# ==================
# Conversion to binary classification task:
# ==================

data = data[np.logical_or.reduce([data[:, label_ind] == x for x in [0, 1, 2, 4, 5, 6]]), :]
print(f"{data.shape[0]} datapoints remain.")
# Convert these labels to 0 and 1.
data[data[:, label_ind] <= 3, label_ind] = 0
data[data[:, label_ind] >= 4, label_ind] = 1
print(f"Labels have been simplified to {np.unique(data[:, label_ind])}")
labels = data[:, -1]
data = data[:, :-1]
# Rescale the data
# data = preprocessing.scale(data)
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
# num_data = data.shape[0]

# Create test-train split
num_data = data.shape[0]
x_train, y_train = data[:int(0.6 * num_data)], labels[:int(0.6 * num_data)]
x_test, y_test = data[int(0.6 * num_data):], labels[int(0.6 * num_data):]

# Determine which indices in the training set correspond to x_sp = y and x_sp != y
pos_train_ind = (x_train[:, spur_ind] == y_train)
neg_train_ind = (x_train[:, spur_ind] != y_train)

# Determine which indices in the test set correspond to x_sp = y and x_sp != y
pos_test_ind = (x_test[:, spur_ind] == y_test)
neg_test_ind = (x_test[:, spur_ind] != y_test)

# Select datapoints in the training set where x_sp = y and x_sp != y
pos_x_train, pos_y_train = x_train[pos_train_ind], y_train[pos_train_ind]
neg_x_train, neg_y_train = x_train[neg_train_ind], y_train[neg_train_ind]

# Select datapoints in the test set where x_sp = y and x_sp != y
pos_x_test, pos_y_test = x_test[pos_test_ind], y_test[pos_test_ind]
neg_x_test, neg_y_test = x_test[neg_test_ind], y_test[neg_test_ind]

print(f'Num training data where x_sp = y: {pos_x_train.shape[0]}')
print(f'Num training data where x_sp = y: {neg_x_train.shape[0]}')
print(f'Num test data where x_sp = y: {pos_x_test.shape[0]}')
print(f'Num test data where x_sp = y: {neg_x_test.shape[0]}')


def fit_max_margin(x_train, y_train):
    """
    Returns max-margin solution on the training datapoint
    # Arguments:
        x_train (np.array): training inputs
        y_train (np.array): 0/1 training labels
    # Returns
        weights (np.array): array of weights
        bias (float): bias value
    """

    # One could also use scipy to do this but
    # with the following code there's greater flexibility to play around with the
    # constraints and see how things change

    x_train = x_train.reshape((x_train.shape[0], -1))
    A = np.hstack([x_train, np.ones((x_train.shape[0], 1))])  # Append a "1" for the bias feature
    Ide = np.identity(x_train.shape[1])
    b_ones = np.ones(A.shape[0])
    cp_weights = cp.Variable(A.shape[1])

    # Quadratic program corresponding to minimizing ||w||^2
    # subject to y (x^T w) >= 1
    prob = cp.Problem(cp.Minimize(cp.quad_form(cp_weights[:-1], Ide)),
                      [np.diag(2 * y_train - 1) @ ((A @ cp_weights)) >= b_ones])
    prob.solve(verbose=False, solver=cp.ECOS)
    weights = cp_weights.value
    return weights[:-1], weights[-1]


def evaluate_max_margin(x_test, y_test, weights, bias):
    """
    Returns accuracy of a linear classifier on test data
    # Arguments:
        x_test (np.array): test inputs
        y_test (np.array): 0/1 test labels
        weights, bias (np.array, float): weights and bias
    # Returns
        accuracy: accuracy of the weights on the given test data
    """
    x_test = x_test.reshape((x_test.shape[0], -1))
    margins = np.matmul(x_test, weights) + bias
    accuracy = np.mean(np.multiply(margins, 2 * y_test - 1) > 0.0)
    return accuracy


mm_weights, mm_bias = fit_max_margin(x_train, y_train)
# If you want to look at how much weight the classifier assigns to different weights, plot this.
# plt.bar(np.arange(mm_weights.shape[0]),np.abs(mm_weights))
# plt.xticks(np.arange(mm_weights.shape[0]));
accuracy = evaluate_max_margin(x_test, y_test, mm_weights, mm_bias)
print(f"Test accuracy = {accuracy}")

pos_accuracy = evaluate_max_margin(pos_x_test, pos_y_test, mm_weights, mm_bias)
print(f"Majority test accuracy = {pos_accuracy}")
neg_accuracy = evaluate_max_margin(neg_x_test, neg_y_test, mm_weights, mm_bias)
print(f"Minority test accuracy = {neg_accuracy}")

(mm_weights[spur_ind]) / np.linalg.norm(mm_weights)

factor = 50  # The factor by which we want to reduce the size of the minority datapoints
print(f"Size of new majority group = {pos_x_train.shape[0]}")
print(f"Size of new minority group = {int(neg_x_train.shape[0] / factor)}")

geom_skewed_x_train = np.vstack([pos_x_train, neg_x_train[:int(neg_x_train.shape[0] / factor)]])
geom_skewed_y_train = np.concatenate([pos_y_train, neg_y_train[:int(neg_y_train.shape[0] / factor)]])

skewed_mm_weights, skewed_mm_bias = fit_max_margin(geom_skewed_x_train, geom_skewed_y_train)
pos_accuracy = evaluate_max_margin(pos_x_test, pos_y_test, skewed_mm_weights, skewed_mm_bias)
print(f"Majority test accuracy = {pos_accuracy}")
neg_accuracy = evaluate_max_margin(neg_x_test, neg_y_test, skewed_mm_weights, skewed_mm_bias)
print(f"Minority test accuracy = {neg_accuracy}")

skewed_mm_weights[spur_ind] / np.linalg.norm(skewed_mm_weights)

n_train_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024]  # Varying number of training dataset sizes.
norms = []
for n_train in n_train_list:
    norms += [[]]
    for k in range(5):
        inds = np.random.choice(np.arange(x_train.shape[0]), n_train, replace=False)  # Randomly select
        # subset of data

        # Fit max-margin on the subset of the data, but only after removing the spurious feature
        temp_mm_weights, temp_mm_bias = fit_max_margin(np.delete(x_train[inds], spur_ind, axis=1), y_train[inds])
        norms[-1] += [np.linalg.norm(temp_mm_weights)]

ys = [np.mean(y) for y in norms]
yerrs = [np.std(y) for y in norms]  # Error bars

mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)
mpl.rcParams['font.family'] = 'monospace'
plt.figure(figsize=(5, 5))
fig, ax = plt.subplots()
ax.set_xlabel('Training set size', fontsize=24)
ax.set_ylabel(r'$\ell_2$ Norm', fontsize=24)
ax.errorbar(x=np.log(n_train_list), y=ys, yerr=yerrs, marker=markers[1], color=colors[1],
            capsize=5, markersize=15)
ax.grid()
plt.tight_layout()
plt.xticks(np.log([16, 64, 256, 1024]), ['16', '64', '256', '1024']);
plt.savefig(f'obesity_geom_skew.png')

from keras.layers import Input, Dense, Activation


def linear_classifier(input_shape, use_bias=True):
    """Linear classifier and its logit output
    # Arguments
        input_shape (tensor): shape of input image tensor
        use_bias (boolean): use bias variable or not
    # Returns
        model (Model): Keras model instance
        logits (Model): Keras model instance for the logit layer
    """
    inputs = Input(shape=input_shape)
    x = Dense(1, kernel_initializer='zeros', use_bias=use_bias)(inputs)
    logits = Model(inputs=inputs, outputs=x)
    outputs = Activation('sigmoid')(x)
    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model, logits


from tensorflow.keras.optimizers import SGD
from keras.losses import BinaryCrossentropy

# The following callback allows us to store and access weights
# throughout training
from keras.callbacks import Callback


class SaveWeights(Callback):
    def __init__(self, verbose=0):
        super(SaveWeights, self).__init__()
        self.epoch = []
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.history.setdefault("Weights", []).append(self.model.layers[1].get_weights())


n_epochs = 10000
n_train = 500  # Number of training points we want in the new dataset
spur_p_list = [0.75, 0.85]  # List of spurious correlation values
# for each value we'll consider a different dataset with that level of spurious correlation
# i.e., Pr[x_sp  = y]

# In all the following lists, there'll be one entry corresponding to each of the datasets
# that we create.
histories = []
callback_list = []
# The following are for weights and biases of the max-margin trained on these datasets
stat_skewed_mm_weights_list = []
stat_skewed_mm_bias_list = []
for p in spur_p_list:
    # Create a statistically skewed training dataset with spurious correlation level = p
    n_pos = int(n_train * p)  # Number of points in the x_sp = y group (majority)
    n_neg = int(n_train * (1 - p))  # Number of points in the x_sp != y group (minority)

    # We'll create the new "duplicated "majority group (ofsize n_pos) by picking n_neg
    # unique points from it and then duplicating it (n_pos/n_neg) times
    stat_skewed_x_train = np.vstack([pos_x_train[:n_neg] for i in range(int(n_pos / n_neg))] +
                                    [neg_x_train[:n_neg]])
    stat_skewed_y_train = np.concatenate([pos_y_train[:n_neg] for i in range(int(n_pos / n_neg))] +
                                         [neg_y_train[:n_neg]])
    print(f"Num points = {stat_skewed_x_train.shape[0]}")

    model, _ = linear_classifier(input_shape=x_train.shape[1:])
    loss = BinaryCrossentropy(from_logits=False)
    model.compile(loss=loss,
                  optimizer=SGD(learning_rate=0.01),
                  metrics=['accuracy'])
    callbacks = [SaveWeights()]
    history = model.fit(stat_skewed_x_train, stat_skewed_y_train,
                        batch_size=32,
                        epochs=n_epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True, workers=4,
                        callbacks=callbacks, verbose=0)
    histories += [history]
    callback_list += [callbacks[0]]

    # Learn the max-margin classifier on this dataset and store it in a list
    stat_skewed_mm_weights, stat_skewed_mm_bias = fit_max_margin(stat_skewed_x_train, stat_skewed_y_train)
    stat_skewed_mm_weights_list += [stat_skewed_mm_weights]
    stat_skewed_mm_bias_list += [stat_skewed_mm_bias]

spur_weights_list = []  # Each entry in this list will correspond
# to one of the spurious correlation levels
for callback in callback_list:
    # Each loop corresponds to training on a particular level of spurious correlation
    spur_weights = []  # This will be a list of scalar values equal to w_sp(t)/||w(t)||
    for weights in callback.history['Weights']:
        spur_weights += [np.abs(weights[0][spur_ind]) / np.linalg.norm(weights[0])]
    spur_weights = np.array(spur_weights)
    spur_weights_list += [spur_weights]

# Plotting function
mpl.rc('xtick', labelsize=18)
mpl.rc('ytick', labelsize=18)
mpl.rcParams['font.family'] = 'monospace'
plt.figure(figsize=(5, 5))
fig, ax = plt.subplots()
ax.set_xlabel('Epoch', fontsize=24)
ax.set_ylabel('Spurious component', fontsize=17)

for k in [0, 1]:
    inds = np.arange(0, n_epochs, 100)
    ax.plot(inds, spur_weights_list[k][inds],
            linestyle='--', color=colors[k])
    inds = np.arange(0, n_epochs, 1000)
    ax.scatter(inds, spur_weights_list[k][inds], marker=markers[k],
               color=colors[k], s=200, label=f'{spur_p_list[k]}')

legend = plt.legend(fontsize=17, ncol=1, loc='upper left', title=r'$Pr[x_{sp}y > 0]$')
plt.setp(legend.get_title(), fontsize='16')
ax.grid()
# plt.ylim([0,1])
plt.tight_layout()
plt.savefig('spurious_componenets.png')

for k in range(len(spur_p_list)):
    print(f"Spurious correlation = {spur_p_list[k]}")
    pos_accuracy = evaluate_max_margin(pos_x_test, pos_y_test,
                                       stat_skewed_mm_weights_list[k], stat_skewed_mm_bias_list[k])
    print(f"Majority test accuracy = {pos_accuracy}")
    neg_accuracy = evaluate_max_margin(neg_x_test, neg_y_test,
                                       stat_skewed_mm_weights_list[k], stat_skewed_mm_bias_list[k])
    print(f"Minority test accuracy = {neg_accuracy}")
    print(f"Spurious component = {stat_skewed_mm_weights_list[k][spur_ind] / np.linalg.norm(stat_skewed_mm_weights_list[k])}")
