import os
import sys
import platform
import json

import munch
import numpy as np
from scipy.stats import norm
import tensorflow as tf
from tensorflow import keras

import util
from DEVAE_network import *
import DEAE_lib

# %%
system = platform.system()
if system == "Windows":
    print("OS is Windows!!!")
    server = False
    verbose = 1


elif system == "Linux":
    print("OS is Linux!!!")
    server = True
    verbose = 2


# hyperpara
if not server:
    gpu = 0
    # example = 'Ex1GeoBrownian'
    # example = 'Ex3OU'
    # example = 'Ex4ExpDiff'
    # example = 'Ex7OU2D'
    example = "Ex4-1-1"
    # example = 'Ex11OU5D'
    # example = 'Ex13OU5D1'
    # example = 'Ex14OU5D2'
    RNN = 1
    b_p1 = 1e-2
    b_pn = 0.0
    b_m = 1e-4
    b_c = 0
    b_d = 0
    l_d = 2
    ensemble_testID = "test2"

else:
    gpu = -1
    RNN = 1
    example = sys.argv[1]
    b_p1 = float(sys.argv[2])
    b_pn = float(sys.argv[3])
    b_m = float(sys.argv[4])
    b_c = float(sys.argv[5])
    b_d = float(sys.argv[6])
    l_d = int(sys.argv[7])
    ensemble_testID = "test" + sys.argv[8]


np_seed, rn_seed, tf_seed = util.randseedIDs[ensemble_testID]
device = util.set_seed_sess(gpu, np_seed, rn_seed, tf_seed)
# %%
# choose numerical example
# example = 'Ex1GeoBrownian'
# example = 'Ex3OU'
# example = 'Ex4ExpDiff'

# %%
# load json file and create variables
json_dir = "configs/" + example + ".json"
with open(json_dir) as json_data_file:
    config = json.load(json_data_file)

config = munch.munchify(config)

DC = config.data_config
NC = config.network_config

NC.beta_kde_1d = [b_p1, b_p1]

NC.beta_moment = [b_m, b_m]
if l_d == 1:
    b_c = 0.0
    b_pn = 0.0
else:
    b_pn *= 5.0 ** (l_d - 2)

NC.beta_cov = [b_c, b_c]
NC.beta_kde_nd = [b_pn, b_pn]
NC.beta_mmd = [b_d, b_d]

NC.latent_dim = l_d


locals().update(DC)
locals().update(NC)

print("latent_dim: ", latent_dim)

model_dir = (
    model_dir
    + data_type
    + f"_z{l_d}_pdf1d{b_p1}_pdfnd{b_pn}_moment{b_m}_cov{b_c}_mmd{b_d}/"
)

# model_dir = model_dir+data_type+"_z{}".format(l_d)+\
#             "_moment{}_cov{}_mmd{}/".format(beta_moment[0], beta_cov[0], beta_mmd[0],)


# %%

# load training data
train_data_1 = np.load("data/" + eqn_name + "/" + data_type + "_train_1.npy")
train_data_2 = np.load("data/" + eqn_name + "/" + data_type + "_train_2.npy")
if n_x == 1:

    train_data_1 = train_data_1[:, :, : RNN + 1]
    train_data_2 = train_data_2[:, :, : RNN + 1]
    train_data = [train_data_1, train_data_2]

else:
    train_data_1 = train_data_1[:, :, :, : RNN + 1]
    train_data_2 = train_data_2[:, :, :, : RNN + 1]
    train_data = [train_data_1, train_data_2]


# %%


moment_normal_c = np.zeros((max_moment))
for m in range(1, max_moment + 1):
    if m % 2 == 0:
        moment_normal_c[m - 1] = (
            2 ** (-m / 2) * np.math.factorial(m) /
            np.math.factorial(int(m / 2))
        )
moment_normal_c = moment_normal_c / moment_weights
moment_normal_c = np.tile(
    moment_normal_c.reshape(1, -1, 1), (batch_size, 1, latent_dim)
)
moment_normal_c = np.reshape(moment_normal_c, (batch_size, -1))

tf.keras.backend.clear_session()

# create and compile model
cvae_2step = []
callback_2step = []
save_dir_2step = []
for i in range(2):

    encoder = make_encoder_model(n_x, n_sample[i], encoder_dim, latent_dim)
    decoder = make_decoder_model(n_x, n_sample[i], decoder_dim, latent_dim)

    kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd = (
        DEAE_lib.kde_preparation(
            example, latent_dim, kde_method, kde_range, kde_num, n_sample[i], batch_size
        )
    )

    moment_layer = MomentLayer1D(max_moment, moment_weights)
    cov_layer = CovZLayer()
    mmd_layer = FastMMDLayer(mmd_sigma, mmd_basis)

    cvae = ConditionalVAE(
        encoder,
        decoder,
        kde_layer_1d,
        kde_layer_nd,
        moment_layer,
        cov_layer,
        mmd_layer,
        pdf_normal_c_1d,
        pdf_normal_c_nd,
        moment_normal_c,
        rnn=RNN,
        beta_kde_1d=beta_kde_1d[i],
        beta_kde_nd=beta_kde_nd[i],
        beta_moment=beta_moment[i],
        beta_cov=beta_cov[i],
        beta_mmd=beta_mmd[i],
    )

    optimizer = keras.optimizers.Adam(learning_rate=lr[i])
    cvae.compile(optimizer, loss=None)
    tmp = cvae(train_data[i][:batch_size])
    cvae.summary()
    cvae_2step.append(cvae)

    save_dir = (
        eqn_name
        + "/"
        + model_dir
        + "RNN{}/sample{}/".format(RNN, n_sample[i])
        + ensemble_testID
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir_2step.append(save_dir)

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_dir + "/best_tra.h5",
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        verbose=verbose,
    )
    callback_2step.append(callback)

    # train and save model

    if i > 0:
        weights = cvae_2step[i - 1].get_weights()
        cvae_2step[i].set_weights(weights)

    with open(eqn_name + "/" + model_dir + "train_config.json", "w") as f:
        json.dump(config, f)

    # cvae_2step[i].load_weights('Ex4ExpDiff/models_2step/RNN1/sample1000/test2/cvae_1000epochs.h5')
    hist = cvae_2step[i].fit(
        x=train_data[i],
        batch_size=batch_size,
        epochs=epochs[i],
        callbacks=callback_2step[i],
        verbose=verbose,
        validation_split=0.1,
    )

    cvae_2step[i].save_weights(
        save_dir_2step[i] + "/cvae_{}epochs.h5".format(epochs[i])
    )

    with open(save_dir_2step[i] + "loss history.json", "w") as f:
        json.dump(hist.history, f)
    np.savetxt(
        save_dir_2step[i] + "/train_loss_{}epochs.csv".format(epochs[i]),
        hist.history["loss"],
    )
    np.savetxt(
        save_dir_2step[i] + "/val_loss_{}epochs.csv".format(epochs[i]),
        hist.history["val_loss"],
    )
