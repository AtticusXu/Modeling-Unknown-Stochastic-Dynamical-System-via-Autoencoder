import os
import json
import argparse

import munch
import numpy as np
import tensorflow as tf
from tensorflow import keras

import DEAE_network
import DEAE_lib


def parse_arguments():
    parser = argparse.ArgumentParser(description="DEAE training script")
    parser.add_argument(
        "--ensemble_seedID", type=str, default="s1", help="Ensemble seed ID"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device number")
    parser.add_argument(
        "--Example", type=str, default="4-1-1", help="Example identifier"
    )
    return parser.parse_args()


def load_config(json_path: str) -> munch.Munch:
    with open(json_path, "r") as f:
        config = json.load(f)
    return munch.munchify(config)


def load_training_data(Example):
    train_data_1 = np.load(f"data/{Example}/train_1.npy")
    train_data_2 = np.load(f"data/{Example}/train_2.npy")
    return [train_data_1, train_data_2]


def create_model(NC, DC, Example, i):
    encoder = DEAE_network.make_encoder_model(
        NC.n_x, DC.n_sample[i], NC.encoder_dim, NC.latent_dim
    )
    decoder = DEAE_network.make_decoder_model(
        NC.n_x, DC.n_sample[i], NC.decoder_dim, NC.latent_dim
    )

    kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd = (
        DEAE_lib.kde_preparation(
            Example,
            NC.latent_dim,
            NC.kde_range,
            NC.kde_num,
            DC.n_sample[i],
            NC.batch_size,
        )
    )
    moment_layer, moment_normal_c = DEAE_lib.moment_preparation(
        NC.max_moment, NC.moment_weights, NC.batch_size, NC.latent_dim
    )

    cov_layer = DEAE_network.CovZLayer()

    return DEAE_network.ConditionalVAE(
        encoder,
        decoder,
        kde_layer_1d,
        kde_layer_nd,
        moment_layer,
        cov_layer,
        pdf_normal_c_1d,
        pdf_normal_c_nd,
        moment_normal_c,
        beta_kde_1d=NC.beta_kde_1d[i],
        beta_kde_nd=NC.beta_kde_nd[i],
        beta_moment=NC.beta_moment[i],
        beta_cov=NC.beta_cov[i],
    )


def train_model(cvae, train_data, NC, save_dir, verbose, i):
    optimizer = keras.optimizers.Adam(learning_rate=NC.lr[i])
    cvae.compile(optimizer, loss=None)
    tmp = cvae(train_data[: NC.batch_size])
    cvae.summary()

    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_dir + "/best_tra.h5",
        save_best_only=True,
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        verbose=verbose,
    )

    hist = cvae.fit(
        x=train_data,
        batch_size=NC.batch_size,
        epochs=NC.epochs[i],
        callbacks=[callback],
        verbose=verbose,
        validation_split=NC.val_split,
    )

    cvae.save_weights(save_dir + f"/cvae_{NC.epochs[i]}epochs.h5")

    with open(save_dir + "loss_history.json", "w") as f:
        json.dump(hist.history, f)

    return cvae, hist


def main():
    args = parse_arguments()
    np_seed, rn_seed, tf_seed = DEAE_lib.randseedIDs[args.ensemble_seedID]
    device = DEAE_lib.set_seed_sess(args.gpu, np_seed, rn_seed, tf_seed)

    json_dir = f"configs/Ex{args.Example}.json"
    config = load_config(json_dir)

    DC = config.data_config
    NC = config.network_config

    model_dir = f"models/Ex{args.Example}/{args.ensemble_seedID}/"

    train_data = load_training_data(args.Example)

    tf.keras.backend.clear_session()

    with tf.device(device):
        cvae_2step = []
        for i in range(2):
            cvae = create_model(NC, DC, args.Example, i)

            save_dir = model_dir + f"sample{DC.n_sample[i]}/"
            os.makedirs(save_dir, exist_ok=True)

            if i > 0:
                cvae.set_weights(cvae_2step[i - 1].get_weights())

            with open(save_dir + "/train_config.json", "w") as f:
                json.dump(config, f)

            cvae, hist = train_model(cvae, train_data[i], NC, save_dir, args.verbose, i)
            cvae_2step.append(cvae)


if __name__ == "__main__":
    main()
