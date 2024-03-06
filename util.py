import sys
import numpy as np
import tensorflow as tf
import random as rn
from tensorflow import keras

randseedIDs = {
    'test1': [17, 11, 1263],
    'test2': [25, 2345, 2091],
    'test3': [3, 313, 342],
    'test4': [416, 4159, 444],
    'test5': [572, 5319, 580],
    'test6': [666, 66, 6],
    'test7': [3453, 27, 98],
    'test8': [3478, 687, 1987],
    'test9': [87, 68, 2],
    'test10': [10, 100, 1000]
} # testID: [np_seed, rn_seed, tf_seed]

def set_seed_sess(gpu, np_seed, rn_seed, tf_seed):
    
    np.random.seed(np_seed)
    rn.seed(rn_seed)
    tf.random.set_seed(tf_seed)
    
    if gpu==-1:
        device = '/CPU:0'
        
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            print("Available GPUs:")
            for gpu in gpus:
                print(gpu)
        else:
            print("No GPU devices found.")
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        device = '/device:GPU:{}'.format(gpu)

    return device
    

class ModelCheckpoint(keras.callbacks.Callback):
    """Save the model after every batch
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        inputs: training inputs to evaluate loss
        outputs: training outputs to evaluate loss
        verbose: verbosity mode, 0 or 1.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, inputs, outputs, 
                 monitor='loss', verbose=1,
                 mode='min', period=1, rnn= False):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.inputs = inputs
        self.outputs = outputs
        self.monitor = monitor
        self.verbose = verbose
        self.period = period
        self.batches_since_last_save = 0
        self.rnn = rnn        
        self.monitor_op = np.less
        self.best = np.Inf
        
        self.history = []
    
    
    def on_batch_end(self, batch, logs=None):
        self.batches_since_last_save += 1
        if self.batches_since_last_save >= self.period:
            self.batches_since_last_save = 0
            if self.rnn:
                current = self.model.evaluate(self.inputs, self.outputs, verbose=0)[-1]
            else:
                current = self.model.evaluate(self.inputs, self.outputs, verbose=0)
            self.history.append(self.model.evaluate(self.inputs, self.outputs, verbose=0))
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print('\nBatch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (batch + 1, self.monitor, self.best,
                                     current, self.filepath))
                self.best = current
                self.model.save(self.filepath, overwrite=True)
#%%
def name_best_model_epoch(epochs):
    return 'best_model_{}epochs/'.format(epochs)

def name_model_epoch(epochs):
    return 'model_{}epochs/'.format(epochs)

def name_loss_hist(epochs):
    return 'val_hist_{}epochs.csv'.format(epochs)

class BestModelCheckpoint_v2(tf.keras.callbacks.Callback):
    """Save best model at the end of each epoch
    """

    def __init__(self, train_inputs, train_outputs, model_dir, epochs_trained,
                 epochs_totrain, save_freq, verbose=0, rnn = False, sub_models=[],save_epochs = []):
        super(BestModelCheckpoint_v2, self).__init__()
        self.train_inputs = train_inputs
        self.train_outputs = train_outputs
        self.model_dir = model_dir
        self.epochs_trained = epochs_trained # before this training begin, from last training
        self.epochs_totrain = epochs_totrain
        self.save_freq = save_freq # save best model as a separate file every save_freq epoch
        self.verbose = verbose
        self.rnn = rnn
        self.sub_models = sub_models
        self.save_epochs = save_epochs

    def on_train_begin(self, logs=None):
        if self.epochs_trained == 0:
            self.best_model = self.model
            model_loss = self.model.evaluate(self.train_inputs, self.train_outputs, verbose=0)
            self.best_sub_models = []
            if self.rnn:
                self.best_loss = model_loss[0]
            else:
                self.best_loss = model_loss
            self.loss_hist = np.array(model_loss).reshape([1,-1]) # [epoch, loss]
        else:
            self.best_model = tf.keras.models.load_model(self.model_dir + name_best_model_epoch(self.epochs_trained))
            self.best_loss = self.best_model.evaluate(self.train_inputs, self.train_outputs, verbose=0)
            self.best_sub_models = []
            if self.rnn:
                self.best_loss = self.best_loss[0]
#          
            self.loss_hist = np.genfromtxt(self.model_dir + name_loss_hist(self.epochs_trained), delimiter=',')

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            print('\nEpoch #: {}'.format(self.epochs_trained + epoch))
        model_loss = self.model.evaluate(self.train_inputs, self.train_outputs, verbose=0)
        if self.rnn:
            model_loss_main = model_loss[0]
        else:
            model_loss_main = model_loss
        epochs_trained_updated = self.epochs_trained + epoch + 1
        self.loss_hist = np.concatenate((self.loss_hist, np.array(model_loss).reshape([1,-1])), axis=0)
        if model_loss_main < self.best_loss:
            if self.verbose:
                print('loss improve from {:e} to {:e}'.format(self.best_loss, model_loss_main))
            self.best_loss = model_loss_main
            self.best_model = self.model
            self.best_sub_models = []
            for sub_model in self.sub_models:
                tmp_model = tf.keras.models.clone_model(sub_model)
                tmp_model.set_weights(sub_model.get_weights())
                self.best_sub_models.append(tmp_model)
        if epochs_trained_updated % self.save_freq == 0:
            self.best_model.save_weights(self.model_dir + name_best_model_epoch(epochs_trained_updated))
            self.model.save_weights(self.model_dir + name_model_epoch(epochs_trained_updated))
            np.savetxt(self.model_dir + name_loss_hist(epochs_trained_updated), self.loss_hist, delimiter=',')
            for best_sub_model in self.best_sub_models:
                best_sub_model.save_weights(self.model_dir + best_sub_model.name+'_best_{}epochs'.format(epochs_trained_updated))
        # for s_e in self.save_epochs:
        #     if epochs_trained_updated ==s_e:
        #         self.best_model.save(self.model_dir + name_best_model_epoch(epochs_trained_updated))
        #         self.model.save(self.model_dir + name_model_epoch(epochs_trained_updated))
        #         np.savetxt(self.model_dir + name_loss_hist(epochs_trained_updated), self.loss_hist, delimiter=',')
        #         for best_sub_model in self.best_sub_models:
        #             best_sub_model.save(self.model_dir + best_sub_model.name+'_best_{}epochs'.format(epochs_trained_updated))
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        epochs_trained_updated = self.epochs_totrain + self.epochs_trained
        self.model.save_weights(self.model_dir + name_model_epoch(epochs_trained_updated))
        self.best_model.save_weights(self.model_dir + name_best_model_epoch(epochs_trained_updated))
        
        np.savetxt(self.model_dir + name_loss_hist(epochs_trained_updated), self.loss_hist, delimiter=',')