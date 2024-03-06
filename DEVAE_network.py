
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#%%
# Define the encoder network
def make_encoder_model(n_x, n_sample, encoder_dim, latent_dim):
    X = keras.Input(shape=(n_sample,n_x))
    X_dt = keras.Input(shape=(n_sample,n_x))

    encoder_0 = layers.Concatenate(axis=-1)([X, X_dt])
    encoder_1 = layers.Conv1D(filters = encoder_dim, kernel_size = 1, activation='elu')(encoder_0)
    encoder_2 = layers.Conv1D(filters = encoder_dim, kernel_size = 1, activation='elu')(encoder_1)
    encoder_3 = layers.Conv1D(filters = encoder_dim, kernel_size = 1, activation='elu')(encoder_2)
    
    Z = layers.Conv1D(filters = latent_dim, kernel_size = 1)(encoder_3)
    
    return keras.Model([X, X_dt], Z, name="encoder")

# Define the decoder network
def make_decoder_model(n_x, n_sample, decoder_dim, latent_dim):
    
    X = keras.Input(shape=(n_sample,n_x))
    Z = keras.Input(shape=(n_sample,latent_dim))
    
    decoder_0 = layers.Concatenate(axis=-1)([X, Z])    
    decoder_1 = layers.Conv1D(filters = decoder_dim, kernel_size = 1, activation='elu')(decoder_0)
    decoder_2 = layers.Conv1D(filters = decoder_dim, kernel_size = 1, activation='elu')(decoder_1)
    decoder_3 = layers.Conv1D(filters = decoder_dim, kernel_size = 1, activation='elu')(decoder_2)
    
    decoder_outputs = layers.Conv1D(filters = n_x, kernel_size = 1)(decoder_3)
    # X_dt = layers.Add()([X,decoder_outputs])
    X_dt = decoder_outputs
    return keras.Model([X, Z], X_dt, name="decoder")

class KDELayer1D(tf.keras.layers.Layer):
    def __init__(self, lower = -3., upper = 3., num=101, n_sample = 1000):
        super(KDELayer1D, self).__init__()
        
        self.lower = lower
        self.upper = upper
        self.num = num
        self.n = n_sample
        self.n_sample = tf.constant(self.n, dtype=tf.float32)
        quantiles = tf.linspace(start=self.lower, stop=self.upper, num = self.num)
        self.quantiles = tf.constant(quantiles, dtype=tf.float32)
        self.bw = tf.pow(self.n_sample, -1.0/5.0)
        self.norm_factor = tf.sqrt(2.0 * np.pi) * self.n_sample * self.bw
        
    def build(self, input_shape):
        q_shape = tf.concat([input_shape[:-1], [self.num]], axis=0)
        self.q_broadcast = tf.expand_dims(tf.broadcast_to(self.quantiles, q_shape), axis=-1)

    
    def call(self, inputs):
        weights = tf.exp(-0.5 * tf.square((tf.expand_dims(inputs, axis=2) - self.q_broadcast) / self.bw))
        
        return tf.reduce_sum(weights, axis=1) / self.norm_factor

class KDELayer(tf.keras.layers.Layer):
    def __init__(self, random_points, **kwargs):
        super(KDELayer, self).__init__(**kwargs)
        self.random_points = tf.cast(random_points, tf.float32)   
    
    def build(self, input_shape):
        batch_size, n_sample, d = input_shape
        self.d = d
        self.n_sample = n_sample
        self.h = (4 / (self.d + 2)) ** (1 / (self.d + 4)) * self.n_sample ** (-1 / (self.d + 4))
        
    def gaussian_kernel(self, diff):
        return (1 / ((2 * np.pi * self.h ** 2) ** (self.d / 2))) * tf.exp(-0.5 * tf.norm(diff, axis=-1) ** 2 / self.h ** 2)

    def call(self, inputs):
        # inputs shape: [batch_size, n_sample, d]

        y = tf.reshape(self.random_points, (-1, 1, 1, self.d))
        
        # Reshape inputs to shape [1, batch_size, n_sample, d]
        x = tf.reshape(inputs, (1, -1, self.n_sample, self.d))
        
        # Compute differences using broadcasting
        # y-x shape: [m, batch_size, n_sample, d]
        # Then compute kernel values
        kernel_vals = self.gaussian_kernel(y - x)
        
        # Average over n_sample and chang shape t0[batch_size, m]
        
        return tf.transpose(tf.reduce_mean(kernel_vals, axis=2))  
        
    
class MMDLayer(tf.keras.layers.Layer):
    def __init__(self, sigma):
        
        super(MMDLayer, self).__init__()
        self.sigma = sigma
        
    
    def RBFkernel(self, x, y):
        # x and y are in dimension of [Nbatch,Numdata,dim]
        dim = x.shape[2]
        batch = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        tiled_x = tf.tile(tf.reshape(x, [batch, x_size, 1, dim]), [1, 1, y_size, 1])
        tiled_y = tf.tile(tf.reshape(y, [batch, 1, y_size, dim]), [1, x_size, 1, 1])
        return tf.exp(-tf.reduce_sum((tiled_x - tiled_y)**2, axis=3) / (2*self.sigma**2))
    
    def call(self, x):
        n_sample = x.shape[1]
        dim = x.shape[2]
        
        x_kernel = (self.sigma**2/(2+self.sigma**2))**(dim/2.0)
        y_kernel = self.RBFkernel(x, x)
        xy_kernel = tf.exp(-tf.reduce_sum(x**2, axis=2)/(2*(1+self.sigma**2)))
        return x_kernel + tf.reduce_sum(y_kernel, axis=[1,2])/(n_sample*(n_sample-1)) - \
            2/n_sample * tf.reduce_sum(xy_kernel, axis=1)*\
                (self.sigma**2/(1+self.sigma**2))**(dim/2.0)

class GroupMMDLayer(tf.keras.layers.Layer):
    def __init__(self, sigma, groups = 10):
        
        super(GroupMMDLayer, self).__init__()
        self.sigma = sigma
        self.groups = groups   #change the group size here. smaller will be faster
        
        
    def RBFkernel(self, x, y):
        # x and y are in dimension of [Nbatch,Numdata,dim]
        dim = x.shape[2]
        batch = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        tiled_x = tf.tile(tf.reshape(x, [batch, x_size, 1, dim]), [1, 1, y_size, 1])
        tiled_y = tf.tile(tf.reshape(y, [batch, 1, y_size, dim]), [1, x_size, 1, 1])
        return tf.exp(-tf.reduce_sum((tiled_x - tiled_y)**2, axis=3) / (2*self.sigma**2))
    
    def call(self, x):
        # print(x.shape)
        n_sample = x.shape[1]
        gs = n_sample//self.groups
        dim = x.shape[2]
        loss_mmd_subgroup_all=0
        
        for i in range(self.groups):
            x_subgroup=x[:,i*gs+1:(i+1)*gs,:]
            x_kernel = (self.sigma**2/(2+self.sigma**2))**(dim/2.0)
            y_kernel = self.RBFkernel(x_subgroup, x_subgroup)
            xy_kernel = tf.exp(-tf.reduce_sum(x_subgroup**2, axis=2)/(2*(1+self.sigma**2)))
            loss_mmd_subgroup=x_kernel + tf.reduce_sum(y_kernel, axis=[1,2])/(gs*(gs-1)) - \
                2/gs* tf.reduce_sum(xy_kernel, axis=1)*\
                    (self.sigma**2/(1+self.sigma**2))**(dim/2.0)
            # loss_mmd_subgroup_all=loss_mmd_subgroup_all+loss_mmd_subgroup/math.sqrt(nb)
            loss_mmd_subgroup_all +=loss_mmd_subgroup/self.groups**2
        return loss_mmd_subgroup_all
   
class FastMMDLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer for Fast Maximum Mean Discrepancy (MMD).
    """
    
    def __init__(self, sigma: float, n_basis: int):
        """
        Initialize the FastMMDLayer.
        
        Parameters:
        - sigma: A scalar value for the Gaussian kernel width.
        - n_basis: Number of basis functions.
        """
        super(FastMMDLayer, self).__init__()
        self.sigma = sigma
        self.n_basis = n_basis

    def build(self, input_shape: tf.TensorShape):
        """
        Build the layer based on the input shape.
        
        Parameters:
        - input_shape: Shape of the input tensor.
        """
        self.batch_size, self.n_sample, self.dim = input_shape
        
        w = tf.random.normal((self.n_basis, self.dim))
        
        zy_exact = self.n_basis**(-1/2)*tf.concat(
                        [tf.exp(-tf.reduce_sum(w**2, axis=1)/2/self.sigma**2),
                         tf.zeros((self.n_basis))], 0)
        
        self.w = tf.constant(w,dtype = tf.float32,name='FastMMD w')
        self.zy_exact = tf.constant(zy_exact,dtype = tf.float32,name='FastMMD zy')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Forward pass for the layer.
        
        Parameters:
        - x: Input tensor.
        
        Returns:
        - Output tensor after applying the FastMMD transformation.
        """
        wx = tf.matmul(self.w, tf.transpose(x, perm=[0, 2, 1])) / self.sigma
        zx = tf.concat([tf.cos(wx), tf.sin(wx)], 1)
        zx_mean = tf.reduce_sum(zx, 2) / self.n_sample * self.n_basis**(-1/2)
        
        return zx_mean - self.zy_exact
                
                
                
class MomentLayer1D(layers.Layer):
    def __init__(self, max_order = 6, moment_weights = [1,1,2,3,8,15]):
        super(MomentLayer1D, self).__init__()
        
        self.max_order = max_order
        self.moment_weights = moment_weights

    def call(self, inputs):
        
        # Calculate the mean
        mean = tf.reduce_mean(inputs, axis=1, keepdims=True)

        # Calculate the centered inputs
        centered_inputs = inputs - mean

        # Calculate the moments
        moments = [mean[:,0]]
        for i in range(1, self.max_order):
            moment = tf.reduce_mean(tf.pow(centered_inputs, i+1), axis=1)
            moments.append(moment/self.moment_weights[i])

        # Concatenate the moments along the last axis
        output = tf.concat(moments, axis=-1)

        return output

class CovarianceLayer1D(layers.Layer):
    def __init__(self):
        super(CovarianceLayer1D, self).__init__()

    def call(self, inputs):
        input_A, input_B = inputs


        # Subtract the means from the inputs
        input_A_mean_subtracted = input_A - tf.reduce_mean(input_A, axis=1, keepdims=True)
        input_B_mean_subtracted = input_B - tf.reduce_mean(input_B, axis=1, keepdims=True)
        
        sig_A = tf.sqrt(tf.reduce_mean(tf.pow(input_A_mean_subtracted, 2), axis=1))
        sig_A = tf.expand_dims(sig_A, axis=-1)
        sig_B = tf.sqrt(tf.reduce_mean(tf.pow(input_B_mean_subtracted, 2), axis=1))
        sig_B = tf.expand_dims(sig_B, axis=-2)

        # Calculate the covariance matrix
        covariance = tf.matmul(input_A_mean_subtracted, input_B_mean_subtracted, transpose_a=True) 
        covariance = covariance / tf.cast(tf.shape(input_A_mean_subtracted)[1] - 1, tf.float32)
        covariance = covariance / (sig_A*sig_B)
        return covariance

class CovZLayer(tf.keras.layers.Layer):
    def __init__(self,):
        super(CovZLayer, self).__init__()
        self.cov_layer = CovarianceLayer1D()
    def call(self, inputs):
        ndim = inputs.shape[-1]
        covZ = []
        for i in range(ndim-1):
            for j in range(i+1,ndim):
                covZ.append(self.cov_layer([inputs[:,:,i:i+1],inputs[:,:,j:j+1]] ))
        return  tf.concat(covZ, axis=-1)    
        
# Define the conditional VAE model
class ConditionalVAE(keras.Model):
    def __init__(self, encoder, decoder, kde_layer_1d, kde_layer_nd, moment_layer, cov_layer, mmd_layer,
                  pdf_normal_1d, pdf_normal_nd, moment_normal, rnn = 1, 
                 beta_kde_1d=1.0, beta_kde_nd = 0.0, beta_moment = 0.0, beta_cov = 0.0, beta_mmd = 0.0):
        super(ConditionalVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        self.kde_layer_1d = kde_layer_1d
        self.pdf_normal_1d = tf.constant(pdf_normal_1d,dtype = tf.float32)
        
        
        
        self.moment_layer = moment_layer
        self.moment_normal = tf.constant(moment_normal,dtype = tf.float32)
        
        
        
        self.rnn = rnn
        self.beta_kde_1d = beta_kde_1d
        self.beta_kde_nd = beta_kde_nd
        self.beta_moment = beta_moment
        self.beta_cov = beta_cov
        self.beta_mmd = beta_mmd
        
        if self.beta_cov > 0:
            self.cov_layer = cov_layer
        
        if self.beta_mmd > 0:
            self.mmd_layer = mmd_layer
            
        if self.beta_kde_nd > 0:
            self.kde_layer_nd = kde_layer_nd
            self.pdf_normal_nd = tf.constant(pdf_normal_nd,dtype = tf.float32)
    
        
    def encode(self, x, x_dt):
        z = self.encoder([x,x_dt])
        return z
    
    def decode(self, x, z):
        x_dt_hat = self.decoder([x, z])
        return x_dt_hat
    
    
    def one_step(self, inputs, step):
        
        # compute z and x'_dt
        x,x_dt = inputs
        dx = x_dt-x
        z = self.encoder([x,dx])
        outputs = self.decoder([x, z])
        
        
        # add loss
        
        z_moment = self.moment_layer(z)
        
        m_loss_2 = keras.losses.MeanSquaredError()(z_moment, self.moment_normal)
        MSE_loss = keras.losses.MeanSquaredError()(x_dt, outputs)
        
        vae_loss = MSE_loss + self.beta_moment*(m_loss_2)
        
        # add metric
        self.add_metric(MSE_loss, name = 'MSE_s{}'.format(step))
        
        self.add_metric(m_loss_2, name = 'moment_s{}'.format(step))

        if self.beta_kde_1d > 0:
            z_pdf_1d = self.kde_layer_1d(z)
            kde_1d_loss_2 = keras.losses.MeanSquaredError()(z_pdf_1d, self.pdf_normal_1d)
            vae_loss = vae_loss + self.beta_kde_1d * (kde_1d_loss_2)
            self.add_metric(kde_1d_loss_2, name = 'kde_1d_s{}'.format(step))
        
        if self.beta_kde_nd > 0:
            z_pdf_nd = self.kde_layer_nd(z)
            kde_nd_loss_2 = keras.losses.MeanSquaredError()(z_pdf_nd, self.pdf_normal_nd)
            vae_loss = vae_loss + self.beta_kde_nd * (kde_nd_loss_2)
            self.add_metric(kde_nd_loss_2, name = 'kde_nd_s{}'.format(step))
        
        if self.beta_cov > 0:
            z_cov = self.cov_layer(z)
            zc_loss_2 = tf.reduce_mean(tf.math.square(z_cov))
            vae_loss = vae_loss + self.beta_cov * (zc_loss_2)
            self.add_metric(zc_loss_2, name = 'z_cov_s{}'.format(step))
            
        if self.beta_mmd > 0:
            z_mmd = self.mmd_layer(z)
            mmd_loss = tf.reduce_mean(tf.math.square(tf.norm(z_mmd,axis=-1)))
            vae_loss = vae_loss + self.beta_mmd * (mmd_loss)
            self.add_metric(mmd_loss, name = 'mmd_s{}'.format(step))
        
        # add loss
        self.add_loss(vae_loss)
        
        
        return outputs
    
    def call(self, inputs):
        if len(inputs.shape) == 3:
            outputs = inputs[:,:,0:1]
            for i in range(self.rnn):
                onestep_inputs = (outputs,inputs[:,:,i+1:i+2])
                outputs = self.one_step(onestep_inputs, i)
            return outputs
        else:
            outputs = inputs[:,:,:,0]
            for i in range(self.rnn):
                onestep_inputs = (outputs,inputs[:,:,:,i+1])
                outputs = self.one_step(onestep_inputs, i)
            return outputs

if __name__ == '__main__':

    data = np.random.normal(0,1,(20,1000,2))
    m_l = MomentLayer1D(6,[1.0,1.0,2.0,3.0,8.0,15.0])
    
    moment = m_l(data).numpy()
    #%%
    from scipy.stats import norm
    import DEAE_lib
    import matplotlib.pyplot as plt
    latent_dim = 5
    n_sample = 1000
    batch = 20
    kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd = DEAE_lib.kde_preparation(
                                           "Ex14OU5D2", latent_dim, "combine",
                                           [-3.0,3.0], 100, n_sample, batch)
    input_A_data = tf.random.normal((batch, n_sample, latent_dim)).numpy()
    # input_A_data[:,:,-1] = input_A_data[:,:,-1]*0.6+input_A_data[:,:,0]*0.4
    pdf_1d =kde_layer_1d(input_A_data).numpy()
    p_loss_1d = keras.losses.MeanSquaredError()(pdf_1d, pdf_normal_c_1d).numpy()*0.1
    
    pdf_nd =kde_layer_nd(input_A_data).numpy()
    p_loss_nd = keras.losses.MeanSquaredError()(pdf_nd, pdf_normal_c_nd).numpy()*5.0**(latent_dim-2)
    
    print(p_loss_1d)
    print(p_loss_nd)
    #%%
    # n_sample = 1000
    # batch = 1000
    # covariance_layer = CovarianceLayer1D()
    # input_A_data = tf.random.normal((batch, n_sample, 1))
    # input_B_data = tf.random.normal((batch, n_sample, 1))
    # output_data = covariance_layer([input_A_data, input_B_data]).numpy()
    # mean_cov = tf.reduce_mean(tf.math.square(output_data)).numpy()
    #%%
    # n_sample = 1000
    # batch = 1000
    # cozv_layer = CovZLayer()
    # input_A_data = tf.random.normal((batch, n_sample, 3))
    # output_data = cozv_layer(input_A_data).numpy()
    # mean_cov = tf.reduce_mean(tf.math.square(output_data)).numpy()
    #%%
    # n_sample = 1000
    # sig =  0.01
    # mmd = MMDLayer(0.01)
    
    # out = mmd(input_B_data).numpy()
    # mean_mmd = tf.reduce_mean(out).numpy()
    # n_sample = 1000
    # sig =  0.01
    # input_B_data = tf.random.normal((20, 1000, 2))
    # mmd = MMDLayer(0.01)
    # import time  
    # mmd_group=GroupMMDLayer(0.01,10)
    # t0 = time.process_time()
    # out = mmd(input_B_data).numpy()
    # t1 =  time.process_time()
    # print(t1-t0)
    # t2 = time.process_time()
    # out_group = mmd_group(input_B_data).numpy()
    # t3 =  time.process_time()
    # print(t3-t2)
    # ou1=np.mean(out)

    # ou2=np.mean(out_group)
    # mean = np.mean(out)
    #%%
    batch = 100
    dim = 5
    n_sample = 1000
    
    sig =  0.1
    n_basis = 256
    fast_mmd = FastMMDLayer(sig,n_basis)
    import time
    
    
    mmd = MMDLayer(sig)

    
    # mmd_group=GroupMMDLayer(sig,20)
    
    
    
    input_B_data = tf.random.normal((batch, n_sample, dim)).numpy()
    #%%
    # input_B_data[:,:,1] = input_B_data[:,:,0]
    rounds = 1
    t0= time.process_time()
    for i in range(rounds):
        fast_out = fast_mmd(input_B_data).numpy()
    t1= time.process_time()
    print(t1-t0)
    
    t0= time.process_time()
    for i in range(rounds):
        out = mmd(input_B_data).numpy()
    t1= time.process_time()
    print(t1-t0)
    
    # t0= time.process_time()
    # for i in range(rounds):
    #     group_out = mmd_group(input_B_data).numpy()
    # t1= time.process_time()
    # print(t1-t0)
   
    mean_fast = tf.reduce_mean(tf.math.square(tf.norm(fast_out,axis=-1))).numpy()
    var_fast = np.var(tf.math.square(tf.norm(fast_out,axis=-1)).numpy())
    print('fast mean {}'.format(mean_fast))
   
    print('var {}'.format(var_fast))
    
    # mean_group = tf.reduce_mean(group_out).numpy()
    mean = tf.reduce_mean(out).numpy()
    print('mean {}'.format(mean))
    print('err {}'.format((mean-mean_fast)/mean))
    #%%
    cozv_layer = CovZLayer()
    output_data = cozv_layer(input_B_data).numpy()
    mean_cov = tf.reduce_mean(tf.math.square(output_data)).numpy()
    #%%
    # m_l = MomentLayer1D(6,[1.0,1.0,2.0,3.0,8.0,15.0])
    # output_ml = cozv_layer(input_B_data).numpy()
    # mean_cov = keras.losses.MeanSquaredError()(output_ml, moment_normal)
    #%%
    # input_B_data = tf.random.normal((20, 1000, 2))
    # x = np.linspace(-3, 3, 101)
    # 
    # norm_pdf = norm.pdf(x)
    
    # kde_nd_layer = KDELayer(x)
    # kde_1d_layer = KDELayer1D()
    
    # pdf_nd = kde_nd_layer(input_B_data).numpy()
    # pdf_1d = kde_1d_layer(input_B_data).numpy()
    # #%%
    # import matplotlib.pyplot as plt
    # id1 = 6
    # plt.plot(x, pdf_nd[id1,:],'r',label='kde nd')
    # plt.plot(x, pdf_1d[id1,:,0],'b',label='kde 1d')
    
    #%%
    # m = 1000
    # batch_size = 100 
    # n_sample = 10000
    # d = 2
    # input_B_data = tf.random.normal((batch_size, n_sample, d))
    # points = np.zeros((m,d))
    # s = 0
    # while s < m:
    #     point = np.random.uniform(-3, 3, d)
    #     if np.linalg.norm(point) <= 3:
    #         points[s,:] = point
    #         s+=1
    # kde_nd_layer = KDELayer(points)
    # pdf_nd = kde_nd_layer(input_B_data).numpy()
    
#%%
    # import matplotlib.pyplot as plt
    # import os

    # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    
    # x = points[:, 0]
    # y = points[:, 1]
    # z = pdf_nd[1,:]
    # z_ = (1 / (2 * np.pi)) * np.exp(-0.5 * np.linalg.norm(points,2,1)**2)
    # z_ = np.tile(z_.reshape(1, m), (batch_size, 1))
    # # ax.scatter(x, y, z, marker='o')
    # # ax.scatter(x, y, z_, marker='*')
    
    # # ax.set_xlabel('X')
    # # ax.set_ylabel('Y')
    # # ax.set_zlabel('PDF Value')
    
    
    # # plt.show()
    
    # loss = keras.losses.MeanSquaredError()(z_,pdf_nd).numpy()

    
    
