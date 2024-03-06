import os
import sys
import platform
import json
import munch

import numpy as np
import scipy
import scipy.io as sio
from scipy.stats import norm
system = platform.system()
import matplotlib
if system == "Linux":
    print("OS is Linux!!!")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from Evaulation import Evaluate
from DEVAE_network import *
from DEAE_lib import *
#%%
# choose numerical example
# example = 'Ex1GeoBrownian'
# example = 'Ex3OU'
# example = 'Ex4ExpDiff'
# example = 'Ex5Trig'
# example = 'Ex6ExpOU'
example = 'Ex7OU2D'
# example = 'Ex8DoubleWell'
# example = 'Ex9Expdis'
# example = 'Ex11OU5D'
# example = 'Ex12OU5D5'
# example = 'Ex13OU5D1'
# example = 'Ex14OU5D2'
# example = 'Ex15OU5D4'
#%%
# load json file and create variables
json_dir = 'jsons/'+example+'.json'
with open(json_dir) as json_data_file:
    config = json.load(json_data_file)

config = munch.munchify(config)

DC = config.data_config
NC = config.network_config

locals().update(DC)
locals().update(NC)

latent_dim = 2

if latent_dim ==1:
    beta_cov = [0.0,0.0] 
    beta_kde_nd = [0.0,0.0] 
else:
    beta_kde_nd[0] *= 5.0**(latent_dim-2)

# beta_kde[0],beta_kde[1] = 0.0, 0.0
# beta_moment[0],beta_moment[1] = 0.0, 0.0
# beta_cov[0],beta_cov[1] = 0.01, 0.01
# beta_mmd[0],beta_mmd[1] = 0.01, 0.01

# beta_kde[0],beta_kde[1] = 1.0, 1.0
# beta_moment[0],beta_moment[1] = 0.01, 0.01
# beta_cov[0],beta_cov[1] = 0.0, 0.0
# beta_mmd[0],beta_mmd[1] = 0.0, 0.0

if n_x ==1:
    
    # case_dir = data_type+"_pdf{}-{}_moment{}-{}_cov{}-{}/".format(beta_kde_1d[0],beta_kde_1d[1],
    #                                                               beta_moment[0],beta_moment[1],
    #                                                               beta_cov[0],beta_cov[1],)
    case_dir =data_type+ f"_z{latent_dim}_pdf1d{beta_kde_1d[0]}_pdfnd{beta_kde_nd[0]}_moment{beta_moment[0]}_cov{beta_cov[0]}_mmd{beta_mmd[0]}/"
elif n_x ==5:
    # case_dir = data_type+"_z{}".format(latent_dim)+"_pdf{}_".format(beta_kde_1d[0])+\
    #             kde_method+"_moment{}_cov{}_mmd{}/".format(beta_moment[0],beta_cov[0],
    #                                                               beta_mmd[0],)
    case_dir =data_type+ f"_z{latent_dim}_pdf1d{beta_kde_1d[0]}_pdfnd{beta_kde_nd[0]}_moment{beta_moment[0]}_cov{beta_cov[0]}_mmd{beta_mmd[0]}/"
else:
    # case_dir = data_type+"_pdf{}".format(beta_kde_1d[0])+\
    #             kde_method+"_moment{}_cov{}_mmd{}/".format(beta_moment[0],beta_cov[0],
    #                                                               beta_mmd[0],)
    case_dir =data_type+ f"_z{latent_dim}_pdf1d{beta_kde_1d[0]}_pdfnd{beta_kde_nd[0]}_moment{beta_moment[0]}_cov{beta_cov[0]}_mmd{beta_mmd[0]}/"

model_dir = model_dir+case_dir
#%%

x_test = np.load('data/'+eqn_name+'/test.npy')


N_seeds = 10 # for ensemble prediction
Seperate_seeds = range(N_seeds)

best_model_dir = eqn_name+'/'+model_dir+'RNN{}/sample{}/'.format(RNN,n_sample[-1])

plot_dir = 'plots/'+example+'/'+case_dir

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)



#%%

moment_normal_c = np.zeros((max_moment))
for m in range(1,max_moment+1):
    if m%2==0:
        moment_normal_c[m-1] = 2**(-m/2)*np.math.factorial(m)/np.math.factorial(int(m/2))
moment_normal_c = moment_normal_c/moment_weights
moment_normal_c = np.tile(moment_normal_c.reshape(1, -1, 1), (batch_size, 1, latent_dim))
moment_normal_c = np.reshape(moment_normal_c, (batch_size,-1))
#%%

NN_save = []
for seed in Seperate_seeds:
    
    best_seeds_dir = best_model_dir+'test{}/'.format(seed+1)+'best_tra.h5'
    encoder = make_encoder_model(n_x, n_sample[1], encoder_dim, latent_dim)
    decoder = make_decoder_model(n_x, n_sample[1], decoder_dim, latent_dim)
    # if n_x ==5 and kde_method =='combine':
    kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd = kde_preparation(
                                                example, latent_dim, kde_method,
                                                kde_range, kde_num, n_sample[1], batch_size)
    # else:
    #     kde_layer_1d, pdf_normal_c_1d = kde_preparation(
    #                                             example, latent_dim, kde_method,
    #                                             kde_range, kde_num, n_sample[1], batch_size)
        # kde_layer_nd, pdf_normal_c_nd =None,0
    moment_layer = MomentLayer1D(max_moment, moment_weights)
    cov_layer = CovZLayer()
    mmd_layer = FastMMDLayer(mmd_sigma,mmd_basis)
    
    vae_seeds = ConditionalVAE(encoder, decoder, kde_layer_1d, kde_layer_nd, moment_layer,
                          cov_layer, mmd_layer, pdf_normal_c_1d, pdf_normal_c_nd,
                          moment_normal_c, rnn = RNN, 
                          beta_kde_1d = beta_kde_1d[1], beta_kde_nd = beta_kde_nd[1],
                          beta_moment = beta_moment[1], beta_cov = beta_cov[1],
                          beta_mmd = beta_mmd[1])
    if n_x ==1:
        tmp = vae_seeds(x_test[:1])
    else:
        tmp = vae_seeds(x_test[:,:,:,:2]).numpy()
    vae_seeds.load_weights(best_seeds_dir)
    NN_save.append(vae_seeds)

#%%

# operator plot with fix x0
if n_x ==1:
    x_1step_NN = np.zeros((N_test,n_sample[1],1,2))
    x_1step_NN[:,:,:,0] = 0.4
    z_G = np.linspace(kde_range[0], kde_range[1],n_sample[1])
    z_NN = np.tile(z_G, (N_test,1)).reshape(N_test,n_sample[1],1,1)
    z_minus = -z_G
    x_G = np.reshape(x_1step_NN[0,:,:,0],(-1))
    for TEST_ID in range(N_seeds):
    
        y_1step = G_1step(x_G,z_G,example)
        y_1step_m = G_1step(x_G,z_minus, example)
        x_1step_NN[:,:,:,1] = NN_save[TEST_ID].decode(x_1step_NN[:,:,:,0],z_NN).numpy()
        y_1step_NN = np.reshape(x_1step_NN[0,:,:,1], (-1))
        
        fig1, ax1 = plt.subplots(figsize=[6,4])
        ax1.plot(z_G,y_1step,'blue', label=r'$\mathrm{D}_{\Delta}\left(x_0, z\right)$')
        ax1.plot(z_G,y_1step_m,'green',label=r'$\mathrm{D}_{\Delta}\left(x_0, -z\right)$')
        ax1.plot(z_G,y_1step_NN,'r--',label=r'$\widetilde{\mathrm{D}}_{\Delta}\left(x_0, z\right)$')
        
        
        ax1.set_xlabel('z', {"size": 14})
        ax1.set_ylabel('$x_1$', {"size": 14})
        # plt.title('D and G with x ={} TESTID{}'.format(x_1step_NN[0,0,0,0],TEST_ID))
        ax1.legend(prop={"size": 11})
        plt.savefig(plot_dir+'/operator_TESTID{}.pdf'.format(TEST_ID),dpi=600, bbox_inches='tight')
        plt.show()
        plt.close('all')
#%%
if n_x ==1:
    z_G = np.linspace(kde_range[0], kde_range[1], n_sample[1])
    x_G = np.linspace(plot_range[0], plot_range[1], 10*N_test)
    z_minus = -z_G
    
    Z_plot, X_plot, = np.meshgrid(z_G, x_G)
    y_1step = G_1step(X_plot,Z_plot,example)
    
    Z_plot_minus, X_plot_minus,  = np.meshgrid(z_minus, x_G)
    y_1step_minus = G_1step( X_plot_minus, Z_plot_minus, example)
    
    
    z_G = np.linspace(kde_range[0], kde_range[1], n_sample[1])
    x_G = np.linspace(plot_range[0], plot_range[1], 10*N_test)
    Z_plot, X_plot, = np.meshgrid(z_G, x_G)
    z_NN = Z_plot.reshape(-1,n_sample[1],1,1)
    x_NN = X_plot.reshape(-1,n_sample[1],1,1)
    y_1step_NN = np.zeros((10*N_test,n_sample[1],1))
    for TEST_ID in range(10):
        for j in range(10):
            y_1step_NN[j*N_test:(j+1)*N_test] = NN_save[TEST_ID].decode(x_NN[j*N_test:(j+1)*N_test],z_NN[j*N_test:(j+1)*N_test]).numpy()
        
        y_plot_NN = y_1step_NN[:,:,0]
        
        print(np.max(abs(y_plot_NN-y_1step)))
        print(np.max(abs(y_plot_NN-y_1step_minus)))
        
        levels = np.linspace(0.16, 1.84, 71)
        err_levels = np.linspace(0, 0.0028, 71)
        if np.max(abs(y_plot_NN-y_1step))<0.003:
            
            fig, axs = plt.subplots(1, 3, figsize=(19, 5))
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.5)
            
            
            # levels = np.linspace(np.min(y_1step), np.max(y_1step), 71)
            # D(x,z)
            ax0 = plt.subplot(gs[0])
            
            contour_G = ax0.contourf(Z_plot, X_plot, y_1step, levels=levels)
            ax0.set_ylabel('$x_0$', {"size": 14})
            ax0.set_xlabel('$z$', {"size": 14})
            ax0.set_title(r'$\mathrm{D}_{\Delta}\left(x_0, f(z)\right)$')
            
            # Dt(x,z)
            ax1 = plt.subplot(gs[1])
            contour_NN = ax1.contourf(Z_plot, X_plot, y_plot_NN, levels=levels)
            ax1.set_xlabel('$z$', {"size": 14})
            ax1.set_title(r'$\widetilde{\mathrm{D}}_{\Delta}\left(x_0, z\right)$')
            
            #color bar
            cbar_ax1 = plt.subplot(gs[2])
            color_bar_1 = fig.colorbar(contour_NN, cbar_ax1)
            color_bar_1.set_ticks(levels[0::10])
            
    
            # D(x,z)-Dt(x,z)
            
            # err_levels = np.linspace(np.min(abs(y_plot_NN-y_1step)), np.max(abs(y_plot_NN-y_1step)), 71)
            ax2 = plt.subplot(gs[3])
            contour_err = ax2.contourf(Z_plot, X_plot, abs(y_plot_NN-y_1step), levels=err_levels, cmap='plasma')
            ax2.set_xlabel('$z$', {"size": 14})
            ax2.set_title(r'$|\widetilde{\mathrm{D}}_{\Delta}\left(x_0, z\right)-\mathrm{D}_{\Delta}\left(x_0, f(z)\right)|$')
            
            #color bar
            cbar_ax2 = plt.subplot(gs[4])
            color_bar_2 = fig.colorbar(contour_err, cbar_ax2)
            color_bar_2.set_ticks(err_levels[0::10])
    
            # plt.subplots_adjust(wspace=0.3)
            plt.savefig(plot_dir+'/D-G_TESTID{}.png'.format(TEST_ID),dpi=500, bbox_inches='tight')
            plt.show()
            plt.close('all')
        elif np.max(abs(y_plot_NN-y_1step_minus))<0.003:
            fig, axs = plt.subplots(1, 3, figsize=(19, 5))
            gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05], wspace=0.5)
            
            # D(x,z)
            ax0 = plt.subplot(gs[0])
            
            contour_G = ax0.contourf(Z_plot, X_plot, y_1step_minus, levels=levels)
            ax0.set_ylabel('$x_0$', {"size": 14})
            ax0.set_xlabel('$z$', {"size": 14})
            ax0.set_title(r'$\mathrm{D}_{\Delta}\left(x_0, f(-z)\right)$')
            
            # Dt(x,z)
            ax1 = plt.subplot(gs[1])
            contour_NN = ax1.contourf(Z_plot, X_plot, y_plot_NN, levels=levels)
            ax1.set_xlabel('$z$', {"size": 14})
            ax1.set_title(r'$\widetilde{\mathrm{D}}_{\Delta}\left(x_0, z\right)$')
            
            #color bar
            cbar_ax1 = plt.subplot(gs[2])
            color_bar_1 = fig.colorbar(contour_NN, cbar_ax1)
            color_bar_1.set_ticks(levels[0::10])
    
            # D(x,z)-Dt(x,z)
           
           
            ax2 = plt.subplot(gs[3])
            contour_err = ax2.contourf(Z_plot, X_plot, abs(y_plot_NN-y_1step_minus), levels=err_levels, cmap='plasma')
            ax2.set_xlabel('$z$', {"size": 14})
            ax2.set_title(r'$|\widetilde{\mathrm{D}}_{\Delta}\left(x_0, z\right)-\mathrm{D}_{\Delta}\left(x_0, f(-z)\right)|$')
            
            #color bar
            cbar_ax2 = plt.subplot(gs[4])
            color_bar_2 = fig.colorbar(contour_err, cbar_ax2)
            color_bar_2.set_ticks(err_levels[0::10])
    
            plt.tight_layout()
            plt.savefig(plot_dir+'/D-Gm_TESTID{}.png'.format(TEST_ID),dpi=500, bbox_inches='tight')
            plt.show()
            plt.close('all')


#%%
# TEST_ID = 5
N_test = x_test.shape[0]
test_step = x_test.shape[-1]-1
x_pred = np.zeros_like(x_test)
x_pred[:,:,:,0] = x_test[:,:,:,0]


z = np.random.normal(size=(N_test,n_sample[1], test_step, latent_dim))
rand_seeds = np.random.randint(N_seeds, size=(test_step,))
for t in range(test_step):
    # z_tmp = NN_save[TEST_ID].encode(x_test[:,:,t:t+1],x_test[:,:,t+1:t+2])
    NN_id = rand_seeds[t]
    # NN_id = 0
    z_tmp = z[:,:,t]
    
    x_pred[:,:,:,t+1] += NN_save[NN_id].decode(x_pred[:,:,:,t],z_tmp)


x_pred_e = np.reshape(x_pred,(-1,n_x,test_step+1))
x_pred_mean = np.mean(x_pred_e,axis=0)
x_pred_std = np.std(x_pred_e,axis=0,ddof=1)

# np.save(f'N3Z{latent_dim}_predmean.npy', x_pred_mean)
# np.save(f'N3Z{latent_dim}_predstd.npy', x_pred_std)



#%%
# x_1step = np.load('data/'+eqn_name+'/1step.npy')
x_1step = x_test[:,:,:,:2]

z = np.random.normal(size=(N_test,n_sample[1], 1, latent_dim))

z_encode = NN_save[NN_id].encode(x_1step[:,:,:,0],x_1step[:,:,:,1])
# x_pred = NN_save[0].decode(x_1step[:,:,:,0],z[:,:,0])
x_pred_1step = NN_save[0].decode(x_1step[:,:,:,0],z_encode)
    
x_pred_1step = np.reshape(x_pred_1step,(-1,n_x))
x_true = np.reshape(x_1step[:,:,:,1],(-1,n_x))

m_pred = np.mean(x_pred_1step,0)
v_pred = np.var(x_pred_1step,0)
m_true = np.mean(x_true,0)
v_true = np.var(x_true,0)
# np.save('prediction.npy',x_pred)
# np.save('reference.npy',x_true)
print(f"true mean: {m_true}, true var: {v_true}")
print(f"pred mean: {m_pred}, pred var: {v_pred}")
print(abs(v_pred-v_true)/v_true)
#%%
fig, axes = plt.subplots(1, n_x, figsize=(5*n_x, 4))
for i in range(n_x):
    mean = m_true[i]
    std = np.sqrt(v_true[i])
    
    if x_true[0,i] == x_true[1,i]:
        axes[i].axvline( x_true[0,i],color ='orange',label = 'reference')
        hist_left, hist_right = x_true[1,i]-0.1*abs(x_true[1,i]),x_true[1,i]+0.1*abs(x_true[1,i])
    else:
        x_axis = np.linspace(scipy.stats.norm.ppf(1e-5),scipy.stats.norm.ppf(1-1e-5), 1000) * std + mean
        axes[i].plot(x_axis, scipy.stats.norm.pdf(x_axis, mean, std),color='#000080',label='Reference')
        hist_left, hist_right = x_axis[0], x_axis[-1]
        
    axes[i].hist(x_pred_1step[:,i], 50, (hist_left,hist_right),alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')
    
    axes[i].set_title(f'x{i+1}')
    axes[i].legend()
plt.tight_layout()
plt.savefig(plot_dir+f'/1step_x_n{latent_dim}.pdf',dpi=600)
plt.show()
plt.close('all')
   #%%
if n_x ==1:
    x_test_e = np.reshape(x_test,(-1,test_step+1))
    x_pred_e = np.reshape(x_pred,(-1,test_step+1))
    
    x_test_mean = np.mean(x_test_e,axis=0)
    x_test_std = np.std(x_test_e,axis=0,ddof=1)
    
    x_pred_mean = np.mean(x_pred_e,axis=0)
    x_pred_std = np.std(x_pred_e,axis=0,ddof=1)
    
    plot_t =  np.arange(x_pred_e.shape[-1])*0.01
    plt.plot(plot_t,np.abs(x_pred_mean-x_test_mean),label='error of mean')
    plt.plot(plot_t,np.abs(x_pred_std-x_test_std),label='error of std')
    plt.xlabel('t')
    plt.legend()
    plt.savefig(plot_dir+'/error of trajectoris.png',dpi=600)
    plt.show()
    plt.close('all')
else:
    x_test_e = np.reshape(x_test,(-1,n_x,test_step+1))
    x_pred_e = np.reshape(x_pred,(-1,n_x,test_step+1))
    
    x_test_mean = np.mean(x_test_e,axis=0)
    x_test_std = np.std(x_test_e,axis=0,ddof=1)
    
    x_pred_mean = np.mean(x_pred_e,axis=0)
    x_pred_std = np.std(x_pred_e,axis=0,ddof=1)
    
    plot_t =  np.arange(x_pred_e.shape[-1])*0.01
    
    
    fig, axes = plt.subplots(1, n_x, figsize=(5*n_x, 4))
    for i in range(n_x):
        axes[i].plot(plot_t,np.abs(x_pred_mean-x_test_mean)[i],label='error of mean')
        axes[i].plot(plot_t,np.abs(x_pred_std-x_test_std)[i],label='error of std')
        axes[i].set_title(f'error of x_{i+1}')
        axes[i].set_xlabel('t')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(plot_dir+'/error of x.png',dpi=600)
    plt.show()
    plt.close('all')
    
    
    
#%%
plot_save = 'png'

if n_x ==1:
    save_path = plot_dir+'/plot_meanstd.'+plot_save
    fig,ax = Evaluate.plot_meanstd(x_test_e,x_pred_e,dt,savepath = save_path)
    plt.show()
    plt.close('all')
    
    
    
    # save_path = plot_dir+'/plot_compare G.'+plot_save
    # fig,ax = Evaluate.plot_compare(example, plot_x0, x_pred, dt, savepath=save_path)
    # plt.show()
    # plt.close('all')
    
    
    # fig,ax =Evaluate.plot_ode_compare(x_test_e,x_pred_e,0.01)
    # fig.savefig('plot_ode_compare.png', dpi=300, transparent=True)
    
    # fig,ax =Evaluate.plot_sample(x_test_e,x_pred_e,0.01)
    # fig.savefig(plot_dir+'/plot_sample.png', dpi=600)
    # fig.clf()
    # plt.show()
    # plt.close('all')

    
    
else:
    save_path = plot_dir+f'/plot_meanstd_n{latent_dim}.'+plot_save
    fig,ax = Evaluate.plot_meanstd(x_test_e,x_pred_e,0.01,savepath = save_path)
    plt.show()
    plt.close('all')
    # for i in range(n_x):
        
        
    #     # fig,ax =Evaluate.plot_ode_compare(x_test_e,x_pred_e,0.01)
    #     # fig.savefig('plot_ode_compare.png', dpi=300, transparent=True)
    #     fig,ax =Evaluate.plot_sample(x_test_e[:100,i],x_pred_e[:100,i],0.01)
    #     fig.savefig(plot_dir+'/plot_sample x{}.png'.format(i+1).format(i+1), dpi=600)
    #     fig.clf()
    #     plt.show()
        # plt.close('all')

#%%
if example == 'Ex8DoubleWell':
    test_kde = KDELayer1D(lower = -4.0, upper = 4.0, num = 801, n_sample = 20000)
    x_test_kde = x_test_e[np.newaxis,:,:]
    x_pred_kde = x_pred_e[np.newaxis,:,:]
    plot_x =  np.linspace(-4.0, 4.0, 801)
    
    # Time_grid = [100,200,300,400,500]
    Time_grid = [50,1000,3000,10000]
    for T0 in Time_grid:
        test_pdf = test_kde(x_test_kde[:,:,T0:T0+1]).numpy()
        pred_pdf = test_kde(x_pred_kde[:,:,T0:T0+1]).numpy()
        
        plt.plot(plot_x,test_pdf[0,:,0],'b',label='Reference')
        plt.plot(plot_x,pred_pdf[0,:,0],'r--', label='Learned')
        
        plt.xlabel('t')
        plt.legend()
        plt.savefig(plot_dir+'/traj_pdf_T{}.pdf'.format(T0*0.01),dpi=600, bbox_inches='tight')
        plt.show()
        plt.close('all')

#%%
# TEST_ID = 2
if n_x ==1:
    M=101
    x_grid=np.linspace(plot_range[0],plot_range[1],M)
    z_sample=np.random.normal(size=(N_test,n_sample[1],latent_dim))
    m_plot = np.zeros_like(x_grid)
    std_plot = np.zeros_like(x_grid)
    m_true = np.zeros_like(x_grid)
    std_true = np.zeros_like(x_grid)
    for i in range(M):
        rand_seeds = np.random.randint(N_seeds, size=(N_test,))
        x_now=x_grid[i]
        x_now_c = np.tile(x_now.reshape(1, -1, 1), (N_test, n_sample[1], latent_dim))
        decode_out = np.zeros_like(x_now_c)
        for j in range(N_test):
            NN_id = rand_seeds[j]
            # NN_id = TEST_ID
            decode_out[j] = NN_save[rand_seeds[j]].decode(x_now_c[j:j+1],z_sample[j:j+1]).numpy()
        decode_out = np.reshape(decode_out, (-1,latent_dim))
        # m_plot[i]=np.mean(decode_out,axis=0)
        m_plot[i]=(np.mean(decode_out,axis=0)-x_now)/dt
        std_plot[i]=np.std(decode_out,axis=0)/np.sqrt(dt)
        if example == 'Ex6ExpOU':
            m_plot[i]=np.log(np.mean(decode_out/x_now,axis=0))/dt
            std_plot[i]=np.std(decode_out,axis=0)
        m_true[i],std_true[i] = Evaluate.condmv_plotting_std_cont(example,x_now,dt)
#%%
if n_x ==1:
    plot_save = 'png'
    if plot_save =='png':
        dx = (beta_range_2[1]-beta_range_2[0])/(M-1)
        drift_err = np.sqrt(np.linalg.norm(m_plot-m_true,2)*dx)
        std_err = np.sqrt(np.linalg.norm(std_plot-std_true,2)*dx)
        
        plt.plot(x_grid,m_plot,'o',markersize = 3,label='generate by decoder')
        plt.plot(x_grid,m_true,label='true dynamic')
        plt.xlabel('x')
        plt.ylabel('drift')
        plt.title('drift L2 err:{}'.format(drift_err) )
        plt.legend()
        plt.savefig(plot_dir+'/drift.png',dpi=600)
        plt.show()
        plt.close('all')
        
        plt.plot(x_grid,std_plot,'o',markersize = 3,label='generate by decoder')
        plt.plot(x_grid,std_true,label='true dynamic')
        plt.xlabel('x')
        plt.title('diffusion L2 err:{}'.format(std_err) )
        plt.ylabel('diffusion')
        plt.legend()
        plt.savefig(plot_dir+'/diffusion.png',dpi=600)
        plt.show()
        plt.close('all')
    else:
        font2 = {'size'   : 14,}
        # drift
        fig1,ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(x_grid,m_true,linestyle='-', linewidth=2.0,color='#000080',label='Reference')
        ax1.plot(x_grid,m_plot,linestyle='dashed', linewidth=2.0,color='#DC143C',label='Learned')
        ax1.set_xlabel('x', font2)
        ax1.set_ylabel('a(x)', font2)
        ax1.legend(prop=font2)
        # diffusion
        fig2,ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(x_grid,std_true,linestyle='-', linewidth=2.0,color='#000080',label='Reference')
        ax2.plot(x_grid,std_plot,linestyle='dashed', linewidth=2.0,color='#DC143C',label='Learned')
        ax2.set_xlabel('x', font2)
        ax2.set_ylabel('b(x)', font2)
        # ax2.set_ylim([0.09,0.11])
        ax2.legend(prop=font2)
        ## save
        fig1.savefig(plot_dir+'/condmean.pdf', bbox_inches='tight')
        fig2.savefig(plot_dir+'/condstd.pdf', bbox_inches='tight')

# #%%

#%%
z_encode = np.zeros((N_test,n_sample[1],latent_dim,2))
test_step = x_test.shape[-1]-1
rand_seeds = np.random.randint(N_seeds, size=(test_step,))
for t in range(2):
    # NN_id = rand_seeds[t]
    NN_id =0
    z_encode[:,:,:,t] = NN_save[NN_id].encode(x_test[:,:,:,t],x_test[:,:,:,t+1])
#%%
if n_x ==1:
    z_encode_e = np.reshape(z_encode,(-1,n_x,test_step+1))
    save_path = plot_dir+'/plot_z.'+plot_save
    fig,ax = Evaluate.plot_encode(example, z_encode_e[:,0,0], dt, savepath=save_path)
    plt.close('all')

#%%
if n_x >1:
    
    test_kde_1d = KDELayer1D(lower = kde_range[0], upper = kde_range[1],
                             num = kde_num, n_sample = n_sample[1])
    x = np.linspace(kde_range[0], kde_range[1], kde_num)
    norm_pdf = norm.pdf(x)
#%%
if n_x >1:
    fig, axes = plt.subplots(1, latent_dim, figsize=(5*latent_dim, 4))
    for i in range(latent_dim):
        z_pdf = test_kde_1d(z_encode[:,:,:,0]).numpy()
        
        axes[i].plot(x,z_pdf[0,:,i],label='encoder output')
        axes[i].plot(x,norm_pdf,label='standard normal')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('pdf')
        axes[i].set_title(f'z_{i+1}')
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(plot_dir+'/encoder_pdf_z.png',dpi=600)
    plt.show()
    plt.close('all')
#%%
if n_x >1:
    kde_num = 2000
    points = np.zeros((kde_num,latent_dim))
    s = 0
    while s < kde_num:
        point = np.random.uniform(kde_range[0], kde_range[1], latent_dim)
        if np.linalg.norm(point) <= 3:
            points[s,:] = point
            s+=1
    
    z_ = (1 / (2 * np.pi)**(latent_dim/2)) * np.exp(-0.5 * np.linalg.norm(points,2,1)**2)
    pdf_normal_c = np.tile(z_.reshape(1, -1), (batch_size, 1))

    test_kde_nd = KDELayer(points)
    z_pdf = test_kde_nd(z_encode[:,:,:,0]).numpy()
    z_true = test_kde_nd(np.random.normal(size = (batch_size,n_sample[1],latent_dim))).numpy()
    z_true_2 = test_kde_nd(np.random.normal(size = (batch_size,n_sample[1],latent_dim))).numpy()
    loss1 = keras.losses.MeanSquaredError()(z_pdf, pdf_normal_c).numpy()
    loss2 = keras.losses.MeanSquaredError()(z_true, pdf_normal_c).numpy()
    
    
#%%
z_plot_12 =np.reshape(z_encode[:,:,:,0],(-1,latent_dim))
if latent_dim==2:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist2d(z_plot_12[:,0],z_plot_12[:,1], 200)
    plt.xlabel('z1')
    plt.ylabel('z2')
    plt.title(f'kde_nd loss: encoder output {loss1:.3g}, true Gaussian {loss2:.3g}', fontsize=16)
    plt.savefig(plot_dir+'/encoder_hist2d.png',dpi=600)
    plt.show()
    plt.close('all')
    
else:
    fig, axes = plt.subplots(1, latent_dim*(latent_dim-1)//2, figsize=(4*latent_dim*(latent_dim-1)//2, 4))
    tmp = 0
    for i in range(latent_dim):
        for j in range(i):
            axes[tmp].hist2d(z_plot_12[:,i],z_plot_12[:,j], 200)
            axes[tmp].set_xlabel(f'z{i+1}')
            axes[tmp].set_ylabel(f'z{j+1}')
            axes[tmp].set_title(f'z{i+1}-z{j+1}')
            tmp +=1


    fig.suptitle(f'kde_nd loss: encoder output {loss1:.3g}, true Gaussian {loss2:.3g}', fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_dir+'/encoder_hist2d.png',dpi=600)
    plt.show()
    plt.close('all')

#%%

TEST_ID = 4
epochs_plot = 500
json_dir = eqn_name+'/'+model_dir+'RNN{}/sample{}/'.format(RNN,n_sample[0])+f'test{TEST_ID+1}loss history.json'
with open(json_dir) as json_data_file:
    loss = json.load(json_data_file)
plt.semilogy(range(epochs_plot),np.array(loss["val_MSE_s0"]), label = 'MSE loss')
# plt.semilogy(range(epochs_plot),np.array(loss["val_kde_s0"])*beta_kde_1d[0], label = 'kde1d loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_kde_1d_s0"])*beta_kde_1d[0], label = 'kde1d loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_kde_nd_s0"])*beta_kde_nd[0], label = 'kdend loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_moment_s0"])*beta_moment[0], label = 'moment loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_z_cov_s0"])*beta_cov[0], label = 'covariance loss')
# plt.semilogy(range(epochs_plot),np.array(loss["val_mmd_s0 "])*beta_mmd[0], label = 'MMD loss')
plt.legend()
plt.savefig(plot_dir+'/training loss first stage.png',dpi=600)
plt.show()

#%%

epochs_plot = 500

TEST_ID = 4
json_dir = best_model_dir+f'test{TEST_ID+1}loss history.json'
with open(json_dir) as json_data_file:
    loss = json.load(json_data_file)
plt.semilogy(range(epochs_plot),np.array(loss["val_MSE_s0"]), label = 'MSE loss')
# plt.semilogy(range(epochs_plot),np.array(loss["val_kde_s0"])*beta_kde_1d[0], label = 'kde1d loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_kde_1d_s0"])*beta_kde_1d[0], label = 'kde1d loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_kde_nd_s0"])*beta_kde_nd[0], label = 'kdend loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_moment_s0"])*beta_moment[0], label = 'moment loss')
plt.semilogy(range(epochs_plot),np.array(loss["val_z_cov_s0"])*beta_cov[0], label = 'covariance loss')
# plt.semilogy(range(epochs_plot),np.array(loss["val_mmd_s0"])*beta_mmd[0], label = 'MMD loss')
plt.legend()
plt.savefig(plot_dir+'/training loss second stage.png',dpi=600)
plt.show()
#%%
z_cov = cov_layer(z_encode[:,:,:,0]).numpy()
cov_loss_z = tf.reduce_mean(tf.math.square(z_cov)).numpy()
n_cov = cov_layer(np.random.normal(size = (20,n_sample[1],latent_dim)))
cov_loss_n = tf.reduce_mean(tf.math.square(n_cov)).numpy()

#%%
np.save('strange.npy',z_encode[0,:,:,0])
# z_encode = np.zeros_like(x_test)

# for t in range(test_step):
#     NN_id = rand_seeds[t]
#     # NN_id = TEST_ID
#     z_encode[:,:,t+1:t+2] = NN_save[NN_id].encode(x_test[:,:,t:t+1],x_test[:,:,t+1:t+2])

# test_moment_layer = MomentLayer(max_moment, moment_weights)
# z_moment = np.zeros((1,max_moment,test_step))
# z_encode = np.reshape(z_encode, (1,-1,test_step+1))
# for t in range(test_step):
#     z_moment[:,:,t] = test_moment_layer(z_encode[:,:,t+1:t+2]).numpy()
    

#%%
# x_train_uni = np.load('data/'+eqn_name+'/uni_train_1.npy')[:20]

# N_train = x_train_uni.shape[0]
# train_step = x_train_uni.shape[2]-1
# x_pred = np.zeros_like(x_train_uni)
# x_pred[:,:,0] = x_train_uni[:,:,0]

# z = np.random.normal(size=(N_train, n_sample[0], train_step))
# rand_seeds = np.random.randint(10, size=(train_step,))
# z_encode = np.zeros_like(x_test)

# test_pdf_layer = Gauss_kde(lower = pdf_range[0], upper = pdf_range[1], num = pdf_num, n_sample = n_sample[0])
# z_pdf = np.zeros((N_train,pdf_num,train_step))
    
# for t in range(train_step):
#     NN_id = rand_seeds[t]
#     # NN_id  = TEST_ID
#     z_encode[:,:,t:t+1] = NN_save[NN_id].encode(x_train_uni[:,:,t:t+1],x_train_uni[:,:,t+1:t+2])
#     # x_pred[:,:,t+1:t+2] += NN_save[TEST_ID].decode(x_pred[:,:,t:t+1],z_encode[:,:,t:t+1])
#     x_pred[:,:,t+1:t+2] += NN_save[NN_id].decode(x_pred[:,:,t:t+1],z[:,:,t:t+1])
#     z_pdf[:,:,t:t+1] = test_pdf_layer(z_encode[:,:,t:t+1]).numpy()
# #%%
# mean_pred = np.mean(x_pred[:,:,1])
# mean_train = np.mean(x_train_uni[:,:,1])

# std_pred = np.std(x_pred[:,:,1])
# std_train = np.std(x_train_uni[:,:,1])
# #%%
# plt.plot(x,z_pdf[0,:,0],label='encoder output with training data')
# plt.plot(x,norm_pdf,label='standard normal')
# plt.xlabel('x')
# plt.ylabel('pdf')
# plt.legend()
# plt.savefig(plot_dir+'/encoder_training_data.png',dpi=600)
# plt.show()
# plt.close('all')
# #%%

# x_train_e = np.reshape(x_train_uni,(-1,train_step+1))
# x_pred_e = np.reshape(x_pred,(-1,train_step+1))

# x_train_mean = np.mean(x_train_e,axis=0)
# x_train_std = np.std(x_train_e,axis=0,ddof=1)

# x_pred_mean = np.mean(x_pred_e,axis=0)
# x_pred_std = np.std(x_pred_e,axis=0,ddof=1)

# plot_t =  np.arange(x_pred_e.shape[-1])*0.01
# plt.plot(plot_t,np.abs(x_pred_mean-x_train_mean),label='error of mean')
# plt.plot(plot_t,np.abs(x_pred_std-x_train_std),label='error of std')
# plt.xlabel('t')
# plt.legend()
# plt.savefig(plot_dir+'/error of training set.png',dpi=600)
# plt.show()
# plt.close('all')
#%%
# fig,ax = Evaluate.plot_meanstd(x_train_e,x_pred_e,0.01)
# fig.savefig(plot_dir+'/plot_meanstd.png', dpi=600)
# # fig.clf()
# plt.show()
# plt.close(fig)







#%%
# test_pdf_layer = gauss_kde(lower = pdf_gl, upper = pdf_gr, num = pdf_num, n_sample = n_sample)
# train_pdf_layer = gauss_kde(lower = pdf_gl, upper = pdf_gr, num = pdf_num, n_sample = n_sample)
# x_pdf = train_pdf_layer(train_inputs[:batch_size]).numpy()
# x_dt_pdf = train_pdf_layer(train_outputs[:batch_size]).numpy()
# dw = ((train_outputs[:batch_size]-train_inputs[:batch_size]) -0.01*(1.2-train_inputs[:batch_size]))/0.03
# dw_pdf = train_pdf_layer(dw).numpy()
# dw_test = ((x_test[:,:,1:2]-x_test[:,:,0:1]) -0.01*(1.2-x_test[:,:,0:1]))/0.03
# dw_test_pdf = test_pdf_layer(dw).numpy()
# x_test_pdf = test_pdf_layer(x_test[:,:,1:2]).numpy()
# #%%

# # plt.plot(x,z_pdf[1,:,0],label='train z')
# plt.plot(x,x_pdf[1,:,0],label='train x1')
# plt.plot(x,x_pdf[2,:,0],label='train x2')
# plt.plot(x,x_pdf[3,:,0],label='train x3')
# # plt.plot(x,x_dt_pdf[1,:,0],label='train x_dt')
# # plt.plot(x,dw_pdf[1,:,0],'--',label='train dw')
# # plt.plot(x,dw_test_pdf[1,:,0],'-.',label='test dw')
# # plt.plot(x,x_test_pdf[1,:,0],'-',label='test x_dt')
# plt.plot(x,norm_pdf,label='standard normal')
# plt.xlabel('x')
# plt.ylabel('pdf')
# plt.legend()
# plt.show()
# plt.close('all')
# #%%
# # z_pred = cvae.encode(x_test[:,:,:1], x_test[:,:,1:2])
# z_pred = cvae.encode(train_inputs[:batch_size], train_outputs[:batch_size])
# test_pdf_layer = gauss_kde(lower = pdf_gl, upper = pdf_gr, num = pdf_num, n_sample = n_sample)
# z_pred_pdf = test_pdf_layer(z_pred).numpy()
# plt.plot(x,z_pred_pdf[1,:,0],label='generate by encoder')

# plt.xlabel('x')
# plt.ylabel('pdf')
# plt.legend()
# plt.show()
# plt.savefig('pdf.png',dpi=600)
# plt.close('all')
# # z_pred = np.reshape(z_pred,(5000,latent_dim))
# # z_sample = np.random.normal(size=(5000,latent_dim))
# # fig,ax = Evaluate.plot_meanstd(z_sample,z_pred,0.01)


# #%%
# # cvae.load_weights('models/cvae.h5')
# # weights = cvae.get_weights()
# # weights[-1] = weights[-1]-.001
# # cvae.set_weights(weights)


