from __future__ import division
import pdb
import os
import munch
import json
import logging

import numpy as np
import numpy.linalg
import matplotlib
from matplotlib import pyplot as plt
import scipy
import scipy.io as sio
import tensorflow as tf

class Evaluate():
	def plot_ode_compare(testdata,predictdata,Delta,savepath=None):
		xt = np.arange(testdata.shape[0])*Delta
		xp = np.arange(predictdata.shape[0])*Delta
		T = max(xt[-1],xp[-1])
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(xt, testdata, color='black')
		ax1.plot(xp, predictdata, 'o', color='#6495ED', markerfacecolor='none')
		plt.xlim([-0.1*T,1.1*T])
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_train_hisGAN(self,Nepoc,G_loss,D_loss,savepath=None):
		x = np.arange(len(G_loss))
		fig1, ax1 = plt.subplots(figsize=[10,7])
		# ax1.plot(x, G_loss, color='#4169E1', label='Generator')
		# ax1.plot(x, D_loss, color='#DC143C', label='Discriminator')
		gp = ax1.plot(x, G_loss, color='#4169E1', label='Generator')
		ax1.tick_params(axis='y', labelcolor='#4169E1')
		ax2 = ax1.twinx()
		dp = ax2.plot(x, -np.array(D_loss), color='#DC143C', label='Negative Discriminator')
		ax2.tick_params(axis='y', labelcolor='#DC143C')
		ax2.set_yscale('log')
		ax1.set_xlim([-100,Nepoc+100])
		# fig1.tight_layout()
		gdps = gp+dp
		labs = [l.get_label() for l in gdps]
		ax1.legend(gdps,labs)
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_index( Nepoc,data,name,savepath=None,log=True):
		x = np.arange(Nepoc)
		fig1, ax1 = plt.subplots(figsize=[10,7])
		ax1.plot(x, data, color='#0000FF', label=name)
		if log:
			ax1.set_yscale('log')
		ax1.legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_sample( testdata,predictdata,Delta,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		# plot
		fig1, ax1 = plt.subplots(1,2,figsize=[20,7])
		for i in range(min(testdata.shape[0],200)):
			ax1[0].plot(xt_test, testdata[i])
			ax1[1].plot(xt_pred, predictdata[i])
		ax1[0].set_title('Ground Truth')
		ax1[1].set_title('Prediction')
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def plot_meanstd(testdata,predictdata,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of Ndata*test
		# Test data
		xt_test = np.arange(testdata.shape[-1])*Delta
		xmean_test = np.mean(testdata,axis=0)
		xstde_test = np.std(testdata,axis=0,ddof=1)
		xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
		# Predict data
		xt_pred = np.arange(predictdata.shape[-1])*Delta
		xmean_pred = np.mean(predictdata,axis=0)
		xstde_pred = np.std(predictdata,axis=0,ddof=1)
		xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
		# Resdata
		if Resdata is not None:
			xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
		# Bound
		test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
		pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
		# plot
		if savepath[-1] =='g':
			n_x = 1
			print(n_x)
			fig1, axes = plt.subplots(1, n_x, figsize=[7*n_x,6])
			for i in range(n_x):
				axes.plot(xt_test, xmean_test,color='#4169E1', label='Ground Truth')
				axes.fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
				axes.plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
				axes.fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
				axes.set_xlabel('T')
				axes.set_title(f'x_{i+1}')
				axes.set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
				axes.legend()
			
			
		else:
			n_x = testdata.shape[1]
			font2 = {'size'   : 14,}
			font3 = {'size'   : 12,}
			fig1, axes = plt.subplots(1, n_x, figsize=[7*n_x,6])
			for i in range(n_x):
				axes[i].plot(xt_test, xmean_test[i], linewidth=2.0, color='#000080', label='Ground Truth Mean')
				axes[i].fill_between(xt_test, test_l[i], test_u[i], color='#000080', alpha=0.2, label='Ground Truth Std')
				axes[i].plot(xt_pred, xmean_pred[i], linewidth=2.0, color='#DC143C', linestyle='dashed', label='Prediction Mean')
				axes[i].fill_between(xt_pred, pred_l[i], pred_u[i], color='#DC143C', alpha=0.2, label='Prediction Std')
				axes[i].set_title(f'x_{i+1}')
				axes[i].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
                # ylow,yup = min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))
    			# ax1.set_ylim([ylow-0.03*abs(ylow),yup+0.03*abs(yup)])
    			# ax1.set_ylim([ylow-0.3*abs(ylow),yup+0.3*abs(yup)])
# 				axes[i].set_xlabel('T', font2)
    			# ax1.set_ylabel('pdf', font2)
				axes[i].legend(prop=font3, ncol=2)
     			# axes.legend(prop=font3, ncol=2,loc=4)
			plt.tight_layout()
    




		if savepath is not None:
			fig1.savefig(savepath, dpi=600)
		return fig1,axes
    
    
	
    
	def plot_encode(name, testdata, Delta, savepath=None):
		# 2. pdf compare plots, e.g. Fig 7 of SDEGANs paper
		font2 = {'size'   : 14,}
		
		x_axis = np.linspace(scipy.stats.norm.ppf(1e-5),scipy.stats.norm.ppf(1-1e-5), 1000)
		fig1,ax1 = plt.subplots(figsize=(6,4))
        # line graph of exact pdf
		ax1.plot(x_axis, scipy.stats.norm.pdf(x_axis),color='#000080',label='Reference')
        # histogram of estimated pdf
		ax1.hist(testdata, bins=50, alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')
		ax1.set_xlabel('z', font2)
		ax1.set_ylabel('pdf', font2)
		ax1.legend(prop=font2)
    
		if savepath is not None:
			fig1.savefig(savepath , bbox_inches='tight', dpi=600)
		return fig1,ax1
        
	def plot_meanstdGeneralD( testdataMD,predictdataMD,dim,Delta,Resdata=None,slice=0,savepath=None):
		# data should be in the form of dim*Ndata*test
		N_plot = min(dim,10)
		fig1, ax1 = plt.subplots(ncols=N_plot, figsize=(10*N_plot, 7), squeeze=False)
		for i in range(N_plot):
			# Test data
			testdata,predictdata = testdataMD[i].T,predictdataMD[i].T
			xt_test = np.arange(testdata.shape[-1])*Delta
			xmean_test = np.mean(testdata,axis=0)
			xstde_test = np.std(testdata,axis=0,ddof=1)
			xt_test,xmean_test,xstde_test = xt_test[slice:],xmean_test[slice:],xstde_test[slice:]
			# Predict data
			xt_pred = np.arange(predictdata.shape[-1])*Delta
			xmean_pred = np.mean(predictdata,axis=0)
			xstde_pred = np.std(predictdata,axis=0,ddof=1)
			xt_pred,xmean_pred,xstde_pred = xt_pred[slice:],xmean_pred[slice:],xstde_pred[slice:]
			# Resdata
			if Resdata is not None:
				xmean_pred = xmean_pred+Resdata[:xmean_pred.shape[0]]
			# Bound
			test_l,test_u = xmean_test - xstde_test, xmean_test + xstde_test
			pred_l,pred_u = xmean_pred - xstde_pred, xmean_pred + xstde_pred
			# plot
			ax1[0,i].plot(xt_test, xmean_test, color='#4169E1', label='Ground Truth')
			ax1[0,i].fill_between(xt_test, test_l, test_u, color='#4169E1', alpha=0.2)
			ax1[0,i].plot(xt_pred, xmean_pred, color='#DC143C', label='Prediction')
			ax1[0,i].fill_between(xt_pred, pred_l, pred_u, color='#DC143C', alpha=0.2)
			ax1[0,i].set_ylim([min(np.min(test_l),np.min(pred_l)),max(np.max(test_u),np.max(pred_u))])
			# ax1.set_ylim([-1.5,2.5])
			ax1[0,i].legend()
		if savepath is not None:
			fig1.savefig(savepath)
		return fig1,ax1

	def readmodel(path,Model,config):
		# This function is designed for test for single models
		ModelX = Model(config)
		# see https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
		ModelXCheckp = tf.train.Checkpoint(G_optimizer=ModelX.G_optimizer,D_optimizer=ModelX.D_optimizer,G=ModelX.G,D=ModelX.D)
		manager = tf.train.CheckpointManager(ModelXCheckp, path, max_to_keep=1)
		ModelXCheckp.restore(manager.latest_checkpoint).expect_partial()
		return ModelX

	def condmv_plotting_std_cont(name,x,Delta):
		if name=='Brownian Motion':
			m,s = x,np.sqrt(Delta)
			return m,s
		elif name=='Ex1GeoBrownian':
			m,s = 2.0*x,1.0*x
			return m,s
		elif name=='Ex3OU':
			m   = 1.0*(1.2-x)
			var = 0.3**2
			return m,np.sqrt(var)
		elif name=='Ex4ExpDiff':
			m   = -5.0*x
			std = 0.5*np.exp(-x**2)
			return m,std
		elif name=='Ex5Trig':
 			m   = np.sin(2*np.pi*x)
 			std = abs(0.5*np.cos(2*np.pi*x))
 			return m,std
		elif name=='Ex6ExpOU':
 			th,dt  = 1.0, 0.01
 			mu,sig = -0.5,0.3
 			MU,SIG = (1-th*dt)*np.log(x)+th*mu*dt,sig*np.sqrt(dt)
 			m = -th*np.log(x)+th*mu+sig**2/2
 			var = (np.exp(SIG**2)-1)*np.exp(2*MU+SIG**2)
 			return m,np.sqrt(var)
		elif name=='Ex8DoubleWell':
 			m   = x-x**3
 			std = 0.5
 			return m,std
		elif name=='Ex9Expdis':
 			m = 1+(-2.0)*x
 			std = 0.1
 			return m,std
		else:
			print('The distribution %s is not supported'%(name))
    
	def plot_compare(name, x0, testdata, Delta, savepath=None):
		# 2. pdf compare plots, e.g. Fig 7 of SDEGANs paper
		font2 = {'size'   : 14,}
		m, std = Evaluate.condmv_plotting_std_cont(name,x0,Delta)
		mean = x0+ m*Delta
		std = std*np.sqrt(Delta)
		if name =='Ex9Expdis':
 			x_axis = np.linspace(-0.02,0.075, 1000) + x0
		elif name =='Ex6ExpOU':
			x_axis = np.linspace(-0.05,0.05, 1000) + x0
		else:
 			x_axis = np.linspace(scipy.stats.norm.ppf(1e-5),scipy.stats.norm.ppf(1-1e-5), 1000) * std + mean
		
		fig1,ax1 = plt.subplots(figsize=(6,4))
        # line graph of exact pdf
		if name =='Ex9Expdis':
 			_id = x_axis>=mean
 			x_plot = x_axis[_id]
 			pdf_plot = np.exp(-(x_axis[_id]-mean)/std)/std
 			ax1.plot(x_plot, pdf_plot,color='#000080',label='Reference')
 			ax1.plot([mean,mean], [0,1/std],color='#000080')
		elif name =='Ex6ExpOU':
			x_axis = np.linspace(x0-0.1/2,x0+0.1/2,200)
			x_axis = x_axis[x_axis>0]
			th,dt  = 1.0, 0.01
			mu,sig = -0.5,0.3
			MU,SIG = (1-th*dt)*np.log(x0)+th*mu*dt,sig*np.sqrt(dt)
			pdf = 1/(x_axis*SIG*np.sqrt(2*np.pi))*np.exp(-(np.log(x_axis)-MU)**2/(2*SIG**2))
			ax1.plot(x_axis, pdf, color='#000080',label='Reference')
		else:
 			ax1.plot(x_axis, scipy.stats.norm.pdf(x_axis, mean, std),color='#000080',label='Reference')
        
        # histogram of estimated pdf
		ax1.hist(testdata, 50, (x_axis[0],x_axis[-1]),alpha=0.6, ec="k", color='#A0A0A0', density=True, histtype='stepfilled',label='Learned')
		ax1.set_xlabel('x', font2)
		ax1.set_ylabel('pdf', font2)
		ax1.legend(prop=font2)
    
		if savepath is not None:
			fig1.savefig(savepath , bbox_inches='tight', dpi=600)
		return fig1,ax1
def cond_meanvar_Enspre(self,model,epoch,savepath,cont=False):
	## compute
	# Npoint = self.monitor_config.cond_mv['Npoint']
	# l1,l2 = self.monitor_config.cond_mv['range']
	if self.eqn_config.dim==1:
		Npoint = 80
		# condition mv functions
		if cont:
			condmvfunc = self.condmv_plotting_std_cont
		else:
			condmvfunc = self.condmv_plotting_std
		# upper and lower limits
		if self.eqn_config.eqn_name=='Geometric Brownian Motion':
			l1,l2 = 0.5,15
		elif self.eqn_config.eqn_name=='OU Process':
			l1,l2 = 0.7,2
		elif self.eqn_config.eqn_name=='Exp_diffusion':
			l1,l2 = -0.6,0.6
		elif self.eqn_config.eqn_name=='Trig_drift':
			l1,l2 = 0.3,0.7
		elif self.eqn_config.eqn_name=='Exp_OU':
			l1,l2 = 0.25,1.7
		elif self.eqn_config.eqn_name=='Double_well':
			l1,l2 = -2,2
			# l1,l2 = -1.8,1.8
		elif self.eqn_config.eqn_name=='Exp_dis':
			l1,l2 = 0.3,0.9
		p_grid = np.linspace(l1,l2,Npoint+1)
		Mean_t, Std_t = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
		Mean_d, Std_d = np.zeros(p_grid.shape),np.zeros(p_grid.shape)
		for i in range(p_grid.shape[0]):
			Mean_t[i],Std_t[i] = condmvfunc(self.eqn_config.eqn_name,p_grid[i],self.Delta)
			Mean_d[i],Std_d[i] = self.condmv_plotting_data(model,p_grid[i],N=500000)
		## change scale
		if self.eqn_config.eqn_name=='Exp_OU':
			Mean_d = (np.log(Mean_d)-np.log(p_grid))/self.eqn_config.Delta
		else:
			Mean_d = (Mean_d-p_grid)/self.eqn_config.Delta
			Std_d = Std_d/np.sqrt(self.eqn_config.Delta)
		
		if not cont:
			if self.eqn_config.eqn_name=='Exp_OU':
				pass
			else:
				Mean_t = (Mean_t-p_grid)/self.eqn_config.Delta
				Mean_d = (Mean_d-p_grid)/self.eqn_config.Delta
				Std_t = Std_t/np.sqrt(self.eqn_config.Delta)
				Std_d = Std_d/np.sqrt(self.eqn_config.Delta)
		else:
			pass
		## draw
		font2 = {'size'   : 14,}
		# drift
		fig1,ax1 = plt.subplots(figsize=(6,4))
		ax1.plot(p_grid,Mean_t,linestyle='-', linewidth=2.0,color='#000080',label='Reference')
		ax1.plot(p_grid,Mean_d,linestyle='dashed', linewidth=2.0,color='#DC143C',label='Learned')
		ax1.set_xlabel('x', font2)
		ax1.set_ylabel('a(x)', font2)
		ax1.legend(prop=font2)
		# diffusion
		fig2,ax2 = plt.subplots(figsize=(6,4))
		ax2.plot(p_grid,Std_t,linestyle='-', linewidth=2.0,color='#000080',label='Reference')
		ax2.plot(p_grid,Std_d,linestyle='dashed', linewidth=2.0,color='#DC143C',label='Learned')
		ax2.set_xlabel('x', font2)
		ax2.set_ylabel('b(x)', font2)
		ax2.set_ylim([0.09,0.11])
		ax2.legend(prop=font2)
		## save
		fig1.savefig(savepath+'/condmean.pdf', bbox_inches='tight')
		fig2.savefig(savepath+'/condstd.pdf', bbox_inches='tight')
	elif self.eqn_config.dim==2:
		Npoint = 20
		# upper and lower limits
		if self.eqn_config.eqn_name=='MdOU':
			l1,l2 = [-2.0,2.0],[-1.0,1.0]
		elif self.eqn_config.eqn_name=='SO':
			l1,l2 = [-2.0,2.0],[-1.0,1.0]
		p_gridx = np.linspace(l1[0],l1[1],Npoint+1)
		p_gridy = np.linspace(l2[0],l2[1],Npoint+1)
		p_gridx,p_gridy = np.meshgrid(p_gridx,p_gridy)
		p_grid = np.array((p_gridx.flatten(),p_gridy.flatten())).T
		PSh = p_grid.shape
		Mean_t, V_t, C_t = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
		Mean_d, V_d, C_d = np.zeros(PSh),np.zeros(PSh),np.zeros(PSh[0])
		for i in range(p_grid.shape[0]):
			Mean_t[i],V_t[i],C_t[i] = self.condmv_plotting_std_cont2D(self.eqn_config.eqn_name,p_grid[i],self.Delta)
			Mean_d[i],V_d[i],C_d[i] = self.condmv_plotting_data2D(model,p_grid[i],N=500000)

		V_t,V_d = np.sqrt(V_t/self.eqn_config.Delta),np.sqrt(V_d/self.eqn_config.Delta)
		C_t,C_d = C_t/self.eqn_config.Delta,C_d/self.eqn_config.Delta
		
		font2 = {'size'   : 14,}
		# drift
		fig1, ax1 = plt.subplots(ncols=4, figsize=[24,4])
		min1 = min(Mean_t[:,0].min(),Mean_d[:,0].min())
		max1 = max(Mean_t[:,0].max(),Mean_d[:,0].max())
		min2 = min(Mean_t[:,1].min(),Mean_d[:,1].min())
		max2 = max(Mean_t[:,1].max(),Mean_d[:,1].max())
		r1,r2 = np.linspace(min1,max1,30),np.linspace(min2,max2,30)
		pgx,pgy = p_grid[:,0].reshape([Npoint+1,Npoint+1]),p_grid[:,1].reshape([Npoint+1,Npoint+1])
		ax1[0].contour(pgx,pgy,Mean_t[:,0].reshape([Npoint+1,Npoint+1]), colors='k',       vmin=min1, vmax=max1, levels=r1)
		ax1[1].contour(pgx,pgy,Mean_d[:,0].reshape([Npoint+1,Npoint+1]), colors='#DC143C', vmin=min1, vmax=max1, levels=r1)
		ax1[2].contour(pgx,pgy,Mean_t[:,1].reshape([Npoint+1,Npoint+1]), colors='k',       vmin=min2, vmax=max2, levels=r2)
		ax1[3].contour(pgx,pgy,Mean_d[:,1].reshape([Npoint+1,Npoint+1]), colors='#DC143C', vmin=min2, vmax=max2, levels=r2)
		fig1.savefig(savepath+'/condmeancontour.pdf')

		# means
		fig, axes = plt.subplots(ncols=4, figsize=(24, 4), constrained_layout=True, subplot_kw={"projection": "3d"})
		self.whitebg(axes)
		axes[0].plot_surface(p_gridx, p_gridy, Mean_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
		axes[1].plot_surface(p_gridx, p_gridy, Mean_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
		axes[2].plot_surface(p_gridx, p_gridy, Mean_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
		axes[3].plot_surface(p_gridx, p_gridy, Mean_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
		axes[0].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
		axes[1].set_zlim([min(Mean_t[:,0]),max(Mean_t[:,0])])
		axes[2].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
		axes[3].set_zlim([min(Mean_t[:,1]),max(Mean_t[:,1])])
		fig.savefig(savepath+'/condmean.pdf')
		# variances
		fig, axes = plt.subplots(ncols=4, figsize=(24, 4), constrained_layout=True, subplot_kw={"projection": "3d"})
		self.whitebg(axes)
		axes[0].plot_surface(p_gridx, p_gridy, V_t[:,0].reshape([Npoint+1,Npoint+1]), cmap='Blues')
		axes[1].plot_surface(p_gridx, p_gridy, V_d[:,0].reshape([Npoint+1,Npoint+1]), cmap='Reds')
		axes[2].plot_surface(p_gridx, p_gridy, V_t[:,1].reshape([Npoint+1,Npoint+1]), cmap='Blues')
		axes[3].plot_surface(p_gridx, p_gridy, V_d[:,1].reshape([Npoint+1,Npoint+1]), cmap='Reds')
		fig.savefig(savepath+'/condstd.pdf')
		# covariance
		fig, axes = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True, subplot_kw={"projection": "3d"})
		self.whitebg(axes)
		axes[0].plot_surface(p_gridx, p_gridy, C_t.reshape([Npoint+1,Npoint+1]), cmap='Blues')
		axes[1].plot_surface(p_gridx, p_gridy, C_d.reshape([Npoint+1,Npoint+1]), cmap='Reds')
		fig.savefig(savepath+'/condcostd.pdf')
		plt.close()

class SdeGanEva(Evaluate):
	def __init__(self,config,result_path,save_path):
		self.eqn_config  = config.eqn_config
		self.net_config  = config.net_config
		self.dat_config  = config.dat_config
		self.result_path = result_path
		self.save_path   = save_path
		self.dim = self.eqn_config.dim
		self.Delta   = self.eqn_config.Delta
		self.n_epochs = self.net_config.N_epochs
		self.test_data_path  = self.dat_config.TestData_dir
		if not os.path.exists(self.save_path):
			os.makedirs(self.save_path)

	def plot_samplecompare(self,save=False):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/S'+str(i+1)+'.pdf') if save else None
			# if i==0:
			# 	pdb.set_trace()
			fig,ax = self.plot_sample(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)

	def plot_losthist(self,save=False):
		try:
			with open(self.result_path+'/Test_history.json') as json_data_file:
				file = json.load(json_data_file)
				G_loss_data = file['G_loss']
				D_loss_data = file['D_loss']
		except:
			raise AttributeError('SdeGanEva::plot_losshist: Fail to find loss data')
		save_ = (self.save_path+'/loss_hist.pdf') if save else None
		fig,ax = self.plot_train_hisGAN(self.n_epochs,G_loss_data,D_loss_data,savepath=save_)

	def plot_Wdistance(self,save=False):
		try:
			with open(self.result_path+'/Test_history.json') as json_data_file:
				file = json.load(json_data_file)
				W_dist_data = file['W_dist']
				save_ = (self.save_path+'/W_dist.pdf') if save else None
				fig,ax = self.plot_index(self.n_epochs,W_dist_data,'Wasserstein Distance',savepath=save_,log=False)
		except:
			pass

	def plot_meancompare(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/'+epoch+'M'+str(i+1)+'.pdf') if save else None
			# pdb.set_trace()
			fig,ax = self.plot_meanstd(test_data[i].T,pre_data[i].T,self.Delta,savepath=save_)

	def plot_meancompare_Resplus(self,save=False,epoch=''):
		test_data = (sio.loadmat(self.test_data_path))['data']
		Res_data = (sio.loadmat(self.eqn_config.Resdata['path']))['pred']
		try:
			pre_data = (sio.loadmat(self.result_path+'/predict.mat'))['pred']
		except:
			raise AttributeError('ResnetEva::plot_single: Fail to find prediction data')
		for i in range(min(self.dim,10)):
			save_ = (self.save_path+'/'+epoch+'M'+str(i+1)+'.pdf') if save else None
			fig,ax = self.plot_meanstd(test_data[i].T,pre_data[i].T,self.Delta,Resdata=Res_data[i],savepath=save_)



