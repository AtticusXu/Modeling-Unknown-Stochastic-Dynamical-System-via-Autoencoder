import numpy as np
import DEVAE_network
from scipy.stats import norm


def G_1step(x,z, example):
    if example == 'Ex3OU':
        th = 1.0
        mu = 1.2
        sig = 0.3
        t = 0.01
        Ext = np.exp(-th*t)
        y = x*Ext+mu*(1-Ext)+sig*np.sqrt(t)*z
        
    elif example == 'Ex4ExpDiff':
        mu = 5
        sigma = 0.5
        t = 0.01
        y = x+(-mu*x)*t+sigma*np.exp(-x**2)*np.sqrt(t)*z
        
    elif example == 'Ex1GeoBrownian':
          mu = 2.0
          sigma = 1.0
          t = 0.01
          y = x*np.exp((mu-sigma**2/2)*t+sigma*np.sqrt(t)*z)
    
    elif example == 'Ex5Trig':
          k = 1
          sigma = 0.5
          t = 0.01
          y = x+np.sin(2*k*np.pi*x)*t+sigma*np.cos(2*k*np.pi*x)*np.sqrt(t)*z
          
    elif example == 'Ex6ExpOU':
          th = 1.0
          mu = -0.5
          sig = 0.3
          t = 0.01
          # y = x**(1-th*t)*mu**(t)*Normal2logN(z)**(sig*np.sqrt(t))
          
          y = x**(1-th*t)*np.exp(th*mu*t+sig*np.sqrt(t)*z)
    
    elif example == 'Ex8DoubleWell':
          sigma = 0.5
          t = 0.01
          y = x+(x-x**3)*t+sigma*np.sqrt(t)*z
    
    elif example == 'Ex9Expdis':
          th =  -2.0
          sig = 0.1
          t = 0.01
          y = x+th*x*t+sig*Normal2Exp(z)*np.sqrt(t)
    
    
    
    return y



def HermiteF(x, degree):
    if degree == 0:
        return np.ones_like(x)
    elif degree == 1:
        return x
    else:
        return x * HermiteF(x, degree-1) - (degree-1) * HermiteF(x, degree-2)
    
def Normal2Exp(z):
    b_coef = [1.0, 
    0.903249477665220,
    0.297985195834260,
    0.0335705586952089,
    -0.00228941798505679,
    -0.000388473483538765]
    
    b = 0
    for i in range(6):
        b += HermiteF(z, i)* b_coef[i]
    
    return b

def Normal2logN(z):
    b_coef = np.ones((6,1))*np.exp(0.5)
    
    b = 0
    for i in range(6):
        b += HermiteF(z, i)* b_coef[i]/np.math.factorial(i)
    
    return b


examples =  ['Ex1GeoBrownian','Ex3OU', 'Ex4ExpDiff', 'Ex5Trig', 
             'Ex6ExpOU', 'Ex8DoubleWell', 'Ex9Expdis',
              'Ex7OU2D','Ex11OU5D','Ex12OU5D5','Ex13OU5D1',
              'Ex14OU5D2','Ex15OU5D4']

def sample_nd_ball(n,m):
    
    N_r = int(pow(m,1/n))+1
    num_points =0
    while num_points < m:
        
        r = [np.linspace(-3, 3, N_r) for i in range(n)]
        
        
        grids = np.meshgrid(*r, indexing='ij')
        
        points = np.stack(grids, axis=-1).reshape(-1, n)
        
        # Compute the Euclidean distance for each point
        distances = np.linalg.norm(points, axis=1)
        
        # Select points that are within the 2D circle of radius 3
        inside_ball = points[distances <= 3]
        num_points = inside_ball[:,0].size
        N_r+=1
    
    return inside_ball
    
    
def kde_preparation(example, latent_dim, kde_method, kde_range, kde_num, n_sample, batch_size):
    if example in examples[:7]:
        x = np.linspace(kde_range[0], kde_range[1], kde_num)
        norm_pdf = norm.pdf(x)
        pdf_normal_c_1d= np.tile(norm_pdf.reshape(1, kde_num, 1), (batch_size, 1, latent_dim))
        kde_layer_1d =DEVAE_network.KDELayer1D(lower = kde_range[0], upper = kde_range[1], 
                                num = kde_num, n_sample = n_sample)
        kde_layer_nd, pdf_normal_c_nd = None, None

    elif example == 'Ex7OU2D':
        x = np.linspace(kde_range[0], kde_range[1], kde_num)
        norm_pdf = norm.pdf(x)
        pdf_normal_c_1d = np.tile(norm_pdf.reshape(1, kde_num, 1), (batch_size, 1, latent_dim))
        
        kde_layer_1d = DEVAE_network.KDELayer1D(lower = kde_range[0], upper = kde_range[1], 
                                num = kde_num, n_sample = n_sample)
        if kde_method =='grid':
            N_theta = int(np.sqrt(kde_num/4))*4
            N_r = int(np.sqrt(kde_num/4))+1
            r = np.linspace(0, kde_range[1], N_r)[1:]  # Exclude the origin for radial points
            theta = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
            
            # Create a meshgrid
            R, Theta = np.meshgrid(r, theta)
        
            # Convert polar coordinates to Cartesian coordinates
            X = R * np.cos(Theta)
            Y = R * np.sin(Theta)
            
            # Flatten and stack the X and Y arrays to get the desired shape
            points = np.stack((X.flatten(), Y.flatten()), axis=-1)
            points = np.concatenate(([[0,0]],points))
        else:
            points = np.zeros((kde_num,latent_dim))
            s = 0
            while s < kde_num:
                point = np.random.uniform(kde_range[0], kde_range[1], latent_dim)
                if np.linalg.norm(point) <= 3:
                    points[s,:] = point
                    s+=1
            
        z_ = (1 / (2 * np.pi)) * np.exp(-0.5 * np.linalg.norm(points,2,1)**2)
        pdf_normal_c_nd = np.tile(z_.reshape(1, -1), (batch_size, 1))
        kde_layer_nd = DEVAE_network.KDELayer(points)

        
    elif example in examples[-5:]:
        
        x = np.linspace(kde_range[0], kde_range[1], kde_num)
        norm_pdf = norm.pdf(x)
        pdf_normal_c_1d = np.tile(norm_pdf.reshape(1, kde_num, 1), (batch_size, 1, latent_dim))
        
        kde_layer_1d = DEVAE_network.KDELayer1D(lower = kde_range[0], upper = kde_range[1], 
                                num = kde_num, n_sample = n_sample)
        
        kde_num *=latent_dim
        
        points = sample_nd_ball(latent_dim,kde_num)
        
        z_ = (1 / (2 * np.pi)**(latent_dim/2)) * np.exp(-0.5 * np.linalg.norm(points,2,1)**2)
        pdf_normal_c_nd = np.tile(z_.reshape(1, -1), (batch_size, 1))
        kde_layer_nd = DEVAE_network.KDELayer(points)
            
        if kde_method =='1dn':
            
            kde_layer_nd, pdf_normal_c_nd = None, None
        elif kde_method =='nd':
            kde_layer_1d, pdf_normal_c_1d = None, None
            
        
    return kde_layer_1d, pdf_normal_c_1d, kde_layer_nd, pdf_normal_c_nd
        
    