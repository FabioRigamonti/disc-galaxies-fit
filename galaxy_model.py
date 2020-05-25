#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cpnest.model
import numpy as np
from utils import model, likelihood
import data_class as dc

class GalaxyModel(cpnest.model.Model):

    names  = [] #'h','om','ol','w0','w1']
    bounds = [] #[0.5,1.0],[0.04,1.0],[0.0,1.0],[-2.0,0.0],[-3.0,3.0]]

    def __init__(self,
                 x,                   # grid coordinates, x
                 y,                   # grid coordinates, y
                 rho,                 # density value at coordinates
                 v_los,               # line of sight velocity value at coordinates
                 sigma_los,           # line of sight velocity dispersion value at coordinates
                 error_rho,           # density error value at coordinates
                 error_v_los,         # line of sight velocity error value at coordinates
                 error_sigma_los,     # line of sight velocity dispersion error value at coordinates
                 *args, **kwargs):

        super(GalaxyModel,self).__init__()
        # Set up the data
        self.data = dc.data(x, y, rho, v_los, sigma_los, error_rho, error_v_los, error_sigma_los)
        self.choice = kwargs.get('choice')
        
        # set up the parameter names and bounds
        # i am assuming that masses are in solar masses and
        # radii are in kpc
        self.names.append('log10_bulge_mass')
        self.bounds.append([10,11])
        self.names.append('log10_bulge_radius')
        self.bounds.append([-3,-1])
        self.names.append('log10_disc_mass')
        self.bounds.append([10,11])
        self.names.append('log10_disc_radius')
        self.bounds.append([-1,0])
        self.names.append('log10_halo_mass')
        self.bounds.append([12,13])
        self.names.append('log10_halo_radius')
        self.bounds.append([0,2])
        self.names.append('xcm')
        self.bounds.append([10,30])#[self.data.x.min(),self.data.x.max()])
        self.names.append('ycm')
        self.bounds.append([0,10])#[self.data.y.min(),self.data.y.max()])
        self.names.append('orientation')
        self.bounds.append([0.0,np.pi])
        self.names.append('inclination')
        self.bounds.append([-np.pi/2.0,np.pi/2.0])
        self._init_grid()
        self._init_data()
    
    def _init_grid(self):
        J = np.size(x[ x == x[0] ])
        N = np.size(y[ y == y[0] ])

        x_true = x[0 : J*N : J]
        y_true = y[0 : J]

        dx = np.abs(x_true[0]-x_true[1])
        dy = np.abs(y_true[0]-y_true[1])

        x_rand = np.zeros((N-1,J-1,100))
        y_rand = np.zeros((N-1,J-1,100))

        size = 10

        for k in range(0,N-1):
                
            a = np.linspace(x_true[k],x_true[k+1],size)
            xrand = np.repeat(a,size).reshape(size,size).T.ravel()

            for j in range(0,J-1):
                    
                b = np.linspace(y_true[j],y_true[j+1],size)
                yrand = np.repeat(b,size)

                x_rand[k,j,:] = xrand
                y_rand[k,j,:] = yrand
        
        self.J = J
        self.N = N
        self.X = (x_rand, y_rand)
    
    def _init_data(self):
        self.ydata = np.zeros((self.N,self.J,3))
        self.yerr = np.zeros((self.N,self.J,3))
        self.ydata[:,:,0] = np.log10(self.data.rho).reshape(self.N,self.J)
        self.ydata[:,:,1] = self.data.v_los.reshape(self.N,self.J)
        self.ydata[:,:,2] = self.data.sigma_los.reshape(self.N,self.J)
        self.yerr[:,:,0] = self.data.error_lrho.reshape(self.N,self.J)
        self.yerr[:,:,1] = self.data.error_v_los.reshape(self.N,self.J)
        self.yerr[:,:,2] = self.data.error_sigma_los.reshape(self.N,self.J)
        
    def log_prior(self,x):
        return super(GalaxyModel,self).log_prior(x)
    
    def log_likelihood(self, x):
        Mb = x['log10_bulge_mass']
        Md = x['log10_disc_mass']
        Mh = x['log10_halo_mass']
        Rb = 10**x['log10_bulge_radius']
        Rd = 10**x['log10_disc_radius']
        Rh = 10**x['log10_halo_radius']
        xcm = x['xcm']
        ycm = x['ycm']
        theta = x['orientation']
        incl  = x['inclination']
        x_g, y_g = self.X
        logL = likelihood(x_g,y_g,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,self.ydata,self.yerr)

        return logL

if __name__=="__main__":
    x, y, rho, v_los, sigma_los = np.loadtxt('linear_data5.txt',unpack = True, usecols = [0,1,2,3,4])
    x, y, err_rho, err_v, err_sigma = np.loadtxt('linear_error5.txt',usecols=(0,1,2,3,4),unpack = True)
    M = GalaxyModel(x, y, rho, v_los, sigma_los, err_rho, err_v, err_sigma, choice='fotometria + cinematica')
    work=cpnest.CPNest(M,
                       verbose  = 2,
                       poolsize = 100,
                       nthreads = 4,
                       nlive    = 1000,
                       maxmcmc  = 100,
                       output   = '.',
                       nhamiltonian = 0,
                       resume   = 1)
    work.run()
