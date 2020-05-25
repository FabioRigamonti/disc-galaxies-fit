#pe fare l'output file del profiling
#python3 -m cProfile -o output script.py
import pstats
from pstats import SortKey
import cProfile
import numpy as np 
import utils
#import model 
import pyximport

x,y,rho,v_los,sigma_los = np.loadtxt('linear_data5.txt',unpack = True, usecols = [0,1,2,3,4])
x, y, err_rho,err_v,err_sigma = np.loadtxt('linear_error5.txt',usecols=(0,1,2,3,4),unpack = True)

err_lrho = err_rho/rho

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

Mb = np.log10(5e10) 
Rb	 =0.01 
Md = np.log10(5e10)  
Rd 	 =0.8  
Mh 	= np.log10(5.e+12)  
Rh =	 10.000  
xcm 	 =20  
ycm 	= 5  
theta =	 2*np.pi/3 
incl =	 np.pi/4

uno = rho > 0
due = np.isnan(v_los) == False
tre = np.isnan(sigma_los) == False
quattro = np.isnan(err_v) == False
cinque  = np.isnan(err_sigma) == False
sei = np.isnan(err_lrho) == False
valuto_data = uno*due*tre*quattro*cinque*sei

ydata = np.zeros((N,J,3))
yerr = np.zeros((N,J,3))
ydata[:,:,0] = np.log10(rho).reshape(N,J)
ydata[:,:,1] = v_los.reshape(N,J)
ydata[:,:,2] = sigma_los.reshape(N,J)
yerr[:,:,0] = err_lrho.reshape(N,J)
yerr[:,:,1] = err_v.reshape(N,J)
yerr[:,:,2] = err_sigma.reshape(N,J)

par = [Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl]
choice = 'fotometria + cinematica'
cProfile.runctx("utils.likelihood(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr)", globals(), locals(), "profile.output")
#cProfile.runctx("utils.model(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl)", globals(), locals(), "profile.output")
#per usare questo va importato model (dopo aver decommentato le funzioni e dopo aver ricompilato utils.pyx con le funzioni che servono)
#cProfile.runctx("model.log_likelihood(par,x_rand,y_rand,np.concatenate((np.log10(rho),v_los,sigma_los)),np.concatenate((err_lrho,err_v,err_sigma)),valuto_data,choice)", globals(), locals(), "profile.output")

p = pstats.Stats('profile.output')

print('prime 25 in tempo:\n')
p.sort_stats(SortKey.TIME).print_stats(25)
print('relative a model:\n')
p.sort_stats(SortKey.TIME).print_stats('model.py')
print('relative a utils:\n')
p.sort_stats(SortKey.TIME).print_stats('utils.pyx')
