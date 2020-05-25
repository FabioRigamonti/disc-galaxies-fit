import numpy as np 
import data_class as dc
import matplotlib.pyplot as plt
from utils import model,likelihood
#from utils import differenza,hernquist_rho,hernquist_sigma
import time

'''this is in Km^2 Kpc / (M_sun s^2).'''
G = 4.299e-6


N = 100
J = 100



#---------------------------------------------------------------------------
                    #HERQUINST
#---------------------------------------------------------------------------

'''

def hernquist(M,Rb,r):
    rho = hernquist_rho(M,Rb,r)
    sigma = hernquist_sigma(M,Rb,rho,r)
    return(rho,sigma)

   

def v_H (M,a,r):
    #circular velocity squared
    #for an herquinst model.
     
    if (M == 0.) or (a == 0.):
        return 0*np.copy(r)

    #circular velocity squared
    v_c2 = (G * M * r) / (r + a)**2

    return v_c2


#---------------------------------------------------------------------------
                    #EXP DISC
#---------------------------------------------------------------------------

def rho_D(Md,R_d,r):
    #surface density for a thin disc with no
    #thickness (no z coord). So I don't have any 
    #integration on the LOS,but only rotation,traslation 
    #and deprojection of the coordinates.
    
    if (Md == 0.) or (R_d == 0.):
        return np.copy(x)*0
    

    rho = (Md / (2 * np.pi * R_d**2) ) * np.exp(-r/R_d)

    return(rho)


def v_D(M,R_d,r):
    #circular velocity for a thin disc with no
    #thickness (no z coord).
    #this was to check only herquinst
    if (M == 0.) or (R_d == 0.):
        return 0*np.copy(r)
    if (np.size(r[r==0]) != 0):
        r[r==0] = 1e-8
    #compute the circular velocity squared moltiplicata già per il raggio
    #computational problem if the argument of bessel function il large (i.e. 600)
    #vc must go to 0
    r_2Rd = r/2*R_d
    v_c2 =  ((G * M * r) / (R_d*R_d))* r_2Rd * differenza(r_2Rd)
    
    
    return(v_c2)

#---------------------------------------------------------------------------
                    #TOTATL QUANTITIES
#---------------------------------------------------------------------------

def rhosigma_tot(Mb,Rb,Md,Rd,c_incl,r_proj,r_true):
    #return the densties e the bulge dispersion
    rhoH,sigma = hernquist(Mb,Rb,r_proj)
    
    rhoD = rho_D(Md,Rd,r_true) /c_incl

    rhotot =  rhoH + rhoD

    return (rhotot,rhoH,rhoD,sigma)



def v_tot(Mb,Rb,Md,Rd,Mh,Rh,r,s_incl,c_phi):
    #return the total circular velocity, in the
    #disc plane. Maybe should return -vsin(i)cos(phi),
    #but it doesn't matter
    if (Md == 0.):
        #se non ho ne disco ne alone la velocità media lungo la linea di vista
        #deve essere nulla, xk bulge isotropo (questo accade in tutti i casi in cui non ho disco). In realtà sto mettendo tutte le 
        #velocità nulle, ma va bene uguale perché in questo caso la sigma e data 
        #da quella di herquinst
        return(np.zeros((r.shape[0],r.shape[1],r.shape[2])))

    vtot = np.sqrt(v_D(Md,Rd,r) + v_H(Mb,Rb,r) + v_H(Mh,Rh,r))

    return vtot*s_incl*c_phi


#---------------------------------------------------------------------------
                    #MODEL
#---------------------------------------------------------------------------

def model_vecchio(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice):
    
    #masse passate in logaritmo
    Mb = 10**Mb
    Md = 10**Md
    Mh = 10**Mh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    c_incl = np.cos(incl)
    s_incl = np.sin(incl)
    #traslo,ruoto,deproietto
    x0 = x-xcm
    y0 = y-ycm
    xr = x0*c_theta + y0*s_theta
    yr = y0*c_theta - x0*s_theta
    yd = yr/c_incl

    phi = np.arctan2(yd,xr)
    c_phi = np.cos(phi)
    r_proj = np.sqrt(x0*x0 + y0*y0)
    r_true = np.sqrt(xr*xr + yd*yd)
    #creo i miei modelli come per la griglia
    rho = np.zeros((N,J))
    v_los = np.zeros((N,J))
    sigma_los = np.zeros((N,J))
    
    #valuto le densità in tutti i punti
    rhotot,rhoH,rhoD,sigmaH2 = rhosigma_tot(Mb,Rb,Md,Rd,c_incl,r_proj,r_true)
    
    #questo è equivalente a ciclo su griglia in cui medio punto per punto
    rho[:-1,:-1] = np.mean(rhotot,axis=2)
    if (choice == 'fotometria'):
        #se ho solo fotometria non calcolo le velocità,sulla griglia
        rho_cons = rho.ravel()
        v_cons = v_los.ravel()
        sigma_cons = sigma_los.ravel()
    else:
        #se il disco è diverso da zero calcolo le velocità
        if Md != 0 :
            #valuto le velocità in tutti i punti
            vtot = v_tot(Mb,Rb,Md,Rd,Mh,Rh,r_true,s_incl,c_phi)
            somma_rho= np.sum(rhoD,axis=2)      #matrice (N,J) con tutte le dens
            #se ce ne è almeno uno nullo, in quelli nulli la velocità deve essere zero
            indice = somma_rho != 0
            num =  np.sum(rhoD*vtot,axis=2)
            #calcolo v sono su quelli con densità non nulla 
            partial_v = np.zeros((N-1,J-1))
            partial_v[indice] = num[indice]/somma_rho[indice]
            v_los[:-1,:-1] = partial_v
        #questo mi serve per calcolare vtot-v_los in pratica a è (N,J,100) e a[0,0,:] contiene 100 volte v_los[0,0] 
        #repeat ripete tutti gli elementi di un vettore k volte se x = [1,2] repeat(x,2)= [1,1,2,2]
        a = np.repeat(v_los[:-1,:-1].ravel(),100).reshape(N-1,J-1,100)
        sigma_D = vtot -a
        sigma_los[:-1,:-1] = np.sqrt(np.sum(rhoD * sigma_D*sigma_D + sigmaH2 * rhoH,axis = 2) /np.sum(rhotot,axis = 2) )
        rho_cons = rho.ravel()
        v_cons = v_los.ravel()
        sigma_cons = sigma_los.ravel()
    
    if choice == 'fotometria':
        return rho_cons
    elif choice == 'cinematica':
        return (v_cons,sigma_cons)
    elif choice == 'fotometria + cinematica':
        return(rho_cons,v_cons,sigma_cons)


'''
def mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice):
    tot = model(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl)
    rho_cons = tot[:,:,0].ravel()
    v_cons = tot[:,:,1].ravel()
    sigma_cons = tot[:,:,2].ravel()
    #print(rho_cons)
    #print(v_cons)
    #print(sigma_cons)
    #print(np.max(rho_cons))
    #print(np.max(v_cons))
    #print(np.max(sigma_cons))
    if choice == 'fotometria':
        return rho_cons
    elif choice == 'cinematica':
        return (v_cons,sigma_cons)
    elif choice == 'fotometria + cinematica':
        return(rho_cons,v_cons,sigma_cons)

#---------------------------------------------------------------------------
                    #FIT CM
#---------------------------------------------------------------------------
'''
def log_likelihood(par,x,y,data,data_err,valuto_data,choice):

    Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = par

    if choice == 'fotometria':
        #rho_m = mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        rho_m = model_vecchio(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        valuto = valuto_data
        mod = np.log10(rho_m[valuto])
        #mod = np.log10(rho_m)
    elif choice == 'cinematica':
        #v_m,sigma_m = mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        v_m,sigma_m = model_vecchio(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        valuto = np.concatenate((valuto_data,valuto_data)) 
        mod = np.concatenate((v_m[valuto_data],sigma_m[valuto_data]))
        #mod = np.concatenate((v_m,sigma_m))
    elif choice == 'fotometria + cinematica':
        #rho_m,v_m,sigma_m = mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        rho_m,v_m,sigma_m = model_vecchio(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        valuto = np.concatenate((valuto_data,valuto_data,valuto_data)) 
        mod = np.concatenate((np.log10(rho_m[valuto_data]),v_m[valuto_data],sigma_m[valuto_data]))
        #mod = np.concatenate((np.log10(rho_m),v_m,sigma_m))

    
    lnp = -0.5 * ( (data[valuto] - mod)**2/data_err[valuto]**2  + np.log(2*np.pi*data_err[valuto]**2) )

    return (np.sum(lnp)) 
'''



  
#---------------------------------------------------------------------------
                        #MAIN
#---------------------------------------------------------------------------
if __name__=="__main__":
    #definisco qui così uso la griglia senza definirla nella funzione
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


    uno = rho > 0
    due = np.isnan(v_los) == False
    tre = np.isnan(sigma_los) == False
    quattro = np.isnan(err_v) == False
    cinque  = np.isnan(err_sigma) == False
    sei = np.isnan(err_lrho) == False

    valuto_data = uno*due*tre*quattro*cinque*sei





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
    
    #old model
    #ti = time.time()
    #rho_vecchio,v_vecchio,sigma_vecchio = model_vecchio(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,'fotometria + cinematica')
    #print(time.time()-ti)
    
    #model with all function from utils
    ti = time.time()
    rho_model,v_model,sigma_model = mymodel(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,'fotometria + cinematica')
    print(time.time()-ti)

    
    #per nuova likelihood
    #in questo caso va bene,va rivisto ad esempio nel caso di lupi dove la griglia
    #può essere ridotta e dim non è 99 ma va calcolato

    ydata = np.zeros((N,J,3))
    yerr = np.zeros((N,J,3))
    ydata[:,:,0] = np.log10(rho).reshape(N,J)
    ydata[:,:,1] = v_los.reshape(N,J)
    ydata[:,:,2] = sigma_los.reshape(N,J)
    yerr[:,:,0] = err_lrho.reshape(N,J)
    yerr[:,:,1] = err_v.reshape(N,J)
    yerr[:,:,2] = err_sigma.reshape(N,J)

    
    for _ in range(10):
        #par = [Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl]
        #ti = time.time()
        #lk_vecchia = log_likelihood(par,x_rand,y_rand,np.concatenate((np.log10(rho),v_los,sigma_los)),np.concatenate((err_lrho,err_v,err_sigma)),valuto_data,'fotometria + cinematica')
        #print(time.time()-ti)
        ti = time.time()
        lk_nuova = likelihood(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,ydata,yerr)
        print(time.time()-ti)
    
    
    
    '''
    mydata1 =dc.data(x,y,np.copy(rho_model),np.copy(v_model),np.copy(sigma_model),0,0,0)

    fig6,ax6,pmesh6,cbar6 = mydata1.surface_density()
    ax6.set_title('nuovo')
    limrho = pmesh6.get_clim()
    fig6.show()

    fig4,ax4,pmesh4,cbar4 = mydata1.velocity_map()
    ax4.set_title('nuovo')
    limv = pmesh4.get_clim()
    fig4.show()

    fig5,ax5,pmesh5,cbar5 = mydata1.dispersion_map()
    ax5.set_title('nuovo')
    limsigma = pmesh5.get_clim()
    fig5.show()
    
    
    mydata2 =dc.data(x,y,np.copy(rho_vecchio),np.copy(v_vecchio),np.copy(sigma_vecchio),0,0,0)

    fig6,ax6,pmesh6,cbar6 = mydata2.surface_density()
    ax6.set_title('vecchio')
    limrho = pmesh6.get_clim()
    fig6.show()

    fig4,ax4,pmesh4,cbar4 = mydata2.velocity_map()
    ax4.set_title('vecchio')
    limv = pmesh4.get_clim()
    fig4.show()

    fig5,ax5,pmesh5,cbar5 = mydata2.dispersion_map()
    ax5.set_title('vecchio')
    limsigma = pmesh5.get_clim()
    fig5.show()


    abs_rho = np.abs(rho_model-rho_vecchio)
    abs_v = np.abs(v_model-v_vecchio)
    abs_sigma = np.abs(sigma_model-sigma_vecchio)

    mydata3 =dc.data(x,y,np.copy(abs_rho),np.copy(abs_v),np.copy(abs_sigma),0,0,0)

    fig6,ax6,pmesh6,cbar6 = mydata3.surface_density()
    ax6.set_title('abs diff')
    limrho = pmesh6.get_clim()
    fig6.show()

    fig4,ax4,pmesh4,cbar4 = mydata3.velocity_map()
    ax4.set_title('abs diff')
    limv = pmesh4.get_clim()
    fig4.show()

    fig5,ax5,pmesh5,cbar5 = mydata3.dispersion_map()
    ax5.set_title('abs diff')
    limsigma = pmesh5.get_clim()
    fig5.show()

    plt.show()


    y = y-5
    a = np.min(abs(y))
    indice = y==a
    plt.plot(x[indice]-20,abs_sigma[indice],'b')
    plt.show()
    '''