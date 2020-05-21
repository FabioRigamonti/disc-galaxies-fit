import numpy as np 
import data_class as dc
import matplotlib.pyplot as plt
from utils import model 
from scipy.special import i0e,i1e,k0e,k1e
import time

'''this is in Km^2 Kpc / (M_sun s^2).'''
G = 4.299e-6


N = 100
J = 100



#---------------------------------------------------------------------------
                    #HERQUINST
#---------------------------------------------------------------------------


'''
def herquinst(M,Rb,x0,y0):
    #function that return density and
    #isotropic velocity dispersion.
    #calcolo tutti insieme
    if (M == 0.) or (Rb == 0.):
        return (0*np.copy(x0),0*np.copy(x0))
	

    R = (x0**2 + y0**2)**0.5

    s = R / Rb
    
    #se s dovesse essere proprio zero
    if (np.size(s[s==0]) != 0):
        s[s==0] = 1e-8

    #definition of variables
    X = np.zeros((N-1,J-1,100))
    A = np.zeros((N-1,J-1,100))
    B = np.zeros((N-1,J-1,100))
    C = np.zeros((N-1,J-1,100))
    D = np.zeros((N-1,J-1,100))
    rho = np.zeros((N-1,J-1,100))
    sigma = np.zeros((N-1,J-1,100))

    #X
    a = s < 1
    b = s == 1
    c = s > 1
    #s < 1
    X[a] = np.log((1 + (1-s[a]**2)**0.5) / s[a]) / ((1 - s[a]**2)**0.5)
    #s = 1
    X[b] = 1.
    #s > 1
    X[c] = np.arccos(1/s[c]) / ((s[c]**2 - 1)**0.5)
    
    #rho 
    d = (s >= 0.98) * (s <= 1.02)
    e = s!=1
    A[e] = M / (2 * np.pi * Rb**2 * (1 - s[e]**2)**2)
    B[e] = (2 + s[e]**2) * X[e] - 3
    rho[e] = A[e] * B[e]
    rho[d] = 4*M / (30*np.pi*Rb**2)

    #sigma
    f = s <700.
    g = s > 700.
    h = (s >= 0.98) * (s <= 1.02)

    # s < 10
    A = (G * M**2 ) / (12 * np.pi * Rb**3 * rho[f])
    B = 1 /  (2*(1 - s[f]**2)**3)
    C = (-3 * s[f]**2) * X[f] * (8*s[f]**6 - 28*s[f]**4 + 35*s[f]**2 - 20) - 24*s[f]**6 + 68*s[f]**4 - 65*s[f]**2 + 6
    sigma[f] = A * (B*C - 6*np.pi*s[f])
    #A = (G * M**2 ) / (12 * np.pi * Rb**3 * rho)
    #B = 1 /  (2*(1 - s**2)**3)
    #C = (-3 * s**2) * X * (8*s**6 - 28*s**4 + 35*s**2 - 20) - 24*s**6 + 68*s**4 - 65*s**2 + 6
    #sigma = A * (B*C - 6*np.pi*s)

    # s > 10
    A = (G * M * 8) / (15 * s[g] * np.pi * Rb)
    B = (8/np.pi - 75*np.pi/64) / s[g]
    C = (64/np.pi**2 - 297/56) / s[g]**2
    D = (512/np.pi**3 - 1199/(21*np.pi) - 75*np.pi/512) / s[g]**3
    sigma[g] = A * (1 + B + C + D)

    #s = 1
    sigma[h] = G*M*(332 - 105*np.pi)/28*Rb

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

def rho_D(Md,R_d,x,y):
    #surface density for a thin disc with no
    #thickness (no z coord). So I don't have any 
    #integration on the LOS,but only rotation,traslation 
    #and deprojection of the coordinates.
    r = (x**2 + y**2)**0.5
    
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
    v_c2 =  ((G * M * r**2) / (2 * R_d**3)) * (i0e(r/(2*R_d)) * k0e(r/(2*R_d)) - i1e(r/(2*R_d)) * k1e(r/(2*R_d))) 
    
    
    return(v_c2)

#---------------------------------------------------------------------------
                    #TOTATL QUANTITIES
#---------------------------------------------------------------------------

def rhosigma_tot(Mb,Rb,Md,Rd,i,x0,y0,xr,yd):
    #return the densties e the bulge dispersion
    rhoH,sigma = herquinst(Mb,Rb,x0,y0)
    
    rhoD = rho_D(Md,Rd,xr,yd) /np.cos(i)

    rhotot =  rhoH + rhoD

    return (rhotot,rhoH,rhoD,sigma)



def v_tot(Mb,Rb,Md,Rd,Mh,Rh,i,xr,yd):
    #return the total circular velocity, in the
    #disc plane. Maybe should return -vsin(i)cos(phi),
    #but it doesn't matter
    if (Md == 0.):
        #se non ho ne disco ne alone la velocità media lungo la linea di vista
        #deve essere nulla, xk bulge isotropo (questo accade in tutti i casi in cui non ho disco). In realtà sto mettendo tutte le 
        #velocità nulle, ma va bene uguale perché in questo caso la sigma e data 
        #da quella di herquinst
        return(0*np.copy(xr))

    r = (xr**2 + yd**2)**0.5

    phi = np.arctan2(yd,xr)

    vtot = (v_D(Md,Rd,r) + v_H(Mb,Rb,r) + v_H(Mh,Rh,r))**0.5

    return vtot*np.sin(i)*np.cos(phi)


#---------------------------------------------------------------------------
                    #MODEL
#---------------------------------------------------------------------------

def model_vecchio(X,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice):
    
    #masse passate in logaritmo
    Mb = 10**Mb
    Md = 10**Md
    Mh = 10**Mh

    #print('{:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3e} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f} \t {:.3f}'.format(Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl))
    
    #x e y sono matrici (N,J,100), x[i,j,0] da la x del punto i,j della griglia
    x,y = X
    
    #traslo,ruoto,deproietto
    x0 = x-xcm
    y0 = y-ycm
    xr = x0*np.cos(theta) + y0*np.sin(theta)
    yr = y0*np.cos(theta) - x0*np.sin(theta)
    yd = yr/np.cos(incl)

    #creo i miei modelli come per la griglia
    rho = np.zeros((N,J))
    v_los = np.zeros((N,J))
    sigma_los = np.zeros((N,J))
    
    #valuto le densità in tutti i punti
    rhotot,rhoH,rhoD,sigmaH2 = rhosigma_tot(Mb,Rb,Md,Rd,incl,x0,y0,xr,yd)

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
            vtot = v_tot(Mb,Rb,Md,Rd,Mh,Rh,incl,xr,yd)
            somma_rho= np.sum(rhoD,axis=2)      #matrice (N,J) con tutte le dens
            if np.size(somma_rho[somma_rho==0]) == 0.:
                #se nessuna densità è nulla calcolo la velocità media
                #in alcuni casi ad alti r/Rd può capitare che la densità sia zero (e^-(r/Rd))
                v_los[:-1,:-1] = np.sum(rhoD*vtot,axis=2)/somma_rho
            else:
                #se ce ne è almeno uno nullo, in quelli nulli la velocità deve essere zero
                partial_v = np.zeros((N-1,J-1))
                indice = somma_rho != 0
                num =  np.sum(rhoD*vtot,axis=2)
                den = somma_rho
                #calcolo v sono su quelli con densità non nulla
                partial_v[indice] = num[indice]/den[indice]
                v_los[:-1,:-1] = partial_v 
        else:
            vtot = 0*np.copy(x)
        #questo mi serve per calcolare vtot-v_los in pratica a è (N,J,100) e a[0,0,:] contiene 100 volte v_los[0,0] 
        #repeat ripete tutti gli elementi di un vettore k volte se x = [1,2] repeat(x,2)= [1,1,2,2]
        a = np.repeat(v_los[:-1,:-1].ravel(),100).reshape(N-1,J-1,100)
        sigma_los[:-1,:-1] = (np.sum(rhoD * (vtot-a)**2 + sigmaH2 * rhoH,axis = 2) /np.sum(rhotot,axis = 2) )**0.5
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

def log_likelyhood(par,x,y,data,data_err,valuto_data,choice):

    Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl = par

    if choice == 'fotometria':
        rho_m = mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        valuto = valuto_data
        mod = np.log10(rho_m[valuto])
        #mod = np.log10(rho_m)
    elif choice == 'cinematica':
        v_m,sigma_m = mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        valuto = np.concatenate((valuto_data,valuto_data)) 
        mod = np.concatenate((v_m[valuto_data],sigma_m[valuto_data]))
        #mod = np.concatenate((v_m,sigma_m))
    elif choice == 'fotometria + cinematica':
        rho_m,v_m,sigma_m = mymodel(x,y,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,choice)
        valuto = np.concatenate((valuto_data,valuto_data,valuto_data)) 
        mod = np.concatenate((np.log10(rho_m[valuto_data]),v_m[valuto_data],sigma_m[valuto_data]))
        #mod = np.concatenate((np.log10(rho_m),v_m,sigma_m))

    
    lnp = -0.5 * ( (data[valuto] - mod)**2/data_err[valuto]**2  + np.log(2*np.pi*data_err[valuto]**2) )

    return (np.sum(lnp)) 


    
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
    Md = -np.inf#np.log10(5e10)  
    Rd 	 =0.8  
    Mh 	= np.log10(5.e+12)  
    Rh =	 10.000  
    xcm 	 =20  
    ycm 	= 5  
    theta =	 2*np.pi/3 
    incl =	 np.pi/4
    

    #rho_vecchio,v_vecchio,sigma_vecchio = model_vecchio((x_rand,y_rand),Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,'fotometria + cinematica')
    #rho_model,v_model,sigma_model = mymodel(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,'fotometria + cinematica')

    for _ in range(10):
        par = [Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl]
        #lk_foto = log_likelyhood(par,(x_rand,y_rand),np.log10(rho),err_lrho,valuto_data,'fotometria')
        #lk_cine = log_likelyhood(par,(x_rand,y_rand),np.concatenate((v_los,sigma_los)),np.concatenate((err_v,err_sigma)),valuto_data,'cinematica')
        #lk_tot = log_likelyhood(par,x_rand,y_rand,np.concatenate((np.log10(rho),v_los,sigma_los)),np.concatenate((err_lrho,err_v,err_sigma)),valuto_data,'fotometria + cinematica')
        rho_model,v_model,sigma_model = mymodel(x_rand,y_rand,Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,'fotometria + cinematica')

       # print(lk_tot)

    #plot eventuale
    #rho_model,v_model,sigma_model = model((x_rand,y_rand),Mb,Rb,Md,Rd,Mh,Rh,xcm,ycm,theta,incl,'fotometria + cinematica')
    
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