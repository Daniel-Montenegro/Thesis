"""
En este codigo se realizan las graficas con las cuales se extrea informacion de los datos.
"""

"""
----------------------------
LIBRERIAS
----------------------------
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

"""                                                                             
===================================                                             
    ->Lectura de los autovectores                                               
"""

from struct import *
import sys
import os

#====================================
"""
************************************
LECTURA Y ALMACENAMIENTO DE DATOS
************************************
"""

data=np.loadtxt("Data(halo-bh-stelar)_new.txt")

print("size data initial:", len(data))

size_data=len(data)

#===========Halos===========#

Mass_halo=data[:,0]*1e10/0.7 #M_sun
vel_dis_halo = data[:,4]
vel_max_halo = data[:,5]
Mass_stelar_halo = data[:,9]*1e10/0.7 #Msun

r_halo = []
spin_halo = []

for i in range(size_data):
     
    r_halo.append([data[i,1],data[i,2],data[i,3]]) 
    spin_halo.append([data[i,6],data[i,7],data[i,8]])

r_halo = np.reshape(r_halo,(size_data,3))
spin_halo = np.reshape(spin_halo,(size_data,3))

#========BHs===========#

Mass_bh=data[:,10]## Ya estan en Masa solar
vel_dis_bh= data[:,14]


r_bh = []
spin_bh = []
spin_disk_bh = []
for i in range(size_data):
     
    r_bh.append([data[i,11],data[i,12],data[i,13]])
    spin_bh.append([data[i,15],data[i,16],data[i,17]])
    spin_disk_bh.append([data[i,18],data[i,19],data[i,20]])

r_bh= np.reshape(r_bh,(size_data,3))
spin_bh = np.reshape(spin_bh,(size_data,3))
spin_disk_bh = np.reshape(spin_disk_bh,(size_data,3))


##*****
#Condicion para que no tome los objetos que tiene
# una masa de halo menor a 1e9 Msun
##****

##==== CONDICION PARA HALOS ===##

"""
condicion = 6e10
#------Halo-----------
vel_dis_halo = vel_dis_halo[Mass_halo > condicion]
vel_max_halo = vel_max_halo[Mass_halo > condicion]
Mass_stelar_halo = Mass_stelar_halo[Mass_halo > condicion]
r_halo = r_halo[Mass_halo > condicion]
spin_halo = spin_halo[Mass_halo > condicion]

#--------BH-----------

Mass_bh = Mass_bh[Mass_halo > condicion]
vel_dis_bh= vel_dis_bh[Mass_halo > condicion]
r_bh = r_bh[Mass_halo > condicion]
spin_bh = spin_bh[Mass_halo > condicion]
spin_disk_bh = spin_disk_bh[Mass_halo > condicion]

Mass_halo = Mass_halo[Mass_halo > condicion]
"""
##=== CONDICION PARA MASA ESTELAR ===##

condicion = 1e8 #Msun
#------Halo-----------
vel_dis_halo = vel_dis_halo[Mass_stelar_halo > condicion]
vel_max_halo = vel_max_halo[Mass_stelar_halo > condicion]
Mass_halo = Mass_halo[Mass_stelar_halo > condicion]
r_halo = r_halo[Mass_stelar_halo > condicion]
spin_halo = spin_halo[Mass_stelar_halo > condicion]

#--------BH-----------

Mass_bh = Mass_bh[Mass_stelar_halo > condicion]
vel_dis_bh= vel_dis_bh[Mass_stelar_halo > condicion]
r_bh = r_bh[Mass_stelar_halo > condicion]
spin_bh = spin_bh[Mass_stelar_halo > condicion]
spin_disk_bh = spin_disk_bh[Mass_stelar_halo > condicion]

Mass_stelar_halo = Mass_stelar_halo[Mass_stelar_halo > condicion]

print("size data final:", len(Mass_halo))
print("size data final:", len(Mass_bh))
size_data = len(Mass_halo)
#----------------------------------------------------
#----------------------------------------------------

"""
GRAFICA BH'S
"""
##HISTOGRAMA##                      
plt.figure()
plt.hist(np.log10(Mass_bh),bins=35, log=True)
plt.xlabel("$\log_{10}(Mass_{bh})[M_{\odot}]$")
plt.savefig("histo_Mass_bh.png")



### ajuste dispersion de la velocidad                                                                                                                 
def func_disp_bh(x,A,B):
    return A*x**2+B

popt, pcov = curve_fit(func_disp_bh, np.log10(vel_dis_bh),np.log10(Mass_bh))

## parametros para la curva teorica

alpha = 8.13
beta = 4.02
sigma_0 = 200. ##km/s
#referencia: http://iopscience.iop.org/article/10.1086/341002/pdf 

plt.figure()
#plt.plot(xdata, func(xdata, *popt), 'g--',                                  
plt.plot(np.log10(vel_dis_bh),func_disp_bh(np.log10(vel_dis_bh), *popt), '.k', label='ajuste')
plt.plot(np.log10(vel_dis_bh),np.log10(Mass_bh),".", label="Data")
plt.plot(np.log10(vel_dis_bh), (alpha+beta*np.log10(vel_dis_bh/sigma_0)), label='linea teorica')
plt.xlabel("$\log(\sigma_{bh})[km/s]$")
plt.ylabel("$\log_{10}(Mass_{bh})[Mass_{\odot}]$")
plt.legend()
plt.savefig('vel_dis_VS_mass_bh.png')



#===============================================================
"""
GRAFICA HALOS
"""

##HISTOGRAMA Halo
plt.figure()
plt.hist(np.log10(Mass_halo),bins=35, log=True)
plt.xlabel("$\log_{10}(Mass_{halo})[M_{\odot}]$")
plt.savefig("histo_Mass_halos.png")

"""
##HISTOGRAMA mass_stelar
plt.figure()
plt.hist(np.log10(Mass_stelar_halo),bins=35, log=True)
plt.xlabel("$\log_{10}(Mass_{halo})[M_{\odot}]$")
plt.savefig("histo_Mass_halos.png")
"""

##GRAFICA DE LA DISTRIBUCION ESPACIAL HALOS Y BH'S

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(r_bh[:,0], r_bh[:,1], r_bh[:,2], s=1, c='r', alpha=0.9,  label="bh")
ax.scatter(r_halo[:,0], r_halo[:,1],  r_halo[:,2], s=8, c='b', alpha=0.4,label="halo")
ax.legend()
ax.view_init(elev=50., azim=35)
plt.savefig("ubicion_espacial_hal0-bh.jpg")

###======###

plt.figure()
plt.plot(np.log10(vel_max_halo),np.log10(Mass_halo),".", label='halo')
plt.xlabel("$Vel_{max,halo}[km/s]$")
plt.ylabel("$\log_{10}(Mass_{halo})[Mass_{\odot}]$")
plt.legend()
plt.savefig('vel_max_VS_mass_Halo.png')

plt.figure()
plt.plot(np.log10(vel_dis_halo),np.log10(Mass_halo),".", label='halo')
#plt.plot(xdata, func(xdata, *popt), 'g--',
plt.xlabel("$\log(\sigma_{halo})[km/s]$")
plt.ylabel("$\log_{10}(Mass_{halo})[Mass_{\odot}]$")
plt.legend()
plt.savefig('vel_dis_VS_mass_Halo.png')


plt.figure()
plt.plot( Mass_stelar_halo,Mass_bh, '.')
plt.yscale("log")
plt.xscale("log")
plt.ylabel("$Mass_{bh}[M_{\odot}]$")
plt.xlabel("$Mass_{\star}[M_{\odot}]$")
plt.savefig("Mass_bhVsMass_stelar_halo.png")

#=== Relacion-halo-bh ===#

## Ajuste                                                                      

def f_MbhVsMhalo(x,a,b,c):
    return a*x**c+b

popt2, pcov2 = curve_fit(f_MbhVsMhalo, Mass_halo, Mass_bh )

plt.figure()
plt.loglog(Mass_halo,Mass_bh,'.', label="Dato")
#plt.plot(xdata, func(xdata, *popt), 'g--',                                 
plt.loglog(Mass_halo, f_MbhVsMhalo(Mass_halo, *popt2), ".k", label='ajuste')
plt.ylabel("$Mass_{bh}$")
plt.xlabel("$Mass_{halo}$")
plt.legend()
plt.savefig("Mass_bh_Vs_Mass_bh.png")


plt.figure()
plt.plot(Mass_halo,(Mass_bh/Mass_halo),'.')
plt.xscale("log")
plt.yscale("log")
plt.ylabel("$Mass_{bh}/Mass_{halo}$")
plt.xlabel("$Mass_{halo}$")
plt.savefig("Mass_bh_div_Mass_haloVsMass_halo.png")


"""
En esta parte del codigo se pretende calcular el angulo que existe entre los 
spines de BHs, halos, disco de acrecion y la direccion de laTweb(EigenVec).
Para esto se calcula el producto punto y cargar los valores de los autovalores.

1. leer los Autovectores(haciendo uso de funciones que apartir del radio pueda
 extraer el valor del Eutovec de esa celda)
2. Calcular el producto punto entre los vectores.
  
"""
#===============================================
"""
*************************
FUNCIONES
************************
"""

"""                                                                            
======================                                                        
Lectura de Datos                                                             
     ---> EigenVectores                                                       
======================                                                        
"""

def read_eigenVec(folder,file,NumEigenVec):
    print("Reading eigenvector file")
    f = open("%s%s%s"%(folder,file,NumEigenVec), "rb")
    #f = open("../Bolshoi/Eigenvec_s1_1", "rb")                               
 
    dumb = f.read(38)

    dumb = f.read(4)
    n_x = f.read(4)
    n_y = f.read(4)
    n_z = f.read(4)
    nodes = f.read(4)
    x0 = f.read(4)
    y0 = f.read(4)
    z0 = f.read(4)
    dx = f.read(4)
    dy = f.read(4)
    dz = f.read(4)
    dumb = f.read(4)

    n_x = (unpack('i', n_x))[0]
    n_y = (unpack('i', n_y))[0]
    n_z = (unpack('i', n_z))[0]
    nodes = (unpack('i', nodes))[0]
    dx = (unpack('f', dx))[0]
    dy = (unpack('f', dy))[0]
    dz = (unpack('f', dz))[0]
    x0 = (unpack('f', x0))[0]
    y0 = (unpack('f', y0))[0]
    z0 = (unpack('f', z0))[0]
    print(n_x, n_y, n_z, nodes, dx, dy, dz)

    total_nodes = 3 * n_x * n_y *n_z
    dumb = f.read(4)
    array_data = f.read(total_nodes*4)
    dumb = f.read(4)
    format_s = str(total_nodes)+'f'
    array_data = unpack(format_s, array_data)
    f.close()
    array_data  = np.array(array_data)
    new_array_data = np.reshape(array_data, (3,n_x,n_y,n_z), order='F')
    return new_array_data, n_x

print("--------------------\n")


#===============================================


"""                                                                            
***********************************                                            
    --> Funcion que devuelve el valor                                          
        de la celda dependiendo de su                                          
        posicion en x,y,z                                                      
***********************************                                            
"""

def Eigen_vec(r,n_x):
    """                                                                       
    Esta funcion retorna el valor del auntoVector                           
    correspondiente a las coordenadas r(x,y,z)                                
    """

    long_box= 25e3 #longitud caja                                              
    dl = n_x/long_box #tamano de cada celda = numero_celdas/long_caja 

    i=np.int(r[0]*dl)
    j=np.int(r[1]*dl)
    k=np.int(r[2]*dl)


    eigen_vec_r = [new_array_data[0,i,j,k],\
                            new_array_data[1,i,j,k],\
                            new_array_data[2,i,j,k]]

    return eigen_vec_r

#===========================================          

"""                                                                            
******************************************                                     
 Asignacion de dek valor de autovec                                            
 y calculo del cos(theta)                                                      
      --->En seccion se asigna el valor del                                    
          autovec(r) dependiendo del radio.                                    
          Ademas, se calcula el producto punto                                 
          y con ello tener el cos(theta)                                       
******************************************                                     
"""

print("---------------------------------\n")
print("Inicia calculo de cos(theta)\n")


Mag_EigenVec_bh=[]
Mag_EigenVec_halo=[]
Mag_Spin_bh=[]
Mag_Spin_halo=[]
Dot_bh=[]
Dot_halo=[]
EigenVec_bh=[]
EigenVec_halo=[]
cos_theta_bh=[]

cos_beta = []
Dot_Sbh_Shalo = []
Mag_Spin_halo = []

cos_alpha = []
Dot_Spin_disk_bh = []
Mag_Spin_disk_bh = []


#===> direccion archivos
folder = '/home/dmontenegro/Data/Sims512/Tweb_512/'
file = 'snap_015.s1.00.eigenvec_'

plt.figure()


counter_halo = 0
counter_bh = 0
Q=1
for j in range(0,3):


    new_array_data, n_x = read_eigenVec(folder,file,"%s"%(Q))
    print("counter = ",counter_halo)
    print("%s%s%s"%(folder,file, Q))


    for i in range(len(r_bh)):


        EigenVec_bh.append(Eigen_vec(r_bh[i],n_x))
        Mag_EigenVec_bh.append(np.linalg.norm(EigenVec_bh[counter_bh])) ##magnitud del autovector                                                   
        Mag_Spin_bh.append(np.linalg.norm(spin_bh[i]))       ##magnitud del Spin_bh                                                                
        #cos_theta
        Dot_bh.append(np.vdot(EigenVec_bh[counter_bh],spin_bh[i]))      ##Productopunto del autovec y spin_bh   
        cos_theta_bh.append(Dot_bh[counter_bh]/(Mag_EigenVec_bh[counter_bh]*Mag_Spin_bh[counter_bh]))
        #cos_beta 
        Dot_Sbh_Shalo.append(np.vdot(spin_bh[i],spin_halo[i])) ##producto punto entre spin_halo y spin_bh                                   
        Mag_Spin_halo.append(np.linalg.norm(spin_halo[i]))
        cos_beta.append(Dot_Sbh_Shalo[counter_bh]/(Mag_Spin_halo[counter_bh]*Mag_Spin_bh[counter_bh]))
        #cos_alpha
        Dot_Spin_disk_bh.append(np.vdot(spin_bh[i],spin_disk_bh[i])) ##producto punto entre spin_halo y spin_bh                                   
        Mag_Spin_disk_bh.append(np.linalg.norm(spin_disk_bh[i]))
        cos_alpha.append(Dot_Spin_disk_bh[counter_bh]/(Mag_Spin_disk_bh[counter_bh]*Mag_Spin_bh[counter_bh]))
        
        counter_bh=counter_bh+1
    Q=Q+1


cos_theta_bh = np.reshape(cos_theta_bh,(3,len(r_bh))) #para cada EigenVec       
cos_beta = np.reshape(cos_beta,(3,len(r_bh)))   #para cada EigenVec             
cos_alpha = np.reshape(cos_alpha,(3,len(r_bh)))   #para cada EigenVec             


#=================================================


"""                                                                                                                                                  
******************************************                                                                                                           
Graficas                                                                                                                                             
******************************************                                                                                                           
"""
print("---------------------------------\n")
print("Inicia Grafica\n")

#================#                                                         
                                                                     
###   THETA   ####                                                          
 ## BH  ## 
                                                                      
plt.figure()
plt.plot(np.log10(Mass_bh),cos_theta_bh[0],'.', label='Eigen_1')
plt.plot(np.log10(Mass_bh),cos_theta_bh[1],'.', label='Eigen_2')
plt.plot(np.log10(Mass_bh),cos_theta_bh[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos( \Theta ) $')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_theta_bh_3_EV.png')




## bh ##                                                                       
"""
fig, axes = plt.subplots(nrows=1, ncols=1)
ax0, ax1, ax2 = axes.flatten()
#ax0 = axes.flatten()
  
ax0.plot(np.log10(Mass_bh),cos_theta_bh[0],'.')
#ax0.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax0.set_ylabel('$\cos(\Theta)$') 
ax0.set_title("Eigen1")

ax1.plot(np.log10(Mass_bh),cos_theta_bh[1],'.')
#ax1.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax1.set_ylabel('$\cos(\Theta)$') 
ax1.set_title("Eigen2")

ax2.plot(np.log10(Mass_bh),cos_theta_bh[2],'.')
ax2.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax2.set_ylabel('$\cos(\Theta)$') 
ax2.set_title("Eigen3")

fig.tight_layout()
"""
plt.figure()
plt.plot(np.log10(Mass_bh),cos_theta_bh[0],'.')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos(\Theta)$') 


plt.savefig('Alinacion_Enviroment_theta_bh_EV.png')



## halo ## 
"""
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()
  
ax0.plot(np.log10(Mass_halo),cos_theta_bh[0],'.')
#ax0.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax0.set_ylabel('$\cos(\Theta)$') 
ax0.set_title("Eigen1")

ax1.plot(np.log10(Mass_halo),cos_theta_bh[1],'.')
#ax1.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax1.set_ylabel('$\cos(\Theta)$') 
ax1.set_title("Eigen2")

ax2.plot(np.log10(Mass_halo),cos_theta_bh[2],'.')
ax2.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax2.set_ylabel('$\cos(\Theta)$') 
ax2.set_title("Eigen3")
fig.tight_layout()
"""
plt.figure()
plt.plot(np.log10(Mass_halo),cos_theta_bh[0],'.')
plt.xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
plt.ylabel('$\cos(\Theta)$') 

plt.savefig('Alinacion_Enviroment_theta_halo_EV.png')




#================#                                                                                                                                  
####   BETA   ####                                                                   

## bh ##                                                                       
"""
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

  
ax0.plot(np.log10(Mass_bh),cos_beta[0],'.')
#ax0.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax0.set_ylabel('$\cos(B)$') 
ax0.set_title("Eigen1")

ax1.plot(np.log10(Mass_bh),cos_beta[1],'.')
#ax1.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax1.set_ylabel('$\cos(B)$') 
ax1.set_title("Eigen2")

ax2.plot(np.log10(Mass_bh),cos_beta[2],'.')
ax2.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax2.set_ylabel('$\cos(B)$') 
ax2.set_title("Eigen3")
fig.tight_layout()
"""
plt.figure()
plt.plot(np.log10(Mass_bh),cos_beta[0],'.')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos(B)$') 

plt.savefig('Alinacion_Enviroment_beta_bh_EV.png')



## halo ## 

"""
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.plot(np.log10(Mass_halo),cos_beta[0],'.')
#ax0.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax0.set_ylabel('$\cos(B)$') 
ax0.set_title("Eigen1")

ax1.plot(np.log10(Mass_halo),cos_beta[1],'.')
#ax1.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax1.set_ylabel('$\cos(B)$') 
ax1.set_title("Eigen2")

ax2.plot(np.log10(Mass_halo),cos_beta[2],'.')
ax2.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax2.set_ylabel('$\cos(B)$') 
ax2.set_title("Eigen3")
fig.tight_layout()
"""
plt.figure()
plt.plot(np.log10(Mass_halo),cos_beta[0],'.')
plt.xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
plt.ylabel('$\cos(B)$') 

plt.savefig('Alinacion_Enviroment_beta_halo_EV.png')


### ALPHA ###

### halo
"""
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.plot(np.log10(Mass_halo),cos_alpha[0],'.')
#ax0.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax0.set_ylabel('$\cos(a)$') 
ax0.set_title("Eigen1")

ax1.plot(np.log10(Mass_halo),cos_alpha[1],'.')
#ax1.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax1.set_ylabel('$\cos(a)$') 
ax1.set_title("Eigen2")

ax2.plot(np.log10(Mass_halo),cos_alpha[2],'.')
ax2.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax2.set_ylabel('$\cos(a)$') 
ax2.set_title("Eigen3")

fig.tight_layout()
"""
plt.figure()
plt.plot(np.log10(Mass_halo),cos_alpha[0],'.')
plt.xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
plt.ylabel('$\cos(a)$') 
plt.savefig('Alinacion_Enviroment_alpha_halo_EV.png')


#bh
"""
fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.plot(np.log10(Mass_bh),cos_alpha[0],'.')
#ax0.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax0.set_ylabel('$\cos(a)$') 
ax0.set_title("Eigen1")

ax1.plot(np.log10(Mass_bh),cos_alpha[1],'.')
#ax1.set_xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
ax1.set_ylabel('$\cos(a)$') 
ax1.set_title("Eigen2")

ax2.plot(np.log10(Mass_bh),cos_alpha[2],'.')
ax2.set_xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
ax2.set_ylabel('$\cos(a)$') 
ax2.set_title("Eigen3")
fig.tight_layout()
"""
plt.figure()
plt.plot(np.log10(Mass_bh),cos_alpha[0],'.')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos(a)$') 
plt.savefig('Alinacion_Enviroment_alpha_bh_EV.png')




"""                                                                            
  ====> HISTOGRAMAS DE LOS ANGULOS                                             
"""

#theta
fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes.flatten()

ax0.hist(cos_theta_bh[0])  # para el Eigenvec  
ax0.set_xlabel("$\cos(\Theta)$")
ax0.set_yscale("log")
ax0.set_title("Eigen1")

ax1.hist(cos_theta_bh[1])  # para el Eigenvec 2                                
ax1.set_xlabel("$\cos(\Theta)$")
ax1.set_yscale("log")
ax1.set_title("Eigen2")

ax2.hist(cos_theta_bh[2])  # para el Eigenvec 3    
ax2.set_xlabel("$\cos(\Theta)$")
ax2.set_yscale("log")
ax2.set_title("Eigen3")
fig.tight_layout()

plt.savefig("histograma_cos_theta.png")


#beta
"""
fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes.flatten()

ax0.hist(cos_beta[0])  # para el Eigenvec  
ax0.set_xlabel("$\cos(B)$")
ax0.set_yscale("log")
ax0.set_title("Eigen1")

ax1.hist(cos_beta[1])  # para el Eigenvec 2                                
ax1.set_xlabel("$\cos(B)$")
ax1.set_yscale("log")
ax1.set_title("Eigen2")

ax2.hist(cos_beta[2])  # para el Eigenvec 3    
ax2.set_xlabel("$\cos(B)$")
ax2.set_yscale("log")
ax2.set_title("Eigen3")
fig.tight_layout()
"""

plt.figure()
plt.hist(cos_beta[0])  
plt.xlabel("$\cos(B)$")
plt.savefig("histograma_cos_beta.png")



#alpha
"""
fig, axes = plt.subplots(nrows=1, ncols=3)
ax0, ax1, ax2 = axes.flatten()

ax0.hist(cos_alpha[0])  # para el Eigenvec 1                                
ax0.set_xlabel("$\cos(a)$")
ax0.set_title("Eigen1")

ax1.hist(cos_alpha[1])  # para el Eigenvec 2                                
ax1.set_xlabel("$\cos(a)$")
ax1.set_title("Eigen2")

ax2.hist(cos_alpha[2])  # para el Eigenvec 3    
ax2.set_xlabel("$\cos(a)$")
ax2.set_title("Eigen3")
fig.tight_layout()
"""
plt.figure()
plt.hist(cos_alpha[0])  
plt.xlabel("$\cos(a)$")
plt.savefig("histograma_cos_alpha.png")


print("---------------------------------\n")
print("Termina Grafica\n")

