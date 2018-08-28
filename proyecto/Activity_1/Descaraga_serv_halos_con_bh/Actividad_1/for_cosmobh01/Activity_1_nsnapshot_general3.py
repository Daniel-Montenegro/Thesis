
"""
En este codigo es capaz de leer los tres auntovectores
o los tres autovalores de las simulaciones. El 
codigo anterior solo leia de uno.
"""


"""
==================================
 LIBRERIAS
===================================
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import gadget 
import arepo

"""
===================================
    ->Lectura de los autovectores
"""

from struct import *
import sys
import os

#===========================================


#===============================================================

"""
***********************************
    --> Lectura snapshots

***********************************
"""
print("-----------------------------------\n")
print("Inicia asignacion de los snapshots\n")


#Sn_bh = gadget.Snapshot('../../Data/snap_015.0.hdf5', parttype=[5], combineFiles=True, verbose=True)

Sn_bh = gadget.Snapshot('/home/dmontenegro/Data/Sims512/cosmobh01/snapdir_015/snap_015.0.hdf5', parttype=[5], combineFiles=True, verbose=True)

#Sn_bh = gadget.Snapshot('/home/dmontenegro/Data/Sims512/cosmobh01/snapdir_015/snap_015.0.hdf5', parttype=[5])  


#====> Asignacion de datos


Spin_bh=Sn_bh.BH_SpinOrientation #Spin de los black hole
r_bh = Sn_bh.Coordinates  # cordanada en x,y,z
Mass_bh = Sn_bh.BH_Mass*1e10/0.7
vel_disp_bh = Sn_bh.SubfindVelDisp  ##km/s
#vel_bh = Sn_bh.Velocities ##km*sqrt(a)/s
Size_data = len(Mass_bh)  ##definir el tamano de los datos


print("-----------------------------------\n")
print("Termina asignacion de los snapshots\n")



print(":::::::::::::::::::::::::::::::::::::\n")
print("Inicia graficas de control BH\n")

"""
Histograma de masa de los BH's
"""
#Bines=np.int(np.sqrt(len(Mass_bh)))
Bines=10
#hist_Mass_halo=np.histogram(np.log10(Mass_halo), bins=17)
plt.figure()
plt.hist(np.log10(Mass_bh),bins=35, log=True)
plt.xlabel("$\log_{10}(Mass_{bh})[M_{\odot}]$")
plt.savefig("histo_Mass_bh.png")


"""
GRAFICAS DE VERIFIVACION
    --> En esta parte del codigo se pretende
        verificar si los datos y los calculos 
        son congruentes.
"""

"""
plt.figure()
plt.plot(np.log10(vel_bh[:,0]),np.log10(Mass_bh),".")
plt.xlabel("$Vel_{bh}[km/s]$")
plt.ylabel("$\log_{10}(Mass_{bh})[Mass_{\odot}]$")
plt.savefig('vel_max_VS_mass_bh.png')
"""

##Calcular el ajuste para bh

### ajuste dispersion de la velocidad 
def func_disp_bh(x,A,B):
    return A*x**2+B

popt, pcov = curve_fit(func_disp_bh, np.log10(vel_disp_bh),np.log10(Mass_bh))


plt.figure()
#plt.plot(xdata, func(xdata, *popt), 'g--',
plt.plot(np.log10(vel_disp_bh),func_disp_bh(np.log10(vel_disp_bh), *popt), '.k', label='ajuste')
plt.plot(np.log10(vel_disp_bh),np.log10(Mass_bh),".", label="Data")
plt.xlabel("$\log(\sigma_{bh})[km/s]$")
plt.ylabel("$\log_{10}(Mass_{bh})[Mass_{\odot}]$")
plt.legend()
plt.savefig('vel_dis_VS_mass_bh.png')

print(":::::::::::::::::::::::::::::::::::::\n")
print("Termina graficas de control halos\n")

#============================================


"""
***********************************
    --> Lectura Subhalos

***********************************
"""

print("Inicia asignacion de los subfind\n")

sub = gadget.Subfind('/home/dmontenegro/Data/Sims512/cosmobh01/groups_015/fof_subhalo_tab_015.0.hdf5' ,combineFiles=True)


Mass_halo= sub.SubhaloMass*1e10/0.7 
vel_dis_halo = sub.SubhaloVelDisp ##km/s 
vel_max_halo = sub.SubhaloVmax  ##km/s
spin_halo=sub.SubhaloSpin ##ckpc/h
r_halo = sub.SubhaloPos ## posicion en x,y,z ckp/h


##========================================##
##Seleccion de los halos que tienen un bh ##

"""
Mass_halo_new = []
vel_dis_halo_new = [] 
vel_max_halo_new =[]
spin_halo_new=[]
r_halo_new = []
    

    
for k in range(len(r_bh)):
    R=[]
    
    x_bh =r_bh[k][0]
    y_bh =r_bh[k][1]
    z_bh =r_bh[k][2]
    
    for j in range(len(r_halo)):
        x_halo =r_halo[j][0]
        y_halo =r_halo[j][1]
        z_halo =r_halo[j][2]
        
        #R.append(np.abs(d_bh[k]-d_halo[j]))
        R.append(np.sqrt((x_bh-x_halo)**2 + (y_bh-y_halo)**2 +(z_bh-z_halo)**2))
        
    id=np.argmin(R)
    print(len(r_bh)-k)
    #print(id)
    
    Mass_halo_new.append(sub.SubhaloMass[id]) ##Msun/h
    vel_dis_halo_new.append(sub.SubhaloVelDisp[id]) ##km/s 
    vel_max_halo_new.append(sub.SubhaloVmax[id])  ##km/s 
    spin_halo_new.append(sub.SubhaloSpin[id]) ##ckpc/h
    r_halo_new.append(sub.SubhaloPos[id]) ## posicion en x,y,z ckp/h

#r_halo_new=np.reshape(r_halo_new,(len(r_bh),3))

Mass_halo_new=np.reshape(Mass_halo_new,(len(r_bh),1))
r_halo_new=np.reshape(r_halo_new,(len(r_bh),3))
vel_dis_halo_new=np.reshape(vel_dis_halo_new,(len(r_bh),1))
vel_max_halo_new=np.reshape(vel_max_halo_new,(len(r_bh),1))
spin_halo_new=np.reshape(spin_halo_new,(len(r_bh),3))


#====Grafica de control ===#
from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure()
ax = Axes3D(fig)

ax.scatter(r_bh[:,0], r_bh[:,1], r_bh[:,2], label="bh" )
ax.scatter(r_halo_new[:,0], r_halo_new[:,1], r_halo_new[:,2], label="halo")
ax.legend()
ax.view_init(elev=15., azim=180)
plt.savefig("Posicion_espacial.png")

############################################

print("-----------------------------------\n")
print("Termina asignacion de los subfind\n")
"""

Data_halo=np.loadtxt("Data_halo_new.txt")

print("tamano de Data halo:",len(Data_halo))
##almacenamiento de los halos que contienen un bh en su interior
Mass_halo_new2 = Data_halo[:,0]*1e10/0.7
vel_dis_halo_new2 = Data_halo[:,4]
vel_max_halo_new2 = Data_halo[:,5]


r_halo_new2=[]
spin_halo_new2 =[]

for i in range(len(r_bh)):
    #a =np. (Data_halo[:i,1])
    #b =np.array_str(Data_halo[:i,2])
    #c =np.array_str(Data_halo[:i,3])
    #r_halo_new2.append(np.array([a, b, c])) ##km/s 
    r_halo_new2.append([Data_halo[i,1],Data_halo[i,2],Data_halo[i,3]]) ##km/s 
    spin_halo_new2.append([Data_halo[i,6],Data_halo[i,7],Data_halo[i,8]])
    
    
r_halo_new2=np.reshape(r_halo_new2,(len(r_bh),3))
spin_halo_new2=np.reshape(spin_halo_new2,(len(r_bh),3))


print(":::::::::::::::::::::::::::::::::::::\n")
print("Inicia graficas de control halos\n")
"""
********************************
Histograma de masa de los halos
********************************
"""
#Bines=np.int(np.sqrt(len(Mass_halo)))
Bines=35
#hist_Mass_halo=np.histogram(np.log10(Mass_halo), bins=17)
plt.figure()
plt.hist(np.log10(Mass_halo_new2),bins=Bines, log=True)
#plt.xlim(9,14)
plt.xlabel("$\log_{10}(Mass_{halo})[M_{\odot}]$")
plt.savefig("histo_Mass_halo.png")



"""
GRAFICAS DE VERIFIVACION
    --> En esta parte del codigo se pretende
        verificar si los datos y los calculos 
        son congruentes.
"""

###==================================###                                                                                                              
###  Ubicacion espacial halos y bh   ###                                                                                                              
"""
from mpl_toolkits.mplot3d import Axes3D

#fig = pyplot.figure()
plt.figure()

Axes3D.scatter(r_bh[:,0], r_bh[:,1], r_bh[:,2],  label="bh")
Axe3D.scatter(r_halo_new2[:,0], r_halo_new2[:,1], r_halo_new2[:,2], label="halo")
Axes3D.legend()
Axes3D.view_init(elev=15., azim=180)
plt.savefig("ubicion_esapcial_halo_y_bh.png")
"""

####*******************###
#==== AJUSTES HALOS =====#

### ajuste dispersion de la velocidad 
def f_disp_halo(x,A,B):
    return A*x**2+B

popt_disp, pcov_disp = curve_fit(f_disp_halo, np.log10(vel_dis_halo_new2),np.log10(Mass_halo_new2))


### ajuste velocidad max
def f_vel_max_halo(x,A,B):
    return A*x**2+B

popt_velmax, pcov_velmax = curve_fit(f_vel_max_halo, np.log10(vel_max_halo_new2),np.log10(Mass_halo_new2))





plt.figure()
plt.plot(np.log10(vel_max_halo_new2),np.log10(Mass_halo_new2),".", label='halo')
#plt.plot(xdata, func(xdata, *popt), 'g--',
plt.plot(np.log10(vel_max_halo_new2), f_vel_max_halo( np.log10(vel_max_halo_new2), *popt_velmax ), '.k', markersize=4, label="ajuste") #GRAFICA AJUSTE
plt.xlabel("$Vel_{max,halo}[km/s]$")
plt.ylabel("$\log_{10}(Mass_{halo})[Mass_{\odot}]$")
plt.legend()
plt.savefig('vel_max_VS_mass_Halo.png')

plt.figure()
plt.plot(np.log10(vel_dis_halo_new2),np.log10(Mass_halo_new2),".", label='halo')
#plt.plot(xdata, func(xdata, *popt), 'g--',
plt.plot(np.log10( vel_dis_halo_new2), f_disp_halo( np.log10(vel_dis_halo_new2), *popt_disp), '.k', label='ajuste')
plt.xlabel("$\log(\sigma_{halo})[km/s]$")
plt.ylabel("$\log_{10}(Mass_{halo})[Mass_{\odot}]$")
plt.legend()
plt.savefig('vel_dis_VS_mass_Halo.png')


print(":::::::::::::::::::::::::::::::::::::\n")
print("Termina graficas de control halos\n")



#===========================================

"""
***********************************
    --> FUNCIONES 

***********************************
"""

#Cargar los datos para 

"""
======================
Lectura de Datos 

    ---> EigenVectores
======================
"""

## Dirección archivos ##
#Direction_eigenvector='../Bolshoi/Eigenvec_s1_1'
#Direction_eigenvalor='../Bolshoi/Eigen_s1_1'



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


#===========================================

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
    """
    eigen_vec_r =np.array([new_array_data[0,i,j,k],\
                            new_array_data[1,i,j,k],\
                            new_array_data[2,i,j,k]])
    """
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

#cos_theta_halo=[]


#folder = '/home/daniel/Documentos/Tesis/Data/Tweb_512/'
#file = 'snap_015.s1.00.eigenvec_'

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
        Mag_Spin_bh.append(np.linalg.norm(Spin_bh[i]))       ##magnitud del Spin_bh
        #cos_theta
        Dot_bh.append(np.vdot(EigenVec_bh[counter_bh],Spin_bh[i]))      ##Productopunto del autovec y spin_bh
        cos_theta_bh.append(Dot_bh[counter_bh]/(Mag_EigenVec_bh[counter_bh]*Mag_Spin_bh[counter_bh])) 
        #cos_beta
        Dot_Sbh_Shalo.append(np.vdot(Spin_bh[i],spin_halo[i])) ##producto punto entre spin_halo y spin_bh
        Mag_Spin_halo.append(np.linalg.norm(spin_halo[i]))
        cos_beta.append(Dot_Sbh_Shalo[counter_bh]/(Mag_Spin_halo[counter_bh]*Mag_Spin_bh[counter_bh])) 

        counter_bh=counter_bh+1
    Q=Q+1

"""    
    for i in range(len(r_halo)):

        
        EigenVec_halo.append(Eigen_vec(r_halo[i],n_x))
        Mag_EigenVec_halo.append(np.linalg.norm(EigenVec_halo[counter_halo])) ##magnitud del autovector
        Mag_Spin_halo.append(np.linalg.norm(spin_halo[i]))       ##magnitud del Spin_bh
        Dot_halo.append(np.vdot(EigenVec_halo[counter_halo],spin_halo[i]))      ##Productopunto del autovec y spin_bh
        cos_theta_halo.append(Dot_halo[counter_halo]/(Mag_EigenVec_halo[counter_halo]*Mag_Spin_halo[counter_halo])) 
        counter_halo=counter_halo+1
    
"""    
    
#print(EigenVec)
#cos_theta_halo = np.reshape(cos_theta_halo,(3,len(r_halo)))    
cos_theta_bh = np.reshape(cos_theta_bh,(3,len(r_bh))) #para cada EigenVec
cos_beta = np.reshape(cos_beta,(3,len(r_bh)))   #para cada EigenVec

##theta: angulo entre EigenVec y spin_bh
##beta: angulo entre spin_bh y spin_halo

#===========================================

"""
******************************************
Graficas
******************************************
"""
print("---------------------------------\n")
print("Inicia Grafica\n")

#================#
####   THETA   ####

## BH  ##
plt.figure()
plt.plot(np.log10(Mass_bh),cos_theta_bh[0],'.', label='Eigen_1')
plt.plot(np.log10(Mass_bh),cos_theta_bh[1],'.', label='Eigen_2')
plt.plot(np.log10(Mass_bh),cos_theta_bh[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos( \Theta ) $')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_theta_bh_3_EV.png')

print("tamano mass_halo=",(len(Mass_halo_new2)))
print("tamano masa_bh=",(len(Mass_bh)))

## HALO ##
plt.figure()
plt.plot(np.log10(Mass_halo_new2),cos_theta_bh[0],'.', label='Eigen_1')
plt.plot(np.log10(Mass_halo_new2),cos_theta_bh[1],'.', label='Eigen_2')
plt.plot(np.log10(Mass_halo_new2),cos_theta_bh[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
plt.ylabel('$\cos( \Theta ) $')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_theta_halo_3_EV.png')

#================#
####   BETA   ####

## bh ##
plt.figure()
plt.plot(np.log10(Mass_bh),cos_beta[0],'.')
#plt.plot(np.log10(Mass_bh),cos_beta[1],'.', label='Eigen_2')
#plt.plot(np.log10(Mass_bh),cos_beta[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos(\gamma)$')
#plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_beta_bh_3_EV.png')
## halo ##

plt.figure()
plt.plot(np.log10(Mass_halo_new2),cos_beta[0],'.')
#plt.plot(np.log10(Mass_halo),cos_beta[1],'.', label='Eigen_2')
#plt.plot(np.log10(Mass_halo),cos_beta[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos(\gamma)$')
#plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_beta_halo_3_EV.png')

plt.figure()
plt.loglog(Mass_bh,Mass_halo_new2,'.')
plt.xlabel("$Mass_{bh}$")
plt.ylabel("$Mass_{halo}$")
plt.savefig("Mass_bh_Vs_Mass_bh.png")



"""
  ====> HISTOGRAMAS DE LOS ANGULOS
"""
plt.figure()
plt.hist(cos_theta_bh[0])  # para el Eigenvec 1
plt.xlabel("$\cos(\Theta)Evec1$")
plt.savefig("histo_cos_thetaEvec1.png")

plt.figure()
plt.hist(cos_theta_bh[2])  # para el Eigenvec 1
plt.xlabel("$\cos(\Theta)Evec2$")
plt.savefig("histo_cos_thetaEvec2.png")


plt.figure()
plt.hist(cos_beta[0]) # para el Eigenvec 1
plt.xlabel("$\cos(\gamma)Evec1$")
plt.savefig("histo_beta_Evec1.png")




print("---------------------------------\n")
print("Termina Grafica\n")


