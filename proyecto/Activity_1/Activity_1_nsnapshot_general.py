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


"""
***********************************
    --> Lectura Subhalos

***********************************
"""
print("-----------------------------------\n")
print("Inicia asignacion de los subfind\n")

sub = gadget.Subfind('../../Data/groups_015/fof_subhalo_tab_015.0.hdf5' ,combineFiles=True)


Mass_halo= sub.SubhaloMass*10e10/0.7 
vel_dis_halo = sub.SubhaloVelDisp ##km/s 
vel_max_halo = sub.SubhaloVmax  ##km/s
spin_halo=sub.SubhaloSpin ##ckpc/h
r_halo = sub.SubhaloPos ## posicion en x,y,z ckp/h


print("-----------------------------------\n")
print("Termina asignacion de los subfind\n")


print(":::::::::::::::::::::::::::::::::::::\n")
print("Inicia graficas de control halos\n")
"""
********************************
Histograma de masa de los halos
********************************
"""
Bines=np.int(np.sqrt(len(Mass_halo)))
#hist_Mass_halo=np.histogram(np.log10(Mass_halo), bins=17)
plt.figure()
plt.hist(np.log10(Mass_halo),bins=Bines)
plt.xlim(9,14)
plt.xlabel("$\log_{10}(Mass_{halo})[M_{\odot}]$")
plt.savefig("histo_Mass_halo.png")



"""
GRAFICAS DE VERIFIVACION
    --> En esta parte del codigo se pretende
        verificar si los datos y los calculos 
        son congruentes.
"""

plt.figure()
plt.plot(np.log10(vel_max_halo),np.log10(Mass_halo),".")
plt.xlabel("$Vel_{max,halo}[km/s]$")
plt.ylabel("$\log_{10}(Mass_{halo})[Mass_{\odot}]$")
plt.savefig('vel_max_VS_mass_Halo.png')

plt.figure()
plt.plot(np.log10(vel_dis_halo),np.log10(Mass_halo),".")
plt.xlabel("$\log(\sigma_{halo})[km/s]$")
plt.ylabel("$\log_{10}(Mass_{halo})[Mass_{\odot}]$")
plt.savefig('vel_dis_VS_mass_Halo.png')


print(":::::::::::::::::::::::::::::::::::::\n")
print("Termina graficas de control halos\n")

#===============================================================


"""
***********************************
    --> Lectura snapshots

***********************************
"""
print("-----------------------------------\n")
print("Inicia asignacion de los snapshots\n")


#Sn_bh = gadget.Snapshot('../../Data/snap_015.0.hdf5', parttype=[5], combineFiles=True, verbose=True)

Sn_bh = gadget.Snapshot('../../Data/snap_015.0.hdf5', parttype=[5])



#====> Asignacion de datos


Spin_bh=Sn_bh.BH_SpinOrientation #Spin de los black hole
r_bh = Sn_bh.Coordinates  # cordanada en x,y,z
Mass_bh = Sn_bh.BH_Mass*10e10/0.7
vel_disp_bh = Sn_bh.SubfindVelDisp  ##km/s


print("-----------------------------------\n")
print("Termina asignacion de los snapshots\n")



print(":::::::::::::::::::::::::::::::::::::\n")
print("Inicia graficas de control BH\n")

"""
Histograma de masa de los BH's
"""
Bines=np.int(np.sqrt(len(Mass_bh)))
#hist_Mass_halo=np.histogram(np.log10(Mass_halo), bins=17)
plt.figure()
plt.hist(np.log10(Mass_bh),bins=Bines)
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

plt.figure()
plt.plot(np.log10(vel_disp_bh),np.log10(Mass_bh),".")
plt.xlabel("$\log(\sigma_{halo})[km/s]$")
plt.ylabel("$\log_{10}(Mass_{bh})[Mass_{\odot}]$")
plt.savefig('vel_dis_VS_mass_bh.png')

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
cos_theta_halo=[]


folder = '/home/daniel/Documentos/Tesis/Data/Tweb_512/'
file = 'snap_015.s1.00.eigenvec_'

plt.figure()


counter_halo = 0
counter_bh = 0
Q=1

"""
theta = angulo entre EigenVec(Tweb), spin_bh 
beta = angulo entre spin_bh, spin_halo
lambda = angulo entre spin_bh, spin_disco
"""

for j in range(0,3):
    
    
    new_array_data, n_x = read_eigenVec(folder,file,"%s"%(Q))
    print("counter = ",counter_halo)
    print("%s%s%s"%(folder,file, Q))
    
    for i in range(len(r_bh)):

        
        EigenVec_bh.append(Eigen_vec(r_bh[i],n_x))
        Mag_EigenVec_bh.append(np.linalg.norm(EigenVec_bh[counter_bh])) ##magnitud del autovector
        Mag_Spin_bh.append(np.linalg.norm(Spin_bh[i]))       ##magnitud del Spin_bh
        Dot_bh.append(np.vdot(EigenVec_bh[counter_bh],Spin_bh[i]))      ##Productopunto del autovec y spin_bh
        cos_theta_bh.append(Dot_bh[counter_bh]/(Mag_EigenVec_bh[counter_bh]*Mag_Spin_bh[counter_bh])) 
        
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
    Q=Q+1
    """
    
#print(EigenVec)
#cos_theta_halo = np.reshape(cos_theta_halo,(3,len(r_halo)))    
cos_theta_bh = np.reshape(cos_theta_bh,(3,len(r_bh)))    




#===========================================

"""
******************************************
Graficas
******************************************
"""
print("---------------------------------\n")
print("Inicia Grafica\n")


plt.figure()
plt.plot(np.log10(Mass_bh),cos_theta_bh[0],'.', label='Eigen_1')
plt.plot(np.log10(Mass_bh),cos_theta_bh[1],'.', label='Eigen_2')
plt.plot(np.log10(Mass_bh),cos_theta_bh[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos( \Theta ) $')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_bh_3_EV.png')


"""
plt.figure()
plt.plot(np.log10(Mass_halo),cos_theta_bh[0],'.', label='Eigen_1')
plt.plot(np.log10(Mass_halo),cos_theta_bh[1],'.', label='Eigen_2')
plt.plot(np.log10(Mass_halo),cos_theta_bh[2],'.', label='Eigen_3')
plt.xlabel('$\log_{10}(M_{halo})[M_{\odot}]$')
plt.ylabel('$\cos( \Theta ) $')
plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.savefig('Alinacion_Enviroment_halo_3_EV.png')
"""

print("---------------------------------\n")
print("Termina Grafica\n")
