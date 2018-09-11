#librerías
import numpy as np
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




##================================##

"""
********************
    BLACK HOLES
*******************
"""
print("-----------------------------------\n")
print("Inicia lectura bh\n")


Sn_bh = gadget.Snapshot('/home/dmontenegro/Data/Sims512/cosmobh01/snapdir_015/snap_015.0.hdf5', parttype=[5],  combineFiles=True, verbose=True)

#Sn_bh = gadget.Snapshot('/home/dmontenegro/Data/Sims512/cosmobh01/snapdir_015/snap_015.0.hdf5', parttype=[5])


Spin_bh=Sn_bh.BH_SpinOrientation #Spin de los black hole
r_bh = Sn_bh.Coordinates  # cordanada en x,y,z
Mass_bh = Sn_bh.BH_Mass*1e10/0.7 ## M(sun)
vel_disp_bh = Sn_bh.SubfindVelDisp  ##km/s
#vel_bh = Sn_bh.Velocities ##km*sqrt(a)/s
#Sn_bh.Coordinates

print("-----------------------------------\n")
print("Termina lectura bh\n")


###============================##

"""
**************************
       HALOS
**************************
"""

print("-----------------------------------\n")
print("Inicia lectura halos\n")


#DataFolder= '../../Data/groups_015/'
#SnapNumber= 1

#sub = arepo.Subfind('%s/'%(DataFolder), SnapNumber ,combineFiles=True)
sub = gadget.Subfind('/home/dmontenegro/Data/Sims512/cosmobh01/groups_015/fof_subhalo_tab_015.0.hdf5' ,combineFiles=True)


size_data = len(Mass_bh)  ##tamano de los datos

Mass_halo= sub.SubhaloMass*1e10/0.7 
vel_dis_halo = sub.SubhaloVelDisp ##km/s 
vel_max_halo = sub.SubhaloVmax  ##km/s 
spin_halo=sub.SubhaloSpin ##ckpc/h
r_halo = sub.SubhaloPos ## posicion en x,y,z ckp/h

print("-----------------------------------\n")
print("Termina lectura halos\n")


###==============================================###

print("-----------------------------------\n")
print("Inicia busqueda de halos con bh\n")

Mass_halo_new = []
vel_dis_halo_new = [] ##km/s 
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
    #print(id)
    print("conteo regresivo: ",len(r_bh)-k)  
    
    Mass_halo_new.append(sub.SubhaloMass[id])
    vel_dis_halo_new.append(sub.SubhaloVelDisp[id]) ##km/s 
    vel_max_halo_new.append(sub.SubhaloVmax[id])  ##km/s 
    spin_halo_new.append(sub.SubhaloSpin[id]) ##ckpc/h
    r_halo_new.append(sub.SubhaloPos[id]) ## posicion en x,y,z ckp/h

Mass_halo_new=np.reshape(Mass_halo_new,(len(r_bh),1))
r_halo_new=np.reshape(r_halo_new,(len(r_bh),3))
vel_dis_halo_new=np.reshape(vel_dis_halo_new,(len(r_bh),1))
vel_max_halo_new=np.reshape(vel_max_halo_new,(len(r_bh),1))
spin_halo_new=np.reshape(spin_halo_new,(len(r_bh),3))

print("-----------------------------------\n")
print("Termina busqueda de halos con bh\n")

print("------------")
print(len(R))
print(len(Mass_halo))
print("------------")
#print(r_bh)
#print(r_halo_new)


"""
    Asignar en memoria el dato de los halo 
"""

print("-----------------------------------\n")
print("Inicia Almacenamiento de los Data halos\n")

np.savetxt('Data(halo-bh)_new.txt',
                   np.column_stack((Mass_halo_new, r_halo_new , vel_dis_halo_new, vel_max_halo_new,spin_halo_new,  Mass_bh, r_bh, vel_disp_bh, Spin_bh)), header= '#',delimiter=" ", comments="# Mass - r_halo(x,y,x) - vel_dis_halo - vel_max - spin_halo(i,j,k) - Mass_bh - r_bh(x,y,x) - vel_disp_bh -  Spin_bh(i,j,k)")
    

print("-----------------------------------\n")
print("Termina Almacenamiento de los Data halos\n")




