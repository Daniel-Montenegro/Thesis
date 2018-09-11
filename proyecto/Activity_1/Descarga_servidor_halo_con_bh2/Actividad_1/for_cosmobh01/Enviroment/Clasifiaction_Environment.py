"""
En este codigo se realiza la clasificación del entorno. Ademas se realiza
la asigancion de las variables (Masa, spin, velocidad, id) para cada particula.

PASOS:

Se cragan los datos para los bh y para los halos que contienen un bh.

Se hacen las funciones que permiten la lectura de los Auto valores y vectores

Se carga los autovalores, con ello se asignan los tres autovalores, que 
ayudaran a la clasificacion. 

Se cargan los AutoVectores, ademas se hace la comparacion entre autovalores que
daran como resultado los tipos de entornos.En espacial se almacena el id de 
cada particula a su entorno respectivo.

Con el id de cada particula para su entorno especifico, se almacena las 
propiedades de cada partucula  (Masa, spin, velocidad, id).

Se realiza el producto punto entre los angulos respectivos, cos theta,
 cos gamma

por ultimo se realizan las graficas.

"""
 
#librerías
import numpy as np
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

#=========================================
#==========================================


"""
BLACK HOLES
"""
print("-----------------------------------\n")
print("Inicia asignacion de los bh\n")

Sn_bh = gadget.Snapshot('/home/dmontenegro/Data/Sims512/cosmobh01/snapdir_015/snap_015.0.hdf5', parttype=[5], combineFiles=True, verbose=True)

Spin_bh=Sn_bh.BH_SpinOrientation #Spin de los black hole
r_bh = Sn_bh.Coordinates  # cordanada en x,y,z kpc
Mass_bh = Sn_bh.BH_Mass*1e10/0.7 ## M(sun)
vel_disp_bh = Sn_bh.SubfindVelDisp  ##km/s

print("-----------------------------------\n")
print("Termina asignacion de los bh\n")


#========================================


print("-----------------------------------\n")
print("Inicia asignacion de los subfind\n")

sub = gadget.Subfind('/home/dmontenegro/Data/Sims512/cosmobh01/groups_015/fof_subhalo_tab_015.0.hdf5' ,combineFiles=True)


Mass_halo= sub.SubhaloMass*1e10/0.7
vel_halo = sub.SubhaloVel ##km/s        
vel_dis_halo = sub.SubhaloVelDisp ##km/s   
vel_max_halo = sub.SubhaloVmax  ##km/s                                         
spin_halo=sub.SubhaloSpin ##ckpc/h                                             
r_halo = sub.SubhaloPos ## posicion en x,y,z ckp/h                            

print("-----------------------------------\n")
print("Termina asignacion de los subfind\n")


#==============================================

"""
Datos de los halos que contienen un bh en su interio
"""
print("-----------------------------------\n")
print("Termina asignacion de los halos con bh\n")


Data_halo=np.loadtxt("Data(halo-bh-stelar)_new.txt")

print("tamano de Data halo:",len(Data_halo))
##almacenamiento de los halos que contienen un bh en su interior
Mass_halo_new2 = Data_halo[:,0]*1e10/0.7
vel_dis_halo_new2 = Data_halo[:,4]
vel_max_halo_new2 = Data_halo[:,5]


r_halo_new2=[]
spin_halo_new2 =[]

for i in range(len(r_bh)):
    r_halo_new2.append([Data_halo[i,1],Data_halo[i,2],Data_halo[i,3]]) ##km/s 
    spin_halo_new2.append([Data_halo[i,6],Data_halo[i,7],Data_halo[i,8]])
    
    
r_halo_new2=np.reshape(r_halo_new2,(len(r_bh),3))
spin_halo_new2=np.reshape(spin_halo_new2,(len(r_bh),3))

print("-----------------------------------\n")
print("Termina asignacion de los halos con bh\n")


##********************************************
##********************************************

"""
    FUNCIONES
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

#print(new_array_data[:,0,0,0])


"""
======================
Lectura de Datos 

    ---> EigenValores
======================
"""

## Dirección archivos ##
#Direction_eigenvector='../Bolshoi/Eigenvec_s1_1'
#Direction_eigenvalor='../Bolshoi/Eigen_s1_1'



def read_eigenVal(folder,file,NumEigenVec):
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

    total_nodes =  n_x * n_y *n_z
    dumb = f.read(4)
    array_data = f.read(total_nodes*4)
    dumb = f.read(4)
    format_s = str(total_nodes)+'f'
    array_data = unpack(format_s, array_data)
    f.close()
    array_data  = np.array(array_data)
    new_array_data = np.reshape(array_data, (n_x,n_y,n_z), order='F')
    return new_array_data, n_x
    
print("--------------------\n")

#==================================================

###calcular los autovelores para los radios respectivos

#r=[]
#eigen_vec_r=[]
def Eigen_vec(r,n_x):
    """
    Esta funcion retorna el valor del auntoVector 
    correspondiente a las coordenadas r(x,y,z)
    """
        
    long_box= 25e3 #longitud caja
    
    #dl = n_x/long_box #tamaño de cada celda = numero_celdas/long_caja
    dl=long_box/n_x_vec
    
    i=np.int(r[0]/dl)
    j=np.int(r[1]/dl)
    k=np.int(r[2]/dl)
    
    """
    eigen_vec_r =np.array([new_array_data[0,i,j,k],\
                            new_array_data[1,i,j,k],\
                            new_array_data[2,i,j,k]])
    """

    eigen_vec_r = [new_array_data_vec[0,i,j,k],\
                            new_array_data_vec[1,i,j,k],\
                            new_array_data_vec[2,i,j,k]]

    return eigen_vec_r

###=======================================================##


def Eigen_val(r,n_x_val):
    """
    Esta funcion retorna el valor del auntoValor 
    correspondiente a las coordenadas r(x,y,z)
    """
        
    long_box= 25e3 #longitud caja
    
    #dl = n_x_val/long_box #tamaño de cada celda = numero_celdas/long_caja
    dl = long_box/n_x_val
    
    i=np.int(r[0]/dl)
    j=np.int(r[1]/dl)
    k=np.int(r[2]/dl)
    
    """
    eigen_vec_r =np.array([new_array_data[0,i,j,k],\
                            new_array_data[1,i,j,k],\
                            new_array_data[2,i,j,k]])
    """
    #print(new_array_data_val[i,j,k])
    #print(i,j,k)
    return new_array_data_val[i,j,k]
       
    

##==========================================##


## asignacion de autovectores con su radio respectivo

"""
Mag_EigenVec = [[Mag_EigenVec1],[Mag_EigenVec2],[Mag_EigenVec3]]
Mag_Spin_bh = [[Mag_Spin_bh1],[Mag_Spin_bh2],[Mag_Spin_bh]]
Dot= [[Dot1],[Dot2],[Dot3]]
EigenVec= [[EigenVec1],[EigenVec2],[EigenVec3]]
cos_theta = []
"""
"""
Mag_EigenVec = []
Mag_Spin_bh = []
Dot= []
EigenVec= []
cos_theta = []
"""
EigenVal= []



##for Voids##
Mag_EigenVec_void = []
Mag_Spin_bh_void = []
Dot_void= []
EigenVec_void= []
cos_theta_void = []
EigenVal_void= []
Mass_bh_void=[]
Mass_halo_void=[]

##for sheet ##
Mag_EigenVec_sheet = []
Mag_Spin_bh_sheet = []
Dot_sheet= []
EigenVec_sheet= []
cos_theta_sheet = []
EigenVal_sheet= []
Mass_bh_sheet=[]
Mass_halo_sheet=[]

##for filament
Mag_EigenVec_filament = []
Mag_Spin_bh_filament = []
Dot_filament= []
EigenVec_filament= []
cos_theta_filament = []
EigenVal_filament= []
Mass_bh_filament=[]
Mass_halo_filament=[]

##for knot
Mag_EigenVec_knot = []
Mag_Spin_bh_knot = []
Dot_knot= []
EigenVec_knot= []
cos_theta_knot = []
EigenVal_knot= []
Mass_bh_knot=[]
Mass_halo_knot=[]



#NumEigenVec = "1" 

folder = '/home/dmontenegro/Data/Sims512/Tweb_512/'
file_vec = 'snap_015.s1.00.eigenvec_'
file_val = 'snap_015.eigen_'


counter = 0
Q=1
##==================================
###Para asignar los 3 autovalores

for j in range(0,3):
    
    
#    new_array_data_vec, n_x_vec = read_eigenVec(folder,file_vec,"%s"%(Q))
    new_array_data_val, n_x_val = read_eigenVal(folder,file_val,"%s"%(Q))

    print("counter = ",counter)
#    print("%s%s%s"%(folder,file_vec, counter))
    print("%s%s%s"%(folder,file_val, Q))

    for i in range(len(r_bh)):
        
        EigenVal.append(Eigen_val(r_bh[i],n_x_val))
                   
        counter=counter+1
    Q=Q+1
    
    
Eigen_val_new = np.reshape(EigenVal,(3,len(r_bh))) #Autovalores de los halos con bh
    
#Asignar los autovalores
Eigen_1=Eigen_val_new[0]
Eigen_2=Eigen_val_new[1]
Eigen_3=Eigen_val_new[2]

print("size Eigen1",(len(Eigen_1)))

#==========================================

void = []
filament = []
knot = []
sheet = []


counter = 0
Q=1
Eigen_th = 0.265
##=============================================


"""
for j in range(0,3):
    
    
    new_array_data_vec, n_x_vec = read_eigenVec(folder,file_vec,"%s"%(Q))
#    new_array_data_val, n_x_val = read_eigenVal(folder,file_val,"%s"%(Q))
    #print("counter = ",counter)
    print("%s%s%s"%(folder,file_vec, Q))
#   print("%s%s%s"%(folder,file_val, counter))
    
    
    for i in range(len(r_bh)):
        #if Eigen_th > 100 :
        
        if Eigen_3[i] <= Eigen_2[i] and Eigen_2[i] <= Eigen_1[i] and Eigen_1[i] <= Eigen_th:
        ##**= Voids ==**##
            void.append(i)
            
        if Eigen_3[i] <= Eigen_2[i] and Eigen_2[i] <= Eigen_th and  Eigen_th <= Eigen_1[i] :
        ##**= sheet ==**##
            sheet.append(i)
        
        if Eigen_3[i] <= Eigen_th and Eigen_th <= Eigen_2[i] and Eigen_2[i] <= Eigen_1[i] :
        ##**= filement ==**##
            filament.append(i)
        
        
        if Eigen_th <= Eigen_3[i] and Eigen_3[i] <= Eigen_2[i] and Eigen_2[i] <= Eigen_1[i] :
        ##**= knot ==**##
            knot.append(i)
        
    Q=Q+1
"""

for j in range(0,3):
    
    
    new_array_data_vec, n_x_vec = read_eigenVec(folder,file_vec,"%s"%(Q))
#    new_array_data_val, n_x_val = read_eigenVal(folder,file_val,"%s"%(Q))
    #print("counter = ",counter)
    print("%s%s%s"%(folder,file_vec, Q))
#   print("%s%s%s"%(folder,file_val, counter))

    Q=Q+1

for i in range(len(r_bh)):
    
    #if Eigen_th > 100 :
    
    if Eigen_3[i] <= Eigen_2[i] and Eigen_2[i] <= Eigen_1[i] and Eigen_1[i] <= Eigen_th:
        ##**= Voids ==**##
        void.append(i)
        
    if Eigen_3[i] <= Eigen_2[i] and Eigen_2[i] <= Eigen_th and  Eigen_th <= Eigen_1[i] :
        ##**= sheet ==**##
        sheet.append(i)
            
    if Eigen_3[i] <= Eigen_th and Eigen_th <= Eigen_2[i] and Eigen_2[i] <= Eigen_1[i] :
        ##**= filement ==**##
        filament.append(i)
        
        
    if Eigen_th <= Eigen_3[i] and Eigen_3[i] <= Eigen_2[i] and Eigen_2[i] <= Eigen_1[i] :
        ##**= knot ==**##
        knot.append(i)



#=================


    ## void ##
for k in range(len(void)):
    
    id_void = void[k] #id del halo que pertenece al entorno void
    
    EigenVec_void.append(Eigen_vec(r_bh[id_void],n_x_vec))
    Mag_EigenVec_void.append(np.linalg.norm(EigenVec_void[k])) ##magnitud del autovector
    Mag_Spin_bh_void.append(np.linalg.norm(Spin_bh[id_void]))    ##magnitud del Spin_bh
    Dot_void.append(np.vdot(EigenVec_void[k],Spin_bh[id_void]))      ##Productopunto del autovec y spin_bh
    cos_theta_void.append(Dot_void[k]/(Mag_EigenVec_void[k]*Mag_Spin_bh_void[k])) 
    Mass_bh_void.append(Mass_bh[id_void])
    Mass_halo_void.append(Mass_halo_new2[id_void])
    
 ## sheet ##
for k in range(len(sheet)):
    id_sheet = sheet[k] #id del halo que pertenece al entorno void
    EigenVec_sheet.append(Eigen_vec(r_bh[id_sheet],n_x_vec))
    Mag_EigenVec_sheet.append(np.linalg.norm(EigenVec_sheet[k])) ##magnitud del autovector
    Mag_Spin_bh_sheet.append(np.linalg.norm(Spin_bh[id_sheet]))    ##magnitud del Spin_bh
    Dot_sheet.append(np.vdot(EigenVec_sheet[k],Spin_bh[id_sheet]))      ##Productopunto del autovec y spin_bh
    cos_theta_sheet.append(Dot_sheet[k]/(Mag_EigenVec_sheet[k]*Mag_Spin_bh_sheet[k])) 
    Mass_bh_sheet.append(Mass_bh[id_sheet])
    Mass_halo_sheet.append(Mass_halo_new2[id_sheet])

        
## filament ##
for k in range(len(filament)):
    id_filament = filament[k] #id del halo que pertenece al entorno void
    EigenVec_filament.append(Eigen_vec(r_bh[id_filament],n_x_vec))
    Mag_EigenVec_filament.append(np.linalg.norm(EigenVec_filament[k])) ##magnitud del autovector
    Mag_Spin_bh_filament.append(np.linalg.norm(Spin_bh[id_filament]))    ##magnitud del Spin_bh
    Dot_filament.append(np.vdot(EigenVec_filament[k],Spin_bh[id_filament]))      ##Productopunto del autovec y spin_bh
    cos_theta_filament.append(Dot_filament[k]/(Mag_EigenVec_filament[k]*Mag_Spin_bh_filament[k])) 
    Mass_bh_filament.append(Mass_bh[id_filament])
    Mass_halo_filament.append(Mass_halo_new2[id_filament])

## knot ##
for k in range(len(knot)):
    id_knot = knot[k] #id del halo que pertenece al entorno void
    EigenVec_knot.append(Eigen_vec(r_bh[id_knot],n_x_vec))
    Mag_EigenVec_knot.append(np.linalg.norm(EigenVec_knot[k])) ##magnitud del autovector
    Mag_Spin_bh_knot.append(np.linalg.norm(Spin_bh[id_knot]))    ##magnitud del Spin_bh
    Dot_knot.append(np.vdot(EigenVec_knot[k],Spin_bh[id_knot]))      ##Productopunto del autovec y spin_bh
    cos_theta_knot.append(Dot_knot[k]/(Mag_EigenVec_knot[k]*Mag_Spin_bh_knot[k])) 
    Mass_bh_knot.append(Mass_bh[id_knot])
    Mass_halo_knot.append(Mass_halo_new2[id_knot])
    




#====================================================
#==================GRAFICAS==========================

##GRAFICAS DE LOS ENTORNOS.

#Void
plt.figure()
plt.plot(Mass_bh_void, cos_theta_void,'.', label='void')
plt.xscale("log")
plt.xlabel("$Mass_{halo}$")
plt.ylabel("$cos(\theta)$")
plt.legend()
plt.savefig("Alineation_Enviroment_theta_Void.png")

#sheet

plt.figure()
plt.plot(Mass_bh_sheet, cos_theta_sheet,'.', label='sheet')
plt.xscale("log")
plt.xlabel("$Mass_{halo}$")
plt.ylabel("$cos(\theta)$")
plt.legend()
plt.savefig("Alineation_Enviroment_theta_sheet.png")

#filament
plt.figure()
plt.plot(Mass_bh_filament, cos_theta_filament,'.', label='filament')
plt.xscale("log")
plt.xlabel("$Mass_{halo}$")
plt.ylabel("$cos(\theta)$")
plt.legend()
plt.savefig("Alineation_Enviroment_theta_filament.png")

#knot
plt.figure()
plt.plot(Mass_bh_knot, cos_theta_knot,'.', label='knot')
plt.xscale("log")
plt.xlabel("$Mass_{halo}$")
plt.ylabel("$cos(\theta)$")
plt.legend()
plt.savefig("Alineation_Enviroment_theta_knot.png")




### GRAFICA DE CALOR, EIGENVALOR ###
"""
plt.plot(np.log10(Mass_bh),cos_theta,'.')
plt.xlabel('$\log_{10}(M_{bh})[M_{\odot}]$')
plt.ylabel('$\cos( \Theta ) $')
plt.savefig('Alinacion_Enviroment_bh.png')
"""

"""
=================================
Asignacion datos y generar Grid 
=================================
"""


X=[]
Y=[]
Z=[]


# Z PROYECTADO SOBRE PLANO X,Y

s = 25   #--> slice en z que pretende tomar

#size_eigenVal = len(new_array_data_val) # Tamano de los datos para autovalores
size_eigenVal = len(Eigen_1) # Tamano de los datos para autovalores
"""
for i in range(n_x_val):
    for j in range(n_x_val):
        X.append(Eigen_1[j])
        Y.append(Eigen_1[i])
        #Z.append(new_array_data[i,j,s])
    
print(len(X),len(Y))

# COMBIERTE ESTRUCTURA A UNA MATRIZ
X=np.reshape(Eigen_1,(np.sqrt(size_eigenVal),np.sqrt(size_eigenVal)))
Y=np.reshape(Eigen_1,(np.sqrt(size_eigenVal),np.sqrt(size_eigenVal)))
#Z=np.reshape(Z,(256,256))


#Y=np.reshape(Y,(size_eigenVal,size_eigenVal))
#Z=np.reshape(Z,(256,256))

# GENERA GRID
posx=np.arange(0,size_eigenVal,1)
posy=np.arange(0,size_eigenVal,1)
#posz=np.arange(0,256,1)

Px,Py=np.meshgrid(posx,posy)

print("estructure ready")
"""
"""
=================================
Graficas
=================================
"""

"""
plt.figure(figsize=[8,8])
#plt.plot(Px,Py)
plt.imshow(X,cmap=plt.cm.BuPu_r)
#plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
#cax = plt.axes([0.85, 0.1, 0.075, 0.8])
#plt.colorbar(cax=cax)
plt.colorbar()

plt.title("Eigenvalor X,Y")
plt.xlabel("position X")
plt.ylabel("position Y")
plt.savefig('Eigen_valores.png')
"""




##********************************************
##********************************************
