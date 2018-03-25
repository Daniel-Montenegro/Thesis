
# coding: utf-8

# In[1]:


#####==Librerias==#####
import numpy as np
import matplotlib.pyplot as plt


# En este programa se va ha calcular la densidad $\rho$ en función del redshift $z$. Se van a considerar las diferentes densidades.
# 
# $$
# \rho_{i}=\rho_{i}(z) \ \ \ \ \ i=\{m,r,\Lambda \}
# $$
# 

# 1) para 
# $$
# \rho_{m}=\rho_{m}(z)=\Omega_{m,0}\rho_{crit}(1+z)^{3} =\frac{3H_{o}}{8\pi G}\Omega_{m}(1+z)^{3}
# $$
# 
# donde 
# 
# $\Omega_{m}=0.32$
# 
# $H_{o}=100h\ km\ s^{-1}\ Mpc^{-1}$
# 
# $G=$

# In[2]:


z=[]
rho=[]
rho_crit=2.8e11 #MasasSolares*Mpc^-3

datos=np.loadtxt("datos.dat")
z=datos[:,] ##redshift

#print(z)
for i in range(len(z)):
    rho.append(0.32*rho_crit*(1+z[i])**3)

plt.plot(z,np.log10(rho),'.')
plt.xlabel("z")
#plt.ylabel('$ \rho $' )
plt.show()
#print(len(rho),len(z))


# 2)
# $$
# \rho_{r}=\rho_{r}(z)=\Omega_{r,0}\rho_{crit}(1+z)^{4} =\frac{3H_{o}}{8\pi G}\Omega_{m}(1+z)^{4}
# $$
# 
# donde 
# 
# $\Omega_{m}=9.4X10^{-5}$
# 

# In[3]:


z=[]
rho=[]
rho_crit=2.8e11 #MasasSolares*Mpc^-3
Omega_m=9.4e-5

datos=np.loadtxt("datos.dat")
z=datos[:,] ##redshift

#print(z)
for i in range(len(z)):
    rho.append(Omega_m*rho_crit*(1+z[i])**4)

plt.plot(z,np.log10(rho),'.')
plt.xlabel("z")
#plt.ylabel('$ \rho $' )
plt.show()
#print(len(rho),len(z))


# 1) para 
# $$
# \rho_{m}=\rho_{k}(z)=\Omega_{k,0}\rho_{crit}(1+z)^{2}= \rho_{crit}(1-\Omega_{0})(1+z)^{2} =\frac{3H_{o}}{8\pi G}(1-\Omega_{0})(1+z)^{2}
# $$
# 
# donde 
# 
# $\Omega_{k} \leq 0.01$
# 
# $H_{o}=100h\ km\ s^{-1}\ Mpc^{-1}$
# 
# $G=$

# In[4]:


z=[]
rho=[]
rho_crit=2.8e11 #MasasSolares*Mpc^-3
Omega_k=0.01

datos=np.loadtxt("datos.dat")
z=datos[:,] ##redshift

#print(z)
for i in range(len(z)):
    rho.append(Omega_k*rho_crit*(1+z[i])**4)

plt.plot(z,np.log10(rho),'.')
plt.xlabel("z")
plt.ylabel("$\rho$")
plt.legend(loc='upper left')
plt.show()
#print(len(rho),len(z))


# In[ ]:




