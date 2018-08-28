import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

x = np.linspace(0,1,10)
#fig = plt.figure()
plt.plot(x)
plt.savefig("kk.png")
