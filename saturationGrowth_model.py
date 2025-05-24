import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def sat(p,f):
    n = len(p)
    z = 1/np.array(f)
    w = 1/np.array(p)
    Sz = np.sum(z)
    Sw = np.sum(w)
    wz = w*z 
    Swz = np.sum(wz)
    w2 = np.pow(w,2)
    Sw2 = np.sum(w2)
    a1 = ((n*Swz)-(Sz*Sw))/((n*Sw2)-(Sw)**2)
    a0 = np.mean(z)-a1*np.mean(w)
    a = 1/a0 
    b = a1*a 
    # predected value
    pred = []
    for i in range(n):
     values = (a*p[i])/(b+p[i])
     pred.append(values)
    plt.scatter(p,f)
    plt.plot(p,np.array(pred), color='red')
    plt.title("Exponential Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    y_mean = np.mean(f)
    ss_total = np.sum((f - y_mean)**2)
    ss_residual = np.sum((f - np.array(pred))**2)
    r2 = 1 - (ss_residual / ss_total)
    print("R square: ",float(r2))
#test case 1
p = [10, 16, 25, 40, 60]
f = [94, 118, 147, 180, 230]
sat(p,f)

#test case 2
p = [0.1, 0.5, 1, 2, 3]
f = [0.09, 0.33, 0.5, 0.67, 0.7]
sat(p,f)

#test case 3
p = [1, 2, 4, 6, 8]
f = [0.5, 0.67, 0.8, 0.86, 0.89]
sat(p,f)