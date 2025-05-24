import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def expo(x,y):
    n = len(x)
    z = np.log(y)
    Sx = np.sum(x)
    Sz = np.sum(z)
    xz = x*z 
    Sxz = np.sum(xz)
    x2 = np.pow(x,2)
    Sx2 = np.sum(x2)
    a1 = ((n*Sxz)-(Sz*Sx))/((n*Sx2)-(Sx)**2)
    a0 = np.mean(z)-a1*np.mean(x)
    a = np.exp(a0)
    b = a1 
    # predected value
    pred = []
    for i in range(n):
     values = a*np.exp(b*t[i])
     pred.append(values)
    plt.scatter(x,y)
    plt.plot(x,np.array(pred), color='red')
    plt.title("Exponential Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum((y - np.array(pred))**2)
    r2 = 1 - (ss_residual / ss_total)
    print("R square: ",float(r2))
#Test case 1
t = [0, 1, 3, 5, 7, 9]
r = [1, 0.891, 0.708, 0.562, 0.447, 0.355]
expo(t,r)

#Test case 2
t = [10, 16, 25, 40, 60]
r = [94, 118, 147, 180, 230]
expo(t,r)

#Test case 3
t = [4, 2.25, 1.45, 1.0, 0.65, 0.25, 0.006]
r = [0.398, 0.298,0.238, 0.198, 0.158, 0.098, 0.048]
expo(t,r)