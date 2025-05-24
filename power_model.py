import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def power(x,y):
    z = np.log(f)
    n = len(x)
    Sz = np.sum(z)
    w = np.log(p)
    Sw = np.sum(w)
    wz = w*z 
    Swz = np.sum(wz)
    w2 = np.pow(w,2)
    Sw2 = np.sum(w2)

    a1 = ((n*Swz)-(Sz*Sw))/((n*Sw2)-(Sw)**2)
    a0 = np.mean(z)-a1*np.mean(w)

    a = np.exp(a0)
    b = a1 

    # predected value
    pred = []
    for i in range(n):
     values = a*np.pow(p[i],b)
     pred.append(values)

    plt.scatter(x,y)
    plt.plot(x,np.array(pred), color='red')
    plt.title("Power Model")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    y_mean = np.mean(y)
    ss_total = np.sum((y - y_mean)**2)
    ss_residual = np.sum((y - np.array(pred))**2)
    r2 = 1 - (ss_residual / ss_total)
    print("R square: ",float(r2))

#test case 1
p = [10, 16, 25, 40, 60]
f = [94, 118, 147, 180, 230]
power(p,f)

#test case 2
p = [1,2,3,4,5]
f = [2,4,9,16,25]
power(p,f)

#test case 3
p=[1,2,3,4,5]
f=[1, 1.414, 1.732, 2.0, 2.236]
power(p,f)
