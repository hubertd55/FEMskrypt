import numpy as np
import matplotlib

c=0
f=0
x_a=0
x_b=1

#wezly=np.array([[1,0],[2,1],[3,0.5],[4, 0.75]])

#elementy= np.array([[1, 1, 3],[2, 4, 2],[3, 3, 4]])

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1


def tablica_geometrii(x_a,x_b,n):
    tymczasowa=(x_b-x_a)/(n-1)
    macierz=np.array([x_a])

    for i in range(1,n,1):
        macierz = np.block([macierz,i*tymczasowa+x_a])
    return macierz

def generujTabliceGeometrii(p,k,n):
    tmp = (k-p) / (n-1)
    macierz = np.array([1,0])

    for i in range(1, n, 1):
        macierz = np.block([
            [macierz],
            [i+1, i * tmp + p],
        ])
    return macierz

#k=tablica_geometrii(0,1,4)
d=generujTabliceGeometrii(0,1,4)
print(k)
print(d)