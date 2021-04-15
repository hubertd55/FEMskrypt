import numpy as np
import matplotlib.pyplot as plt

#wartosci poczatkowe
c=0
f=0
x_a=1
x_b=15
liczba_wezlow=7

#wezly=np.array([[1,0],[2,1],[3,0.5],[4, 0.75]])
#elementy= np.array([[1, 1, 3],[2, 4, 2],[3, 3, 4]])

twb_L = 'D'
twb_R = 'D'

wwb_L = 0
wwb_R = 1

#funkcja generowania tablicy geometrii
def GenerujTabliceGeometrii(xa,xb,n):
    temp = (xb-xa) / (n-1)
    macierz = np.array([1,xa])

    for i in range(1, n, 1):
        macierz = np.block([
            [macierz],
            [i+1, i * temp + xa],
        ])

    macierz2 = np.array([1, 1, 2])
    for i in range(2, n, 1):
        macierz2 = np.block([
            [macierz2],
            [i, i, i + 1]
        ])

    return macierz,macierz2

#wywolanie funkcji
wezly,elementy=GenerujTabliceGeometrii(x_a,x_b,liczba_wezlow)
print("WEZLY:\n",wezly)
print("\nELEMENTY:\n",elementy)

#------tworzenie wektora z 0 dla punktow
y=[0]
for i in range(1, liczba_wezlow, 1):
    y.append(0)


#------wykres geometrii

#punkty
plt.plot(wezly[:,1],y,marker='o')

#podpisy punktow
for i in range(0, liczba_wezlow, 1):
    f="x"+str(i+1)
    plt.annotate(f, (wezly[i,1]-0.25, -0.006), color='green')

#podpisy elementów
for i in range(0, liczba_wezlow-1, 1):
    g = "s" + str(i + 1)
    plt.annotate(g, ((wezly[i, 1]+wezly[i+1,1])/2-0.2, 0.01), color='blue')

plt.show()