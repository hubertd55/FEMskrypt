import numpy as np
import matplotlib.pyplot as plt

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

#rysowanie wezlow
def rysowanie(wezly,liczba_wezlow):
    # ------tworzenie wektora z 0 dla wezlow
    y = [0]
    for i in range(1, liczba_wezlow, 1):
        y.append(0)

    # punkty dla wezlow
    plt.plot(wezly[:, 1], y, marker='o')

    # podpisy punktow(wezlow)
    for i in range(0, liczba_wezlow, 1):
        f = "x" + str(i + 1)
        plt.annotate(f, (wezly[i, 1], -0.006), color='green')

    # podpisy elementów
    for i in range(0, liczba_wezlow - 1, 1):
        g = "s" + str(i + 1)
        plt.annotate(g, ((wezly[i, 1] + wezly[i + 1, 1]) / 2, 0.005),
        color='blue')

    plt.grid(True)
    plt.show()

def FunkcjeBazowe(x):
    if x==0:
        f = (lambda x: 1+0*x)
        df = (lambda x: 0*x)

    elif x==1:
        f=(lambda x: -1/2*x + 1/2, lambda x: 0.5*x + 0.5)
        df = (lambda x: -1/2 + 0 * x, lambda x: 0.5 + 0 * x )

    elif x == 2:
        f = (lambda x: 1 / 2 * x * (x - 1),
             lambda x: -x ** 2 + 1,
             lambda x: 1 / 2 * x * (x + 1))
        df = (lambda x: x - 1 / 2,
              lambda x: -2 * x,
              lambda x: x + 1 / 2)

    else:
        raise Exception("blad w funkcji - funkcje bazowe")

    return f,df

def rysujRozwiazanie(wezly,elementy,WB,u):

    y = [0]
    for i in range(1, np.shape(wezly)[0], 1):
        y.append(0)

    # punkty dla wezlow
    plt.plot(wezly[:, 1], y, marker='o')

    wewx=wezly[:,1]
    plt.plot(wewx, u)
    plt.plot(wewx,u,'m*')
    plt.show()

def Alokacja(x):
    tmp = (x,x)
    tmp1 = (x,1)
    A = np.zeros(tmp)
    b = np.zeros(tmp1)
    return A, b

def Aij(df_i, df_j, c, f_i, f_j):
    fun_podc = lambda x: -df_i(x) * df_j(x) + c * f_i(x) * f_j(x)
    return fun_podc