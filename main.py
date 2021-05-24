import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as spint

#wartosci poczatkowe
c=0
f=0
x_a=0
x_b=1
liczba_wezlow=8

#Warunki Brzegowe
WB= [{"ind":1, "typ":'D', "wartosc":1},
     {"ind":liczba_wezlow, "typ":'D', "wartosc":2}]
#print(WB[1]["ind"]) - wywolanie konkretnego elementu z tablicy slownikow

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

def rysowanie(wezly,liczba_wezlow):
    # ------tworzenie wektora z 0 dla punktow
    y = [0]
    for i in range(1, liczba_wezlow, 1):
        y.append(0)

    # ------wykres geometrii

    # punkty
    plt.plot(wezly[:, 1], y, marker='o')

    # podpisy punktow
    for i in range(0, liczba_wezlow, 1):
        f = "x" + str(i + 1)
        plt.annotate(f, (wezly[i, 1], -0.006), color='green')

    # podpisy elementów
    for i in range(0, liczba_wezlow - 1, 1):
        g = "s" + str(i + 1)
        plt.annotate(g, ((wezly[i, 1] + wezly[i + 1, 1]) / 2, 0.005), color='blue')

    plt.grid(True)
    plt.show()

# wywolanie funkcji generoujacej oraz rysujacej
wezly, elementy = GenerujTabliceGeometrii(x_a, x_b, liczba_wezlow)
print("WEZLY:\n", wezly)
print("\nELEMENTY:\n", elementy)

rysowanie(wezly,liczba_wezlow)

#------------------------------------------------------------------------ Pierwsza czesc koniec ---------------------------------------------------------------------------------------------------

def FunkcjeBazowe(x):
    #n-stopien wymaganych funkcji ksztaltu
    #zwraca: f-(n+1) elementowa lista funkcji bazowych. df - (n+1_ pochodnych funkcji bazowych

    if x==0:
        f = (lambda x: 1+0*x)
        df = (lambda x: 0*x)
    elif x==1:
        f=(lambda x: -1/2*x + 1/2, lambda x: 0.5*x + 0.5)
        df = (lambda x: -1/2 + 0 * x, lambda x: 0.5 + 0 * x )
    else:
        raise Exception("blad w funkcji funcje bazowe")

    return f,df

stopien_funkcji_bazowych=1
phi,dphi = FunkcjeBazowe(stopien_funkcji_bazowych)

print(phi)
print(dphi)

xx = np.linspace(-1,1,101)
plt.plot(xx,phi[0](xx),'r')
plt.plot(xx,phi[1](xx),'g')
plt.plot(xx,dphi[0](xx),'b')
plt.plot(xx,dphi[1](xx),'c')
plt.show()


def Alokacja(x):
    tmp = (x,x)
    tmp1 = (x,1)
    A = np.zeros(tmp)
    b = np.zeros(tmp1)
    return A, b

A,b = Alokacja(liczba_wezlow)
print("------------")
print(A)
print(b)

#------------------Processing--------------------------------

def Aij(df_i, df_j, c, f_i, f_j):
    fun_podc = lambda x: -df_i(x) * df_j(x) + c * f_i(x) * f_j(x)
    return fun_podc

#A,b = Alokacja(liczba_wezlow)

liczbaElementow = np.shape(elementy)[0]

for ee in np.arange(0,liczbaElementow):

    elemRowInd=ee
    elemGlobalInd=elementy[ee,0]
    elemWezel1 = elementy[ee,1]  #indeks poczatkowego wezla elementu ee
    elemWezel2 = elementy[ee, 2] #indeks koncowego wezla elementu ee
    indGlobalneWezlow=np.array([elemWezel1,elemWezel2])

    Ml = np.zeros((stopien_funkcji_bazowych + 1, stopien_funkcji_bazowych + 1))#nawiasy

    p_a = wezly[elemWezel1-1,1]
    p_b = wezly[elemWezel2-1,1]

    J = (p_b-p_a)/2

    m=0
    n=0
    Ml[m,n] = J * spint.quad(Aij(dphi[m],dphi[n],c,phi[m],phi[n]),-1,1)[0]

    m=0
    n=1
    Ml[m,n] = J * spint.quad(Aij(dphi[m],dphi[n],c,phi[m],phi[n]),-1,1)[0]

    m=1
    n=0
    Ml[m,n] = J * spint.quad(Aij(dphi[m],dphi[n],c,phi[m],phi[n]),-1,1)[0]

    m=1
    n=1
    Ml[m,n] = J * spint.quad(Aij(dphi[m],dphi[n],c,phi[m],phi[n]),-1,1)[0]

    A[np.ix_(indGlobalneWezlow-1,indGlobalneWezlow-1)] = A[np.ix_(indGlobalneWezlow-1,indGlobalneWezlow-1)]+Ml

print(WB)

if WB[0]['typ'] == 'D':
    ind_wezla = WB[0]['ind']
    wart_war_brzeg = WB[0]['wartosc']

    iwp = ind_wezla-1

    WZMACNIACZ = 10**14

    b[iwp] = A[iwp,iwp]*WZMACNIACZ*wart_war_brzeg
    A[iwp,iwp] = A[iwp,iwp]*WZMACNIACZ

if WB[1]['typ'] == 'D':
    ind_wezla = WB[1]['ind']
    wart_war_brzeg = WB[1]['wartosc']

    iwp = ind_wezla - 1

    WZMACNIACZ = 10**14

    b[iwp] = A[iwp, iwp] * WZMACNIACZ * wart_war_brzeg
    A[iwp, iwp] = A[iwp, iwp] * WZMACNIACZ


if WB[0]['typ'] == 'N':
    print("Nie zaimplementowano")

if WB[1]['typ'] == 'N':
    print("Nie zaimplementowano")

u=np.linalg.solve(A,b)


def rysujRozwiazanie(wezly,elementy,WB,u):
    wewx=wezly[:,1]
    plt.plot(wewx,u,'m*')
    plt.show()

rysujRozwiazanie(wezly,elementy,WB,u)

print("------------")
print(A)
print(b)
