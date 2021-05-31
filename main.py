from funkcje import *
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

#wywolanie funkcji generujacej oraz rysujacej
wezly, elementy = GenerujTabliceGeometrii(x_a, x_b, liczba_wezlow)
print("\nWEZLY:\n", wezly)
print("\nELEMENTY:\n", elementy)

rysowanie(wezly,liczba_wezlow)

#określenie stopnia funkcji bazowych oraz wywołanie funkcji
stopien_funkcji_bazowych=1
phi,dphi = FunkcjeBazowe(stopien_funkcji_bazowych)

#wyrysowanie funkcji bazowych
col=['r','g','m','b','c','y','k','lime','navy','purple','coral','peru']
z=0
xx = np.linspace(-1,1,101)
for i in range(0,stopien_funkcji_bazowych+1):
    plt.plot(xx, phi[i](xx), col[z], label="φ"+str(i+1))
    plt.plot(xx, dphi[i](xx), col[z+6],label="dφ"+str(i+1))
    plt.legend()
    z=z+1

plt.show()


#alokacja pamieci macierzy A,b
A,b = Alokacja(liczba_wezlow)
print("\nAlokacja A:")
print(A)
print("\nAlokacja b:")
print(b)

liczbaElementow = np.shape(elementy)[0]

for ee in np.arange(0,liczbaElementow):
    elemRowInd=ee
    elemGlobalInd=elementy[ee,0]
    elemWezel1 = elementy[ee,1]  #indeks poczatkowego wezla elementu ee
    elemWezel2 = elementy[ee, 2] #indeks koncowego wezla elementu ee
    indGlobalneWezlow=np.array([elemWezel1,elemWezel2])

    Ml = np.zeros((stopien_funkcji_bazowych+1,stopien_funkcji_bazowych+1))

    p_a = wezly[elemWezel1-1,1]
    p_b = wezly[elemWezel2-1,1]

    J = (p_b-p_a)/2

    for m in range(stopien_funkcji_bazowych + 1):
        for n in range(stopien_funkcji_bazowych + 1):
            Ml[m,n]=J*spint.quad(Aij(dphi[m],dphi[n],c,phi[m],phi[n]),-1,1)[0]

    if stopien_funkcji_bazowych == 1:
        A[np.ix_(indGlobalneWezlow-1,indGlobalneWezlow-1)]+=Ml
    elif stopien_funkcji_bazowych == 2:
        A[np.ix_(indGlobalneWezlow-1,indGlobalneWezlow-1)]+=Ml[:-1,:-1]

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

rysujRozwiazanie(wezly,elementy,WB,u)

print("\nMacierz A:\n",A)
print("\nMacierz b:\n",b)