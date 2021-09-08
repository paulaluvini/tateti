#!/usr/bin/env python
# coding: utf-8

# ## Práctica Metaheurísticas

# El objetivo de esta práctica es lograr un agente que juegue (bien) al TaTeTi, entrenado mediante un algoritmo genético.
# En este caso la población será un conjunto de agentes que competirán entre sí. Cada individuo será un agente que, básicamente, debe responder a la pregunta “¿Cuál es tu próxima jugada?”

# ### Definiendo al agente

# Un agente será una clase en Python que pueda responder a un método next_move(tablero). Dicho método tomará un tablero de TaTeTi con las jugadas realizadas hasta el momento y deberá devolver un número de fila y un número de columna (las filas se enumeran desde 1 contando desde arriba y las columnas también desde 1 contando desde la izquierda), que indica en qué casillero se hará su próxima jugada.
# 
# El tablero estará representado por una lista de listas. Cada lista representará una fila del tablero y contendrá en sus posiciones alguno de los siguientes 3 caracteres: “.”, que representa una casilla vacía. “x” que representa una jugada del jugador Cruz. “o” que representa una jugada del Círculo.

# In[61]:


import random
import numpy as np


# In[1]:


class agente:
    
    def next_move(tablero):
        print(tablero)
        


# In[14]:


tablero1 = [[".",".","."],[".",".","."],[".",".","."]]
tablero2 = [["x",".","."],[".","x","."],["x","x","."]]
print(tablero2)


# In[15]:


agente.next_move(tablero = tablero2)


# In[68]:


tablero3 = [['.', '.', '.'], ['.', '.', '.'], ['.', '.', '.']]
tablero_vacio = ['.', '.', '.','.', '.', '.', '.', '.', '.']

for i in range(0, 9):
    posiciones = []
    for fila in tablero3:
        for posicion in fila:
            posiciones.append(posicion)
if posiciones == tablero_vacio:
    posiciones[random.choice(np.arange(0,8))]= jugador
    print(posiciones)


# In[99]:


#Primero lo que voy a hacer es buscar el primer lugar vacio y completarlo con mi jugador. Luego guardo ese tablero y voy a completar
#el segundo lugar vacio, asi de manera de tener todos los tableros existentes posibles. Luego evaluo con mis caracteristicas
#a los tableros y me quedo con el mejor.

jugador = "x"
competidor = "o"
vacio = "."

movimiento = []
for i in range(0, 9):
    posiciones = []
    for fila in tablero2:
        for posicion in fila:
            posiciones.append(posicion)
    p = []
    if posiciones[i] == ".":
        posiciones[i] = jugador
        p.append(posiciones)
        p = p[0]
    else:
        p = posiciones

    new_posiciones = [p[0:3], p[3:6], p[6:9]]
    print(new_posiciones)

    #CARACTERISTICAS
    puntajes = []
    
    #Segundo movimiento
    fila1_v1 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == (vacio or competidor) and new_posiciones[0][2] == (vacio or competidor)
    fila1_v2 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[0][1] == (vacio or competidor) and new_posiciones[0][2] == jugador
    fila1_v3 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[0][1] == jugador and new_posiciones[0][2] == (vacio or competidor)
    
    fila2_v1 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == (vacio or competidor) and new_posiciones[1][2] == (vacio or competidor)
    fila2_v2 = new_posiciones[1][0] == (vacio or competidor) and new_posiciones[1][1] == (vacio or competidor) and new_posiciones[1][2] == jugador
    fila2_v3 = new_posiciones[1][0] == (vacio or competidor) and new_posiciones[1][1] == jugador and new_posiciones[1][2] == (vacio or competidor)

    fila3_v1 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == (vacio or competidor) and new_posiciones[2][2] == (vacio or competidor)
    fila3_v2 = new_posiciones[2][0] == (vacio or competidor) and new_posiciones[2][1] == (vacio or competidor) and new_posiciones[2][2] == jugador
    fila3_v3 = new_posiciones[2][0] == (vacio or competidor) and new_posiciones[2][1] == jugador and new_posiciones[2][2] == (vacio or competidor)
    
    columna1_v1 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == (vacio or competidor) and new_posiciones[2][0] == (vacio or competidor)
    columna1_v2 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[1][0] == jugador and new_posiciones[2][0] == (vacio or competidor)
    columna1_v3 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[1][0] == (vacio or competidor) and new_posiciones[2][0] == jugador
    
    columna2_v1 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == (vacio or competidor) and new_posiciones[2][1] == (vacio or competidor)
    columna2_v2 = new_posiciones[0][1] == (vacio or competidor) and new_posiciones[1][1] == jugador and new_posiciones[2][1] == (vacio or competidor)
    columna2_v3 = new_posiciones[0][1] == (vacio or competidor) and new_posiciones[1][1] == (vacio or competidor) and new_posiciones[2][1] == jugador
    
    columna3_v1 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == (vacio or competidor) and new_posiciones[2][2] == (vacio or competidor)
    columna3_v2 = new_posiciones[0][2] == (vacio or competidor) and new_posiciones[1][2] == jugador and new_posiciones[2][2] == (vacio or competidor)
    columna3_v3 = new_posiciones[0][2] == (vacio or competidor) and new_posiciones[1][2] == (vacio or competidor) and new_posiciones[2][2] == jugador
    
    #Dos por fila/columna
    fila1_v4 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == jugador and new_posiciones[0][2] == (vacio or competidor)
    fila1_v5 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[0][1] == jugador and new_posiciones[0][2] == jugador
    
    fila2_v4 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[1][2] == (vacio or competidor)
    fila2_v5 = new_posiciones[1][0] == (vacio or competidor) and new_posiciones[1][1] == jugador and new_posiciones[1][2] == jugador

    fila3_v4 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == jugador and new_posiciones[2][2] == (vacio or competidor)
    fila3_v5 = new_posiciones[2][0] == (vacio or competidor) and new_posiciones[2][1] == jugador and new_posiciones[2][2] == jugador
    
    columna1_v4 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == jugador and new_posiciones[2][0] == (vacio or competidor)
    columna1_v5 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[1][0] == jugador and new_posiciones[2][0] == jugador
        
    columna2_v4 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][1] == (vacio or competidor)
    columna2_v5 = new_posiciones[0][1] == (vacio or competidor) and new_posiciones[1][1] == jugador and new_posiciones[2][1] == jugador
        
    columna3_v4 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == jugador and new_posiciones[2][2] == (vacio or competidor)
    columna3_v5 = new_posiciones[0][2] == (vacio or competidor) and new_posiciones[1][2] == jugador and new_posiciones[2][2] == jugador
    
    #Ganadores
    fila1 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == jugador and new_posiciones[0][2] == jugador
    fila2 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[1][2] == jugador
    fila3 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == jugador and new_posiciones[2][2] == jugador
    columna1 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == jugador and new_posiciones[2][0] == jugador
    columna2 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][1] == jugador
    columna3 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == jugador and new_posiciones[2][2] == jugador
    diagonal1 = new_posiciones[0][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][2] == jugador
    diagonal2 = new_posiciones[2][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[0][2] == jugador
    
    
    if fila1_v1 or fila1_v2 or fila1_v3 or fila2_v1 or fila2_v2 or fila2_v3 or fila3_v1 or fila3_v2 or fila3_v3 or columna1_v1 or columna1_v2 or columna1_v3 or columna2_v1 or columna2_v2 or columna2_v3 or columna3_v1 or columna3_v2 or columna3_v3:
        puntajes.append(sum([fila1_v1, fila1_v2, fila1_v3, fila2_v1, fila2_v2, fila2_v3, fila3_v1, fila3_v2, fila3_v3,
                            columna1_v1, columna1_v2, columna1_v3, columna2_v1, columna2_v2, columna2_v3, columna3_v1,
                            columna3_v2, columna3_v3]))
    else:
        puntajes.append(0)
    if fila1_v4 or fila1_v5 or fila2_v4 or fila2_v5 or fila3_v4 or fila3_v5 or columna1_v4 or columna1_v5 or columna2_v4 or columna2_v5 or columna3_v4 or columna3_v5:
        puntajes.append(sum([fila1_v4, fila1_v5, fila1_v6, fila2_v4, fila2_v5, fila2_v6, fila3_v4, fila3_v5, fila3_v6, 
                             columna1_v4, columna1_v5, columna2_v4, columna2_v5, columna3_v4, columna3_v5]))
    else:
        puntajes.append(0)
    if fila1 or fila2 or fila3 or columna1 or columna2 or columna3 or diagonal1 or diagonal2:
        puntajes.append(sum([fila1, fila2, fila3, columna1, columna2, columna3, diagonal1, diagonal2]))
    else:
        puntajes.append(0)
    movimiento.append(sum(puntajes))
    print(puntajes)


print(movimiento)
movimiento.index(max(movimiento))


# In[88]:


sum([fila1_v1, fila1_v2, fila1_v3])

