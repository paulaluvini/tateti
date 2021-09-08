#!/usr/bin/env python
# coding: utf-8

# ## Práctica Metaheurísticas

# El objetivo de esta práctica es lograr un agente que juegue (bien) al TaTeTi, entrenado mediante un algoritmo genético.
# En este caso la población será un conjunto de agentes que competirán entre sí. Cada individuo será un agente que, básicamente, debe responder a la pregunta “¿Cuál es tu próxima jugada?”

# ### Definiendo al agente

# Un agente será una clase en Python que pueda responder a un método next_move(tablero). Dicho método tomará un tablero de TaTeTi con las jugadas realizadas hasta el momento y deberá devolver un número de fila y un número de columna (las filas se enumeran desde 1 contando desde arriba y las columnas también desde 1 contando desde la izquierda), que indica en qué casillero se hará su próxima jugada.
# 
# El tablero estará representado por una lista de listas. Cada lista representará una fila del tablero y contendrá en sus posiciones alguno de los siguientes 3 caracteres: “.”, que representa una casilla vacía. “x” que representa una jugada del jugador Cruz. “o” que representa una jugada del Círculo.

# In[1]:


import random
import numpy as np


# In[2]:


#Primero lo que voy a hacer es buscar el primer lugar vacio y completarlo con mi jugador. 
#Luego guardo ese tablero y voy a generar un nuevo tablero con el segundo lugar vacio, asi de manera de tener todos los tableros existentes 
#posibles en una jugada. 
#Luego evaluo con mis caracteristicas (hasta ahora puse 3) a los tableros y me quedo con la mejor jugada.

class agente:
    def next_move(tablero, competidor, jugador, ponderaciones):
        vacio = "."
        movimiento = []
        mov = []
        for i in range(0, 9):
            posiciones = []
            for fila in tablero:
                for posicion in fila:
                    posiciones.append(posicion)
            p = []
            #No considero los lugares del tablero ya completos
            if posiciones[i] == "x":
                continue
            if posiciones[i] == "o":
                continue
            if posiciones[i] == ".":
                posiciones[i] = jugador
                p.append(posiciones)
                p = p[0]
            else:
                p = posiciones
            mov.append(i)
            new_posiciones = [p[0:3], p[3:6], p[6:9]]
            print(new_posiciones)

            #CARACTERISTICAS
            puntajes = []
    
            #1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
            fila1_v1 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == jugador and new_posiciones[0][2] == (vacio or competidor)
            fila1_v2 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[0][1] == jugador and new_posiciones[0][2] == jugador
    
            fila2_v1 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[1][2] == (vacio or competidor)
            fila2_v2 = new_posiciones[1][0] == (vacio or competidor) and new_posiciones[1][1] == jugador and new_posiciones[1][2] == jugador
    
            fila3_v1 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == jugador and new_posiciones[2][2] == (vacio or competidor)
            fila3_v2 = new_posiciones[2][0] == (vacio or competidor) and new_posiciones[2][1] == jugador and new_posiciones[2][2] == jugador
    
            columna1_v1 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == jugador and new_posiciones[2][0] == (vacio or competidor)
            columna1_v2 = new_posiciones[0][0] == (vacio or competidor) and new_posiciones[1][0] == jugador and new_posiciones[2][0] == jugador
        
            columna2_v1 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][1] == (vacio or competidor)
            columna2_v2 = new_posiciones[0][1] == (vacio or competidor) and new_posiciones[1][1] == jugador and new_posiciones[2][1] == jugador
        
            columna3_v1 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == jugador and new_posiciones[2][2] == (vacio or competidor)
            columna3_v2 = new_posiciones[0][2] == (vacio or competidor) and new_posiciones[1][2] == jugador and new_posiciones[2][2] == jugador
    
            #Ganadores: que completan fila, columna o diagonal
            fila1 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == jugador and new_posiciones[0][2] == jugador
            fila2 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[1][2] == jugador
            fila3 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == jugador and new_posiciones[2][2] == jugador
    
            columna1 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == jugador and new_posiciones[2][0] == jugador
            columna2 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][1] == jugador
            columna3 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == jugador and new_posiciones[2][2] == jugador
    
            diagonal1 = new_posiciones[0][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][2] == jugador
            diagonal2 = new_posiciones[2][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[0][2] == jugador
    
            #Que evitan que el otro rellene su linea
            fila1_competidor = new_posiciones[0].count(competidor) == 2 and new_posiciones[0].count(jugador) == 1
            fila2_competidor = new_posiciones[1].count(competidor) == 2 and new_posiciones[1].count(jugador) == 1
            fila3_competidor = new_posiciones[2].count(competidor) == 2 and new_posiciones[2].count(jugador) == 1
    
            columna1_competidor = ((new_posiciones[0][0],new_posiciones[1][0],new_posiciones[2][0]).count(competidor) == 2 and (new_posiciones[0][0],new_posiciones[1][0],new_posiciones[2][0]).count(jugador) == 1)
            columna2_competidor = (new_posiciones[0][1],new_posiciones[1][1],new_posiciones[2][1]).count(competidor) == 2 and (new_posiciones[0][1],new_posiciones[1][1],new_posiciones[2][1]).count(jugador) == 1
            columna3_competidor = (new_posiciones[0][2],new_posiciones[1][2],new_posiciones[2][2]).count(competidor) == 2 and (new_posiciones[0][2],new_posiciones[1][2],new_posiciones[2][2]).count(jugador) == 1
            
            #Ocurrencias: 
    
            #1era caracteristica
            if fila1_v1 or fila1_v2 or fila2_v1 or fila2_v2 or fila3_v1 or fila3_v2 or columna1_v1 or columna1_v2 or columna2_v1 or columna2_v2 or columna3_v1 or columna3_v2:
                puntajes.append(ponderaciones[0]*sum([fila1_v1, fila1_v2, fila2_v1, fila2_v2, fila3_v1, fila3_v2, columna1_v1, columna1_v2, columna2_v1, columna2_v2, columna3_v1, columna3_v2]))
            else:
                puntajes.append(0)
            
            #2da caracteristica
            if fila1 or fila2 or fila3 or columna1 or columna2 or columna3 or diagonal1 or diagonal2:
                puntajes.append(ponderaciones[1]*sum([fila1, fila2, fila3, columna1, columna2, columna3, diagonal1, diagonal2]))
            else:
                puntajes.append(0)
            
            #3era caracteristica
            if fila1_competidor or fila2_competidor or fila3_competidor or columna1_competidor or columna2_competidor or columna3_competidor:
                puntajes.append(ponderaciones[2]*sum([fila1_competidor, fila2_competidor, fila3_competidor,columna1_competidor, columna2_competidor, columna3_competidor]))
            else:
                puntajes.append(0)
            print(puntajes)
            movimiento.append(sum(puntajes))
        #print(movimiento)
        #Aca no importa si el puntaje maximo se da en dos o mas tableros, nos quedamos con cualquiera de ellos de igual forma
        indice = mov[movimiento.index(max(movimiento))]
        if indice == 0:
            jugada = [1,1]
        if indice == 1:
            jugada = [1,2]
        if indice == 2:
            jugada = [1,3]
        if indice == 3:
            jugada = [2,1]
        if indice == 4:
            jugada = [2,2]
        if indice == 5:
            jugada = [2,3]
        if indice == 6:
            jugada = [3,1]
        if indice == 7:
            jugada = [3,2]
        if indice == 8:
            jugada = [3,3]
        print("La jugada debe ser: ")
        return jugada


# In[3]:


agente.next_move(tablero = [['x', '.', 'x'], ['.', 'o', 'o'], ['.', '.', '.']], competidor = "o", jugador = "x", 
                 ponderaciones = [1,5,4])


# In[5]:


agente.next_move(tablero = [['x', '.', 'o'], ['.', 'x', '.'], ['x', 'x', '.']], competidor = "o", jugador = "x", 
                 ponderaciones = [1,2,3])

