#!/usr/bin/env python
# coding: utf-8

# <h1> <center> Práctica Metaheurísticas </center> </h1> 
# 
# El objetivo de esta práctica es lograr un agente que juegue (bien) al TaTeTi, entrenado mediante un algoritmo genético.
# En este caso la población será un conjunto de agentes que competirán entre sí. Cada individuo será un agente que, básicamente, debe responder a la pregunta “¿Cuál es tu próxima jugada?”

# In[1]:


import random
import numpy as np


# ### Características
# 
# 1. Definir las características que conformarán la función de evaluación de los agentes:
# 
#     1- Cantidad de grupos de 2 cruces consecutivas por fila/columna<br>
#     2- Ganadores: que completan fila, columna o diagonal<br>
#     3- Que evitan que el otro rellene su linea<br>
#     4- Cantidad de jugadas del tablero<br>
#     5- Cantidad de grupos de 2 círculos consecutivas por fila/columna

# ## Algoritmo Genético
# 
# Acá voy a desarrollar un algoritmo genético basado en el que encontramos acá: https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/master/Tutorial%20Project
# https://towardsdatascience.com/genetic-algorithm-explained-step-by-step-65358abe2bf
# 
# En un segundo paso deberíamos crear más funciones de fitness, agregarle criterios de selección, etc. como la consigna requiere.

# ### Definición de Funciones

# #### Fitness
# 
# En primer lugar vamos a definir las distintas funciones de fitness a utilizar. Comenzaremos por tomar a la función objetivo como primer opción.

# In[2]:


#Comenzamos por plantear una función de fitness que es la función objetivo y un término 
def fitness_v1(equation_inputs, population):
    #f = []
    #max_jugadas = 9
    #for i in np.arange(1,max_jugadas+1):
    #    f.append(np.log(i))
    #c = list(reversed(f))[count_jugadas-1]
    #caracteristica4 = []
    #for i in range(len(population)):
    #    caracteristica4.append(population[i][3])
    #caract_jugada = np.multiply(caracteristica4, c)
    fitness = np.sum(population*equation_inputs, axis=1)#+np.multiply(caract_jugada,equation_inputs[1])
    return fitness


# In[3]:


#Acá voy a agregar un término cuadrático en la característica "ganar"
def fitness_v2(equation_inputs, population):
    #f = []
    #for i in np.arange(1,max_jugadas+1):
    #    f.append(np.log(i))
    #c = list(reversed(f))[count_jugadas-1]
    #caracteristica4 = []
    #for i in range(len(population)):
    #    caracteristica4.append(population[i][3])
    #caract_jugada = np.multiply(caracteristica4, c)
    caracteristica2 = []
    for i in range(len(population)):
        caracteristica2.append(population[i][1])
    fitness = np.sum(population*equation_inputs, axis=1)+np.multiply(caracteristica2,equation_inputs[1]**2)#+np.multiply(caract_jugada,equation_inputs[1])
    return fitness


# #### De selección

# In[4]:


# SELECCIóN PONDERADA
# A continuación elegimos a la cantidad que queremos de individuos de la generación actual para que sean "padres" de la siguiente.
# Estos van a ser los individuos con mayor puntaje en el fitness calculado en la función anterior

def select_mating_sort(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


# In[5]:


# SELECCIóN MIXTA
# A continuación elegimos algunos individuos al azar de la generación actual para que sean "padres" de la siguiente.

def select_mating_random(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    
    #Primero los selecciono con un sort
    parents_sort = np.empty((round(num_parents/2), pop.shape[1]))
    for parent_num in range(round(num_parents/2)):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents_sort[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    
    #Agarro algunos otros aleatoriamente
    parents_rand = np.empty(((num_parents-round(num_parents/2)), pop.shape[1]))
    for parent_num2 in range(num_parents-round(num_parents/2)):
        random_index = np.random.choice(len(pop))
        parents_rand[parent_num2, :] = pop[random_index, :]
    
    parents[0:parents_sort.shape[0], :] = parents_sort
    parents[parents_sort.shape[0]:, :] = parents_rand
    return parents


# #### Crossover
# 
# El crossover es intercambiar un gen o un grupo de genes por combinar dos cromosomas padres. El nuevo cromosoma producto de esta combinación se llama "offspring".

# In[6]:


#Hacemos un crossover intercambiando características entre sí y de esta manera creamos nuevos individuos. Tomamos la mitad de las
#caracteristicas, o 2 de 3.

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring


# In[7]:


#Probamos también cambiando solo 1 característica al azar

def crossover_last(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = random.choice(np.arange(1,3))

    for k in range(offspring_size[0]):
        parent1_idx = k%parents.shape[0]
        parent2_idx = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring


# #### Mutación

# In[8]:


#Una mutación intercambia una pequeña proporción de características del gen. Vamos a determinar que solo haya una mutacion

def mutation(offspring_crossover):
    mutation_parameter = round(offspring_crossover.shape[0]*offspring_crossover.shape[1]*0.1) #Solo dejo que el 10% del total de genes mute
    #Ahora por cada uno de las posibles mutaciones voy a sacar al azar una posición dentro de todos los cromosomas y genes y
    #reemplazarla por un valor aleatorio
    for i in range(mutation_parameter):
        gen_id = (random.choice(np.arange(0, offspring_crossover.shape[0])), random.choice(np.arange(0, offspring_crossover.shape[1])))
        offspring_crossover[gen_id[0], gen_id[1]] = random_value = np.random.uniform(0, 5.0, 1)
    return offspring_crossover


# ### Población inicial
# 
# La población inicial van a ser distintos agentes cuya diferencia entre sí va a estar definida por los *alpha* que cada uno tenga. 

# In[9]:


#Creamos la población inicial, con los distintos alfa para las distintas características

num_alpha = 3 #Esto va a depender de la cantidad de características que definamos
agents = 20 # Definimos el tamaño de la población.
population_size = (agents,num_alpha) 
population_zero = np.random.uniform(low=0, high=5.0,size=population_size)
print(population_zero)

#Creamos los puntajes correspondientes a las 3 características
#1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
#2- Ganadores: que completan fila, columna o diagonal
#3- Que evitan que el otro rellene su linea
#4- Cantidad de jugadas del tablero
#5- Cantidad de grupos de 2 círculos consecutivas por fila/columna
caracteristicas = [10,50,50]#,15,-10]


# ### Corriendo el algoritmo

# In[10]:


# Fitness 1 y select sorted
# Best solution :  [[[1.77039356 4.23467875 7.68155172]]]
# Best solution fitness :  [459.88442493]

# Fitness 1 y select mix
# Best solution :  [[[-0.88903542  4.3231201   9.41263685]]]
# Best solution fitness :  [489.64475622]

# Fitness 2  y select sorted
# Best solution :  [[[ 1.0715854   4.78466731 10.46561068]]]
# Best solution fitness :  [586.81058118]

# Fitness 2  y select mix
# Best solution :  [[[ 2.88305206  3.86156895 11.10734357]]]
# Best solution fitness :  [570.04099011]


# ### Definiendo al agente

# Un agente será una clase en Python que pueda responder a un método next_move(tablero). Dicho método tomará un tablero de TaTeTi con las jugadas realizadas hasta el momento y deberá devolver un número de fila y un número de columna (las filas se enumeran desde 1 contando desde arriba y las columnas también desde 1 contando desde la izquierda), que indica en qué casillero se hará su próxima jugada.
# 
# El tablero estará representado por una lista de listas. Cada lista representará una fila del tablero y contendrá en sus posiciones alguno de los siguientes 3 caracteres: “.”, que representa una casilla vacía. “x” que representa una jugada del jugador Cruz. “o” que representa una jugada del Círculo.

# In[11]:


class algoritmo():
    def train():

        #Creamos la población inicial, con los distintos alfa para las distintas características
        num_alpha = 3 #Esto va a depender de la cantidad de características que definamos
        agents = 500 # Definimos el tamaño de la población.
        population_size = (agents,num_alpha) 
        new_population = np.random.uniform(low=0, high=5.0,size=population_size)

        #Creamos los puntajes correspondientes a las 3 características
        #1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
        #2- Ganadores: que completan fila, columna o diagonal
        #3- Que evitan que el otro rellene su linea
        caracteristicas = [10,50,30]
        
        #ALGORITMO GENETICO
        best_outputs = []
        num_generations = 15
        num_parents_mating = 100 #La cantidad de padres que son tomados de cada generación
        
        for generation in range(num_generations):
            print("Generacion: ", generation)
            fitness = fitness_v1(equation_inputs = caracteristicas, population = new_population)#,count_jugadas= nro_jugada)

            # Mostramos el mejor resultado
            best_outputs.append(np.max(fitness))
            print("Mejor resultado: ", np.max(fitness))
    
            # Usamos la función de selección ordenada
            parents = select_mating_random(new_population, fitness,num_parents_mating)

            # Realizamos el crossover.
            offspring_crossover = crossover(parents,offspring_size=(population_size[0]-parents.shape[0], num_alpha))

            # Cambiamos algunos genes por mutacion
            offspring_mutation = mutation(offspring_crossover)

            # Finalmente la población nueva que sucedera a la generacion previa va a ser una combinacion de los parents (los
            # mejores individuos) y del crossover/mutation.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
    
        # El mejor resultado
        fitness = fitness_v1(caracteristicas, new_population)#, nro_jugada)
        best_match_idx = np.where(fitness == np.max(fitness))

        #print("Best solution : ", new_population[best_match_idx, :])
        #print("Best solution fitness : ", fitness[best_match_idx])
        ponderaciones = new_population[best_match_idx, :][0][0]
        
        return ponderaciones


# ### Ta-te-ti 3x3

# In[12]:


class agente():
    def next_move(tablero):
        #TABLERO
        #Primero lo que voy a hacer es buscar el primer lugar vacio y completarlo con mi jugador. 
        #Luego guardo ese tablero y voy a generar un nuevo tablero con el segundo lugar vacio, asi de manera de tener todos los tableros existentes posibles en una jugada. 
        #Luego evaluo con mis caracteristicas (hasta ahora puse 3) a los tableros y me quedo con la mejor jugada.
        ponderaciones = algoritmo.train()
        caracteristicas = [10,50,30]
        vacio = "."
        jugador = "x"
        competidor = "o"
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
                posiciones[i] = "x"
                p.append(posiciones)
                p = p[0]
            else:
                p = posiciones
            mov.append(i)
            new_posiciones = [p[0:3], p[3:6], p[6:9]]
            print("Posibles tableros: ")
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
    
            #2- Ganadores: que completan fila, columna o diagonal
            fila1 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == jugador and new_posiciones[0][2] == jugador
            fila2 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[1][2] == jugador
            fila3 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == jugador and new_posiciones[2][2] == jugador
    
            columna1 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == jugador and new_posiciones[2][0] == jugador
            columna2 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][1] == jugador
            columna3 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == jugador and new_posiciones[2][2] == jugador
    
            diagonal1 = new_posiciones[0][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][2] == jugador
            diagonal2 = new_posiciones[2][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[0][2] == jugador
    
            #3- Que evitan que el otro rellene su linea, columna o diagonal
            fila1_competidor = new_posiciones[0].count(competidor) == 2 and new_posiciones[0].count(jugador) == 1
            fila2_competidor = new_posiciones[1].count(competidor) == 2 and new_posiciones[1].count(jugador) == 1
            fila3_competidor = new_posiciones[2].count(competidor) == 2 and new_posiciones[2].count(jugador) == 1
    
            columna1_competidor = ((new_posiciones[0][0],new_posiciones[1][0],new_posiciones[2][0]).count(competidor) == 2 and (new_posiciones[0][0],new_posiciones[1][0],new_posiciones[2][0]).count(jugador) == 1)
            columna2_competidor = (new_posiciones[0][1],new_posiciones[1][1],new_posiciones[2][1]).count(competidor) == 2 and (new_posiciones[0][1],new_posiciones[1][1],new_posiciones[2][1]).count(jugador) == 1
            columna3_competidor = (new_posiciones[0][2],new_posiciones[1][2],new_posiciones[2][2]).count(competidor) == 2 and (new_posiciones[0][2],new_posiciones[1][2],new_posiciones[2][2]).count(jugador) == 1
            
            diagonal1_competidor = new_posiciones[0][0] == competidor and new_posiciones[1][1] == competidor and new_posiciones[2][2] == jugador
            diagonal2_competidor = new_posiciones[0][0] == jugador and new_posiciones[1][1] == competidor and new_posiciones[2][2] == competidor
            diagonal3_competidor = new_posiciones[0][0] == competidor and new_posiciones[1][1] == jugador and new_posiciones[2][2] == competidor
            
            diagonal4_competidor = new_posiciones[2][0] == competidor and new_posiciones[1][1] == competidor and new_posiciones[0][2] == jugador
            diagonal5_competidor = new_posiciones[2][0] == competidor and new_posiciones[1][1] == jugador and new_posiciones[0][2] == competidor
            diagonal6_competidor = new_posiciones[2][0] == jugador and new_posiciones[1][1] == competidor and new_posiciones[0][2] == competidor
            
            #Ocurrencias: 
    
            #1era caracteristica
            
            if fila1_v1 or fila1_v2 or fila2_v1 or fila2_v2 or fila3_v1 or fila3_v2 or columna1_v1 or columna1_v2 or columna2_v1 or columna2_v2 or columna3_v1 or columna3_v2:
                puntajes.append(ponderaciones[0]*caracteristicas[0]*sum([fila1_v1, fila1_v2, fila2_v1, fila2_v2, fila3_v1, fila3_v2, columna1_v1, columna1_v2, columna2_v1, columna2_v2, columna3_v1, columna3_v2]))
            else:
                puntajes.append(0)
            
            #2da caracteristica
        
            if fila1 or fila2 or fila3 or columna1 or columna2 or columna3 or diagonal1 or diagonal2:
                puntajes.append(ponderaciones[1]*caracteristicas[1]*sum([fila1, fila2, fila3, columna1, columna2, columna3, diagonal1, diagonal2]))
            else:
                puntajes.append(0)
            
            #3era caracteristica
            if fila1_competidor or fila2_competidor or fila3_competidor or columna1_competidor or columna2_competidor or columna3_competidor or diagonal1_competidor or diagonal2_competidor or diagonal3_competidor or diagonal4_competidor or diagonal5_competidor or diagonal6_competidor:
                puntajes.append(ponderaciones[2]*caracteristicas[2]*sum([fila1_competidor, fila2_competidor, fila3_competidor,columna1_competidor, columna2_competidor, columna3_competidor,
                                                                        diagonal1_competidor, diagonal2_competidor, diagonal3_competidor, diagonal4_competidor, diagonal5_competidor, diagonal6_competidor]))
            else:
                puntajes.append(0)
            
            print(puntajes)
            
            movimiento.append(sum(puntajes))
        #print(movimiento)
        #Aca no importa si el puntaje maximo se da en dos o mas tableros, nos quedamos con cualquiera de ellos de igual forma
        indice = mov[movimiento.index(max(movimiento))]
        lista = []
        for i in range(3):
            for j in range(3):
                lista.append((i+1,j+1))
        jugada = lista[indice]
        print("La jugada debe ser: ")
        return jugada


# In[13]:


agente.next_move(tablero = [['o', 'x', 'x'], ['o', '.', 'o'], ['x', 'o', '.']])


# ### Ta-te-ti 5x5
# 
# En el caso del tateti de 5x5 tenemos más características dado que ahora no nos va a interesar tener 2 cruces sino 2, 3 o 4 por fila, columna o diagonal. De esta manera las características serán:
# 
# * Filas/columnas/diagonales con 2 cruces y el resto de espacios vacíos.
# * Filas/columnas/diagonales con 3 cruces y el resto de espacios vacíos.
# * Filas/columnas/diagonales con 4 cruces y el resto de espacios vacíos.
# * Filas/columnas/diagonales con 5 cruces.
# * Evitar que el otro jugador rellene su línea.

# In[14]:


class algoritmo():
    def train():

        #Creamos la población inicial, con los distintos alfa para las distintas características
        num_alpha = 5 #Esto va a depender de la cantidad de características que definamos
        agents = 500 # Definimos el tamaño de la población.
        population_size = (agents,num_alpha) 
        new_population = np.random.uniform(low=0, high=5.0,size=population_size)

        #Creamos los puntajes correspondientes a las 3 características
        #1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
        #2- Ganadores: que completan fila, columna o diagonal
        #3- Que evitan que el otro rellene su linea
        caracteristicas = [10,20,30,60,50]
        
        #ALGORITMO GENETICO
        best_outputs = []
        num_generations = 15
        num_parents_mating = 100 #La cantidad de padres que son tomados de cada generación
        
        for generation in range(num_generations):
            print("Generacion: ", generation)
            fitness = fitness_v1(equation_inputs = caracteristicas, population = new_population)#,count_jugadas= nro_jugada)

            # Mostramos el mejor resultado
            best_outputs.append(np.max(fitness))
            print("Mejor resultado: ", np.max(fitness))
    
            # Usamos la función de selección ordenada
            parents = select_mating_random(new_population, fitness,num_parents_mating)

            # Realizamos el crossover.
            offspring_crossover = crossover(parents,offspring_size=(population_size[0]-parents.shape[0], num_alpha))

            # Cambiamos algunos genes por mutacion
            offspring_mutation = mutation(offspring_crossover)

            # Finalmente la población nueva que sucedera a la generacion previa va a ser una combinacion de los parents (los
            # mejores individuos) y del crossover/mutation.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
    
        # El mejor resultado
        fitness = fitness_v1(caracteristicas, new_population)#, nro_jugada)
        best_match_idx = np.where(fitness == np.max(fitness))

        #print("Best solution : ", new_population[best_match_idx, :])
        #print("Best solution fitness : ", fitness[best_match_idx])
        ponderaciones = new_population[best_match_idx, :][0][0]
        
        return ponderaciones


# In[15]:


class agente():
    def next_move(tablero):
        #TABLERO
        #Primero lo que voy a hacer es buscar el primer lugar vacio y completarlo con mi jugador. 
        #Luego guardo ese tablero y voy a generar un nuevo tablero con el segundo lugar vacio, asi de manera de tener todos los tableros existentes posibles en una jugada. 
        #Luego evaluo con mis caracteristicas a los tableros y me quedo con la mejor jugada.
        ponderaciones = algoritmo.train()
        print(ponderaciones)
        caracteristicas = [10,20,30,60,50]
        
        vacio = "."
        jugador = "x"
        competidor = "o"
        movimiento = []
        mov = []
        for i in range(0, 25):
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
                posiciones[i] = "x"
                p.append(posiciones)
                p = p[0]
            else:
                p = posiciones
            mov.append(i)
            new_posiciones = [p[0:5], p[5:10], p[10:15], p[15:20], p[20:25]]
            print("Posibles tableros: ")
            print(new_posiciones)
            
            #CARACTERISTICAS
            #Las calculamos en términos de columna, fila y diagonal. Vamos a contar: 2, 3 y 4 cruces y 5 como ganador.
            #Ademas contamos como caracteristica no permitir que el otro gane, cuando hay 4 circulos y una cruz.
            
            puntajes = []
                   
            #Filas
            occurences_1 = []
            occurences_2 = []
            occurences_3 = []
            occurences_4 = []
            occur_competidor = []
            for g in range(len(new_posiciones)):
                occurences_1.append(new_posiciones[g].count("x") == 2 and new_posiciones[g].count(".") == 3)
                occurences_2.append(new_posiciones[g].count("x") == 3 and new_posiciones[g].count(".") == 2)
                occurences_3.append(new_posiciones[g].count("x") == 4 and new_posiciones[g].count(".") == 1)
                occurences_4.append(new_posiciones[g].count("x") == 5)
                occur_competidor.append(new_posiciones[g].count("o") == 4 and new_posiciones[g].count("x") == 1)
            
            #Columnas
            grupos = []
            for r in range(len(new_posiciones)):
                g = []
                for l in range(5):
                    g.append(new_posiciones[l][r])
                grupos.append(g)
    
            for g in range(len(grupos)):
                occurences_1.append(grupos[g].count("x") == 2 and grupos[g].count(".") == 3)
                occurences_2.append(grupos[g].count("x") == 3 and grupos[g].count(".") == 2)
                occurences_3.append(grupos[g].count("x") == 4 and grupos[g].count(".") == 1)
                occurences_4.append(grupos[g].count("x") == 5)
                occur_competidor.append(grupos[g].count("o") == 4 and grupos[g].count("x") == 1)
                
            #Diagonal
            
            grupo_diag = []
            for i in range(5):
                grupo_diag.append(new_posiciones[i][i])
            occurences_1.append(grupo_diag.count("x") == 2 and grupo_diag.count(".") == 3)
            occurences_2.append(grupo_diag.count("x") == 3 and grupo_diag.count(".") == 2)
            occurences_3.append(grupo_diag.count("x") == 4 and grupo_diag.count(".") == 1)
            occurences_4.append(grupo_diag.count("x") == 5)
            occur_competidor.append(grupo_diag.count("o") == 4 and grupo_diag.count("x") == 1)
            
            grupo2_diag = new_posiciones[4][0], new_posiciones[3][1],new_posiciones[2][2], new_posiciones[1][3],new_posiciones[0][4]
            occurences_1.append(grupo2_diag.count("x") == 2 and grupo2_diag.count(".") == 3)
            occurences_2.append(grupo2_diag.count("x") == 3 and grupo2_diag.count(".") == 2)
            occurences_3.append(grupo2_diag.count("x") == 4 and grupo2_diag.count(".") == 1)
            occurences_4.append(grupo2_diag.count("x") == 5)
            occur_competidor.append(grupo2_diag.count("o") == 4 and grupo2_diag.count("x") == 1)
            
            #Ocurrencias: 
    
            #1era caracteristica: 2 cruces
            
            if (sum(occurences_1) > 0):
                puntajes.append(ponderaciones[0]*caracteristicas[0]*(sum(occurences_1)))
            else:
                puntajes.append(0)
                    
            #2da caracteristica: 3 cruces
            
            if (sum(occurences_2) > 0):
                puntajes.append(ponderaciones[1]*caracteristicas[1]*(sum(occurences_2)))
            else:
                puntajes.append(0)
                    
            #3era caracteristica: 4 cruces
            
            if (sum(occurences_3) > 0):
                puntajes.append(ponderaciones[2]*caracteristicas[2]*(sum(occurences_3)))
            else:
                puntajes.append(0)
                    
            #4ta caracteristica: Ganadores
            
            if (sum(occurences_4) > 0):
                puntajes.append(ponderaciones[3]*caracteristicas[3]*(sum(occurences_4)))
            else:
                puntajes.append(0)
                            
            #5ta caracteristica: no permitir que el otro gane
            if (sum(occur_competidor) > 0):
                puntajes.append(ponderaciones[4]*caracteristicas[4]*(sum(occur_competidor)))
            else:
                puntajes.append(0)
            
            print(puntajes)
            movimiento.append(sum(puntajes))
    
        #Aca no importa si el puntaje maximo se da en dos o mas tableros, nos quedamos con cualquiera de ellos de igual forma
        indice = mov[movimiento.index(max(movimiento))]
        lista = []
        for i in range(5):
            for j in range(5):
                lista.append((i+1,j+1))
        jugada = lista[indice]
        print("La jugada debe ser: ")
        return jugada


# In[16]:


agente.next_move(tablero = [['x', 'x', 'x', 'o', 'o'], ['x', '.', '.', 'o','.'], ['o', 'o', 'o', 'o','x'],
                                ['x', 'x', 'x', 'o', '.'],['o', '.', '.', '.', '.']])


# In[17]:


#BORRADORES ACA ABAJOOO


# In[18]:


#from itertools import permutations, cycle, islice
    
#idx = [list(islice(cycle([0,1, 2, 3,4]), i, i+4)) for i in range(len([0,1, 2, 3, 4]))][:-3]
#idx
#grupos = []
#for r in range(len(a)):
#    for i in idx:
#        grupos.append(a[r][i[0]:(i[2]+1)])
#print(grupos)
#occurences1 = []
#for g in range(len(a)):
#    occurences1.append(a[g].count("x") == 3 and a[g].count(".") == 2)
#    print(a[g])
#sum(occurences1)


# In[19]:


#grupos = []
#idx = [0,1,2,3,4]
#for r in range(len(a)):
#    g = []
#    for l in range(5):
#        g.append(a[l][r])
#    grupos.append(g)
       
#print(grupos)

#occurences2 = []
#for g in range(len(grupos)):
#    occurences2.append(grupos[g].count("x") == 4)
#sum(occurences2)


# In[20]:


#grupo2_diag = a[4][0], a[3][1],a[2][2], a[1][3],a[0][4]
#c = []
#c.append(grupo2_diag.count("x") == 4 and grupo2_diag.count(".") == 1)


# In[21]:


#grupo_diag = []
#c = []
#for i in range(5):
#    grupo_diag.append(a[i][i])
#c.append(grupo_diag.count("x") == 5)
#print(c)

