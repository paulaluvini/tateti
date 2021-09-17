#!/usr/bin/env python
# coding: utf-8

# <h1> <center>  Práctica Metaheurísticas </center> </h1> 
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
def fitness_v1(equation_inputs, population, count_jugadas):
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
def fitness_v2(equation_inputs, population, count_jugadas):
    #f = []
    #for i in np.arange(1,max_jugadas+1):
    #    f.append(np.log(i))
    #c = list(reversed(f))[count_jugadas-1]
    #caracteristica4 = []
    #for i in range(len(population)):
    #    caracteristica4.append(population[i][3])
    #caract_jugada = np.multiply(caracteristica4, c)
    #caracteristica2 = []
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

    parents_sort = np.empty((round(num_parents/2), pop.shape[1]))
    for parent_num in range(round(num_parents/2)):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents_sort[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999

    parents_rand = np.empty(((num_parents-round(num_parents/2)), pop.shape[1]))
    for parent_num2 in range(num_parents-round(num_parents/2)):
        random_index = np.random.choice(len(pop))
        parents_rand[parent_num2, :] = pop[random_index, :]
    
    parents[0:parents_sort.shape[0], :] = parents_sort
    parents[parents_sort.shape[0]:, :] = parents_rand
    return parents


# #### Crossover
# 
# What is crossover? Crossover is ‘the change of a single (0 or 1) or a group of genes (e.g. [1,0,1])’ occurred because of mating between two parent chromosomes. The new chromosome produced after crossover operation is called ‘offspring’. Following illustration explains crossover process. Always remember that crossover happens between parent chromosomes.
# 

# In[6]:


#Hacemos un crossover intercambiando características entre sí y de esta manera creamos nuevos individuos

def crossover(parents, offspring_size):
    #ACA VER: QUIERO SOLO QUEDARME CON LOS CROSSOVER O CON LOS ORIGINALES TAMBIEN? MEDIO QUE ACUMULARIA VARIABLES
    #offspring = np.empty((offspring_size[0]+parents.shape[0], offspring_size[1]))
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring


# #### Mutación

# In[7]:


#Una mutación intercambia una pequeña proporción de características del gen.

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover


# ### Población inicial
# 
# La población inicial van a ser distintos agentes cuya diferencia entre sí va a estar definida por los *alpha* que cada uno tenga. 

# In[8]:


#Creamos la población inicial, con los distintos alfa para las distintas características

num_alpha = 5 #Esto va a depender de la cantidad de características que definamos
agents = 15 # Definimos el tamaño de la población.
population_size = (agents,num_alpha) 
population_zero = np.random.uniform(low=0, high=5.0,size=population_size)
print(population_zero)

#Creamos los puntajes correspondientes a las 3 características
#1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
#2- Ganadores: que completan fila, columna o diagonal
#3- Que evitan que el otro rellene su linea
#4- Cantidad de jugadas del tablero
#5- Cantidad de grupos de 2 círculos consecutivas por fila/columna
caracteristicas = [10,50,50,15,-10]


# ### Corriendo el algoritmo

# In[9]:


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

# In[10]:


class agente():
    def next_move(tablero):

        #Creamos la población inicial, con los distintos alfa para las distintas características
        num_alpha = 3 #Esto va a depender de la cantidad de características que definamos
        agents = 15 # Definimos el tamaño de la población.
        population_size = (agents,num_alpha) 
        new_population = np.random.uniform(low=0, high=5.0,size=population_size)
        print(new_population)
        
        #Creamos los puntajes correspondientes a las 3 características
        #1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
        #2- Ganadores: que completan fila, columna o diagonal
        #3- Que evitan que el otro rellene su linea
        caracteristicas = [10,50,30]
        
        max_jugadas = 9 #Para la funcion fitness, tamaño del trablero
        
        #ALGORITMO GENETICO
        best_outputs = []
        num_generations = 15
        num_parents_mating = 2 #La cantidad de padres que son tomados de cada generación
        
        jugadas_viejas = []
        for a in tablero:
            for i in a:
                if i != ".":
                    jugadas_viejas.append(i)
        nro_jugada = len(jugadas_viejas)+1
        
        for generation in range(num_generations):
            print("Generacion: ", generation)
            fitness = fitness_v1(equation_inputs = caracteristicas, population = new_population,count_jugadas= nro_jugada)

            # Mostramos el mejor resultado
            best_outputs.append(np.max(fitness))
            print("Mejor resultado: ", np.max(fitness))
    
            # Usamos la función de selección ordenada
            parents = select_mating_random(new_population, fitness,num_parents_mating)

            # Generating next generation using crossover.
            offspring_crossover = crossover(parents,offspring_size=(population_size[0]-parents.shape[0], num_alpha))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = mutation(offspring_crossover, num_mutations=1)

            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation
    
        # Getting the best solution after iterating finishing all generations.
        #At first, the fitness is calculated for each solution in the final generation.
        fitness = fitness_v1(caracteristicas, new_population, nro_jugada)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = np.where(fitness == np.max(fitness))

        print("Best solution : ", new_population[best_match_idx, :])
        print("Best solution fitness : ", fitness[best_match_idx])
        ponderaciones = new_population[best_match_idx, :][0][0]
        
        #TABLERO
        #Primero lo que voy a hacer es buscar el primer lugar vacio y completarlo con mi jugador. 
        #Luego guardo ese tablero y voy a generar un nuevo tablero con el segundo lugar vacio, asi de manera de tener todos los tableros existentes posibles en una jugada. 
        #Luego evaluo con mis caracteristicas (hasta ahora puse 3) a los tableros y me quedo con la mejor jugada.

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
            print(diagonal1_competidor)
            print(diagonal2_competidor)
            print(diagonal3_competidor)
            print(diagonal4_competidor)
            print(diagonal5_competidor)
            print(diagonal6_competidor)
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


# In[11]:


agente.next_move(tablero = [['x', 'x', 'o'], ['o', 'o', '.'], ['x', '.', '.']])


# In[ ]:




