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
    
            #2- Ganadores: que completan fila, columna o diagonal
            fila1 = new_posiciones[0][0] == jugador and new_posiciones[0][1] == jugador and new_posiciones[0][2] == jugador
            fila2 = new_posiciones[1][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[1][2] == jugador
            fila3 = new_posiciones[2][0] == jugador and new_posiciones[2][1] == jugador and new_posiciones[2][2] == jugador
    
            columna1 = new_posiciones[0][0] == jugador and new_posiciones[1][0] == jugador and new_posiciones[2][0] == jugador
            columna2 = new_posiciones[0][1] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][1] == jugador
            columna3 = new_posiciones[0][2] == jugador and new_posiciones[1][2] == jugador and new_posiciones[2][2] == jugador
    
            diagonal1 = new_posiciones[0][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[2][2] == jugador
            diagonal2 = new_posiciones[2][0] == jugador and new_posiciones[1][1] == jugador and new_posiciones[0][2] == jugador
    
            #3- Que evitan que el otro rellene su linea
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


agente.next_move(tablero = [['x', '.', 'o'], ['.', 'x', '.'], ['.', '.', 'o']], competidor = "o", jugador = "x", 
                 ponderaciones = [1,1,1])


# ## Algoritmo Genético
# 
# Acá voy a desarrollar un algoritmo genético basado en el que encontramos acá: https://github.com/ahmedfgad/GeneticAlgorithmPython/tree/master/Tutorial%20Project
# 
# En un segundo paso deberíamos crear más funciones de fitness, agregarle criterios de selección, etc. como la consigna requiere.

# #### Población inicial

# In[4]:


num_alpha = 3
sol_per_pop = 100
#Creamos la población inicial, con los distintos alfa para las distintas características
# Definimos el tamaño de la población.
pop_size = (sol_per_pop,num_alpha) 
new_population = np.random.uniform(low=-4.0, high=4.0,size=pop_size)
new_population


# In[5]:


#Creamos los puntajes correspondientes a las 3 características
#1- Cantidad de grupos de 2 cruces consecutivas #Dos por fila/columna
#2- Ganadores: que completan fila, columna o diagonal
#3- Que evitan que el otro rellene su linea
equation_inputs = [10,50,30]


# ### Definición de Funciones

# #### Fitness
# 
# En primer lugar vamos a definir las distintas funciones de fitness a utilizar. Comenzaremos por tomar a la función objetivo como primer opción.

# In[6]:


def pop_fitness(equation_inputs, population):
    fitness = np.sum(population*equation_inputs, axis=1)
    return fitness


# In[7]:


fitness = pop_fitness(equation_inputs,new_population)
fitness


# #### De selección

# In[8]:


# A continuación elegimos a la cantidad que queremos de individuos de la generación actual para que sean "padres" de la siguiente.
# Estos van a ser los individuos con mayor puntaje en el fitness calculado en la función anterior.

def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents


# In[9]:


parents = select_mating_pool(new_population, fitness, 4)
parents


# #### Crossover

# In[10]:


#Hacemos un crossover intercambiando características entre sí y de esta manera creamos nuevos individuos

def crossover(parents, offspring_size):
    #ACA VER: QUIERO SOLO QUEDARME CON LOS CROSSOVER O CON LOS ORIGINALES TAMBIEN? MEDIO QUE ACUMULARIA VARIABLES
    #offspring = np.empty((offspring_size[0]+parents.shape[0], offspring_size[1]))
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
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


# In[11]:


offspring_crossover = crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_alpha))
offspring_crossover


# In[12]:


#a = crossover(parents =  parents, offspring_size=(3, num_alpha))
#for k in range(offspring_size[0], parents.shape[0]+offspring_size[0]):
#    a[k] = parents[k-offspring_size[0]]

#print(a)
#print(parents)


# #### Mutación

# In[13]:


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


# In[14]:


mutations_counter = np.uint8(offspring_crossover.shape[1] / 1)
mutations_counter
for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(1):
            # The random value to be added to the gene.
            random_value = np.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
        
offspring_crossover


# ### Corriendo el algoritmo

# In[15]:


best_outputs = []
num_generations = 15
num_parents_mating = 4 #La cantidad de padres que son tomados de cada generación

for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.
    fitness = pop_fitness(equation_inputs, new_population)
    print("Fitness")
    print(fitness)

    best_outputs.append(np.max(np.sum(new_population*equation_inputs, axis=1)))
    # The best result in the current iteration.
    print("Best result : ", np.max(np.sum(new_population*equation_inputs, axis=1)))
    
    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness,num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_alpha))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations=1)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
fitness = pop_fitness(equation_inputs, new_population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.where(fitness == np.max(fitness))

print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])


# In[16]:


agente.next_move(tablero = [['x', '.', 'o'], ['.', 'x', '.'], ['x', 'x', '.']], competidor = "o", jugador = "x", 
                 ponderaciones = new_population[best_match_idx, :][0][0])


# In[ ]:




