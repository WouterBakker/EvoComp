
#General packages
import numpy as np
import tqdm

# imports framework
import sys, os
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller


#DEAP
import array
import random
import json

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

toolbox = base.Toolbox()

##

experiment_name = "MuCommaLambda1"
n_hidden_neurons = 10
n_enemy = [6]

if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)
    
headless = True

if headless:
    test_the_best = False
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    
else:
  test_the_best = True


# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name= experiment_name,
                  enemies=n_enemy,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  logs="off",
                  randomini = "yes")
                  



# Simulates a game for individual x
def simulation(env,individual):
    fitness,_player,_enemy,_time = env.play(pcont= np.array(individual))
    return fitness

# Runs a simulated game for each individual, calculates fitness
def evaluate(individual):
    #sim = simulation(env,individual)
    return [simulation(env,individual)]
  
  

#Run a test with the best 
# # loads file with the best solution for testing
if test_the_best:
  
    best_solution = np.loadtxt(experiment_name+'/best.txt')
    print( '\n Test the Best Saved Solution \n')
    env.update_parameter('speed','normal')
    evaluate(best_solution)
    sys.exit(0)

### Important parameters

#probability of two individuals mating
p_mate = 0.2
p_mutate = 0.2
generations = 30

#population size
npop = 80

#Lower/upper bound for output value
low = -1
high = 1

#Individual size. ----Note: 20 sensors in total
# Number of weights for multilayer with x hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

#DEAP init

#Initial population generation
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("indices", np.random.uniform, low = low, high = high, size = n_vars)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

# make a random number of individuals by calling toolbox.population()
toolbox.register("population", tools.initRepeat, list, toolbox.individual, n = npop)

#Add evaluate function to the DEAP toolbox, calculates fitness
toolbox.register("evaluate", evaluate)
#Mate/crossover parameters
toolbox.register("mate", tools.cxOnePoint)
#Mutation: generate variation in individual's 'DNA' with probability indpb
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
#Selection: determines how best individuals are selected
toolbox.register("select", tools.selTournament, tournsize=2)

mu = 30
lambda_ = 60

def main():

    pop = toolbox.population()
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    #algorithms.eaSimple(pop, toolbox, p_mate, p_mutate, generations, stats = stats, halloffame=hof)
    #algorithms.eaMuPlusLambda(pop, toolbox, mu, lambda_, p_mate, p_mutate, generations, stats = stats, halloffame=hof)
    algorithms.eaMuCommaLambda(pop, toolbox, mu, lambda_, p_mate, p_mutate, generations, stats = stats, halloffame=hof)

    
    np.savetxt(experiment_name+'/best.txt',hof[:])
    
    return pop, stats, hof

if __name__ == "__main__":
    main()





