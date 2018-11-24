import operator
import math
import random
import matplotlib.pyplot as plt

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import arff



def protectedDiv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 1


def split_data(data, split_percentage):
    x = int(len(data)*split_percentage)
    return data[0:x], data[x:]

def get_primitive_set_china():
    pset = gp.PrimitiveSet("MAIN", 15)   # number of inputs should be length of variables
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)

    pset.addEphemeralConstant("rand101", lambda: random.randint(-100, 100)) #do i need????
    pset.renameArguments(ARG0='AFP') #do i need????
    pset.renameArguments(ARG1='input')
    pset.renameArguments(ARG2='output')
    pset.renameArguments(ARG3='enquiry')
    pset.renameArguments(ARG4='file')
    pset.renameArguments(ARG5='interface')
    pset.renameArguments(ARG6='added')
    pset.renameArguments(ARG7='changed')
    pset.renameArguments(ARG8='deleted')
    pset.renameArguments(ARG9='PDR_AFP')
    pset.renameArguments(ARG10='PDR_UFP')
    pset.renameArguments(ARG11='NPDR_UFP')
    pset.renameArguments(ARG12='Resource')
    pset.renameArguments(ARG13='devtype')
    pset.renameArguments(ARG14='duration')
    return pset

def get_primitive_set_albrecht():
    pset = gp.PrimitiveSet("MAIN", 7)   # number of inputs should be length of variables
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)

    pset.addEphemeralConstant("rand101", lambda: random.randint(-100, 100))
    pset.renameArguments(ARG0='Input')
    pset.renameArguments(ARG1='Output')
    pset.renameArguments(ARG2='Inquiry')
    pset.renameArguments(ARG3='File')
    pset.renameArguments(ARG4='FPAdj')
    pset.renameArguments(ARG5='RawFPcounts')
    pset.renameArguments(ARG6='AdjFP')

    return pset

def get_primitive_set_miyazaki():
    pset = gp.PrimitiveSet("MAIN", 7)   # number of inputs should be length of variables
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)

    pset.addEphemeralConstant("rand101", lambda: random.randint(-100, 100))
    pset.renameArguments(ARG0='KLOC')
    pset.renameArguments(ARG1='SCRN')
    pset.renameArguments(ARG2='FORM')
    pset.renameArguments(ARG3='FILE')
    pset.renameArguments(ARG4='ESCRN')
    pset.renameArguments(ARG5='EFORM')
    pset.renameArguments(ARG6='EFILE')

    return pset

def get_fitness_function_china(toolbox):

    def evaluate(data, individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # randomly select some samples
        #fitness =  mean error for samples
        # r_samples_size = 30
        # r_samples = random.sample(data, r_samples_size)
        total_percentage_error = 0
        for s in data:
            total_percentage_error += 100* abs(func(s[1], s[2], s[3], s[4], s[5], s[6], s[7], \
                    s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15]) - s[-1]) / s[-1]

        return  (total_percentage_error/len(data),)

    return evaluate

def get_fitness_function_albrecht(toolbox):

    def evaluate(data, individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # randomly select some samples
        #fitness =  mean error for samples
        #r_samples_size = 30
        #r_samples = random.sample(data, r_samples_size)
        total_percentage_error = 0
        for s in data:
            total_percentage_error += 100*abs(func(s[0], s[1], s[2], s[3], s[4], s[5], s[6]) - s[-1]) / s[-1]

        return  (total_percentage_error/len(data),)

    return evaluate

def get_fitness_function_miyazaki(toolbox):

    def evaluate(data, individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # randomly select some samples
        #fitness =  mean error for samples
        #r_samples_size = 30
        #r_samples = random.sample(data, r_samples_size)
        total_percentage_error = 0
        for s in data:
            total_percentage_error += 100*abs(func(s[1], s[2], s[3], s[4], s[5], s[6], s[7]) - s[-1]) / s[-1]

        return  (total_percentage_error/len(data),)

    return evaluate

def main():
    data = arff.load(open('Data/miyazaki94.arff'))
    # variables  = data
    training_data, test_data = split_data(data['data'], 0.75)
    pset = get_primitive_set_miyazaki()
    print()

    creator.create("FitnessMin" , base.Fitness, weights=(-1.0,))  # -1.0 as its a minimise function
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    fit_func = get_fitness_function_miyazaki(toolbox)
    toolbox.register("evaluate", fit_func, training_data)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=5))


    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    no_generations = 100
    no_runs = 1
    eaSimple_data = np.zeros((no_generations, no_runs))
    #for j in range(no_runs):
    #    for i in range(no_generations):
    #        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 1, stats=mstats,
    #                               halloffame=hof, verbose=True)
    #        x = fit_func(test_data, hof.items[0])[0]
    #        eaSimple_data[i, j] = x
    #    print(j)

    random.seed(318)
    pop = toolbox.population(n=300)
    aMuPlusLambda_data = np.zeros((no_generations, no_runs))
    for j in range(no_runs):
        for i in range (no_generations):
            pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 100, 300, 0.25, 0.75, 1, stats=mstats, \
                                   halloffame=hof, verbose=False)
            eaSimple_data[i,j] = hof.items[0].fitness.values[0]
            aMuPlusLambda_data[i,j] = fit_func(test_data, hof.items[0])[0]
        print(j)
        
    plt.plot(range(no_generations), np.mean(aMuPlusLambda_data, axis=1), color='b') #test
    plt.plot(range(no_generations), np.mean(eaSimple_data, axis=1), color='r') # train

    plt.show()
    # print log
    #print(str(hof.items[0]) + "     " + str(hof.items[0].fitness))
    #print(fit_func(test_data, hof.items[0]))
    
    return pop, log, hof

if __name__ == "__main__":
    main()