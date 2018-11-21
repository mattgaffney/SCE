import operator
import math
import random

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

import arff



pset = gp.PrimitiveSet("MAIN", 1) # number of argument should be length of variables??  list of variables?
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1)) #do i need????
pset.renameArguments(ARG0='x') #do i need????

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


#
def evaluate(individual, variables, effort):
    # randomly select a number of entries

    # compile tree and plug in variables
    func = toolbox.compile(expr=individual)
    #fitness =(sum(func(variables[i]) - efffort[i])**2)**0.5

    return (fitness) # must be tuple


toolbox.register("evaluate", evaluate, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)


def main():
    data = arff.load(open('Data/china.arff'))
    ##seperate data into variables and effort

    ## run population as in example
    print()


if __name__ == "__main__":
    main()