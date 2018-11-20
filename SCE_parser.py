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



pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
#pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1)) #do i need????
pset.renameArguments(ARG0='x') #do i need????



def main():
    data = arff.load(open('Data/china.arff', 'rb'))
    print()


if __name__ == "__main__":
    main()