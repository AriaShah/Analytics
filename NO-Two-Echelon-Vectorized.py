
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
from deap import base
from deap import creator
from deap import tools
import warnings
import mpu
import math
import time

start = time.time()
warnings.filterwarnings("ignore")

# Read data
# Read locations
df = pd.read_csv('C:/Users/HP/Desktop/DataIran.csv')
# Read demands
df1 = pd.read_csv('C:/Users/HP/Desktop/ir2.csv')
# Read transportation costs between customers and DCs
ctr1 = np.array(pd.read_csv('C:/Users/HP/Desktop/ctr-c-dc.csv'))
# Read transportation costs between DCs and factories
ctr2 = np.array(pd.read_csv('C:/Users/HP/Desktop/ctr-dc-f.csv'))
# Read carrying costs
ch = np.array(pd.read_csv('C:/Users/HP/Desktop/ch.csv'))

# Variable with the Longitude and Latitude
X_weighted = df.loc[:, ['Latitude', 'Longitude']]

ProdN = 3  # Number of products
ProdW = [1, 1, 1]  # Products weights
FactN = 2  # Number of factories
TDCN = 10  # Total number of DCs
DCN = 10  # Selected number of DCs
TimeN = 31

# Separate lat & lng of factories, DCs, customers
lat_long = np.array(X_weighted[X_weighted.columns[0:2]])
Fact_lat_long = lat_long[0:FactN, :]
DC_lat_long = lat_long[FactN:FactN + TDCN, :]
Cust_lat_long = lat_long[FactN + TDCN:, :]

InitialInv = np.zeros(shape=(TDCN, ProdN, TimeN))
InitialInv[:, :, 0] += 1000 # Initial Inventory

SafetyStock = np.zeros(shape=(TDCN, ProdN, TimeN))
SafetyStock[:, :, 0] += 1000 # Safety Stock

CustN = len(Cust_lat_long)  # Number of customers

# Separate Demands
Demands = np.array(df1[df1.columns[9:102]])  # Demands separated by time units
Demands = np.array(np.hsplit(Demands, ProdN))  # Dividing demands per products
D = np.zeros(shape=(CustN,ProdN,TimeN))  # Demands per customer per product per time
for c in range(CustN):
    for p in range(ProdN):
        D[c][p] = Demands[p][c]

# Distance between customers and DCs
dist_cust_dc = np.zeros(shape=(CustN, TDCN))
for c in range(CustN):
    for dc in range(TDCN):
        dist_cust_dc[c][dc] = mpu.haversine_distance(Cust_lat_long[c, :], DC_lat_long[dc, :])

# Distance between DCs and factories
dist_dc_fact = np.zeros(shape=(TDCN, FactN))
for dc in range(TDCN):
    for fact in range(FactN):
        dist_dc_fact[dc][fact] = mpu.haversine_distance(DC_lat_long[dc, :], Fact_lat_long[fact, :])

# Sort DCs for customers based on min distance*transportation costs
distTr_cust_dc = ctr1 * dist_cust_dc
argmindistTrCDC = np.zeros(shape=(CustN, TDCN))
for c in range(CustN):
    for dc in range(TDCN):
        argminC = np.argmin(distTr_cust_dc[c, :])
        argmindistTrCDC[c][dc] = argminC
        distTr_cust_dc[c][argminC] = math.inf

# Carrying costs between customers and DCs
ch_cust_dc = np.zeros(shape=(CustN, TDCN))
for c in range(CustN):
    for dc in range(TDCN):
        ch_cust_dc[c][dc] = sum(np.sum(D,axis=2) [c] * ch [dc,:])

# Sort DCs for customers based on min carrying costs
ch_cust_dc1 = copy.copy(ch_cust_dc)
argminchCDC = np.zeros(shape=(CustN, TDCN))
for c in range(CustN):
    for dc in range(TDCN):
        argminC = np.argmin(ch_cust_dc1[c, :])
        argminchCDC[c][dc] = argminC
        ch_cust_dc1[c][argminC] = math.inf

# This function  assigns flows and calculates total carrying costs
def AsgnFlow(ind):

    if np.random.rand() <= 0.85:
        argminSelect = argmindistTrCDC
    else:
        argminSelect = argminchCDC
    FDCPT = np.zeros(shape=(FactN, TDCN, CustN, ProdN, TimeN))
    DC_cust_Tr = np.zeros(shape=(FactN, TDCN, CustN, ProdN, TimeN))
    fact_DC_Tr = np.zeros(shape=(FactN, TDCN, CustN, ProdN, TimeN))
    InitialInv_Tr = np.zeros(shape=(FactN, TDCN, ProdN, TimeN))
    for c in range(CustN):
        for dc in range(TDCN):
            dd = int(argminSelect[c][dc])
            if dd in ind:
                f = np.argmin(dist_dc_fact[dd,:])
                FDCPT [f][dd][c] = D [c]
                DC_cust_Tr [f][dd][c] = FDCPT [f][dd][c] * dist_cust_dc[c][dd] * ctr1 [c][dd]
                fact_DC_Tr [f][dd] = FDCPT [f][dd] * dist_dc_fact[dd][f] * ctr2 [dd][f]
                InitialInv_Tr [f][dd] = InitialInv[dd] * dist_dc_fact[dd][f] * ctr2 [dd][f]
                break
    return FDCPT, DC_cust_Tr, fact_DC_Tr, InitialInv_Tr

# This function calculates total carrying costs
def CarryingCost(ind, FDCPT):
    invLeft = np.zeros(shape=(TDCN, ProdN, TimeN))
    invLeftCost = np.zeros(shape=(TDCN, ProdN, TimeN))
    invUsed = np.sum(FDCPT, axis=(0, 2))/2
    invUsedCost = np.zeros(shape=(TDCN, ProdN, TimeN))
    invLeft += SafetyStock
    for t in range(TimeN):
        if t == 0:
            invLeftCost[...,t] = invLeft[...,t] * ch/2
            invUsedCost[...,t] = invUsed[...,t] * ch
        else:
            invLeft[...,t] = invLeft[...,t-1] + InitialInv[...,t-1]
            invLeftCost[...,t] = invLeft[...,t] * ch/2
            invUsedCost[..., t] = invUsed[..., t] * ch
    notinind = list(set(range(TDCN)).difference(set(ind)))
    invLeft [notinind,...] = 0
    invLeftCost [notinind,...] = 0
    return invLeft, invLeftCost, invUsed, invUsedCost


# This function evaluates the sum of transportation and holding and site opening costs
def evalcost(ind):

    fdcpt, TC1, TC2, TC3 = AsgnFlow(ind)
    TC = TC1.sum() + TC2.sum() + TC3.sum()  # Total transportation costs
    HC1, HC2, HC3, HC4 = CarryingCost(ind, fdcpt)
    HC = HC2.sum() #+ HC4.sum()  # Total carrying costs

    Totalc = TC + HC

    return Totalc,


def evalcost1(ind):
    fdcpt, TC1, TC2, TC3 = AsgnFlow(ind)
    TC = TC1.sum() + TC2.sum() + TC3.sum()  # Total transportation costs
    HC1, HC2, HC3, HC4 = CarryingCost(ind, fdcpt)
    HC = HC2.sum() #+ HC4.sum()  # Total holding costs

    Totalc = TC + HC
    return fdcpt, TC1, TC2, TC3, HC,


# This function corrects infeasible individuals after crossover or mutation
def childCORRECTION(ind):
    i = 0
    while i in range(len(ind)):
        if ind[i] in ind[:i]:
            del ind[i]
        else:
            i += 1
    # [ind.remove(i) for i in ind if ind.count(i) > 1] # remove duplicates
    permchild = list(range(TDCN))
    random.shuffle(permchild)
    [ind.append(j) for j in permchild if len(ind) < DCN if j not in ind]  # add new items instead of removed duplicates

    return ind


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr", random.sample, range(TDCN), DCN)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalcost)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("CORRECTION", childCORRECTION)
toolbox.register("MutUnif", tools.mutUniformInt)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA algorithm
pop = toolbox.population(n=40)
CXPB, MUTPB, NGEN = 0.9, 0.5, 40

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

FinalObj = []
BestObjSave = []
FinalSol = []

for g in range(NGEN):

    print("-- Generation %i --" % g)
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            toolbox.CORRECTION(child1)
            toolbox.CORRECTION(child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.MutUnif(mutant, 0, TDCN - 1, MUTPB)
            toolbox.CORRECTION(mutant)
            del mutant.fitness.values
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    # Gather all the fitnesses in one list and print the stats
    fits = [ind.fitness.values[0] for ind in pop]

    FinalObj.append(min(fits))
    BestObjSave.append(min(FinalObj))
    FinalSol.append(pop[fits.index(min(fits))])

end = time.time()
print(end - start)
print(f"Best objective: {min(FinalObj)}")

GAplot = [i for i in range(1, NGEN + 1)]
plt.plot(GAplot, BestObjSave)
plt.xlabel('Number of Generations', fontsize=18, fontweight='bold')
plt.ylabel('Total Cost', fontsize=18, fontweight='bold')
plt.title('GA Objective', fontsize=18, fontweight='bold')
plt.show()







