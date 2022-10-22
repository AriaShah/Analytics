
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
df = pd.read_csv('C:/Users/HP/Desktop/NO-Inputs.csv')
# df.dropna(axis=0,how='any',subset=['population'],inplace=True)
df1 = pd.read_csv('C:/Users/HP/Desktop/ir2.csv')

# Variable with the Longitude and Latitude
X_weighted = df.loc[:, ['Latitude', 'Longitude', 'Assign']]

# X_weighted.head(4)

ProdN = 3  # Number of products

ProdW = [1, 1, 1]  # Products weights

FactN = 0  # Number of factories
FactNN = max(1, FactN)

ctr = 0.05  # Transportation costs per weight unit

ch = 0.01  # Holding costs per weight unit

TDCN = 10  # Total number of DCs

DCN = 10  # Selected number of DCs

lat_long = np.array(X_weighted[X_weighted.columns[0:2]])
finalind = np.array(X_weighted[X_weighted.columns[2]])

Fact_lat_long = lat_long[0:FactN, :]
DC_lat_long = lat_long[FactN:FactN + TDCN, :]
Cust_lat_long = lat_long[FactN + TDCN:, :]

Timehorizon = 31

InitialInv = np.zeros(shape=(TDCN, ProdN, Timehorizon))
# InitialInv += 1000
# InitialInv [:,:,0] += 1000

SafetyStock = np.zeros(shape=(TDCN, ProdN, Timehorizon))
SafetyStock += 100000

CustN = len(Cust_lat_long)  # Number of customers

Demands = np.array(df1[df1.columns[9:102]])  # Demands separated by time units

Demands = np.array(np.hsplit(Demands, ProdN))  # Deviding demands per products

ind1 = finalind[FactN:TDCN]
ind2 = finalind[FactN + TDCN:]
a11 = np.zeros((TDCN, FactN), dtype=np.uint8)
a12 = np.zeros((CustN, TDCN), dtype=np.uint8)

for c in range(CustN):
    for dc in range(TDCN):
        if ind2[c] == dc:
            a12[c][dc] = 1

for dc in range(TDCN):
    for fact in range(FactN):
        if ind1[dc] == fact:
            a11[dc][fact] = 1

dist_cust_dc = np.zeros(shape=(CustN, TDCN))
for c in range(CustN):
    for dc in range(TDCN):
        dist_cust_dc[c][dc] = mpu.haversine_distance(Cust_lat_long[c, :], DC_lat_long[dc, :])

dist_cust_dc1 = copy.copy(dist_cust_dc)
argminMatrix = np.zeros(shape=(CustN, TDCN))
for c in range(CustN):
    for dc in range(TDCN):
        argminC = np.argmin(dist_cust_dc1[c, :])
        argminMatrix[c][dc] = argminC
        dist_cust_dc1[c][argminC] = math.inf

dist_dc_fact = np.zeros(shape=(TDCN, FactNN))
for dc in range(TDCN):
    for fact in range(FactNN):
        if FactN == 0:
            dist_dc_fact[dc][fact] = 0
        else:
            dist_dc_fact[dc][fact] = mpu.haversine_distance(DC_lat_long[dc, :], Fact_lat_long[fact, :])


def AsgnFunc(ind):
    AsgnCustDC = np.zeros(shape=(CustN, TDCN))
    for c in range(CustN):
        for dc in range(TDCN):
            Mindist = int(argminMatrix[c][dc])
            if Mindist in ind:
                AsgnCustDC[c][Mindist] = 1
                break

    AsgnDCFact = np.zeros(shape=(TDCN, FactNN))
    for dc in ind:
        for fact in range(FactNN):
            if fact == np.argmin(dist_dc_fact[dc, :]):
                AsgnDCFact[dc][fact] = 1
                break

    return [AsgnDCFact, AsgnCustDC]


def Fact_DC_Cust_Prod_Time(ind):
    FDPCT = np.zeros(shape=(FactNN, TDCN, CustN, ProdN, Timehorizon))
    FDPCT_DC_cust_Tr = np.zeros(shape=(FactNN, TDCN, CustN, ProdN, Timehorizon))
    FDPCT_fact_DC_Tr = np.zeros(shape=(FactNN, TDCN, CustN, ProdN, Timehorizon))
    InitialInv_Tr = np.zeros(shape=(FactNN, TDCN, ProdN, Timehorizon))
    for fact in range(FactNN):
        for dc in range(TDCN):
            if ind[0][dc][fact] == 1:
                for c in range(CustN):
                    if ind[1][c][dc] == 1:
                        for p in range(ProdN):
                            for t in range(Timehorizon):
                                FDPCT[fact][dc][c][p][t] = Demands[p][c][
                                    t]  # Demand per customer per product per time per DC
                                FDPCT_DC_cust_Tr[fact][dc][c][p][t] = FDPCT[fact][dc][c][p][t] * dist_cust_dc[c][
                                    dc] * ctr
                                FDPCT_fact_DC_Tr[fact][dc][c][p][t] = FDPCT[fact][dc][c][p][t] * dist_dc_fact[dc][
                                    fact] * ch
                                if t == 0:
                                    InitialInv_Tr[fact][dc][p][t] = InitialInv[dc][p][t] * dist_dc_fact[dc][fact] * ch

    return FDPCT, FDPCT_DC_cust_Tr, FDPCT_fact_DC_Tr, InitialInv_Tr


# This function finds the coordinates of the DCs given by GA
def DCLocations(ind):
    DClocs = []  # Location of the selected DCs in ind
    [DClocs.append(DC_lat_long[i]) for i in ind]
    return DClocs


# This function calculates total carrying costs
def DCTotalInv_cost(ind, FDCPT):
    InvLeft = np.zeros(shape=(TDCN, ProdN, Timehorizon))
    InvUsed = np.zeros(shape=(TDCN, ProdN, Timehorizon))
    for dc in range(TDCN):
        if dc in ind:
            for p in range(ProdN):
                for t in range(Timehorizon):
                    InvLeft[dc][p][t] = InvLeft[dc][p][t - 1] + InitialInv[dc][p][t - 1] + SafetyStock[dc][p][t] - \
                                        np.sum(FDCPT, axis=(0, 2))[dc][p][t]
                    InvUsed[dc][p][t] = np.sum(FDCPT, axis=(0, 2))[dc][p][t]
    return InvLeft * ch, (InvUsed / 2) * ch


# This function evaluates the sum of transportation and holding and site opening costs
def evalcost(ind):
    TC1, TC2, TC3, TC4 = Fact_DC_Cust_Prod_Time(AsgnFunc(ind))
    TC = TC2.sum() + TC3.sum() + TC4.sum()  # Total transportation costs
    TC5, TC6 = DCTotalInv_cost(ind, TC1)
    HC = TC5.sum() + TC6.sum()  # Total holding costs

    Totalc = TC + HC

    return Totalc,


def evalcost1(ind):
    TC1, TC2, TC3, TC4 = Fact_DC_Cust_Prod_Time(AsgnFunc(ind))
    TC = TC2.sum() + TC3.sum() + TC4.sum()  # Total transportation costs
    TC5, TC6 = DCTotalInv_cost(ind, TC1)
    HC = TC5.sum() + TC6.sum()  # Total holding costs

    # Totalc = TC+HC

    return TC, HC,


# This function delets duplicates in a list
def childCORRECTION(ind):
    # Remove duplicates
    i = 0
    while i in range(len(ind)):
        if ind[i] in ind[:i]:
            del ind[i]
        else:
            i += 1
    # [ind.remove(i) for i in ind if ind.count(i) > 1] # remove duplicates

    # Add new attr instead of removed ones
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
pop = toolbox.population(n=5)
CXPB, MUTPB, NGEN = 0.9, 0.5, 5

# Evaluate the entire population
fitnesses = map(toolbox.evaluate, pop)
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

FinalObj = []
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
    FinalSol.append(pop[fits.index(min(fits))])

end = time.time()
print(end - start)

print(f"Best objective: {min(FinalObj)}")

GAplot = [i for i in range(1, NGEN + 1)]
plt.plot(GAplot, FinalObj)
plt.xlabel('Number of Generations', fontsize=18, fontweight='bold')
plt.ylabel('Total Cost', fontsize=18, fontweight='bold')
plt.title('GA Objective', fontsize=18, fontweight='bold')
plt.show()







