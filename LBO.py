## ========================================================================
#   Ladybug Beetle Optimization (LBO) algorithm
#
#   Developed in Python 3.9.7
#
#   Author and programmer: Saadat Safiri
#
#         e-Mail: saadatsafiri@gmail.com
#
#
#   Main paper:
#   "Ladybug Beetle Optimization algorithm: application for real‑world problems"
#
#   In order to use this optimization algorithm code, only change the
#   'sphare' function as you wish
# =========================================================================

from cmath import inf
import numpy as np
import random
import math

def run(problem, params):
    global NFE
    NFE=0

    # Problem Information
    costfunc = problem['costfunc']
    nvar = problem['nvar']
    varmin = problem['varmin']
    varmax = problem['varmax']

    # Parameters
    max_NFE= params['max_NFE']
    npop = params['npop']
    npop_init = npop
    beta = params['beta']
    sigma = params['sigma']

    # Best Solution Ever Found
    bestsol = {}
    bestsol['position'] = [0] * npop
    bestsol['cost']= np.inf

    # Initialize Population
    pop = []
    for i in range(npop):
        p = {}

        p['position'] = np.random.uniform(varmin, varmax, nvar).tolist()
        p['cost'] = costfunc(p['position'])
        pop.append(p)

        NFE +=1
        if pop[i]['cost'] < bestsol['cost']:
            bestsol['position'] = pop[i]['position']
            bestsol['cost'] = pop[i]['cost']

    # Best Cost of Iterations
    bestcost = []
    bestcost.append(bestsol['cost'])
    

    it = 0
    # Main Loop
    while NFE<max_NFE:
    # for it in range(maxit):
        it += 1
        costs = np.array([x['cost'] for x in pop])
        SoC = sum(costs)
        avg_cost = np.mean(costs)
        if avg_cost != 0:
            costs = costs/avg_cost
        probs = np.exp(-beta*costs)


        newSol = []

        for i in range(npop):
            new = {}
            new['cost'] = inf
            # Perform Roulette Wheel Selection
            j=0
            while (j<2 or j>npop):
                j=roulette_wheel_selection(probs)
            
            # Perform Crossover
            if random.random()>0.2:
                Rnd = np.random.random()-0.5
                new['position'] = (np.array(pop[j]['position'])+np.array((np.random.random(size=(1, nvar))*(np.array(pop[j]['position'])-np.array(pop[i]['position'])).tolist()).tolist()[0])+np.array((np.random.random(size=(1, nvar))*(np.array(pop[j-1]['position'])-np.array(pop[j]['position'])).tolist()).tolist()[0])+np.array(([((abs(pop[i]['cost']/SoC))**(-it/npop))*(Rnd)* el for el in ([element/varmax for element in pop[j]['position']])]))).tolist()
            else:
                new['position'] = mutate(pop[i], 0.05*nvar, sigma)
            
            new = apply_bound(new, varmin, varmax)
        
            new['cost'] = costfunc(new['position'])
            newSol.append(new)
            NFE +=1
            if new['cost'] < bestsol['cost']:
                bestsol['position'] = new['position']
                bestsol['cost'] = new['cost']

        npop = npop-0.1*np.random.random()*npop*(NFE/max_NFE)
        npop = round(npop)
        npop = max(math.floor(npop_init/4),npop)

        # Merge, Sort and Select
        pop += newSol
        pop = sorted(pop, key=lambda x: x['cost'])
        pop = pop[0:npop]

        # Store Best Cost
        bestcost.append(bestsol['cost'])

        # Show Iteration Information
        print("Iteration {}: Best Cost = {}".format(it, bestcost[it]))

    # Output
    out = {}
    out['pop'] = pop
    out['bestsol'] = bestsol
    out['bestcost'] = bestcost
    out['npop'] = npop
    return out

def mutate(x, mu, sigma):
    y = {}
    y['position'] = x['position']
    flag = np.random.rand(len(x['position'])) <= mu
    ind = np.argwhere(flag)
    index = [i[0] for i in ind]
    addingValue = [sigma*elmnt for elmnt in [i[0] for i in np.random.randn(*ind.shape).tolist()]]

    count=0
    for k in index:
        y['position'][k] += addingValue[count]
        count +=1

    return y['position']

def apply_bound(x, varmin, varmax):
    # print(x.position)
    x['position'] = np.maximum(x['position'], varmin).tolist()
    x['position'] = np.minimum(x['position'], varmax).tolist()
    return x

def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]
