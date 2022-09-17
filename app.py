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

import matplotlib.pyplot as plt
import LBO

global NFE
NFE = 0
# Cost Function
def sphere(x):
    return sum([ elmnt**2 for elmnt in x ]) # Sphare

# Problem Definition
problem = {}
problem['costfunc'] = sphere
problem['nvar'] = 30
problem['varmin'] = -100
problem['varmax'] =  100

# GA Parameters
params = {}
params['max_NFE'] = 1e4
params['npop'] = 200
params['beta'] = 8
params['sigma'] = 0.05

# Run GA
out = LBO.run(problem, params)

# Results
plt.plot(out['bestcost'])
plt.xlabel('NFE')
plt.ylabel('Best Cost')
plt.title('LBO algorithm')
plt.grid(True)
plt.show()

