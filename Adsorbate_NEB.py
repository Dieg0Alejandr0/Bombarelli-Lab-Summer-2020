import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import pi
import argparse
import NEB_Methods
from NEB_Methods import AdsorbateString, AdsorbateNEB, plotPerformance

from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc100, add_adsorbate
from ase.constraints import FixAtoms
from ase.visualize import view
from ase.io import read, write

# 2x2-Al(001) surface with 3 layers and an
# Au atom adsorbed in a hollow site:
slab = fcc100('Al', size=(2, 2, 3))
add_adsorbate(slab, 'Au', 1.7, 'hollow')
slab.center(axis=2, vacuum=4.0)

# Make sure the structure is correct:
view(slab, viewer="x3d")

# Fix second and third layers:
mask = [atom.tag > 1 for atom in slab]
#print(mask)
slab.set_constraint(FixAtoms(mask=mask))

# Use EMT potential:
slab.set_calculator(EMT())

# Initial state:
qn = QuasiNewton(slab, trajectory='initial.traj')
qn.run(fmax=0.05)

# Final state:
slab[-1].x += slab.get_cell()[0, 0] / 2
qn = QuasiNewton(slab, trajectory='final.traj')
qn.run(fmax=0.05)

initial = read('initial.traj')
initial.set_calculator(EMT())
final = read('final.traj')
final.set_calculator(EMT())
saddle = torch.FloatTensor( [  0.0, 0.0, 4.0, 2.8637824058532715, 0.0, 4.0, 0.0, 2.8637824058532715, 4.0, 2.8637824058532715, 
            2.8637824058532715, 4.0, 1.4318912029266357, 1.4318912029266357, 6.025000095367432, 4.295673847198486, 
            1.4318912029266357, 6.025000095367432, 1.4318912029266357, 4.295673847198486, 6.025000095367432, 
            4.295673847198486, 4.295673847198486, 6.025000095367432, -0.0033709672279655933, 0.023305919021368027, 
            8.124732971191406, 2.8614587783813477, -0.037766292691230774, 7.955101013183594, -0.0033709669951349497, 
            2.8404767513275146, 8.124732971191406, 2.8614587783813477, 2.901547431945801, 7.955101013183594, 
            2.8864598274230957, 1.4318909645080566, 9.999818801879883 ] )

#We Define The Behavior Of Our Simulation Based On User-Input
if (__name__ == '__main__'):
    
    parser = argparse.ArgumentParser(description = "The Parameters Of Our NEB Simulation")
    parser.add_argument('--number', default = 5, type=int)
    parser.add_argument('--k', default = 1.0, type=float)
    parser.add_argument('--dt', default = 3e-2, type=float)
    parser.add_argument('--iterations', default = 150, type=int)
    parser.add_argument('-plotting', default = False, action = 'store_true')
    args = parser.parse_args()
    
    #We Define Our Elastic Band
    Band = AdsorbateString(initial, final, args.number, args.k)
    #We Define Our Simulation And Performance Plot
    NEB = AdsorbateNEB(Band, args.dt, args.iterations, plotting = args.plotting)
    #plotPerformance( NEB, 3, args.number, saddle, [args.dt, args.iterations], "Adsorbate", simple=False)