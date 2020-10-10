import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import pi
import argparse
import NEB_Methods
from NEB_Methods import DifferentiableString, DifferentiableNEB, NEBLoss
    
#We Define Our Cosine-Wells Potential
class COS(nn.Module):
    def __init__(self):
        super(COS, self).__init__()
        self.a = [1, 1]
        self.b = [-2, 1]
        self.c = 0
        
    def getEnergy(self, r):       
        """
        potential energy as a function of position
        for the LEPS potential on a line
        python version
        """
        x=r[:, 0]
        y=r[:, 1]
        
        V = self.a[0]*torch.cos(self.b[0]*x) + self.a[1]*torch.cos(self.b[1]*y + self.c)
        
        return V
    
    def forward(self, r):
        
        return self.getEnergy(r)
    
    def forward(self, r):
        
        return self.getEnergy(r)
    
#The Following Are Hardcoded Variables Meant For NEB Simulations
r = [torch.linspace(0, 6, 60), torch.linspace(0, 12, 60)]
bounds = [0, 6, 0, 12]
ticks = [ [i for i in range(0, 7)], [i for i in range(0, 13)] ]
levels = 20
initial = torch.FloatTensor([1.55, 3.17])
final = torch.FloatTensor([4.71, 9.6])
potential = COS()
name = "Cosine"

#We Define The Behavior Of Our Simulation Based On User-Input
if (__name__ == '__main__'):
    
    parser = argparse.ArgumentParser(description = "The Parameters Of Our NEB Simulation")
    parser.add_argument('--images', default = 10, type=int)
    parser.add_argument('--method', default = "rk4", type=str)
    parser.add_argument('--lr', default = 5e-2, type=float)
    parser.add_argument('--t', default = 1e-2, type=float)
    parser.add_argument('--epochs', default = 200, type=int)
    parser.add_argument('-plotting', default = False, action = 'store_true')
    parser.add_argument('-last', default = False, action = 'store_true')
    parser.add_argument('-loss', default = False, action = 'store_true')
    parser.add_argument('-reaction', default = False, action = 'store_true')
    args = parser.parse_args()
    
    t = torch.linspace(0, args.t, 2)
    ODEBand = DifferentiableString(initial, final, t, args.method, potential, args.images)
    optimizer = torch.optim.Adam(ODEBand.parameters(), lr=args.lr)
    S = NEBLoss(potential)
    
    DifferentiableNEB(ODEBand, optimizer, S, levels, r, bounds, ticks, args.epochs, last = args.last, 
                      loss = args.loss, reaction = args.reaction)