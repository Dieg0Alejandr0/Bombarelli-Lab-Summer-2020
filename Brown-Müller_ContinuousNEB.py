import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import pi
import argparse
import NEB_Methods
from NEB_Methods import ContinuousString, ContinuousNEB, NEBLoss
    
#We Define Our Brown-Müller Potential
class BM(nn.Module):
    def __init__(self):
        super(BM, self).__init__()
        self.A = [-200, -100, -170, 15]
        self.a = [-1, -1, -6.5, 0.7]
        self.b = [0, 0, 11, 0.6]
        self.c = [-10, -10, -6.5, 0.7]
        self.X = [1, 0, -0.5, -1]
        self.Y = [0, 0.5, 1.5, 1]
        
    def getEnergy(self, r):       
        """
        potential energy as a function of position
        for the LEPS potential on a line
        python version
        """
        x=r[:, 0]
        y=r[:, 1]
        
        V = 0
        
        for i in range(4):
            
            V += self.A[i]*torch.exp( self.a[i]*( (x-self.X[i])**2 ) + 
                                            self.b[i]*(x-self.X[i])*(y-self.Y[i]) + 
                                            self.c[i]*( (y-self.Y[i])**2 )  )
        
        return V
    
    def forward(self, r):
        
        return self.getEnergy(r)
    
#The Following Are Hardcoded Variables Meant For NEB Simulations
r = [torch.linspace(-1.5, 1.1, 60), torch.linspace(-0.5, 2.0, 60)]
bounds = [-1.5, 1.0, -0.5, 2.0]
ticks = [ [i/2 for i in range(-3, 3)], [i/2 for i in range(-1, 5)] ]
levels = 180
initial = torch.FloatTensor([-0.53, 1.47])
final = torch.FloatTensor([0.65, 0.04])
potential = BM()
name = "Brown-Müller"

#We Define The Behavior Of Our Simulation Based On User-Input
if (__name__ == '__main__'):
    
    parser = argparse.ArgumentParser(description = "The Parameters Of Our NEB Simulation")
    parser.add_argument('--images', default = 8, type=int)
    parser.add_argument('--method', default = "rk4", type=str)
    parser.add_argument('--lr', default = 5e-3, type=float)
    parser.add_argument('--t', default = 1e-3, type=float)
    parser.add_argument('--epochs', default = 500, type=int)
    parser.add_argument('-plotting', default = False, action = 'store_true')
    parser.add_argument('-last', default = False, action = 'store_true')
    parser.add_argument('-loss', default = False, action = 'store_true')
    parser.add_argument('-reaction', default = False, action = 'store_true')
    args = parser.parse_args()
    
    t = torch.linspace(0, args.t, 2)
    ODEBand = ContinuousString(initial, final, t, args.method, potential, args.images)
    optimizer = torch.optim.Adam(ODEBand.parameters(), lr=args.lr)
    S = NEBLoss(potential)
    
    ContinuousNEB(ODEBand, S, optimizer, potential, levels, r, bounds, ticks, args.epochs, plotting = args.plotting, 
                  last = args.last, loss = args.loss, reaction = args.reaction)