import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from math import pi, floor
from torchdiffeq.torchdiffeq._impl import odeint_adjoint, odeint

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
#view(slab, viewer="x3d")

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

#We Define How To Make The Contour Plot Of A Potential
def contour(potential, levels, x, y):
    #We Get Our Coordinates For Contouring
    x, y = torch.meshgrid( [x, y] )
    
    """
    The Following Reshaping Is First For The Purposes Of The Potential And
    Then For The Potential Found To Be The Same Shape As The Coordinates For The
    Contour Plotting Function
    """
    size = x.shape[0]*x.shape[1]
    rx = x.reshape( (size, 1) )
    ry = y.reshape( (size, 1) )
    r = torch.cat( (rx, ry), dim = 1).reshape( (size, 2) )

    z = potential(r)
    z = z.reshape(60, 60)
    
    #We Plot
    X = x.numpy()
    Y = y.numpy()
    Z = z.numpy()

    plt.contour(X, Y, Z, levels, colors='black')
    
#We Define How To Plot Our Trajectory 
def plotMEP(trajectory, potential, levels, r, bounds, ticks, name, figure=0):
    
    plt.figure(figure)
    plt.title("Minimum Energy Path Found")
    plt.axis(bounds)
    plt.xlabel('rBC [Å]')
    plt.ylabel('rAB [Å]')
    plt.xticks(ticks[0])
    plt.yticks(ticks[1])
    contour(potential, levels, r[0], r[1])
    visualX = (trajectory)[:, 0].numpy()
    visualY = (trajectory)[:, 1].numpy()
    plt.plot(visualX, visualY, ".-b", label = "MEP")
    plt.legend()
    #plt.savefig("Figures/" + name + " Trajectory.png")
    plt.close(figure)
    plt.show()
    
#We Define How We Plot Our Losses
def plotLosses(epochs, Losses, name):
    
    plt.figure(-1)
    epochs = [epoch for epoch in range(1, epochs + 1)]
    plt.title("Losses Of Trajectory")
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.plot(epochs, Losses, "-b")
    #plt.savefig("Figures/" + name + " Losses.png")
    plt.close(-1)
    plt.show()
    
#We Define How To Plot The 1D Potential Of Our MEP
def plotPotential(path, name):
    
    V = path.potentials().numpy()
    I = [i for i in range(1, path.N+1)]
    V -= V[0]
    
    plt.figure(-2)
    plt.title("Potential Energy Of Band")
    plt.xlabel('Path [Images]')
    plt.ylabel('Potential [eV]')
    plt.plot(I, V, ".-m")
    #plt.savefig("Figures/" + name + " Potential.png")
    plt.close(-2)
    plt.show()
    
#We Get Our Band's Potential And Length
def AdsorbatePotential(path):
    V = path.potentials().numpy()
    Offset = V[0]
    V -= Offset
    L = np.linspace(0.0, 3.0, len(V) )
    P = np.poly1d( np.polyfit(L, V, 4) )
    l = np.linspace(0.0, 3.0, 200)
    #We Define Our Plot
    plt.figure(-1)
    plt.scatter( L, V, c="m" )
    plt.plot( l, P(l), "-m", label = "MEP")
    plt.title("Potential Energy Of Band")
    plt.axis([-0.6, 3.6, -0.02, 0.39])
    plt.xlabel("Path [Å]")
    plt.ylabel("Potential Energy [eV]")
    plt.xticks( [ i/2 for i in range(-1, 8) ] )
    plt.yticks( [ i/20 for i in range(0, 8) ] )
    plt.legend()
    #plt.savefig("Figures/Adsorbate Potential.png")
    plt.show(block=True)
    plt.close(-1)
    
#We Define Our Loss
class StandardLoss(nn.Module):
    
    def __init__(self, potential):
        super(StandardLoss, self).__init__()
        self.potential = potential
        
    def tangent(self, z):
        #The Following Approximates The Tangent For Different Atoms At Different Images
        Zero = torch.zeros( (1, z.shape[1]) )
        length = z.shape[0]
        #Eqn.(3)
        tangent = (z[2:, :] - z[:length-2, :])
        norm = torch.norm(tangent, dim=1)
        norm = torch.reshape(norm, (tangent.shape[0], 1))
        tau = torch.cat( (Zero, tangent/norm, Zero), dim = 0)
        
        return tau
        
    
    def DV(self, z):
        """
        The Following Returns The Perpendicular Component of The Potential Gradient
        """
        #We Properly Set Up The Differentiability Of The Potential And Find It
        z = z.detach().requires_grad_(True)
        Zero = torch.zeros(z.shape[1])
        
        with torch.enable_grad():
            V = self.potential( z )
            #We Get Back Our Gradient And Set Up A Graph Of Gradients
            V.backward( torch.ones_like( V ), retain_graph = True, create_graph = True)
            grad = z.grad
            
            grad[0, :] = Zero
            grad[-1, :] = Zero
            
        #We Find The Perpendicular Component Of The Potential Gradient
        tau = self.tangent(z)
        product = torch.mm( grad, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        perpendicular = grad - parallel
        
        return perpendicular
        
    def forward(self, string):
        
        N = string.R.shape[0]
        V = string.potentials()
        S = N*0.5*(string.R[1:, :] - string.R[:N-1, :])**2
        D = self.DV(string.R)
        
        Loss = (torch.sum(V) + torch.sum(S) + torch.sum(torch.abs(D) ) )/N
        return Loss
    
#We Define Our Loss
class NEBLoss(nn.Module):
    
    def __init__(self, potential):
        super(NEBLoss, self).__init__()
        self.potential = potential
        
    def tangent(self, z):
        #The Following Approximates The Tangent For Different Atoms At Different Images
        Zero = torch.zeros( (1, z.shape[1]) )
        length = z.shape[0]
        #Eqn.(3)
        tangent = (z[2:, :] - z[:length-2, :])
        norm = torch.norm(tangent, dim=1)
        norm = torch.reshape(norm, (tangent.shape[0], 1))
        tau = torch.cat( (Zero, tangent/norm, Zero), dim = 0)
        
        return tau
        
    
    def DV(self, z):
        """
        The Following Returns The Perpendicular Component of The Potential Gradient
        """
        #We Properly Set Up The Differentiability Of The Potential And Find It
        z = z.detach().requires_grad_(True)
        Zero = torch.zeros(z.shape[1])
        
        with torch.enable_grad():
            V = self.potential( z )
            #We Get Back Our Gradient And Set Up A Graph Of Gradients
            V.backward( torch.ones_like( V ), retain_graph = True, create_graph = True)
            grad = z.grad
            
            grad[0, :] = Zero
            grad[-1, :] = Zero
            
        #We Find The Perpendicular Component Of The Potential Gradient
        tau = self.tangent(z)
        product = torch.mm( grad, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        perpendicular = grad - parallel
        
        return perpendicular
        
    def forward(self, string):
        
        N = string.zN.shape[0]
        V = string.potentials()
        S = N*0.5*(string.zN[1:, :] - string.zN[:N-1, :])**2
        D = self.DV(string.zN)
        
        Loss = (torch.sum(V) + torch.sum(S) + torch.sum(torch.abs(D) ) )/N
        return Loss
    
class CubicSpline(nn.Module):
    
    def __init__(self, trajectory):
        super(CubicSpline, self).__init__()
        self.trajectory = trajectory
        self.n = trajectory.shape[0]-1
        self.d = trajectory.shape[1]
        self.parameters = []
        
    def compute(self, subtrajectory):
        
        with torch.no_grad():
            n = self.n
            x = torch.linspace(0, 1, self.n+1)
            a = subtrajectory
            h = x[1:] - x[:self.n]
            alpha = (3/h[1:])*(a[2:] - a[1:n]) - (3/h[:n-1])*(a[1:n] - a[:n-1])

            l = torch.ones((n+1,))
            mu = torch.zeros((n+1,))
            z = torch.zeros((n+1,))
            for i in range(1, n):
                l[i] = 4/self.n - h[i-1]*mu[i-1]
                mu[i] = h[i]/l[i]
                z[i] = (alpha[i-1] - h[i-1]*z[i-1])/l[i]

            c = torch.zeros((n+1,))
            for j in range(n-1, -1, -1):
                c[j] = z[j] - mu[j]*c[j+1]

            b = (a[1:] - a[:n])/h - (h*(c[1:]+2*c[:n]))/3
            d = (c[1:]-c[:n])/(3*h)
            
        return [a, b, c, d, x]
    
    def populate(self):
        
        D = self.d
        self.parameters = []
        
        for degree in range(D):
            
            degreeParams = self.compute(self.trajectory[:, degree])
            (self.parameters).append(degreeParams)
            
    def evaluate(self, Lambda):
        
        if (Lambda < 0 or Lambda > 1):
            raise ValueError("Lambda Must Be Between The Values Of 0 And 1")
            
        R = torch.zeros( (1, self.d) )
        if (Lambda < 1.00):
            index = floor(Lambda*self.n)
        else:
            index = (self.n - 1)
        params = self.parameters
        
        for degree in range(self.d):

            R[0, degree] = params[degree][0][index] + params[degree][1][index]*(Lambda - params[degree][-1][index]) + params[degree][2][index]*((Lambda - params[degree][-1][index])**2) + params[degree][3][index]*((Lambda - params[degree][-1][index])**3)
            
        return R
    
    def forward(self, Lambdas):
        
        self.populate()
        
        final = torch.zeros((1, self.d))

        for Lambda in Lambdas:

            if (Lambda == 0):
                final = self.evaluate(Lambda)

            else:
                final = torch.cat( (final, self.evaluate(Lambda)), dim=0 )
        
        return final
    
#We Define Our String Of Points
class SimpleString(nn.Module):
    """The String Is Initialized With An Initial And Final Point 
    Of The Path And The Number Of Images Desired
    """
    def __init__(self, initial, final, potential, number=5, k = 1.00):
        
        """
        self.initial: The Initial Image's Position Vector
        self.final: The Final Image's Position Vector
        self.N: The Number Of Images 
        self.k: The Spring Constant Of The String
        self.R: Every Image's Current Position Vector
        self.u: Every Image's Current Velocity Vector
        self.F: Every Image's Current Force Vector
        self.prevF: Most Recent Recorded Image Force Vectors
        self.potential: The Analytical Function Of The Potential
        """
        
        super(SimpleString, self).__init__()
        self.initial = initial
        self.final = final
        self.N = number
        self.k = k
        self.R = []
        self.u = []
        self.F = []
        self.prevF = []
        self.potential = potential
        
    def potentials(self):
        #The Following Returns The Potential Along The Curve
        V = self.potential( self.R ) 
        return torch.reshape( V, (V.shape[0],) )
        
    def start(self):        
        #We Properly Define Our Initial, Final, And General Position Vectors
        (x0, y0, xf, yf) = (self.initial[0], self.initial[1], self.final[0], self.final[1])
        self.R = torch.zeros(self.N, 2)
        
        #We Linearly Interpolate For Both Components
        self.R[:, 0] = torch.linspace(x0, xf, self.N)
        self.R[:, 1] = torch.linspace(y0, yf, self.N)
        
        #We Initialize Some Pivotal Attributes
        Zero = torch.zeros( (self.N, self.R.shape[1]) )
        self.u = Zero
        self.F = Zero
        self.prevF = Zero
    
    def tangent(self):
        #The Following Approximates The Tangent For Different Atoms At Different Images
             
        Zero = torch.zeros( (1, self.R.shape[1]) )
        length = self.N
        #Eqn.(3)
        tangent = (self.R[2:, :] - self.R[:length-2, :])
        norm = torch.norm(tangent, dim=1)
        norm = torch.reshape(norm, (tangent.shape[0], 1))
        tau = torch.cat( (Zero, tangent/norm, Zero), dim = 0)
        
        return tau
    
    def DV(self):
        """
        The Following Returns The Perpendicular Component of The Potential Gradient
        """
        #We Properly Set Up The Differentiability Of The Potential And Find It
        R = self.R.detach().requires_grad_(True)
        Zero = torch.zeros(R.shape[1])
        V = self.potential(R)
        #We Get Back Our Gradient
        V.backward( torch.ones_like(V) )
        grad = R.grad
        grad[0, :] = Zero
        grad[-1, :] = Zero
        #We Find The Perpendicular Component Of The Potential Gradient
        tau = self.tangent()
        product = torch.mm( grad, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        Nambla = grad - parallel

        return Nambla
            
    
    def springForce(self):
        """
        Recall that by Hooke's Law F_s = kx
        """
        path = self.R
        Zero = torch.zeros( (1, path.shape[1]) )
        length = self.N
        #Eqn.(6)
        displacement = (path[2:, :] - path[1:length-1, :]) - (path[1:length-1, :] - path[:length-2, :])
        force = self.k*displacement
        F = torch.cat( (Zero, force, Zero), dim = 0)
        
        return F
    
    def switch(self):
        
        #The Following Computes The f(theta) Detailed In Page 396
        f = torch.zeros( (self.N, 1) )
        R = self.R
        length = self.N
        
        #We Calculate Eqn. (10)
        t1 = R[2:, :] - R[1:length-1, :]
        t2 = R[1:length-1, :] - R[:length-2, :]
        prod = torch.mm(t1, torch.transpose(t2, 0, 1) )
        numerator = torch.diagonal(prod)

        norm = torch.norm(t1)*torch.norm(t2)
        one = torch.ones( (norm.shape))
        denominator = torch.where(norm != 0, norm, one)
        
        cosine = numerator/denominator
        f_t = 0.5*(1 + torch.cos(pi*cosine) )
        f[1:length-1, :] = torch.reshape(f_t, (length-2, 1) )
        
        return f
    
    def springComponents(self):
        """
        The Following Computes The Parallel And Perpendicular Component Of 
        The Spring Force Up To A Factor Of f(theta)
        """
        #We Find Our Needed Attributes
        f = self.switch()
        spring = self.springForce()
        tau = self.tangent()
        
        #As Described In Eqn.(9), We Find Each Component Of The Spring Force For All Atoms 
        product = torch.mm( spring, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        
        difference = spring - parallel
        perpendicular = f*difference
        
        return (parallel, perpendicular)
            
    def setForce(self, F):
        #The Force Referred To Here Is Found In Eqn.(9)
        self.F = F
        
    def checkVelocity(self):
        #N Is The Number Of Images
        N = self.N
        C = self.R.shape[1] 
        zero = torch.zeros((N, C))

        #We Run Each Velocity Through Eqn.(11)
        norm = torch.norm(self.F, dim=1)
        norm = torch.reshape(norm, (len(norm), 1))
        one = torch.ones_like(norm)
        denominator = torch.where(norm == 0.0, one, norm)
        Fhat = self.F/denominator
    
        Zero = torch.zeros( (N,C) )
        product = torch.diagonal( torch.mm( self.u, torch.transpose(Fhat, 0, 1) ) ).reshape( N, 1 )
        self.u = torch.where(product > 0, product*Fhat, Zero)
    
    def Verlet(self, mass = 1.00, dt = 3e-2):
        """
        The following updates the positions of the images
        """
        #We Integrate For Our New Position
        deltaR= self.u*dt + 0.5*(self.F/mass)*(dt**2)
        self.R += deltaR
        #We Integrate Our Acceleration
        deltau = 0.5*(self.prevF/mass + self.F/mass)*dt
        self.u += deltau
        #Eqn.(11)
        self.checkVelocity()
        #We Update Our Previous Forces
        self.prevF = self.F 
    
    def forward(self, mass, dt):
        #We Define Our Forces
        DV = self.DV()
        (spring, nudge) = self.springComponents()
        #We Find Our Force
        F_i = -DV + spring + nudge
        self.setForce(F_i)
        #When We've Traced The Entire Path We Perform Verlet Integration
        self.Verlet(mass = mass, dt = dt)
        
#We Define Our String Of Points
class AdsorbateString(nn.Module):
    """The String Is Initialized With An Initial And Final Point 
    Of The Path And The Number Of Images Desired
    """
    def __init__(self, initial, final, number=5, k = 1.00):
        
        """
        self.initial: The Initial Image's Position Vector
        self.final: The Final Image's Position Vector
        self.N: The Number Of Images 
        self.k: The Spring Constant Of The String
        self.R: Every Image's Current Position Vector
        self.u: Every Image's Current Velocity Vector
        self.VGrad: Every Image's Potential Gradient Vector
        self.F: Every Image's Current Force Vector
        self.prevF: Most Recent Recorded Image Force Vectors
        self.images: The ASE Atoms Objects Of Each Image
        self.masses: An Array Of The Masses Of The Atoms
        self.shape: The Shape Of The Atoms Object Attributes
        """
        
        super(AdsorbateString, self).__init__()
        self.initial = initial
        self.final = final
        self.N = number
        self.k = k
        self.R = []
        self.u = []
        self.VGrad = []
        self.F = []
        self.prevF = []
        self.images = []
        self.masses = []
        self.shape = ()
        
    #The Following Just Returns The Potential Energy For Each Image
    def potentials(self):
        
        energies = torch.FloatTensor( [image.get_potential_energy() for image in self.images] )

        return energies
        
    #The Following Is Meant To Define Important Attributes
    def update(self, begin = False):       
        
        #If We Are Starting To Build Our String
        if (begin):
            
            L = list( (self.initial).get_masses() )
            self.masses = torch.reshape( torch.FloatTensor(3*L), (1, 3*len(L) ) )
            
            #We Create Atoms Instance For Each Image
            constraint = FixAtoms(mask=[atom.tag > 1 for atom in initial])
            self.images = [self.initial]
            
            for i in range(self.N-2):
                
                image = self.initial.copy()
                image.set_calculator(EMT())
                image.set_constraint(constraint)
                self.images.append(image)

            self.images.append(final)
        
        #We Updates The Positions, Potential Gradients, And Velocities Of Our Images
        for i in range(1, self.N - 1):
            
            
            positions = np.reshape(self.R[i].numpy(), (self.shape[0], self.shape[1] ) )
            (self.images[i]).set_positions( positions )
            
            force = np.reshape( (self.images[i]).get_forces(), ( 1, self.shape[0]*self.shape[1] ) )
            self.VGrad[i, :] = torch.FloatTensor( force )
            
            #If The Atoms Object Has No Velocities Attribute, We Define It, Else We Take It And Fix Its Magnitude
            #if ( type(self.images[i].get_velocities()) == type(None) ):
            velocities = np.reshape( self.u[i].numpy(), (self.shape[0], self.shape[1]) )
            (self.images[i]).set_velocities( velocities ) 
            #else:
            #    velocities = torch.FloatTensor(
            #        np.reshape( (self.images[i]).get_velocities(), ( 1, self.shape[0]*self.shape[1] ) )
            #        )
            #    self.u[i] = velocities
            #    self.checkVelocity()
        
        #We Just Have Defined The Gradient As The Forces, But Recall That F = -DV
        self.VGrad = torch.where(self.VGrad != 0, -self.VGrad, self.VGrad)
            
        
    def start(self):
        
        #We Properly Define Our Initial, Final, And General Position Vectors
        initialR = torch.FloatTensor( self.initial.get_positions() )
        self.shape = (initialR.shape[0], initialR.shape[1])
        initialR = torch.reshape( initialR, (1, initialR.shape[0]*initialR.shape[1]) )
        finalR = torch.FloatTensor( self.final.get_positions() )
        finalR = torch.reshape( finalR, (1, finalR.shape[0]*finalR.shape[1]) )
        self.R = torch.zeros( ( self.N, initialR.shape[1] ) )
        
        #We Linearly Interpolate For All Images
        for x in range(self.R.shape[1]):
            self.R[:, x] = torch.linspace(initialR[0, x], finalR[0, x], self.N)
        
        #We Initialize Some Pivotal Attributes
        Zero = torch.zeros( (self.N, self.R.shape[1]) )
        self.u = Zero
        self.F = Zero
        self.prevF = Zero
        self.VGrad = Zero
        self.update(begin = True)

    def tangent(self):
        
        #The Following Approximates The Tangent For Different Atoms At Different Images
            
        Zero = torch.zeros( (1, self.R.shape[1]) )
        length = self.N
        #Eqn.(3)
        tangent = (self.R[2:, :] - self.R[:length-2, :])
        norm = torch.norm(tangent, dim=1)
        norm = torch.reshape(norm, (tangent.shape[0], 1))
        tau = torch.cat( (Zero, tangent/norm, Zero), dim = 0)
        
        return tau
        
    
    def DV(self):
        """
        The Following Returns The Perpendicular Component of The Potential Gradient
        """
        grad = self.VGrad
        tau = self.tangent()
        #We Find The Perpendicular Component Of The Potential Gradient
        product = torch.mm( grad, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        Nambla = grad - parallel
            
        return Nambla
            
    
    def springForce(self):
        """
        Recall that by Hooke's Law F_s = kx
        """
        path = self.R
        Zero = torch.zeros( (1, path.shape[1]) )
        length = self.N
        #Eqn.(6)
        displacement = (path[2:, :] - path[1:length-1, :]) - (path[1:length-1, :] - path[:length-2, :])
        force = self.k*displacement
        F = torch.cat( (Zero, force, Zero), dim = 0)
        
        return F
    
    def switch(self):
        
        #The Following Computes The f(theta) Detailed In Page 396
        f = torch.zeros( (self.N, 1) )
        R = self.R
        length = self.N
        
        #We Calculate Eqn. (10)
        t1 = R[2:, :] - R[1:length-1, :]
        t2 = R[1:length-1, :] - R[:length-2, :]
        prod = torch.mm(t1, torch.transpose(t2, 0, 1) )
        numerator = torch.diagonal(prod)

        norm = torch.norm(t1)*torch.norm(t2)
        one = torch.ones( (norm.shape))
        denominator = torch.where(norm != 0, norm, one)
        
        cosine = numerator/denominator
        f_t = 0.5*(1 + torch.cos(pi*cosine) )
        f[1:length-1, :] = torch.reshape(f_t, (length-2, 1) )
        
        return f
    
    def springComponents(self):
        """
        The Following Computes The Parallel And Perpendicular Component Of 
        The Spring Force Up To A Factor Of f(theta)
        """
        parallel = torch.zeros( (self.N, self.R.shape[1]) )
        perpendicular = torch.zeros( (self.N, 1) )

        f = self.switch()
        spring = self.springForce()
        tau = self.tangent()
        
        #As Described In Eqn.(9), We Find Each Component Of The Spring Force For All Atoms 
        product = torch.mm( spring, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        
        difference = spring - parallel
        perpendicular = f*difference
        
        return (parallel, perpendicular)
            
    def setForce(self, F):
        #The Force Referred To Here Is Found In Eqn.(9)
        self.F = F
        
    def checkVelocity(self):
        #N Is The Number Of Images And A The Number Of Atoms
        N = self.N
        A = self.shape[0] 

        #We Run Each Velocity Component Through Eqn.(11)
        norm = torch.norm(self.F, dim=1)
        norm = torch.reshape(norm, (len(norm), 1))
        one = torch.ones_like(norm)
        denominator = torch.where(norm == 0.0, one, norm)
        Fhat = self.F/denominator
    
        Zero = torch.zeros( (N,3*A) )
        product = torch.diagonal( torch.mm( self.u, torch.transpose(Fhat, 0, 1) ) ).reshape( N, 1 )
        self.u = torch.where(product > 0, product*Fhat, Zero)
    
    def Verlet(self, dt = 1e-1):
        """
        The following updates the positions of the images
        """
        #We Integrate For Our New Position
        deltaR= self.u*dt + 0.5*(self.F/self.masses)*(dt**2)
        self.R += deltaR
        #We Integrate Our Acceleration
        deltau = 0.5*(self.prevF/self.masses + self.F/self.masses)*dt
        self.u += deltau
        #Eqn.(11)
        self.checkVelocity()
        #We Update Our Class Attributes
        self.prevF = self.F
        self.update()  
    
    def forward(self, dt):
        #We Define Our Forces
        DV = self.DV()
        (spring, nudge) = self.springComponents()
        #We Find Our Force
        F_i = -DV + spring + nudge
        self.setForce(F_i)
        #When We've Traced The Entire Path We Perform Verlet Integration
        self.Verlet(dt = dt)
        self.update()
        
def SimpleNEB(path, levels, r, bounds, ticks, name, 
              mass = 1.0, dt = 3e-2, iterations = 250, 
              plotting = False, last = False, loss = False, reaction = False):
    
    if (path.N < 3):
        raise ValueError("Too few images requested!")
    
    #We Define Our String Of Points
    path.start()
    losses = []
    lossfunction = StandardLoss(path.potential)
        
    #Iterations Of Updating The Entire String
    for iteration in range(iterations):
        #We Define Our Plot
        if (plotting):
            plotMEP(path.R, path.potential, levels, r, bounds, ticks, name, figure = iteration)
        
        #We Update Our Path
        path(mass, dt)
        
        L = lossfunction(path)
        losses.append(L)

    if (last or plotting):
        plotMEP(path.R, path.potential, levels, r, bounds, ticks, name, figure = iteration)
        
    if (loss):
        plotLosses(iterations, losses, name)
        
    if (reaction):
        plotPotential(path, name)
        
    return path, losses

def AdsorbateNEB(path, dt = 2e-1, iterations = 200, plotting = False):
    
    if (path.N < 3):
        raise ValueError("Too few images requested!")
    
    #We Define Our String Of Points
    path.start()
    #Iterations Of Updating The Entire String
    for iteration in range(iterations):              
        path(dt)
        
    if (plotting):
        if (path.N == 5):
            AdsorbatePotential(path)
        else:
            plotPotential(path, "Adsorbate") 
        
    return path

def plotPerformance( path, start, end, x, params, name, simple = True, single = True):
    #We Define A Plot Of The Barrier Of Our MEP Over The Number Of Images Used
    I = [i for i in range(start, end+1)]
    predicted = []
    observed = x
    
    #We Find The Performance Of NEB On Bands Of Different Image Numbers
    for images in range(start, end+1):
        
        path.N = images
        if (simple):
            MEP = SimpleNEB(path, params[0], params[1], params[2], params[3], name,
                            mass = params[4], dt = params[5], iterations = params[6] )[0]
        else:
            MEP = AdsorbateNEB(path, params[0], params[1] ) [0]
        index = torch.argmax( torch.FloatTensor( MEP.potentials() ) )
        predicted.append( MEP.R[index, :].tolist() )
        
    predicted = torch.FloatTensor(predicted)
    if (single):
        D = torch.norm( (predicted-observed), dim=1).tolist()
    else:
        D1 = torch.norm( (predicted-observed[0]), dim=1)
        D2 = torch.norm( (predicted-observed[1]), dim=1)
        D = torch.where(D1>D2, D2, D1).tolist()
    plt.figure(-10)
    plt.title("Performance Over Images")
    plt.xlabel('Images')
    plt.ylabel('||Barrier Found - Ground Truth|| [Å]')

    plt.plot(I, D, "o--g")
    #plt.savefig("Figures/" + name + " Performance.png")
    plt.close(-10)
    plt.show()
    
    
#We Define Our ODE For The Differentiable NEB
class ODE(nn.Module):
    
    def __init__(self, potential, scalar):
        super(ODE, self).__init__()
        
        self.scalar = torch.nn.Parameter(scalar)
        self.potential = potential
        
    def tangent(self, z):
        
        #The Following Approximates The Tangent For Different Images
        Zero = torch.zeros( (1, z.shape[1]) )
        length = z.shape[0]
        #Eqn.(3)
        tangent = (z[2:, :] - z[:length-2, :])
        norm = torch.norm(tangent, dim=1)
        norm = torch.reshape(norm, (tangent.shape[0], 1))
        tau = torch.cat( (Zero, tangent/norm, Zero), dim = 0)
        
        return tau
    
    def DV(self, z):
        """
        The Following Returns The Perpendicular Component of The Potential Gradient
        """
        #We Properly Set Up The Differentiability Of The Potential And Find It
        z = z.detach().requires_grad_(True)
        Zero = torch.zeros(z.shape[1])
        
        with torch.enable_grad():
            V = self.potential( z )
            #We Get Back Our Gradient And Set Up A Graph Of Gradients
            V.backward( torch.ones_like( V ), retain_graph = True, create_graph = True)
            grad = z.grad
            
            grad[0, :] = Zero
            grad[-1, :] = Zero
            
        #We Find The Perpendicular Component Of The Potential Gradient
        tau = self.tangent(z)
        product = torch.mm( grad, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        perpendicular = grad - parallel
        
        return (parallel, perpendicular)
        
    def forward(self, t, z):

        f = -self.DV(z)[1] - (self.scalar)*self.DV(z)[0]

        return f
    
#We Define Our Differentiable NEB String
class DifferentiableString(nn.Module):
    """
    The String Is Initialized With An Initial And Final Point 
    Of The Path And The Number Of Images Desired
    """
    def __init__(self, initial, final, t, method, potential, number):
        
        """
        self.z0: The Initial Hidden State, Or Beginning Transition Path Of Images
        self.zN: The Final Hidden State, Or Our Predicted Transition Via ODE Solving
        self.t0: The Times For ODE Integration
        self.method: Our Method Of Integration
        self.potential: Our Analytical Function Of A Potential Energy Surface 
        self.N: The Number Of Images In Our Trajectory
        self.ODE: The Ordinary Differential Equation Governing The Dynamics Of Our System
        """
        super(DifferentiableString, self).__init__()
        self.initial = initial
        self.final = final
        self.z0 = []
        self.zN = []
        self.t = t
        self.method = method
        self.potential = potential
        self.N = number
        self.ODE = ODE( self.potential, torch.zeros((self.N, 1)) )
        
        #We Properly Define Our Initial, Final, And General Position Vectors
        (x0, y0, xf, yf) = (self.initial[0], self.initial[1], self.final[0], self.final[1])
        z0 = torch.zeros(self.N, 2).requires_grad_(True)
        
        #We Linearly Interpolate For Both Components
        z0[:, 0] = torch.linspace(x0, xf, self.N)
        z0[:, 1] = torch.linspace(y0, yf, self.N)
        self.zN = z0.detach()
        self.z0 = torch.nn.Parameter( z0 ) 
        
    #The Following Returns The Potential Along The Curve
    def potentials(self):
        
        return self.potential( self.zN )         
        
    def step(self):
        
        self.z0 = torch.nn.Parameter(self.zN)
        
    def forward(self):
        
        solution = odeint_adjoint(self.ODE, self.z0, self.t, method = self.method)
        self.zN = solution[-1]
        
        return self.zN
    
def DifferentiableNEB(path, optimize, lossfunction, levels, r, bounds, ticks, epochs = 200, plotting = False, last = False, loss = False, 
               reaction = False):
    
    if (path.N < 3):
        raise ValueError("Too few images requested!")
    
    #We Define Our String Of Points And LEPS Potential
    losses = []
        
    #Iterations Of Updating The Entire String
    for epoch in range(epochs):

        if (plotting):
            plotMEP(path.zN.detach(), path.potential, levels, r, bounds, ticks, figure=epoch)
        
        path()
        L = lossfunction(path)
        losses.append(L)

        optimize.zero_grad()
        L.backward()
        optimize.step()
        path.step()
            

    if (last or plotting):
        plotMEP(path.zN.detach(), path.potential, levels, r, bounds, ticks, figure=epoch)
    
    if (loss):
        plotLosses(epochs, losses)
    
    if (reaction):
        plotPotential(path)

    return path, losses

#We Define Our Naive Model
class NaiveNN(nn.Module):
    
    def __init__(self, N, n):
        super(NaiveNN, self).__init__()
        self.model = torch.nn.Sequential(nn.Linear(N, N),
                                         nn.Linear(N, n),
                                         nn.ReLU(),
                                         nn.Linear(n, n),
                                         nn.ReLU() )
        
    def forward(self, z0):

        zN = self.model( torch.transpose(z0, 0, 1) ).transpose(0, 1)
        
        return zN

#We Define Our Loss For Regressive Cases
class RegressiveLoss(nn.Module):
    
    def __init__(self):
        super(RegressiveLoss, self).__init__()
        self.interpolator = CubicSpline
        
    def interpolation(self, trajectory, number):
        
        interpolating = self.interpolator(trajectory)
        Lambdas = torch.linspace(0, 1, number)
        
        interpolation = interpolating(Lambdas)
        return interpolation
        
    def forward(self, observed, prediction):
        
        lossfunction = torch.nn.MSELoss()
        interpolated = self.interpolation(observed, prediction.shape[0])
        Loss = lossfunction(interpolated, prediction)
        
        return Loss
    
def NaiveNEB(trajectory, model, optimize, potential, levels, r, bounds, ticks, epochs = 200, plotting = False, last = False, 
               loss = False, reaction = False):
    
    if (trajectory.N < 3):
        raise ValueError("Too few images requested!")
    
    #We Define Our Loss Function And Losses Array
    lossfunction = RegressiveLoss()
    losses = []
    
    #Iterations Of Updating The Entire String
    for epoch in range(epochs):

        if (plotting and epoch > 0):
            plotMEP(predicted.detach(), potential, levels, r, bounds, ticks, figure=epoch)
        
        predicted = model(trajectory)
        L = lossfunction(trajectory, predicted)
        losses.append(L)

        optimize.zero_grad()
        L.backward()
        optimize.step()          

    if (last or plotting):
        plotMEP(predicted.detach(), potential, levels, r, bounds, ticks, figure=epoch)
    
    if (loss):
        plotLosses(epochs, losses)
    
    if (reaction):
        plotPotential(path)

    return predicted, losses

#We Define The Spring Term Of Our Neural ODE
class NeuralSpring(nn.Module):
    
    def __init__(self, n):
        super(NeuralSpring, self).__init__()
        self.model = torch.nn.Sequential(nn.Linear(n+1, n+1), 
                                         nn.ReLU(),
                                         nn.Linear(n+1, n),
                                         nn.ReLU(),
                                         nn.Linear(n, n),
                                         nn.ReLU() )

    def forward(self, t, z):
        
        T = t*torch.ones((1, z.shape[1]))
        x = torch.cat((T, z), dim = 0)
        
        f = self.model( torch.transpose(x, 0, 1) ).transpose(0, 1)      
        return f
    
#We Define Our ODE
class NeuralODE(nn.Module):
    
    def __init__(self, n, potential):
        super(NeuralODE, self).__init__()
        self.neuralspring = NeuralSpring(n)
        self.potential = potential
        
    def tangent(self, z):
        
        #The Following Approximates The Tangent For Different Images
        Zero = torch.zeros( (1, z.shape[1]) )
        length = z.shape[0]
        #Eqn.(3)
        tangent = (z[2:, :] - z[:length-2, :])
        norm = torch.norm(tangent, dim=1)
        norm = torch.reshape(norm, (tangent.shape[0], 1))
        tau = torch.cat( (Zero, tangent/norm, Zero), dim = 0)
        
        return tau
    
    def DV(self, z):
        """
        The Following Returns The Perpendicular Component of The Potential Gradient
        """
        #We Properly Set Up The Differentiability Of The Potential And Find It
        z = z.detach().requires_grad_(True)
        Zero = torch.zeros(z.shape[1])
        
        with torch.enable_grad():
            V = self.potential( z )
            #We Get Back Our Gradient And Set Up A Graph Of Gradients
            V.backward( torch.ones_like( V ), retain_graph = True, create_graph = True)
            grad = z.grad
            
            grad[0, :] = Zero
            grad[-1, :] = Zero
            
        #We Find The Perpendicular Component Of The Potential Gradient
        tau = self.tangent(z)
        product = torch.mm( grad, tau.transpose(0, 1) )
        dot = torch.diagonal(product)
        dot = torch.reshape(dot, (len(dot), 1) )
        parallel = dot*tau
        perpendicular = grad - parallel
        
        return perpendicular
        
    def forward(self, t, z):

        f = -self.DV(z) - self.neuralspring(t, z)

        return f
    
#We Define Our Neural ODE Based String
class ContinuousString(nn.Module):
    """
    The String Is Initialized With An Initial And Final Point 
    Of The Path And The Number Of Images Desired
    """
    def __init__(self, initial, final, t, method, potential, number):
        
        """
        self.z0: The Initial Hidden State, Or Beginning Transition Path Of Images
        self.zN: The Final Hidden State, Or Our Predicted Transition Via ODE Solving
        self.t0: The Times For ODE Integration
        self.method: Our Method Of Integration
        self.potential: Our Analytical Function Of A Potential Energy Surface 
        self.N: The Number Of Images In Our Trajectory
        self.ODE: The Ordinary Differential Equation Governing The Dynamics Of Our System
        """
        super(ContinuousString, self).__init__()
        self.initial = initial
        self.final = final
        self.z0 = []
        self.zN = []
        self.t = t
        self.method = method
        self.potential = potential
        self.N = number
        self.ODE = NeuralODE( self.N, self.potential )
        
        #We Properly Define Our Initial, Final, And General Position Vectors
        (x0, y0, xf, yf) = (self.initial[0], self.initial[1], self.final[0], self.final[1])
        z0 = torch.zeros(self.N, 2).requires_grad_(True)
        
        #We Linearly Interpolate For Both Components
        z0[:, 0] = torch.linspace(x0, xf, self.N)
        z0[:, 1] = torch.linspace(y0, yf, self.N)
        self.zN = z0.detach()
        self.z0 = torch.nn.Parameter( z0 )
        
    #The Following Returns The Potential Along The Curve
    def potentials(self):
        
        return self.potential( self.zN )         
        
    def step(self):
        
        #We Properly Define Our Initial, Final, And General Position Vectors
        (x0, y0, xf, yf) = (self.initial[0], self.initial[1], self.final[0], self.final[1])
        z0 = torch.zeros(self.N, 2).requires_grad_(True)
        
        #We Linearly Interpolate For Both Components
        z0[:, 0] = torch.linspace(x0, xf, self.N)
        z0[:, 1] = torch.linspace(y0, yf, self.N)
        self.z0 = torch.nn.Parameter(z0)
        
    def forward(self):
        
        solution = odeint_adjoint(self.ODE, self.z0, self.t, method = self.method)
        self.zN = solution[-1]
        
        return self.zN
    
def ContinuousNEBGT(path, trajectory, optimize, potential, levels, r, bounds, ticks, epochs = 200, plotting = False, 
                  last = False, loss = False, reaction = False):
    
    if (path.N < 3):
        raise ValueError("Too few images requested!")
    
    #We Define Our Loss Function And Losses Array
    lossfunction = RegressiveLoss()
    losses = []
    
    #Iterations Of Updating The Entire String
    for epoch in range(epochs):

        if (plotting and epoch > 0):
            plotMEP(path.zN.detach(), potential, levels, r, bounds, ticks, figure=epoch)
        
        predicted = path()
        L = lossfunction(trajectory, predicted)
        losses.append(L)

        optimize.zero_grad()
        L.backward()
        optimize.step() 
        path.step()

    if (last or plotting):
        plotMEP(path.zN.detach(), potential, levels, r, bounds, ticks, figure=epoch)
    
    if (loss):
        plotLosses(epochs, losses)
    
    if (reaction):
        plotPotential(path)

    return path, losses
    
def ContinuousNEB(path, lossfunction, optimize, potential, levels, r, bounds, ticks, epochs = 200, plotting = False, last = False, 
            loss = False, reaction = False):
    
    if (path.N < 3):
        raise ValueError("Too few images requested!")
    
    #We Define Our Losses Array
    losses = []
    
    #Iterations Of Updating The Entire String
    for epoch in range(epochs):

        if (plotting and epoch > 0):
            plotMEP(path.zN.detach(), potential, levels, r, bounds, ticks, figure=epoch)
        
        path()
        L = lossfunction(path)
        losses.append(L)

        optimize.zero_grad()
        L.backward()
        optimize.step() 
        path.step()

    if (last or plotting):
        plotMEP(path.zN.detach(), potential, levels, r, bounds, ticks, figure=epoch)
    
    if (loss):
        plotLosses(epochs, losses)
    
    if (reaction):
        plotPotential(path)

    return path, losses