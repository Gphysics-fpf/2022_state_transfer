import numpy as np
from numpy import floor, sqrt
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import pi as π
import copy


class Exp2QB2Cavities:
    
    def __init__(self, δ1=2*π*8.406, δ2=2*π*8.406, g1=2*π*0.0086, g2=2*π*0.0086,
                 ω1=2*π*8.406, ω2=2*π*8.406, κ1=2*π*0.0086, κ2=2*π*0.0086, l=5, mmax=100,
                 Gconstant=False, δLamb=0.0):
        
        """Experiment with 2 qubits, 2 cavities and waveguide.
        ..."""
        self.δ1 = δ1
        self.δ2 = δ2
        self.δLamb = δLamb
        self.g1 = g1
        self.g2 = g2
        self.ω1 = ω1
        self.ω2 = ω2
        self.κ1 = κ1 = κ1*10**9
        self.κ2 = κ2 = κ2*10**9
        self.l = l
        self.mmax = mmax
        self.Gconstant = Gconstant
        self.compute_matrices_()

    def compute_matrices_(self):
        # We then set the parameters of the waveguide a = width
        m = np.arange(self.l*80)                               # Number of modes in the bandwidth
        a = 0.02286                                            # Width of our waveguide (in principle fixed)
        c = 299792458                                          # Speed of light in vacuum
        self.ω = (2*π)*(c*np.sqrt((1/(2*a))**2 + (m/(2*self.l))**2))/10**9 # ω of the modes (in GHz)

        #we obtain the mode that is resonant with cavity 1
        self.mcentral = np.abs(self.ω-self.ω1).argmin()
        mmin = max(self.mcentral - int(self.mmax/2), 0)
        self.mrelevant = np.arange(mmin, mmin + self.mmax)
        self.ωrelevant = self.ω[mmin:mmin+self.mmax]
        
        νrelevant = self.ωrelevant*1e9/(2*π)
        vgroup = c*np.sqrt(1 - (c/(2*a*νrelevant))**2)
        
        # Index of the mode that is resonant with the qubit 1
        resonant_ndx = np.abs(self.ωrelevant-self.ω1).argmin()
        G1 = np.sqrt(((self.κ1)*vgroup)/(2*self.l))/10**9
        G2 = np.sqrt(((self.κ2)*vgroup)/(2*self.l))/10**9
        
        if self.Gconstant: 
            self.G1 = G1[resonant_ndx]
            self.G2 = G2[resonant_ndx]
        else:
            self.G1 = G1[resonant_ndx]*np.sqrt(self.ωrelevant/self.ωrelevant[resonant_ndx])
            self.G2 = G2[resonant_ndx]*np.sqrt(self.ωrelevant/self.ωrelevant[resonant_ndx])
            
        #self.tprop = tprop = (self.l/(vgroup[mcentral]))*10**9. Since we decide to define the group velocity
        #in terms of ωrelevant, we cannot use vgroup[mcentral] anymore.
        
        self.tprop = (self.l/(vgroup[resonant_ndx]))*10**9
        
        # initialize the matrix A and write the values for the cavity frequencies
        H = np.zeros((self.mmax+4, self.mmax+4))
        # Fill the matrix with the Waveguide frequencies
        H[4:,4:] = np.diag(self.ωrelevant)
        H[0,0] = self.δ1 + self.δLamb
        H[1,1] = self.δ2 + self.δLamb
        H[2,2] = self.ω1
        H[3,3] = self.ω2
        if self.κ1:
            H[2,4:] = H[4:,2]= self.G1 * self.mode(self.mrelevant, self.l, x=0)
        if self.κ2:
            H[3,4:] = H[4:,3]= self.G2 * self.mode(self.mrelevant, self.l, x=self.l)        
        self.H = sp.csr_matrix(H)
        self.size = self.H.shape[0]

        # Qubit1 - cavity1 interaction
        self.Hqbcav1 = sp.csr_matrix(([1, 1], ([0,2],[2,0])), shape=self.H.shape)
        # Qubit2 - cavity2 interaction
        self.Hqbcav2 = sp.csr_matrix(([1, 1], ([1,3],[3,1])), shape=self.H.shape)

    def Hamiltonian(self, g1=None, g2=None):
        """Adds the qubit part to the hamiltonian matrix"""
        return self.H + (self.g1 if g1 is None else g1) * self.Hqbcav1 \
                      + (self.g2 if g2 is None else g2) * self.Hqbcav2
        
    def voltage_positition(self, ψtk, t, x):
        """
        Input:
        ψtk[i,k] = matrix with states at t_i for momentum 'k'
        t[i] = different instants of time
        x[n] = positions to evaluate the wavefunction

        Output:
        V[n,i] = voltages at positions x_n and times t_i
        """
        phases = np.exp(-1j*self.ωrelevant*t.reshape(-1,1))
        weights = np.abs(ψtk[:,4:])**2
        k = π * self.mrelevant / self.l
        modes = np.sqrt(self.ωrelevant) * np.cos(k * x.reshape(-1,1))
        V = modes @ (phases * weights).T
        return V

    def mode(self, m, l, x):
        """Return waveguide mode evaluated at position x for all frequencies"""
        return np.cos((π*m/l)*x)
    
    def cavity_excited(self, which=0):
        if which not in [0,1]:
            raise Exception(f'Cavity number {which} not in [0,1]')
        C = np.zeros(self.size)
        C[which + 2] = 1
        return C  # state with cavity 'which' excited with 1 photon
    
    def qubit_excited(self, which=0):
        if which not in [0,1]:
            raise Exception(f'Qubit number {which} not in [0,1]')
        C = np.zeros(self.size)
        C[which] = 1
        return C  # state with qubit 'which' excited with 1 photon
    
    def sum_EM_modes(self, output):
        ψ = []
        for i in range(output.shape[0]):
            ψ.append(np.sum(np.abs(output[i, 4:])**2))
        return ψ
    
    def change_parameters(self, **kwdargs):
        output = copy.deepcopy(self)
        for k in kwdargs:
            if k not in output.__dict__:
                raise Exception(f'Unknown parameter {k}')
            output.__dict__[k] = kwdargs[k]
        output.compute_matrices_()
        return output

def evolve_time_dependent(Ht, ψ0, t, nsteps=100):
    """Solve a time-dependent Schrödinger equation
    
    Inputs:  
    Ht: function of time that returns a new sparse matrix H(t) for each time 
    ψ0: vector containing the initial value of the coefficients
    t: vector of all the time intervals we want to take
    """
    
    if not isinstance(t, np.ndarray):
        t = np.linspace(0, t, nsteps)
    output = np.zeros((t.size,) + ψ0.shape, dtype=np.complex128)
    t0 = 0.0
    for (i, tn) in enumerate(t):
        Δt = tn - t0
        ψ0 = scipy.sparse.linalg.expm_multiply((-1j*Δt)*Ht(tn), ψ0)
        output[i] = ψ0
        t0 = tn
    return output

def control_constant(system, t):
    return system.g1

def control_test(system, t):
    return (system.g1)*\
            np.exp(-((system.g1)*4*t)**2)

def control_ETH(system, t):
    return ((system.g1)/2)*\
            (np.cosh(((system.g1)/2)*t))**(-1) 
def control_ETH_truncated(system, t, tmax, tdelay, topen):

    gt1 = πecewise(t, [t < tmax, t >= tmax, t >= tmax + topen], 
                       [lambda t:(system.g1/2)*(np.cosh(((system.g1)/2)*(t-tmax))**(-1)),
                                        lambda t: system.g1/2, lambda t: 0])

    gt2 =  πecewise(t, [t<= tmax + tdelay ,  tmax + tdelay - topen < t , t >= tmax + tdelay], 
                        [lambda t: 0, lambda t: system.g2/2,
                         lambda t: (system.g2/2)*(np.cosh(((system.g2)/2)*(-t+tmax+tdelay))**(-1))])
    return gt1, gt2


def control_stannigel(system, t, tmax, tdelay, topen):
    
    gt1 = πecewise(t, [t < tmax, t >= tmax, t >= tmax + 36], 
              [lambda t: system.g1*(np.exp(system.g1*(t-tmax))/(2-np.exp(system.g1*(t-tmax)))),
                lambda t: system.g1, lambda t: 0])

    gt2 =   πecewise(t, [t<= tmax + tdelay ,  tmax + tdelay -36 < t , t >= tmax + tdelay], 
                         [lambda t: 0, lambda t: system.g2,
                         lambda t: system.g2*(np.exp(system.g2*(-t+tmax+tdelay))/(2-np.exp(system.g2*(-t+tmax+tdelay))))])
    return gt1, gt2


def control_readout(system, t, tmax = 200, tdelay = 300 , topen = 100):
    
    """The intervals guarantee that the protocol is symmetrical. tmax and topen last the same on both ends 
    DURATION = 2*(tmax+topen) + tdelay. The experiment must last longer than Duration  """
    
    gt1 = πecewise(t, [t > 0, t >= tmax, t >= tmax + topen, t >= tmax + topen +tdelay, t >= tmax + topen +tdelay +topen],         
                           [lambda t:(system.g1/2)*(np.cosh(((system.g1)/2)*(t-tmax))**(-1)),
                                            lambda t: system.g1/2, lambda t: 0,
                            lambda t: system.g1/2, lambda t: (system.g1/2)*(np.cosh(((system.g1)/2)*(-t+tmax+topen+tdelay+topen))**(-1))])
    
    return gt1, 0
def step_photon(mmax): 
    #Equiprobable superposition of modes
    γ1 = np.ones(mmax+4)
    γ1[:4] = 0
    γ1 = γ1/np.sqrt(mmax)
    return γ1

def monochromatic_photon(mmax, m):
    #perfectly monochromatic photon (probably not a photon)
    γ2 = np.zeros(mmax+4)
    γ2[m] = 1
    return γ2    

def gaussian_photon(mmax, width):
    # gaussian distribution
    x = np.arange(mmax)
    γ3 = (1/(width*np.sqrt(2*π)))*np.exp(-0.5*((x - (mmax)/2)/width)**2)
    #γ3 = γ3/(γ3@γ3)**(0.5)
    γ3 = np.concatenate((np.zeros(4), γ3))
    # we concatenate to obtain an array of size mmax+4. Only population in photonic modes
    return γ3**0.5

def sech_photon(mmax, width):
    x = np.arange(np.round(-mmax/2), np.round(mmax/2))
    γ4 = 1/np.cosh(x/width)
    γ4 = γ4/(γ4@γ4)**(0.5)    
    γ4 = np.concatenate((np.zeros(4), γ4))
    return γ4

def sech_photon2(mmax, σ):
    x = np.arange(mmax)
    γ5 = (1/(2*σ)) * np.cosh( (π/2)*((x-(mmax)/2)/σ) )**(-1)
    γ5 = np.concatenate((np.zeros(4), γ5))
    return γ5**0.5
 
