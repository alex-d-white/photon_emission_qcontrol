#!/usr/bin/python

import numpy as np
import scipy.optimize
from matplotlib import pyplot as plt

import cavity_qed_systems
import pulse
import cmps_utils
import tncontract as tn
import parameterized_pulse as ppulse

import sys


def optimize_overlap(state, target_mps, pulse):
    """Right now we assume that there is only one pulse."""
    def _obj_fun(x):
        pulse.update(x)
        olap = np.abs(state.get_inner_product(target_mps, [pulse]))**2
        print("Current overlap = ", olap)
        return -olap
    
    def _gradient(x):
        pulse.update(x)
        inner_prod = state.get_inner_product(target_mps, [pulse])
        grad_inner_prod = state.get_inner_prod_gradient(
            target_mps, [pulse])[0]
        grad = -2 * pulse.get_gradient(state.times) @ np.real(
            grad_inner_prod * np.conj(inner_prod))
        #print(np.shape(grad))
        return grad.astype(float)

    print(pulse.bounds())
    print(np.shape(pulse.state()))
    scipy.optimize.minimize(_obj_fun, pulse.state(), bounds=pulse.bounds(),jac=_gradient, method="L-BFGS-B", 
                            options={'disp':None, 'maxiter':200, 'ftol':1e-8, 'gtol':1e-8, 'maxcor':50})
    

    
def setupOpt(delta, g, kappa, bounds):
    #run opt
    targetState = cavity_qed_systems.ModulatedTavisCumming(0.01, 2000, [0,0], g, kappa)
    targetMPS = targetState.get_mps([pulse.ConstantPulse(0)])
    state = cavity_qed_systems.ModulatedTavisCumming(0.01, 2000, [-delta/2,delta/2], g, kappa)
    
    deltaPulse = ppulse.DirectParameterizedPulse(20, 2000, [-bounds/2, bounds/2])
    
    n=5
    x = np.random.uniform(-bounds/2, bounds/2, 2000 + 1)
    LPF = 1/n*np.ones(n)
    x = np.convolve(x, LPF, mode = 'same')
    x = bounds/2 * x/max(x)
    
    deltaPulse.update(x)
    
    
#     plt.plot(deltaPulse.state())
    optimize_overlap(state, targetMPS, deltaPulse)
#     plt.plot(deltaPulse.state())
#     plt.show()

    #run precise overlap
    targetState = cavity_qed_systems.ModulatedTavisCumming(0.001, 20000, [0,0], g, kappa)
    targetMPS = targetState.get_mps([pulse.ConstantPulse(0)])
    state = cavity_qed_systems.ModulatedTavisCumming(0.001, 20000, [-delta/2,delta/2], g, kappa)
    overlapStartValue = (np.abs(state.get_inner_product(targetMPS, [pulse.ConstantPulse(0)]))**2)
    overlapEndValue = (np.abs(state.get_inner_product(targetMPS, [deltaPulse]))**2)
    
    return(overlapStartValue, overlapEndValue)
    
    
if __name__ == "__main__":
    # execute only if run as a script
    #print(sys.argv)
    #Takes command line arguements delta, g, kappa, bounds (all floats)
    delta = float(sys.argv[1])
    g = float(sys.argv[2])
    kappa = float(sys.argv[3])
    bounds =  float(sys.argv[4])
    
    print(setupOpt(delta, g, kappa, bounds))
    
    