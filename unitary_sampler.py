import numpy as np
import pickle
import os
from functools import reduce

def haar_qr(N, rng: np.random.default_rng):
    """" Generate Haar unitary of size NxN using QR decomposition """
    A, B = rng.normal(0,1,size=(N,N)), rng.normal(0,1,size=(N,N))
    Z = A + 1j * B
    Q, R = np.linalg.qr(Z)
    Lambda = np.diag(R.diagonal()/np.abs(R.diagonal()))
    return np.dot(Q,Lambda)


def U_1_sym_gate_sampler(rng: np.random.default_rng):
    U = np.zeros((4,4), dtype=complex)
    phase1 = rng.uniform(0,1)
    phase2 = rng.uniform(0, 1)
    U[0, 0] = np.exp(-1j * 2 * np.pi * phase1)
    U[3, 3] = np.exp(-1j * 2 * np.pi * phase2)
    U_10 = haar_qr(2,rng)
    U[np.array([1,2]).reshape((2,-1)),np.array([1,2])] = U_10
    return U


def Z_2_sym_gate(rng: np.random.default_rng):
    U = np.zeros((4,4), dtype=complex)
    U_00 = haar_qr(2,rng)
    U[np.array([0,3]).reshape((2,-1)),np.array([0,3])] = U_00
    U_10 = haar_qr(2,rng)
    U[np.array([1,2]).reshape((2,-1)),np.array([1,2])] = U_10
    return U


## This function was used previously. But this is not the general form of MG. In fact this has particle-hole symmetry
def matchgate_sampler_particle_hole_symmetric(rng: np.random.default_rng):
    U = np.zeros((4,4), dtype=complex)
    global_phase = np.exp(-1j * 2 * np.pi * rng.uniform(0,1))
    U[0, 0] = 1 
    U[3, 3] = 1 
    U_10 = haar_qr(2,rng)
    U_10 = U_10/np.sqrt(np.linalg.det(U_10))
    U[np.array([1,2]).reshape((2,-1)),np.array([1,2])] = U_10
    U = U*global_phase
    return U

def matchgate_sampler_Z_2(rng: np.random.default_rng):
    U = np.zeros((4,4), dtype=complex)
    U_00 = haar_qr(2,rng)
    U_00 = U_00/np.sqrt(np.linalg.det(U_00))
    U[np.array([0,3]).reshape((2,-1)),np.array([0,3])] = U_00
    U_10 = haar_qr(2,rng)
    U_10 = U_10/np.sqrt(np.linalg.det(U_10))
    U[np.array([1,2]).reshape((2,-1)),np.array([1,2])] = U_10

    return U


def weak_interacting_gates(g,seed,num_of_gates):
    U_list = []
    rng = np.random.default_rng(seed=seed)
    for i in range(num_of_gates):
        ran_num = rng.uniform(0,1)
        if ran_num < g:
            U_list.append(Z_2_sym_gate(rng))
        else:
            U_list.append(matchgate_sampler_Z_2(rng))
    return U_list