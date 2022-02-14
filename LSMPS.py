#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 10:00:13 2022

@author: adhipatiunus
"""

from generate_particles import generate_particle_singular, generate_particle_multires
from neighbor_search import neighbor_search_cell_list
import numpy as np

x_min, x_max, y_min, y_max = 0, 10, 0, 10
R = 1
x_center, y_center = 5, 5
sigma = 0.5
R_e = 3.5
cell_size = R_e * sigma

particle = generate_particle_multires(x_min, x_max, y_min, y_max,
                            x_center, y_center, R, sigma)

neighbor_search_cell_list(particle, cell_size, x_max, x_min, y_max, y_min)

def LSMPS(particle, R_e, typ):
    if typ == 'x':
        EtaDxPos, EtaDxxPos = calculate_derivative(particle, R_e, particle.neighbor_xpos, 'x')
        EtaDxNeg, EtaDxxNeg = calculate_derivative(particle, R_e, particle.neighbor_xneg, 'x')
        return EtaDxPos, EtaDxxPos, EtaDxNeg, EtaDxxNeg
    if typ == 'y':
        EtaDyPos, EtaDyyPos = calculate_derivative(particle, R_e, particle.neighbor_ypos, 'y')
        EtaDyNeg, EtaDyyNeg = calculate_derivative(particle, R_e, particle.neighbor_yneg, 'y')
        return EtaDyPos, EtaDyyPos, EtaDyNeg, EtaDyyNeg
    if typ == 'all':
        EtaDxAll, EtaDyAll, EtaDxxAll, EtaDxyAll, EtaDyyAll, index_all = calculate_derivative(particle, R_e, particle.neighbor_all, 'all')
        return EtaDxAll, EtaDyAll, EtaDxxAll, EtaDxyAll, EtaDyyAll
    
    
def calculate_derivative(particle, R_e, neighbor_list, typ):
    N = len(particle.x)
    b_data = [np.array([])] * N
    EtaDx   = np.zeros((N, N))
    EtaDy   = np.zeros((N, N))
    EtaDxx  = np.zeros((N, N))
    EtaDxy  = np.zeros((N, N))
    EtaDyy  = np.zeros((N, N))
    index   = np.zeros((N, N))
    
    index_inner = [particle.index[i] for i in range(len(particle.x)) if particle.boundary[i] == False]

    for i in index_inner:
        H_rs = np.zeros((6,6))
        M = np.zeros((6,6))
        P = np.zeros((6,1))
        b_temp = [np.array([])] * len(neighbor_list[i])
        #print(i)
        
        #print(i)
        
        Li = particle.diameter[i]
        
        H_rs[0, 0] = 1
        H_rs[1, 1] = Li**-1
        H_rs[2, 2] = Li**-1
        H_rs[3, 3] = 2 * Li**-2
        H_rs[4, 4] = Li**-2
        H_rs[5, 5] = 2 * Li**2
        
        neighbor_idx = neighbor_list[i]
        
        idx_i = i
        x_i = particle.x[idx_i]
        y_i = particle.y[idx_i]
        R_i = particle.diameter[idx_i]
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            x_j = particle.x[idx_j]
            y_j = particle.y[idx_j]
            R_j = particle.diameter[idx_j]
            
            R_ij = R_e * (R_i + R_j) / 2
            x_ij = x_j - x_i
            y_ij = y_j - y_i
            r_ij = np.sqrt((x_ij)**2 + (y_ij)**2)
            
            if r_ij < R_ij:
                w_ij = (r_ij / R_ij - 1)**2
            else:
                w_ij = 0
             
            p_x = x_ij / R_i
            p_y = y_ij / R_i
            
            P[0, 0] = 1.0
            P[1, 0] = p_x
            P[2, 0] = p_y
            P[3, 0] = p_x**2
            P[4, 0] = p_x * p_y
            P[5, 0] = p_y**2
            
            M = M + w_ij * np.matmul(P, P.transpose())
            b_temp[j] = w_ij * P
        #print(M)
        try:
            M_inv = np.linalg.inv(M)
        except:
            M_inv = M
        MinvHrs = H_rs * M_inv
        b_data[i] = b_temp
        
        for j in range(len(neighbor_idx)):
            idx_j = neighbor_idx[j]
            #i[indexdx_i].append(idx_j)
            Eta = np.matmul(MinvHrs, b_data[i][j])
            EtaDx[idx_i,idx_j] = Eta[1]
            EtaDy[idx_i,idx_j] = Eta[2]
            EtaDxx[idx_i,idx_j] = Eta[3]
            EtaDxy[idx_i,idx_j] = Eta[4]
            EtaDyy[idx_i,idx_j] = Eta[5]
            
    if typ == 'all':
        return EtaDx, EtaDy, EtaDxx, EtaDxy, EtaDyy
    elif typ == 'x':
        return EtaDx, EtaDxx
    elif typ == 'y':
        return EtaDy, EtaDyy
        
#EtaDxPos, EtaDxxPos, EtaDxNeg, EtaDxxNeg = LSMPS(particle, R_e, 'x')
#%%
#N = len(particle.index)    
#rho = 1.225
#dt = 0.1
#LHS = 3 * rho / (2 * dt) * np.eye(N) 
EtaDxPos, EtaDxxPos, EtaDxNeg, EtaDxxNeg = LSMPS(particle, R_e, 'x')
        
        



