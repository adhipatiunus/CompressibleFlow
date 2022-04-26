#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 20:29:29 2022
@author: adhipatiunus
"""

from generate_particles import generate_particle_singular, generate_particle_multires, generate_particle_bump
from neighbor_search import neighbor_search_cell_list
from LSMPS import LSMPS
import matplotlib.pyplot as plt
import numpy as np
import sys

x_min, x_max, y_min, y_max = 0, 5, 0, 5
R = 0.5
x_center, y_center = 1.0, 1.0
sigma = 0.05
R_e1 = 3.5
R_e2 = 3.5
cell_size = 3.5 * sigma

particle, n_boundary, n1, n2 = generate_particle_bump(x_min, x_max, y_min, y_max, sigma)

#particle, n_boundary = generate_particle_singular(x_min, x_max, y_min, y_max, x_center, y_center, R, sigma)
#%%
neighbor_search_cell_list(particle, cell_size, y_max, y_min, x_max, x_min)

N = len(particle.x)

gamma = 1.4
R = 287
M0 = 0.6
T0 = 288.15
rho0 = 1.225
mu0 = 1.789 * 10**-5
eta = 10**-4
eta_T = 10**-2

T = T0 * np.ones(N)
u = M0 * np.sqrt(gamma * R * T)
v = np.zeros_like(u)
rho = rho0 * np.ones(N)
p = rho * R * T
mu = mu0 * (T / T0)**1.5 * (T0+110) / (T + 110)
omega = 0

t = 0
t_end = 10
dt = 5*10**-4
#%%
EtaDxAll, EtaDyAll, EtaDxxAll, EtaDxyAll, EtaDyyAll = LSMPS(particle, R_e2, 'all')
#%%

dx_2d_all = EtaDxAll.copy()
dy_2d_all = EtaDxAll.copy()
dxx_2d_all = EtaDxxAll.copy()
dxy_2d_all = EtaDxyAll.copy()
dyy_2d_all = EtaDyyAll.copy()

n_total = N
# variable (n-1)
u_prev = u.copy()
v_prev = v.copy()
p_prev = p.copy()
T_prev = T.copy()
rho_prev = rho.copy()

# Matrix and vector
A = np.zeros((n_total, n_total))
B = np.zeros((n_total, n_total))
C = np.zeros(n_total)
D = np.zeros((n_total, n_total))
E = np.zeros((n_total, n_total))
F = np.zeros(n_total)

#%%
def get_brinkman_penalization(particle, eta, omega, n_boundary, n_total, u):
    penalization = np.zeros(n_total)
    for i in range(n_total):
        if particle.solid[i] == True:
            penalization[i] = 1 / eta * u[i]
    return penalization

def get_brinkman_penalization_temperature(particle, eta, n_boundary, n_total, T, T_obs):
    penalization = np.zeros(n_total)
    for i in range(n_total):
        if particle.solid[i] == True:
            penalization[i] = 1 / eta * (T[i]-T_obs)
    return penalization
    
#%%
gamma = 1.4
R = 287
M0 = 0.5
T0 = 288.15
rho0 = 1.225
mu0 = 1.789 * 10**-5
eta = 10**-4
C_p = gamma * R / (gamma - 1)
C_v = R / (gamma - 1)
Pr = 0.71
T_obs = T0

T = T0 * np.ones(N)
u = M0 * np.sqrt(gamma * R * T)
v = np.zeros_like(u)
rho = rho0 * np.ones(N)
p = rho * R * T
mu = mu0 * (T / T0)**1.5 * (T0+110) / (T + 110)
omega = 0
k = mu * C_p / Pr

u_bound = u[:n_boundary]
u_bound[n1:n2] = 0
v_bound = v[:n_boundary]
v_bound[n1:n2] = 0
T_bound = T[:n_boundary]
p_bound = p[:n_boundary]
rho_bound = rho[:n_boundary]

# Matrix and vector
A = np.zeros((n_total, n_total))
B = np.zeros((n_total, n_total))
C = np.zeros(n_total)
D = np.zeros((n_total, n_total))
E = np.zeros((n_total, n_total))
F = np.zeros(n_total)

t = 0
t_end = 10
alpha_C = 0.1
dt = 10**-5

# Matrix A
A = 3.0 * (np.eye(n_total).T * rho).T / (2 * dt) \
    + np.multiply(dx_2d_all, rho * u) \
    + np.multiply(dy_2d_all, rho * v) \
    - 4.0 / 3.0 * np.multiply(dxx_2d_all, mu) \
    - np.multiply(dyy_2d_all, mu)
    
# Matrix B
B = - 1.0 / 3.0 * np.multiply(dxy_2d_all, mu)

# Matrix C
C = 2 * (rho * u) / dt - (rho_prev * u_prev) / (2 * dt) \
    - np.matmul(dx_2d_all, p) \
    - get_brinkman_penalization(particle, eta, omega, n_boundary, n_total, u)

# Matrix D
D = - 1.0 / 3.0 * np.multiply(dxy_2d_all, mu)

# Matrix D
E = 3.0 * (np.eye(n_total).T * rho).T / (2 * dt) \
    + np.multiply(dx_2d_all, rho * u) \
    + np.multiply(dy_2d_all, rho * v) \
    - 4.0 / 3.0 * np.multiply(dyy_2d_all, mu) \
    - np.multiply(dxx_2d_all, mu)
    
F = 2 * (rho * v) / dt - (rho_prev * v_prev) / (2 * dt) \
    - np.matmul(dy_2d_all, p) \
    - get_brinkman_penalization(particle, eta, omega, n_boundary, n_total, v)
    
M = np.matmul(E, A) - np.matmul(B, D)
b = np.matmul(E, C) - np.matmul(B, F)
u_pred = np.linalg.solve(M, b)

M = np.matmul(D, B) - np.matmul(A, E)
b = np.matmul(D, C) - np.matmul(A, F)

v_pred = np.linalg.solve(M, b) 

# Update B.C.
u_pred[:n_boundary] = u_bound
v_pred[:n_boundary] = v_bound

dm = 3 * rho / (2 * dt) - 2 * rho / dt + rho_prev / (2 * dt) \
        + np.matmul(dx_2d_all, rho * u_pred) \
        + np.matmul(dy_2d_all, rho * v_pred)
        
C_rho = 1 / (R * T)
        
LHS_pc = 3 * C_rho * np.eye(n_total) / (2 * dt) \
            + np.multiply(dx_2d_all, C_rho * u_pred) \
            + np.multiply(dy_2d_all, C_rho * v_pred) \
            - 2 * dt / 3 * (dxx_2d_all + dyy_2d_all)

p_change = np.linalg.solve(LHS_pc, -dm)

u_corr = u_pred - 2 * dt / (3 * rho) * np.matmul(dx_2d_all, p_change)
v_corr = v_pred - 2 * dt / (3 * rho) * np.matmul(dy_2d_all, p_change)

u_corr[:n_boundary] = u_bound
v_corr[:n_boundary] = v_bound

LHS_rho = 3 * np.eye(n_total) / (2 * dt) \
            + (dx_2d_all.T * u_corr).T \
            + (dy_2d_all.T * v_corr).T
            
u_change = u_corr - u_pred
v_change = v_corr - v_pred

rho_change = C_rho * p_change

rho_pred = rho + rho_change
p_corr = p + p_change

p_corr[:n_boundary] = p_bound
rho_pred[:n_boundary] = rho_bound

LHS_T = 3 *  (np.eye(n_total).T * (rho_pred * C_v)).T / (2 * dt) \
        + np.multiply(dx_2d_all, rho_pred * C_v * u_corr) \
        + np.multiply(dy_2d_all, rho_pred * C_v * v_corr) \
        - np.multiply(dxx_2d_all, k) - np.multiply(dyy_2d_all, k)
        
tau_xx = 2.0 / 3.0 * (2 * np.multiply(dx_2d_all, mu * u_corr) \
                      - np.multiply(dy_2d_all, mu * v_corr))

tau_xy = np.multiply(dy_2d_all, mu * u_corr) \
        + np.multiply(dx_2d_all, mu * v_corr)
            
tau_yy = 2.0 / 3.0 * (2 * np.multiply(dy_2d_all, mu * v_corr) \
                      - np.multiply(dx_2d_all, mu * u_corr))
        
RHS_T = 2 * rho_pred * C_v * T / dt \
        - rho_pred * C_v * T_prev / (2 * dt) \
        + tau_xx * np.matmul(dx_2d_all, u_corr) \
        + tau_xy * np.matmul(dy_2d_all, u_corr) \
        + tau_xy * np.matmul(dx_2d_all, v_corr) \
        + tau_yy * np.matmul(dy_2d_all, v_corr) \
        - p_corr * (np.matmul(dx_2d_all, u_corr) + np.matmul(dy_2d_all, v_corr)) \
        - get_brinkman_penalization_temperature(particle, eta_T, n_boundary, n_total, T, T_obs)

T_corr = np.linalg.solve(LHS_T, RHS_T)
T_corr[:n_boundary] = T_bound

rho_corr = p_corr / (R * T_corr)
rho_corr[:n_boundary] = rho_bound 

u_prev = u
v_prev = v
p_prev = p
T_prev = T
rho_prev = rho

u = u_corr
v = v_corr
p = p_corr
T = T_corr
rho = rho_corr

i = 0

while(t < t_end):
    print('Simulating, t = ' + str(t))
    # Matrix A
    A = 3.0 * (np.eye(n_total).T * rho).T / (2 * dt) \
        + np.multiply(dx_2d_all, rho * u) \
        + np.multiply(dy_2d_all, rho * v) \
        - 4.0 / 3.0 * np.multiply(dxx_2d_all, mu) \
        - np.multiply(dyy_2d_all, mu)
        
    # Matrix B
    B = - 1.0 / 3.0 * np.multiply(dxy_2d_all, mu)

    # Matrix C
    C = 2 * (rho * u) / dt - (rho_prev * u_prev) / (2 * dt) \
        - np.matmul(dx_2d_all, p) \
        - get_brinkman_penalization(particle, eta, omega, n_boundary, n_total, u)

    # Matrix D
    D = - 1.0 / 3.0 * np.multiply(dxy_2d_all, mu)

    # Matrix D
    E = 3.0 * (np.eye(n_total).T * rho).T / (2 * dt) \
        + np.multiply(dx_2d_all, rho * u) \
        + np.multiply(dy_2d_all, rho * v) \
        - 4.0 / 3.0 * np.multiply(dyy_2d_all, mu) \
        - np.multiply(dxx_2d_all, mu)
        
    F = 2 * (rho * v) / dt - (rho_prev * v_prev) / (2 * dt) \
        - np.matmul(dy_2d_all, p) \
        - get_brinkman_penalization(particle, eta, omega, n_boundary, n_total, v)
        
    M = np.matmul(E, A) - np.matmul(B, D)
    b = np.matmul(E, C) - np.matmul(B, F)
    u_pred = np.linalg.solve(M, b)

    M = np.matmul(D, B) - np.matmul(A, E)
    b = np.matmul(D, C) - np.matmul(A, F)

    v_pred = np.linalg.solve(M, b) 

    # Update B.C.
    u_pred[:n_boundary] = u_bound
    v_pred[:n_boundary] = v_bound

    dm = 3 * rho / (2 * dt) - 2 * rho / dt + rho_prev / (2 * dt) \
            + np.matmul(dx_2d_all, rho * u_pred) \
            + np.matmul(dy_2d_all, rho * v_pred)
            
    C_rho = 1 / (R * T)
            
    LHS_pc = 3 * C_rho * np.eye(n_total) / (2 * dt) \
                + np.multiply(dx_2d_all, C_rho * u_pred) \
                + np.multiply(dy_2d_all, C_rho * v_pred) \
                - 2 * dt / 3 * (dxx_2d_all + dyy_2d_all)

    p_change = np.linalg.solve(LHS_pc, -dm)

    u_corr = u_pred - 2 * dt / (3 * rho) * np.matmul(dx_2d_all, p_change)
    v_corr = v_pred - 2 * dt / (3 * rho) * np.matmul(dy_2d_all, p_change)

    u_corr[:n_boundary] = u_bound
    v_corr[:n_boundary] = v_bound

    LHS_rho = 3 * np.eye(n_total) / (2 * dt) \
                + (dx_2d_all.T * u_corr).T \
                + (dy_2d_all.T * v_corr).T
                
    u_change = u_corr - u_pred
    v_change = v_corr - v_pred

    rho_change = C_rho * p_change

    rho_pred = rho + rho_change
    p_corr = p + p_change

    p_corr[:n_boundary] = p_bound
    rho_pred[:n_boundary] = rho_bound

    LHS_T = 3 *  (np.eye(n_total).T * (rho_pred * C_v)).T / (2 * dt) \
            + np.multiply(dx_2d_all, rho_pred * C_v * u_corr) \
            + np.multiply(dy_2d_all, rho_pred * C_v * v_corr) \
            - np.multiply(dxx_2d_all, k) - np.multiply(dyy_2d_all, k)
            
    tau_xx = 2.0 / 3.0 * (2 * np.multiply(dx_2d_all, mu * u_corr) \
                          - np.multiply(dy_2d_all, mu * v_corr))

    tau_xy = np.multiply(dy_2d_all, mu * u_corr) \
            + np.multiply(dx_2d_all, mu * v_corr)
                
    tau_yy = 2.0 / 3.0 * (2 * np.multiply(dy_2d_all, mu * v_corr) \
                          - np.multiply(dx_2d_all, mu * u_corr))
            
    RHS_T = 2 * rho_pred * C_v * T / dt \
            - rho_pred * C_v * T_prev / (2 * dt) \
            + tau_xx * np.matmul(dx_2d_all, u_corr) \
            + tau_xy * np.matmul(dy_2d_all, u_corr) \
            + tau_xy * np.matmul(dx_2d_all, v_corr) \
            + tau_yy * np.matmul(dy_2d_all, v_corr) \
            - p_corr * (np.matmul(dx_2d_all, u_corr) + np.matmul(dy_2d_all, v_corr)) \
            - get_brinkman_penalization_temperature(particle, eta_T, n_boundary, n_total, T, T_obs)
            
    T_corr = np.linalg.solve(LHS_T, RHS_T)    
    T_corr[:n_boundary] = T_bound
    
    rho_corr = p_corr / (R * T_corr)
    rho_corr[:n_boundary] = rho_bound 

    if min(T_corr) <= 0:
        sys.exit()
    
    u_prev = u
    v_prev = v
    p_prev = p
    T_prev = T
    rho_prev = rho

    u = u_corr
    v = v_corr
    p = p_corr
    T = T_corr
    rho = rho_corr
    
    mu = mu0 * (T / T0)**1.5 * (T0+110) / (T + 110)
    k = mu * C_p / Pr
    
    t += dt
    if i % 10==0:
        np.savez('File' + str(int(i/10)) + '.npz', particle.x,particle.y,u,v,p,T,rho)
    i += 1
    dt = 10**-5
