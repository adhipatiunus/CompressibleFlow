#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 13:15:05 2022

@author: adhipatiunus
"""

import numpy as np

def neighbor_search_naive(particle, cell_size):
    N = len(particle.index)
    particle.neighbor = [[] for i in range(N)]
    for i in range(N):
        for j in range(N):
            if distance(particle.x[i], particle.x[j], particle.y[i], particle.y[j]) < cell_size:
                particle.neighbor[i].append(particle.index[j])
                
def neighbor_search_cell_list(particle, cell_size, y_max, y_min, x_max, x_min):
    nrows = int((y_max - y_min) / cell_size) + 1
    ncols = int((x_max - x_min) / cell_size) + 1

    cell = [[[] for i in range(ncols)] for j in range(nrows)]

    N = len(particle.index)

    for i in range(N):
        listx = int((particle.x[i] - x_min) / cell_size)
        listy = int((particle.y[i] - y_min) / cell_size)
        cell[listx][listy].append(particle.index[i])
        
    particle.neighbor_all   = [[] for i in range(N)]
    particle.neighbor_xpos  = [[] for i in range(N)]
    particle.neighbor_xneg  = [[] for i in range(N)]
    particle.neighbor_ypos  = [[] for i in range(N)]
    particle.neighbor_yneg  = [[] for i in range(N)]
    
    pcnt = 0

    for i in range(nrows):
        for j in range(ncols):
            for pi in cell[i][j]:
                neigh_row, neigh_col = i - 1, j - 1
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i - 1, j
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i - 1, j + 1
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i, j - 1
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i, j
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i, j + 1
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i + 1, j - 1
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i + 1, j
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                
                neigh_row, neigh_col = i + 1, j + 1
                push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols)
                print(str(pcnt/N*100)+'%')
                pcnt += 1
    print(str(pcnt/N*100)+'%')
                            
def push_back_particle(particle, cell_size, cell, pi, neigh_row, neigh_col, nrows, ncols):
    if neigh_row >= 0 and neigh_col >= 0 and neigh_row < nrows and neigh_col < ncols:
        for pj in cell[neigh_row][neigh_col]:
            if distance(particle.x[pi], particle.x[pj], particle.y[pi], particle.y[pj]) < cell_size and pi != pj:
                x_ij = particle.x[pi] - particle.x[pj]
                y_ij = particle.y[pi] - particle.y[pj]
                if x_ij < 10**-6:
                    particle.neighbor_xpos[pi].append(pj)
                elif x_ij > -10**-6:
                    particle.neighbor_xneg[pi].append(pj)
                if y_ij < 10**-6:
                    particle.neighbor_ypos[pi].append(pj)
                elif y_ij > -10**-6:
                    particle.neighbor_yneg[pi].append(pj)
                particle.neighbor_all[pi].append(pj)
                    
def distance(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

