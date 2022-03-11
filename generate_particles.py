#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 15:13:03 2022

@author: adhipatiunus
"""

from particle import Particle
import matplotlib.pyplot as plt
import numpy as np

def generate_particle_singular(x_min, x_max, y_min, y_max, x_center, y_center, R, sigma):
    lx = x_max - x_min
    ly = y_max - y_min

    nx = int(lx / sigma) + 1
    ny = int(ly / sigma) + 1

    particle = Particle()
    
    y_west = np.linspace(y_min, y_max, ny)
    x_west = x_min * np.ones_like(y_west)
    
    y_east = np.linspace(y_min, y_max, ny)
    x_east = x_max * np.ones_like(y_east)
    
    x_north = np.linspace(x_min + sigma, x_max - sigma, nx - 2)
    y_north = y_max * np.ones_like(x_north)
    
    x_south = np.linspace(x_min + sigma, x_max - sigma, nx - 2)
    y_south = y_min * np.ones_like(x_south)
    
    particle.x = np.concatenate((x_west, x_east, x_north, x_south))
    particle.y = np.concatenate((y_west, y_east, y_north, y_south))
    
    n_boundary = len(particle.x)
    
    x_inner = np.linspace(x_min + sigma, x_max - sigma, nx - 2)
    y_inner = np.linspace(x_min + sigma, x_max - sigma, nx - 2)

    X, Y = np.meshgrid(x_inner, y_inner)

    X_inner = X.flatten()
    Y_inner = Y.flatten()
    
    print(np.shape(X_inner))
    
    particle.x = np.concatenate((particle.x, X_inner))
    particle.y = np.concatenate((particle.y, Y_inner))

    sphere = (particle.x - x_center)**2 + (particle.y - y_center)**2 <= R**2
    N = nx * ny
    particle.solid = np.zeros(N)
    particle.boundary = np.zeros_like(particle.solid)
    particle.boundary[:n_boundary] = True
    particle.diameter = sigma * np.ones_like(particle.solid)
    particle.index = np.arange(0, N)
    particle.solid[sphere] = True

    return particle, n_boundary

def generate_node_spherical(x_center, y_center, R, R_in, R_out, h):
    x_min = x_center - 2 * R_out
    x_max = x_center + 2 * R_out
    y_min = y_center - 2 * R_out
    y_max = y_center + 2 * R_out
    
    lx = x_max - x_min
    ly = y_max - y_min

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    
    X, Y = np.meshgrid(x, y)
    
    node_x = X.flatten()
    node_y = Y.flatten()
    
    delete_inner = (node_x - x_center)**2 + (node_y - y_center)**2 <= R_in
    delete_outer = (node_x - x_center)**2 + (node_y - y_center)**2 > R_out
    delete_node = delete_inner + delete_outer
    
    node_x = node_x[~delete_node]
    node_y = node_y[~delete_node]
    sp = h * np.ones_like(node_x)
    
    return node_x, node_y, sp

def generate_particle_multires(x_min, x_max, y_min, y_max, x_center, y_center, R, sigma):
    h1 = 1/64
    h2 = 1/32
    h3 = 1/16
    h4 = 1/8 
    h5 = 1/4 
    h6 = 1/2  

    h = h1 * sigma
    lx = x_max - x_min
    ly = y_max - y_min

    nx = int(lx / h) + 1
    ny = int(ly / h) + 1

    particle = Particle()

    # West Boundary
    y_west = np.linspace(y_min, y_max, ny)
    x_west = x_min * np.ones_like(y_west)
    sp_west = sigma * np.ones_like(y_west)

    # East Boundary
    y_east = np.linspace(y_min, y_max, ny)
    x_east = x_max * np.ones_like(y_east)
    sp_east = sigma * np.ones_like(y_east)

    # North Boundary
    x_north = np.linspace(x_min + h, x_max - h, nx - 2)
    y_north = y_max * np.ones_like(x_north)
    sp_north = sigma * np.ones_like(x_north)

    # South Boundary
    x_south = np.linspace(x_min + h, x_max - h, nx - 2)
    y_south = y_min * np.ones_like(x_south)
    sp_south = sigma * np.ones_like(x_south)

    particle.x = np.concatenate((x_west, x_east, x_north, x_south))
    particle.y = np.concatenate((y_west, y_east, y_north, y_south))
    particle.diameter = np.concatenate((sp_west, sp_east, sp_north, sp_south))
    
    n_boundary = len(particle.x)

    # Inside Sphere
    R_in = 0
    R_out = R / 4
    h = h5 * sigma

    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    R_in = R_out
    R_out = R / 2
    h = h4 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    R_in = R_out
    R_out = 3 * R / 4
    h = h3 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    R_in = R_out
    R_out = 7 * R / 8
    h = h2 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    R_in = R_out
    R_out = R
    h = h1 * sigma
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    n_sphere = len(particle.x)
    
    # Outside Sphere
    n_layer = 4
    h = h1 * sigma
    R_in = R_out
    R_out = R_in + n_layer * h
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    n_layer = 4
    h = h2 * sigma
    R_in = R_out
    R_out = R_in + n_layer * h
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))

    n_layer = 4
    h = h3 * sigma
    R_in = R_out
    R_out = R_in + n_layer * h
    node_x, node_y, sp = generate_node_spherical(x_center, y_center, R, R_in, R_out, h)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Intermediate Particle
    h = h3 * sigma
    n_layer = 4;
    xmin = x_center - R_out - n_layer * h
    xmax = x_center + R_out + n_layer * h
    ymin = y_center - R_out - n_layer * h
    ymax = y_center + R_out + n_layer * h

    lx_ = xmax - xmin
    ly_ = ymax - ymin

    nx = int(lx_ / h) + 1
    ny = int(ly_ / h) + 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    node_x = X.flatten()
    node_y = Y.flatten()

    delete_inner = (node_x - x_center)**2 + (node_y - y_center)**2 <= R_out**2
    delete_outer = (node_x < x_min) + (node_x > x_max) +  (node_y < y_min) + (node_y > y_max)
    delete = delete_inner + delete_outer
    node_x = node_x[~delete]
    #print(node_x)
    node_y = node_y[~delete]
    sp = h * np.ones_like(node_x)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    # Box Particle
    x_min_inner = xmin
    x_max_inner = xmax
    y_min_inner = ymin
    y_max_inner = ymax
    
    n_layer = 16
    
    h = h4 * sigma
    xmin = x_min + h
    xmax = x_max - h
    ymin = y_min + h
    ymax = y_max - h
    
    lx_ = xmax - xmin
    ly_ = ymax - ymin

    nx = int(lx_ / h) + 1
    ny = int(ly_ / h) + 1

    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)

    X, Y = np.meshgrid(x, y)

    node_x = X.flatten()
    node_y = Y.flatten()

    delete_inner = (node_x > x_min_inner) * (node_x < x_max_inner) * (node_y > y_min_inner) * (node_y < y_max_inner)
    node_x = node_x[~delete_inner]
    node_y = node_y[~delete_inner]
    sp = h * np.ones_like(node_x)

    particle.x = np.concatenate((particle.x, node_x))
    particle.y = np.concatenate((particle.y, node_y))
    particle.diameter = np.concatenate((particle.diameter, sp))
    
    N = len(particle.x)
    particle.index = np.arange(0, N)
    particle.boundary = np.full(N, False)
    particle.boundary[:n_boundary] = True
    particle.solid = np.full(N, False)
    particle.solid[n_boundary:n_sphere] = True
    
    
    return particle, n_boundary