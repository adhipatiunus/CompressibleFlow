#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:39:38 2022

@author: adhipatiunus
"""
import numpy as np

class Particle:
    def __init__(self):
        self.x = np.array([])
        self.y = np.array([])
        self.index = np.array([])
        self.diameter = np.array([])
        self.boundary = np.array([])
        self.solid = np.array([])
        self.neighbor_all = []
        self.neighbor_xpos = []
        self.neighbor_xneg = []
        self.neighbor_ypos = []
        self.neighbor_yneg = []
        self.u = np.array([])
        self.v = np.array([])
        self.T = np.array([])
        self.p = np.array([])
        self.rho = np.array([])
        