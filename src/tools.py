import matplotlib.pyplot as plt
import matplotlib

import os
import numpy as np
import scipy.optimize as op
from scipy.integrate import solve_ivp
import pandas as pd
import datetime
import ast

def get_conservation(model):
    known = [[1,1,1,0,0],
             [0,0,0,2,1],
             [7,3,2,-2,0],
            ]
    if model == "01":
        intmed = [[1,1,0],
                  [0,0,2],
                  [6,5,-1],
                 ]
    elif model == "02":
        intmed = [[1,1,0],
                  [0,0,2],
                  [6,4,-1],
                 ] 
    elif model == "03":
        intmed = [[1,1,0],
                  [0,0,2],
                  [5,4,-1],
                 ]  
    elif model == "04":
        intmed = [[1,1,0],
                  [0,0,1],
                  [6,5,-1],
                 ]
    elif model == "05":
        intmed = [[1,1,0],
                  [0,0,1],
                  [6,4,-1],
                 ] 
    elif model == "06":
        intmed = [[1,1,0],
                  [0,0,1],
                  [5,4,-1],
                 ] 
    elif model == "07":
        intmed = [[1,1,0,0],
                  [0,0,2,1],
                  [6,5,-1,-1],
                 ] 
    elif model == "08":
        intmed = [[1,1,0,0],
                  [0,0,2,1],
                  [6,4,-1,-1],
                 ] 
    elif model == "09":
        intmed = [[1,1,0,0],
                  [0,0,2,1],
                  [5,4,-1,-1],
                 ] 
    elif model == "10":
        intmed = [[1,1,1,0],
                  [0,0,0,2],
                  [6,5,4,-1],
                 ] 
    elif model == "11":
        intmed = [[1,1,1,0],
                  [0,0,0,1],
                  [6,5,4,-1],
                 ] 
    elif model == "12":
        intmed = [[1,1,1,0,0],
                  [0,0,0,2,1],
                  [6,5,4,-1,-1],
                 ] 
    return [k + i for k, i in zip(known, intmed)]

def chem_equation_with_oxalate(l, r, k=None, chemformula=None, conservation=None):
    if conservation is not None:
        _l = compensate_oxalate_l(l,r,conservation)
    else:
        _l = l
    return chem_equation_str(_l, r, k=k, chemformula=chemformula)

def chem_equation_str(l, r, k=None, chemformula=None):
    chem_eq_all = ""
    length = 15
    if k is None:
        k = np.zeros(l.shape[0])
    if chemformula is None:
        known_name = ["Mn7", "Mn3", "Mn2"]
        chemformula = known_name + (["I" + str(i) for i in range(l.shape[1]-len(known_name))])
    
    for i in range(l.shape[0]):
        chem_eq = "(" + str(i+1).rjust(2) + ") "
        for j in range(l.shape[1]):
            if l[i,j] > 1.:
                chem_eq += str(int(l[i,j])) + " " + chemformula[j]
                chem_eq += " + "
            elif l[i,j] == 1.:
                chem_eq += chemformula[j]
                chem_eq += " + "
        chem_eq = chem_eq[:-3].ljust(length+5)
        chem_eq += " -> "

        for j in range(r.shape[1]):
            if r[i,j] > 1.:
                chem_eq += str(int(r[i,j])) + " " + chemformula[j]
                chem_eq += " + "
            elif r[i,j] == 1.:
                chem_eq += chemformula[j]
                chem_eq += " + "
        chem_eq = chem_eq[:-3].ljust(int(length*2+9))
        chem_eq += " k = " + "{:.2f}".format(k[i]) + "\n"
        
        chem_eq_all += chem_eq
    return chem_eq_all

def compensate_oxalate_l(l,r,ci):
    eq_Mn = ci[0]
    eq_C = ci[1]
    ox = ci[2]
    l2 = l.copy()
    for i in range(l.shape[0]):
        num_C2O4 = int(np.sum(r[i,:]*eq_C - l[i,:]*eq_C)/2)
        if num_C2O4 != 0:
            l2[i,3] += num_C2O4
    return l2       

def generate_chem_formula(conservation):
    eq_Mn = conservation[0]
    eq_C = conservation[1]
    ox = conservation[2]
    chemformula = []
    for i in range(len(eq_Mn)):
        name = ""
        if eq_Mn[i] > 1:
            name = name + "Mn" + str(int(eq_Mn[i]))
        elif eq_Mn[i] == 1:
            name = name + "Mn"
        
        if eq_C[i] > 1:
            name = name + "C" + str(int(eq_C[i])) + "O" + str(int(eq_C[i]*2))
        elif eq_C[i] == 1:
            name = name + "CO2"
        
        if ox[i] > 0:
            name = name + "+" + str(int(ox[i]))
        elif ox[i] < 0:
            name = name + str(int(ox[i]))        
        chemformula.append(name)
    return chemformula