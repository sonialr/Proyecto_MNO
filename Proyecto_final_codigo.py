"""Minimal example to call the GLOP solver."""

# [START program]
from __future__ import print_function

from ortools.linear_solver import pywraplp

import pandas as pd
import numpy as np
import scipy
import scipy.optimize
from scipy import optimize
from math import log
from math import log1p


#def carga_red():
rt = r'C:\Users\jsanjuan\Desktop\MNO\proyecto final\Codigo/'
xl = pd.read_excel(rt + 'BD_MNO4.xlsx', sheet_name = None)
print(xl['Hoja1'].head(20))
print(xl['Hoja2'].head())

df_nodos = xl['nodos']
df_enlaces = xl['enlaces']
df_enlaces['tup'] = df_enlaces[['origen', 'destino']].apply(tuple, axis = 1)

print('Suma neta:', df_nodos['RED_NETO2'].sum() )

dc_bs = dict(zip(df_nodos['nodo'], df_nodos['RED_NETO2']))
dc_bs_enum = dict(zip( df_nodos['nodo'].tolist(), range(0, df_nodos['nodo'].size)))
dc_vars = {od: 0 for od in df_enlaces['tup'].tolist()}
dc_vars_enum = dict(zip(df_enlaces['tup'], range(0, df_enlaces['tup'].size)))

dc_costos = dict(zip( df_enlaces['tup'], df_enlaces['costo']))

A = np.zeros( (len(dc_bs), len(dc_vars)))
b = np.zeros(len(dc_bs))
c = np.zeros(len(dc_vars))

for nodo in dc_bs_enum:
    #Llenamos el vector b
    b[dc_bs_enum[nodo]] = dc_bs[nodo]
    # Llenamos la matriz A
    for (o,d) in dc_vars:
            if o == nodo:
                #constraints[b].SetCoefficient(dc_vars[(o, d)], 1)
                A[dc_bs_enum[nodo], dc_vars_enum[(o, d)]] = 1
            if d == nodo:
                A[dc_bs_enum[nodo], dc_vars_enum[(o, d)]] = -1
                #constraints[b].SetCoefficient(dc_vars[(o, d)], -1)

# Llenamos el vector c
for var in dc_vars:
    c[dc_vars_enum[var]] = dc_costos[var]

print(A[:10, :10])
print('b:', b)
print('c:', c)

# # # # # # # # # # # SOLUCIÓN:  METODO DE PUNTOS INTERIORES # # # # # # # # # 

def primalNewtonBarrier(c, A, b, N_iter=20):
    """"
    Algoritmo tomado de aquí:
    https://www.cs.toronto.edu/~robere/paper/interiorpoint.pdf        
        
    Algorithm 1 
    Interior Point Method 1: Primal Newton Barrier Method
    Choose ρ ∈ (0, 1) and µ0 > 0 sufficiently large.
    Choose a point x0 so that Ax0 = b and x ≥ 0 are both satisfied.
    k ← 0
    for k = 0, 1, 2, . . . do
        µk = ρµk−1
        Compute the constrained Newton direction pB using (9)
        Solve: minα B(xk, µk), where xk = xk−1 + αpB, subject to Axk = b
        xk ← xk−1 + αpB
        k ← k + 1
    """
    rho = 0.5
    muk_1 = 1e10
    
    # Buscamos una solución factible
    # esto lo haremos con un algoritmo de mínimos cuadrados con restricción
    # de no-negatividad en la solución.  
    # Dicho rutina está incluida en scipy con el método scipy.optimize.nnls
    n = 78
    x_sol = scipy.optimize.lsq_linear(A, b, bounds = (np.ones(n)+ 1, [np.inf]* n))
    xk = x_sol.x
    #xk, _ = scipy.optimize.nnls(A, b)
    
    #n = xk.size
    e = np.ones(n)
    
    for i in range(1, N_iter + 1):
        muk = rho * muk_1
        # Calculamos la dirección restringida de Newton
        
        X = np.diag(xk)
        
        #XX = X
        #print('shape de X:', X.shape)
        X2 = np.diag(xk*xk)
        
        lambda_star = np.linalg.solve(np.matmul(np.matmul(A, X2), A.T),
                                      np.matmul(np.matmul(A, X2), c) 
                                      - muk * np.matmul(np.matmul(A , X), e))
        
        pB = xk + (1/muk) * np.matmul(X2, np.matmul(A.T,lambda_star) - c)
        
        # Resolvemos el sistema B_{alpha}(x, y)
        B = lambda x_var, mu: np.dot(c,x_var) - muk*sum(log1p(xi) for xi in x_var)
        B_alpha = lambda alpha: B(xk + alpha * pB, muk)
        
        best_alpha = scipy.optimize.fminbound(B_alpha, -.1, .1)
        
        # Actualizamos xk 
        xk = xk + best_alpha * pB
        if np.linalg.norm(xk) < 1e-7:
            return xk
        #print('Iteración:', i, 'xk=', xk)
    return xk

solucion = primalNewtonBarrier(c, A, b, N_iter=50)
print('Valor optimo:{:,.2f}'.format(np.dot(c, solucion) ))

# # # # # # # # # # # SOLUCIÓN: SCIPY - OPTIMIZE # # # # # # # # # # # # # # # 
#
#
x = scipy.optimize.linprog(c, A_ub=A, b_ub=b, A_eq=None, b_eq=None,
                           method='simplex')
print('Valor optimo [SCIPY]:{:,.2f}'.format(x.fun))

print(x.fun)


# # # # # # # # # # # SOLUCIÓN: MODULO OR-TOOLS # # # # # # # # # # # # # # # #
#
# Copyright 2010-2018 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def LP_solver():
    # Create the linear solver with the GLOP backend.
    solver = pywraplp.Solver('simple_lp_program',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    # Create the variables x and y.
    #lsxs = [0, 0, 0] 
    
    for var in dc_vars:
        dc_vars[var] = solver.NumVar(0.0, solver.infinity(),  '-'.join(var))
    
    # Creamos un diccionario de restricciones
    constraints = {b:0 for b in dc_bs}
    
    for b in dc_bs:
        constraints[b] = solver.Constraint(dc_bs[b],dc_bs[b],'ct')        
        # Los coeficientes de los enlaces que entran es 1
        for (o,d) in dc_vars:
            if o == b:
                constraints[b].SetCoefficient(dc_vars[(o, d)], 1)
            if d == b:
                constraints[b].SetCoefficient(dc_vars[(o, d)], -1)
    
    objective = solver.Objective()
    
    for var in dc_vars:
        
        objective.SetCoefficient(dc_vars[var], dc_costos[var])
    
    objective.SetMinimization()

    # Call the solver and display the results.
    solver.Solve()
    print('Solution:')
    print('Objective value = ', objective.Value())
    
    df_sol_ortools = {var:dc_vars[var].solution_value() for var in dc_vars}
    for var in dc_vars:
        print('{} = {:.2f}'.format(var,  dc_vars[var].solution_value()))
    
    return pd.DataFrame.from_dict(df_sol_ortools, orient='index')
    
df_sol_ortools = LP_solver()
df_sol_ortools.reset_index(inplace = True)
df_sol_ortools.rename(columns={0:'flujo'}, inplace = True)
df_sol_ortools['origen'] = df_sol_ortools['index'].map(lambda x: x[0])
df_sol_ortools['destino'] = df_sol_ortools['index'].map(lambda x: x[1])

df_sol_ortools.drop('index',axis=1, inplace = True)

df_sol_ortools = df_sol_ortools[df_sol_ortools['origen']!='SOURCE']
#df_sol_ortools = df_sol_ortools[df_sol_ortools['flujo'] > 0]

# Pegamos las coordenadas de los nodos origen destino

centroides = pd.read_excel(rt + r'\centroides.xlsx', sheet_name = 'nodos')
dc_centroides = dict( zip(  centroides['nodo'], centroides[['x', 'y']].apply(tuple, axis=1)))

df_sol_ortools['or_x'] = df_sol_ortools['origen'].map(lambda x: dc_centroides[x][0])
df_sol_ortools['or_y'] = df_sol_ortools['origen'].map(lambda x: dc_centroides[x][1])

df_sol_ortools['dest_x'] = df_sol_ortools['destino'].map(lambda x: dc_centroides[x][0])
df_sol_ortools['dest_y'] = df_sol_ortools['destino'].map(lambda x: dc_centroides[x][1])

df_sol_ortools.to_csv(rt + r'\bd_mapa_final.csv', index=False)