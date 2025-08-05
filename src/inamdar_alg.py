import numpy as np
import gurobipy as gp
from gurobipy import GRB
# Compute the LP solution for the first instance
import os
import sys
# Get the absolute path of the src directory
current_dir = os.getcwd()
src_path = os.path.join(current_dir, '../src')
sys.path.append(src_path)
from setcover_lp import solve_setcover_lp


# Compute the subsets A1 and A2

import random
import math

# def setCoverRounding(x, instance,H  ):
#   """
#   Set Cover Rounding Algorithm on the projected subsets S_{|H} 
#   and the heavy element set H.
#   """
#   S,d =  [set(S_i) & H for S_i in instance['subsets']], instance['parameters']['d']
#   Cover = set()
#   lenX = len(x)
#   for r in range(int(d * math.log(lenX))):
#     for i in range(lenX):
#       if S[i] != set() and random.random() < x[i]:
#         Cover.add(i)
#   return Cover
def setCoverRounding(x, instance,H  ):
  """
  Set Cover Rounding Algorithm on the projected subsets S_{|H} 
  and the heavy element set H.
  """
  S,d =  [set(S_i) & H for S_i in instance['subsets']], instance['parameters']['d']
  Cover = set()
  f = instance['parameters']['f']
  for r in range(len(x)):
    if x[r]>=1/f:
        Cover.add(r)
  return Cover

def A_Subroutine(solution, instance,alpha=6):
  """
  A subroutine to find the sets A1 and A2 
  Return A = A1 \cup A2
  """
  S, P = instance['subsets'], instance['universe']
  m, n = len(S), len(P)
  #Construct heavy element set H
#   print(f"Constructing heavy element set H {instance['element_probs']}" )
  H = set( [ j for j in range(n) if instance['element_probs'][j] >= (1 / (6 * alpha)) ] )
  #Obtaining x_tilde
  x_tilde = [min(6 * alpha * solution[i], 1) for i in range(m)]
  #Using setCoverRounding to get A1
  A1 = setCoverRounding(x_tilde, instance,H)
  #Getting A2
  A2 = [i for i in range(m) if solution[i] >= 1 / (6 * alpha)]
  #A = A1 U A2
  return A1 | set(A2)

def solve_lp(instance):
    m = len(instance['subsets'])
    n = len(instance['universe'])
    model = gp.Model("modified_set_cover_lp")
    model.Params.OutputFlag = 0  # Suppress Gurobi output

    # Variables: x_i for each subset S_i
    x = model.addVars(m, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x")

    # Constraints: For each element j, sum_{i: j in S_i} x_i >= element_probs[j]
    for j in range(n):
        subset_indices = instance['element_to_subsets'][j]
        model.addConstr(gp.quicksum(x[i] for i in subset_indices) >= instance['element_probs'][j], name=f"cover_{j}")
    
    # Objective: Minimize sum_i w_i x_i
    weights = instance['subset_weights']
    model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(m)), GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return [x[i].X for i in range(m)]
    else:
        return None


def separation_oracle(solution,instance,Delta,tolerance=1e-6, debug=False):
    """
    Separation oracle for the Set Cover problem with additional constraints.
    Verify if the candidate solution violates any Covering Constraints: y_j - \sum_{i:j \in S_i}x_i <= 0,
    Coloring Constraints: -\sum_{j \in P_c} y_j + k_c <= 0, and
    Individual Fairness Constraints: -y_j + element_probs[j] <= 0.
    """
    n,m = len(instance['universe']), len(instance['subsets'])

    if debug:
        print(f"Checking solution: x[0:5]={solution[:5]}, y[0:5]={solution[m:m+5]}")

    for i in range(m+n):
        if solution[i] <= 0 - tolerance:
            # -x <= 0
            # If the candidate solution violates the non-negativity constraints, return the constraint that has been violated by the solution
            if debug:
                print(f"Non-negativity violation: variable {i} = {solution[i]}")
            return False, np.array([-1.0 if j == i else 0.0 for j in range(m+n)]), 0.0

        if solution[i] >= 1 + tolerance:
            # If the candidate solution violates the upper bound constraints, return the constraint that has been violated by the solution
            if debug:
                print(f"Upper bound violation: variable {i} = {solution[i]}")
            return False, np.array([1.0 if j == i else 0.0 for j in range(m+n)]), 1.0
    """
    Covering Constraints: - \sum_{i:j \in S_i}x_i + y_j <= 0
    """
    for j in range(n):
        sum_x = sum(solution[i] for i in instance['element_to_subsets'][j])  # sum of x_i where j is in S_i
        if -sum_x + solution[m+j] >= tolerance:
            return False, np.array([-1.0 if i in instance['element_to_subsets'][j] else 0.0 for i in range(m)] + [1.0 if k==j else 0.0 for k in range(n)]), 0.0 

    """
    Coloring Constraints: -\sum_{j \in P_c} y_j  <= - k_c 
    """
    k = [instance['parameters']["k1"], instance['parameters']["k2"],instance['parameters'] ["k3"]] 
    for c in range(len(instance['partitions'])):
        # Check if the candidate solution violates the coloring constraints
        # For each partition, we need to ensure that at most one element from the partition is selected.
        partition = instance['partitions'][c+1]
        sum_partition = sum(solution[m+j] for j in partition)
        # If the sum of the selected elements in the partition P_c is less than k_c, we have a violation
        if  k[c] - sum_partition >= tolerance: # Defecit is > tolerance
            if debug:
                print(f"Coloring constraint violation: partition {c+1}, required: {k[c]}, got: {sum_partition}")
            return False, np.array([0.0 for i in range(m)] + [-1.0 if j in partition else 0.0 for j in range(n)]), - k[c] 
                # Separating plane:- \sum_{j\in P[c]} y_j <= -k[c]
    """
    Individual Fairness Constraints: -y_j   <= -element_probs[j] => y_j   >= element_probs[j]
    """
    for j in range(n):
        if solution[m+j] <= instance['element_probs'][j]- tolerance:
            # If the candidate solution violates the non-negativity constraints, return the constraint that has been violated by the solution
            return False, np.array([0.0 for i in range(m)] + [-1.0 if k == j else 0.0 for k in range(n)]),  - instance['element_probs'][j] # Separating plane: x_j >= 0

    """
    Check if the objective value exceeds the tolerance + Delta
    Objective: \sum_{i\in [m]}w_ix_i <= Delta
    """
    objective_value = sum(solution[i]*instance['subset_weights'][i] for i in range(m) )
    if objective_value >= tolerance + Delta:
        return False, np.array([instance['subset_weights'][i] for i in range(m)] + [0.0 for j in range(n)]),  Delta 
    # Separating plane: \sum_{i=1}^{m}w_i x_i <= Delta
    """
    Check the knapsack covering constraints
    """
    A = A_Subroutine(solution, instance)
    SetCover_A = [instance['subsets'][i] for i in A]
    P_covered = set()
    P_covered.update(*SetCover_A)
    k_c  = [max(k[c] - len(P_covered & set(instance['partitions'][c+1])), 0) for c in range(3)]

    # Residual number of elements that S_i contributes to each partition
    P_uncovered = set(instance['universe']) - P_covered
    deg_ic = { 
       i: [min(k_c[c], len(P_uncovered & set(instance['subsets'][i]) & set(instance['partitions'][c+1]))) for c in range(3)]
       for i in range(m) }
    A_complement = [i for i in range(m) if i not in A]  # Indices of subsets not in A

    """
    Check if  - \sum_{i \notin A} x_i deg_ic(S_i,A) <= - k_c[c] 
    """
    for c in range(3):
        if sum([deg_ic[i][c]*solution[i] for i in A_complement]) <= k_c[c]-tolerance:
            # If the candidate solution violates the knapsack covering constraints, return the constraint that has been violated by the solution
            return False, np.array([-deg_ic[i][c] for i in range(m)] + [0.0 for j in range(n)]), -k_c[c]

    # If all checks pass
    return True, None, None # x_candidate is feasible, no separating hyperplane



def ellipsoid_algorithm(instance,lp_solution,  max_iterations=50000, tolerance=1e-9, debug=False):
    # Initial ellipsoid: a large sphere centered at the origin
    m,n = len(instance['subsets']), len(instance['universe'])
    # Set y values high enough to potentially satisfy coloring constraints
    k_values = [instance['parameters']['k1'], instance['parameters']['k2'], instance['parameters']['k3']]
    # y = list(instance['element_probs'].values())  # Use element probabilities directly for y values
    f = instance['parameters']['f']
    x,y = lp_solution[0],lp_solution[1]
    # solve_setcover_lp([instance])[0]  # Solve the LP to get initial x values
    # x = solve_lp(instance)  # Solve the LP to get initial x values
    # print(x,y)
    # Better initialization - create a solution that might satisfy coloring constraints
    current_solution = np.array(x+y)
    initial_radius = np.sqrt(m+n)  # Use the maximum subset weight as the initial radius
    # max(10.0, np.sqrt(m+n))  # Larger initial radius
    current_P = initial_radius**2 * np.identity(m+n) # Shape matrix (P = R^2 * I for a sphere)
    feasible_point = None
    low , high  =   sum([instance['subset_weights'][i]*x[i] for i in range(m)]) , sum(instance['subset_weights'])
    feasible_ = False
    
    # More relaxed tolerance for beta_k
    beta_tolerance = max(1e-10, tolerance * (m+n))
    
    if debug:
        print(f"Starting ellipsoid algorithm: m={m}, n={n}, total_dim={m+n}")
        print(f"Initial solution: x={current_solution[:m]}, y={current_solution[m:]}")
        print(f"Initial radius: {initial_radius}, beta_tolerance: {beta_tolerance}")
        print(f"Weight range: [{low}, {high}]")
        print(f"Required k values: {k_values}")
    
    # print(low,high)
    while low< high and ( high >= tolerance+ 2*low):
        Delta = (low + high) / 2
        if debug:
            print(f"\nBinary search: Delta = {Delta}, range = [{low}, {high}]")
        for k in range(max_iterations):
            # print(f"Iteration {k}: Center = {current_solution}")
            # Call the separation oracle to check feasibility and get a separating hyperplane if needed
            # print(f"Checking feasibility for Delta = {Delta}")
            # print(current_solution)
            is_feasible, cut_normal, cut_rhs = separation_oracle(current_solution, instance, Delta)
            
            if is_feasible:
                feasible_point = current_solution.copy()
                # if debug:
                # print(f"Found a feasible point at iteration {k}",feasible_point)
                feasible_ = True
                break
                
            # Calculate g: normalized direction of the cut
            # The inequality is cut_normal^T x <= cut_rhs, if current_x violates it, cut_normal^T current_x > cut_rhs
            g = cut_normal
            # The ellipsoid update formulas often use g as a vector, not necessarily normalized
            # but the scaling factors take care of it.
            # Calculate beta_k = g^T P_k g
            g_Pk = np.dot(g, current_P)
            beta_k = np.dot(g_Pk, g)
            # Update center x_{k+1}
            step_size = (1.0 / (n+m + 1))
            if beta_k <= beta_tolerance:
                if debug:
                    print(f"Beta_k ({beta_k}) is too small (tolerance: {beta_tolerance}), stopping iterations at iteration {k}.")
                # else:
                    # print(f"Beta_k ({beta_k}) is too small (tolerance: {beta_tolerance}), stopping iterations.")
                break
            current_solution = current_solution - step_size * np.dot(current_P, g) / np.sqrt(beta_k)
            # Update shape matrix P_{k+1}
            Pk_g_outer = np.outer(np.dot(current_P, g), np.dot(current_P, g))
            current_P = ((m+n)**2 / ((m+n)**2 - 1)) * (current_P - 2*step_size * Pk_g_outer / beta_k)
        if feasible_:
            feasible_ = False
            high = Delta
        else:
            low = Delta
        # print(f"After iteration {k}, new range: [{low}, {high}]")
    if feasible_point is None:
        feasible_point = np.ones(m+n)  # Return numpy array, not list
    # print("Final feasible point:", feasible_point)
    return feasible_point


def inamdar_rounding(instance,lp_solution,alpha=6):
    """
    Inamdar Rounding Algorithm for the Set Cover Problem.
    """
    solution = ellipsoid_algorithm(instance,lp_solution)
    # print("Ellipsoid solution:", solution[:len(instance['subsets'])])  # Print only the x part of the solution
    if solution is None:
        return None
    # Round the solution using the A subroutine
    A = A_Subroutine(solution, instance, alpha)
    # Do the randomized rounding for the sets in A_complement
    A_complement = [i for i in range(len(instance['subsets'])) if i not in A]
    P_covered = set()
    P_covered.update(*[instance['subsets'][i] for i in A])

    P_covered = {  c: len(P_covered & set(instance['partitions'][c+1])) for c in range(3) }
    if P_covered[0] >= instance['parameters']['k1'] and P_covered[1] >= instance['parameters']['k2'] and P_covered[2] >= instance['parameters']['k3']:
        # If the current solution already satisfies the coloring constraints, return A
        return A
    
    for k in range(3):
        for i in A_complement:
            if random.random() < 6*solution[i]:
                A.add(i)
    # print("Final rounded set A:", A)
    return A
