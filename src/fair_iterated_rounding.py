from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.linalg import null_space 

def get_constraint_matrix(instance, floating_x, floating_y, k1, k2, k3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (A, b) for the system A @ [x, y] <= b for the set cover LP in any given iteration h.
    Variable order: [x_0, ..., x_{mh-1}, y_0, ..., y_{nh-1}]
    floating_x:     subsets with values in (0,1)
    floating_y:     elements which are not fully covered by any single subset 
    x_i :           subset selection variable for subset i
    y_e :           element coverage variable for element e
    k1,k2,k3 :      remaining number of elements to cover for colors 1, 2, and 3 respectively in the iteration h
    """
    element_colors = instance['element_colors']
    # Initialize constraint matrix A and vector b
    A, b = [], []
    # Constraints:
    # 1. For each color c: sum_{e in color_c} y_e >= k_c  -->   -sum y_e <= -k_c
    mh, nh = len(floating_x), len(floating_y)
    
    # Count how many floating elements we have of each color
    for color, k_c in zip([1, 2, 3], [k1, k2, k3]):
        if k_c > 0:
            row = [0] * (mh + nh)
            color_count = 0
            
            # For each element j that is floating and has color c, we add -1 to the row
            for j in range(len(floating_y)):
                if element_colors[floating_y[j]] == color:
                    row[mh + j] = -1  # y_e coefficient
                    color_count += 1
            
            # Only add the constraint if there are elements of this color
            if color_count > 0:
                # Adjust the right-hand side to be a reasonable value
                # At most we can satisfy k_c = color_count (all y values = 1)
                adjusted_k = min(k_c, color_count)
                A.append(row)
                b.append(-adjusted_k)

    # 2. For each element e:  - sum_{i: e in subset_i} x_i + y_e <= 0
    for j in range(len(floating_y)):
        row = [0] * (mh + nh)
        has_covering_subset = False
        
        for i in range(len(floating_x)):
            if floating_y[j] in instance['subsets'][floating_x[i]]:
                row[i] = -1  # -x_i
                has_covering_subset = True
        
        row[mh + j] = 1  # y_e
        
        # Only add this constraint if there's at least one subset that can cover this element
        # or if there are no floating subsets at all (in which case we need to constrain y directly)
        if has_covering_subset or mh == 0:
            A.append(row)
            b.append(0)
    
    A = np.array(A)
    b = np.array(b)
    
    # Verify that we have at least one constraint
    if A.shape[0] == 0:
        # If there are no constraints, add a dummy constraint that's always satisfied
        A = np.zeros((1, mh + nh))
        b = np.array([0])
        
    return A, b

def cover_round(floating_x: List[int], floating_y: List[int], instance, floating_vector: List[float]) -> List[float]:
    """
    Cover rounding function that marks subsets and elements based on the floating vector.
    It marks subsets that are fully covered and elements that are covered by these subsets.
    It returns the updated  floating_vector.
    floating_vector: List of fractional values corresponding to floating_x and floating_y.
    """
    marked_x = set()
    f = max(list(instance['element_frequencies'].values()))  # Maximum frequency of any element in the subsets
    for i in range(len(floating_x)):
        # If the subset is fully covered, we mark it
        if floating_vector[i] >= 1/f:
            floating_vector[i] = 1  # Mark the subset as fully covered
            marked_x.add(floating_x[i])
    marked_y = set()
    for i in marked_x:
        for j in instance['subsets'][i]:
            if j in floating_y:
                floating_vector[len(floating_x) + floating_y.index(j)] = 1
                marked_y.add(j)  # Mark the element as covered by the subset
    
    return floating_vector

def check_underdetermined(A: np.ndarray) -> bool:
    """
    Checks if the system of equations Ax <= b is underdetermined.
    An underdetermined system has more variables than constraints.
    A: Constraint matrix
    b: Constraint vector
    """
     # Check if the constraints are linearly dependent
    return A.shape[0] < A.shape[1]  # More variables than constraints

def check_feasibility(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> bool:
    """
    Checks if the solution x satisfies the constraints Ax <= b.
    A: Constraint matrix
    b: Constraint vector
    x: Solution vector
    """
    
    epsilon = 1e-5  # Allow a small error term for numerical stability
    assert A.shape[0] == b.shape[0], "The number of constraints must match the number of elements in b."  
    assert A.shape[1] == x.shape[0], "The number of variables in x must match the number of columns in A."  
    return np.all(np.dot(A, x) <= b + epsilon)

def lp_rounding(eta, solution: List[List[float]], instance) -> List[int]:
    if eta == None:
        eta = 1 # Default value for eta if not provided
    # (x,y) is the fractional solution of the LP relaxation of the set cover instance
    x, y = solution[0], solution[1]
    # print(solution)
    setcover = [] 
    # elements_to_subsets = subsets_to_element(instance)
    k1, k2, k3 = instance['parameters']['k1'], instance['parameters']['k2'], instance['parameters']['k3']
    residual_colors = {1: k1, 2: k2, 3: k3}
    
    # Get floating variables
    floating_x, floating_y = [i for i in range(len(x)) if 0 < x[i] < 1], [j for j in range(len(y)) if 0 < y[j] < 1]

    # If no floating variables, just return the selected subsets
    if len(floating_x) == 0:
        return [i for i in range(len(x)) if x[i] == 1]
        
    # Apply cover rounding
    floating_vector = cover_round(floating_x, floating_y, instance, [x[i] for i in floating_x] + [y[j] for j in floating_y])
    
    # Update the x and y vectors with the floating vector values
    for i in range(len(floating_x)):
        x[floating_x[i]] = floating_vector[i]
    for j in range(len(floating_y)):
        y[floating_y[j]] = floating_vector[len(floating_x) + j]
    
    # Update floating variables after cover rounding
    floating_x, floating_y = [i for i in range(len(x)) if 0 < x[i] < 1], [j for j in range(len(y)) if 0 < y[j] < 1]
    floating_vector = [x[i] for i in floating_x] + [y[j] for j in floating_y]

    # Main loop - continue while we have enough floating x variables
    element_colors = instance['element_colors']
    while len(floating_x) > eta * 3 and (len(floating_y) > 0):
        # Update residual colors
        for color, k_c in residual_colors.items():
            k_c -= sum([1 for j in instance['universe'] if element_colors[j] == color and y[j] == 1])
            residual_colors[color] = k_c

        k1, k2, k3 = residual_colors[1], residual_colors[2], residual_colors[3]
        
        # Handle empty floating variables case
        if len(floating_x) == 0 or len(floating_y) == 0:
            break
            
        # Get constraint matrix
        Ah, bh = get_constraint_matrix(instance, floating_x, floating_y, k1, k2, k3) 
        
        # Ensure dimensions match
        if Ah.shape[1] != len(floating_vector):
            print(f"Dimension mismatch: A has {Ah.shape[1]} columns but floating_vector has {len(floating_vector)} elements")
            # Reconstruct our floating_vector to match
            floating_vector = [x[i] for i in floating_x] + [y[j] for j in floating_y]
        
        try:
            # Check if the current floating vector satisfies the constraints
            if not check_feasibility(Ah, bh, np.array(floating_vector)):
                # Try to fix the floating vector by clipping values
                floating_vector = np.clip(floating_vector, 0, 1)
                if not check_feasibility(Ah, bh, floating_vector):
                    # If still not feasible, we can't continue
                    break
        except Exception as e:
            print(f"Feasibility check error: {e}")
            break

        # Compute the null space of the constraint matrix
        try:
            null_space_basis = null_space(Ah)
            if null_space_basis.shape[1] == 0:
                # No null space, can't continue
                break
                
            random_vector = null_space_basis @ np.random.randn(null_space_basis.shape[1])
            random_vector /= np.linalg.norm(random_vector)
            floating_vector = np.array(floating_vector)
        except Exception as e:
            print(f"Null space calculation error: {e}")
            break

        # Compute alpha and beta to ensure floating_vector remains within [0, 1]
        alpha = float('inf')
        beta = float('inf')

        for i in range(len(floating_vector)):
            if 0 < floating_vector[i] < 1:
                if random_vector[i] > 0:
                    alpha = min(alpha, abs((1 - floating_vector[i]) / random_vector[i]))
                    beta = min(beta, abs(floating_vector[i] / random_vector[i]))
                elif random_vector[i] < 0:
                    alpha = min(alpha, abs(floating_vector[i] / (-random_vector[i])))
                    beta = min(beta, abs((1 - floating_vector[i]) / (-random_vector[i])))
        
        # Handle degenerate cases
        if alpha == float('inf') or beta == float('inf'):
            # Can't find valid perturbation, randomly round some variables
            idx = np.random.randint(0, len(floating_x))
            x[floating_x[idx]] = round(x[floating_x[idx]])
        else:
            # Update the floating vector with perturbation
            prob = alpha / (alpha + beta)
            q = np.random.binomial(1, prob)
            
            if q == 1:
                # with probability α/(α + β) we update the floating vector by adding alpha*random_vector
                floating_vector += alpha * random_vector
            else:
                # with probability β/(α + β) we update the floating vector by subtracting beta*random_vector
                floating_vector -= beta * random_vector
            
            # Ensure values stay in [0,1]
            floating_vector = np.clip(floating_vector, 0, 1)
            
            # Update x and y from floating_vector
            for i in range(len(floating_x)):
                x[floating_x[i]] = floating_vector[i]
            for j in range(len(floating_y)):
                y[floating_y[j]] = floating_vector[len(floating_x) + j]
        
        # Update floating variables for next iteration
        floating_x, floating_y = [i for i in range(len(x)) if 0 < x[i] < 1], [j for j in range(len(y)) if 0 < y[j] < 1]
        floating_vector = [x[i] for i in floating_x] + [y[j] for j in floating_y]
    
    # Return the final selected subsets
    for i in range(len(floating_x)):
        if 0 < x[floating_x[i]] < 1 or x[floating_x[i]] == 1:
            setcover.append(floating_x[i])
    
    return setcover + [i for i in range(len(x)) if x[i] == 1]
