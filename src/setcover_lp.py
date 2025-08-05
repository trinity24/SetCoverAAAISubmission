from gurobipy import Model, GRB
from typing import List, Dict, Any

def solve_setcover_lp(dataset: List[Dict[str, Any]]) -> List[List[float]]:
    """
    Solve the LP relaxation of the fair set cover problem for each instance in the dataset.
    Returns a list of solutions, each a list of x_i values (fractional subset selections).
    """
    solutions = [] 
    for instance in dataset:

        #Unpack the instance 
        subsets = instance['subsets']
        # print(len(subsets))
        weights = instance['subset_weights']
        element_colors = {int(elem): color for elem, color in instance['element_colors'].items()}
        n_subsets = len(subsets)
        n_elements = len(instance['universe'])

        params = instance['parameters']
        k1, k2, k3 = params['k1'], params['k2'], params['k3']

        # Build color-to-elements mapping
        color_to_elements = {1: [], 2: [], 3: []}
        for elem, color in element_colors.items():
            color_to_elements[color].append(elem)

        # Build element-to-subsets mapping
        element_to_subsets = {elem: [] for elem in range(n_elements)}
        # print(f"Length of the element_to_subsets :  {len(element_to_subsets)}")
        for i, subset in enumerate(subsets):
            for elem in subset:
                element_to_subsets[elem].append(i)
        # Start Gurobi model
        model = Model()
        model.Params.OutputFlag = 0  # silent

        # Variables: x_i in [0,1] for each subset
        x = model.addVars(n_subsets, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x") 
        y = model.addVars(n_elements, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")  # [0,1] variables for elements
        # Objective: minimize total weight
        model.setObjective(sum(weights[i] * x[i] for i in range(n_subsets)), GRB.MINIMIZE)
        # For each color c, ensure at least k_c elements of color c are covered
        for color, k_c in zip([1, 2, 3], [k1, k2, k3]):
            elements_c = color_to_elements[color]
            model.addConstr(sum([y[elem] for elem in elements_c]) >= k_c)
        # Ensure that each element is covered at least a probability of p_elem
        

        element_probs = { int(elem): prob for elem, prob in instance['element_probs'].items() }
        for elem, prob in element_probs.items():
            # For each element, ensure its coverage probability is at least prob
            cover_expr = sum(x[i] for i in element_to_subsets[elem])
            model.addConstr(y[elem] <= cover_expr)
            # Ensure that y[elem] >= prob for each element      
            model.addConstr(y[elem] >= prob)
        model.optimize()
        solution_x = [x[i].X for i in range(n_subsets)]
        solution_y = [y[elem].X for elem in range(n_elements)]
        solutions.append((solution_x, solution_y))
    return solutions
