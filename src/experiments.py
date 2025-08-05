import numpy as np
import json
import pickle
import setcover_dataset 
import fair_iterated_rounding
import inamdar_alg
import setcover_lp
import importlib

importlib.reload(fair_iterated_rounding)
importlib.reload(inamdar_alg)
importlib.reload(setcover_dataset)
importlib.reload(setcover_lp)

from setcover_dataset import generate_setcover_dataset
from fair_iterated_rounding import lp_rounding
from inamdar_alg import inamdar_rounding
from setcover_lp import solve_setcover_lp
import seaborn as sns
import time


def weight_solution(solution, instance):
    """
    Calculate the total weight of the solution for a given instance.
    """
    total_weight = 0
    for j in solution:
        total_weight += instance['subset_weights'][j]
    return total_weight
def get_lp_objective(lp_solution, instance):
    """
    Calculate the LP objective value for a given instance.
    """
    total_weight = 0
    for j in range(len(lp_solution[0])):
        total_weight += instance['subset_weights'][j] * lp_solution[0][j]
    return total_weight

def read_dataset(file_path):
    """
    Read the dataset from a pickle file.
    """
    with open(file_path, 'rb') as file:
        dataset = pickle.load(file)
    return dataset

def write_dataset(file_path, dataset):
    """
    Write the dataset to a pickle file.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)

def get_inamdar_objectives(dataset, lp_solutions):
    """
    Get the Inamdar solutions for the dataset.
    """
    n = dataset[0]['parameters']['n']
    inamdar_objectives = []
    coverage_reqs = []
    inamdar_times = []
    for instance,solution in zip(dataset, lp_solutions):
        start_time = time.time()
        # Generate 100 rounded solutions using Inamdar's algorithm
        rounded_solutions = [inamdar_rounding(instance,solution) for _ in range(100)]
        inamdar_times.append(time.time() - start_time)
        coverage_probs = { 
            j : (sum([covered_elements(s, instance)[j] for s in rounded_solutions]) / len(rounded_solutions))
                             for j in range(n) }
        # Number of elements that are covered at least as much as their probability
        cov_req = [1 if coverage_probs[j]>= instance['element_probs'][j] else 0.0 for j in range(n)]
        coverage_reqs.append(sum(cov_req))
        # Calculate the weighted solution for each rounded solution
        objs = [weight_solution(s, instance) for s in rounded_solutions]
        inamdar_objectives.append(np.mean(objs))
    return inamdar_objectives,coverage_reqs, inamdar_times

def covered_elements(solution, instance):
    """
    Get the covered elements for a given solution and instance.
    """
    n = instance['parameters']['n']
    covered = { j : 0 for j in range(n) }
    for j in solution:
        for element in instance['subsets'][j]:
            covered[element] = 1
    return covered

def get_lp_objectives(dataset, lp_solutions):
    """
    Get the LP solutions for the dataset.
    """
    lp_objectives = []
    for instance,solution in zip(dataset, lp_solutions):
        objective_value = get_lp_objective(solution, instance)
        lp_objectives.append(objective_value)
    return lp_objectives

def get_max_objectives(dataset):
    """
    Get the maximum objectives for the dataset.
    """
    max_objectives = []
    for instance in dataset:
        max_weight = sum(instance['subset_weights'])
        max_objectives.append(max_weight)
    return max_objectives

def plot_coverage_reqs(fair_iter_coverage_reqs, inamdar_coverage_reqs,frequency=None):
    """
    Plot the coverage requirements for Fair Iterated Rounding and Inamdar Rounding.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    x = range(len(fair_iter_coverage_reqs))
    plt.plot(x, fair_iter_coverage_reqs, marker='o', label='Fair Iterated Rounding')
    plt.plot(x, inamdar_coverage_reqs, marker='o', label='Inamdar Rounding')

    plt.title('Coverage Requirements Comparison')
    plt.xlabel('Instance Index')
    plt.ylabel('Coverage Requirement')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot before showing it
    if frequency is not None:
       plt.savefig(f'coverage_requirements_comparison_f_{frequency}.png', dpi=400)
    else:
        plt.savefig('coverage_requirements_comparison.png', dpi=400)
    
    plt.show()


def plot_approximation_ratios(fair_iter_ratios, inamdar_ratios, frequency=None):
    """
    Plot the approximation ratios for Fair Iterated Rounding and Inamdar Rounding.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    x = range(len(fair_iter_ratios))
    plt.plot(x, fair_iter_ratios, marker='o', label='Fair Iterated Rounding')
    plt.plot(x, inamdar_ratios, marker='o', label='Inamdar Rounding')

    plt.title('Approximation Ratios Comparison')
    plt.xlabel('Instance Index')
    plt.ylabel('Approximation Ratio')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot before showing it
    if frequency is not None:
        plt.savefig(f'approximation_ratios_comparison_f_{frequency}.png', dpi=400)
    else:
        plt.savefig('approximation_ratios_comparison.png', dpi=400)
    
    plt.show()

def plot_objectives(fair_iter_objs, inamdar_objs, lp_objs,max_objs=None, frequency=None):
    """
    Plot the objectives for the given dataset using seaborn for nicer colors.
    """
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", palette="muted", color_codes=True)
    plt.figure(figsize=(10, 6))

    x = range(len(fair_iter_objs))
    sns.lineplot(x=x, y=fair_iter_objs, marker='o', label='Fair Iterated Rounding')
    sns.lineplot(x=x, y=inamdar_objs, marker='o', label='Inamdar Rounding')
    sns.lineplot(x=x, y=lp_objs, marker='o', label='LP Relaxation')
    if max_objs is not None:
        sns.lineplot(x=x, y=max_objs, marker='o', label='Max Objective', linestyle='--')

    plt.title('Objective Values Comparison')
    plt.xlabel('Instance Index')
    plt.ylabel('Objective Value')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot before showing it
    if frequency is not None:
        plt.savefig(f'objectives_comparison_f_{frequency}.png', dpi=400)
    else:
        plt.savefig('objectives_comparison.png', dpi=400)
    
    plt.show()

def get_fair_iter_objectives(dataset, lp_solutions):
    """
    Get the Fair Iterated Rounding solutions for the dataset.
    """
    n = dataset[0]['parameters']['n']
    fair_iter_objectives = []
    coverage_reqs = []
    fair_iter_times = []
    for instance, solution in zip(dataset, lp_solutions):
        start_time = time.time()
        rounded_solutions = [lp_rounding(1, solution, instance) for _ in range(100)]
        fair_iter_times.append(time.time() - start_time)
        coverage_probs = {j:
                          sum([covered_elements(s, instance)[j] for s in rounded_solutions]) / len(rounded_solutions)
                          for j in range(n)}
        cov_req = [1 if coverage_probs[j] >= instance['element_probs'][j] else 0.0 for j in range(n)]
        coverage_reqs.append(sum(cov_req))
        objs = [weight_solution(s, instance) for s in rounded_solutions]
        fair_iter_objectives.append(np.mean(objs))
    return fair_iter_objectives, coverage_reqs, fair_iter_times
    # for instance,solution in zip(dataset, lp_solutions):
    #     rounded_solutions = [lp_rounding(1, solution, instance) for _ in range(100)]
    #     coverage_probs = { j :
    #                     sum([covered_elements(s, instance)[j] for s in rounded_solutions]) /len(rounded_solutions)
    #                          for j in range(n) }
    #     # Number of elements that are covered at least as much as their probability
    #     cov_req = [1 if coverage_probs[j]>= instance['element_probs'][j] else 0.0 for j in range(n)]
    #     coverage_reqs.append(sum(cov_req))
    #     objs =  [ weight_solution(s, instance) for s in rounded_solutions ]
    #     fair_iter_objectives.append(np.mean(objs))
    # return fair_iter_objectives, coverage_reqs

def normalize_weights(instance):
    """
    Normalize the weights of the subsets in the instance.
    """
    rounded_weights = [round(w, 3) for w in instance['subset_weights']]

    max_weight = max(rounded_weights)
    if max_weight > 0:
        instance['subset_weights'] = [w / max_weight for w in rounded_weights]
    else:
        instance['subset_weights'] = [0] * len(rounded_weights)

def plot_running_times(fair_iter_time, inamdar_time, frequency=None):
    """
    Plot the running times for Fair Iterated Rounding and Inamdar Rounding.
    """
    import matplotlib.pyplot as plt

    sns.set(style="whitegrid", palette="muted", color_codes=True)
    plt.figure(figsize=(10, 6))

    x = range(len(fair_iter_time))
    sns.lineplot(x=x, y=fair_iter_time, marker='o', label='Fair Iterated Rounding')
    sns.lineplot(x=x, y=inamdar_time, marker='o', label='Inamdar Rounding')
    plt.title('Running Times Comparison')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Rounding Method')
    plt.tight_layout()

    # Save the plot before showing it
    if frequency is not None:
        plt.savefig(f'running_times_comparison_f_{frequency}.png', dpi=400)
    else:
        plt.savefig('running_times_comparison.png', dpi=400)

def run_experiments(dataset,f):
    
    for instance in dataset:
        # Normalize the weights of the subsets in the instance
        normalize_weights(instance)

    lp_solutions = solve_setcover_lp(dataset)
    lp_objectives = get_lp_objectives(dataset, lp_solutions)
    # start_fair_iter = time.time()
    fair_iter_objectives, coverage_reqs_fair_iter,fair_iter_times = get_fair_iter_objectives(dataset, lp_solutions)
    # end_fair_iter = time.time()
    # print(f"Fair Iterated Rounding time: {end_fair_iter - start_fair_iter:.4f} seconds")

    # start_inamdar = time.time()
    inamdar_objectives, coverage_reqs_inamdar,inamdar_times = get_inamdar_objectives(dataset, lp_solutions)
    # end_inamdar = time.time()
    # print(f"Inamdar Rounding time: {end_inamdar - start_inamdar:.4f} seconds")
    approximation_ratio_fair_iter = [fair_iter_objectives[i] / lp_objectives[i] for i in range(len(fair_iter_objectives))]
    approximation_ratio_inamdar = [inamdar_objectives[i] / lp_objectives[i] for i in range(len(inamdar_objectives))]
    
    return {

        "time_fair_iter": fair_iter_times,
        "time_inamdar": inamdar_times,
        "max_objectives": get_max_objectives(dataset),
        "lp_objectives": lp_objectives,
        "fair_iter_objectives": fair_iter_objectives,
        "inamdar_objectives": inamdar_objectives,
        "approximation_ratio_fair_iter": approximation_ratio_fair_iter,
        "approximation_ratio_inamdar": approximation_ratio_inamdar,
        "coverage_reqs_fair_iter": coverage_reqs_fair_iter,
        "coverage_reqs_inamdar": coverage_reqs_inamdar
    }
def run_experiments_for_frequencies(frequencies,dataset_paths):
    """
    Run experiments for a list of frequencies.
    """
    for f_ in range(len(frequencies)):
        f = frequencies[f_]
        loaded_dataset = read_dataset(dataset_paths[f_])
        print(f"Running experiments for frequency: {f}")
        lp_solutions_dataset  = solve_setcover_lp(loaded_dataset)
        write_dataset()
        lp_objective_dataset = [ get_lp_objective(lp_solutions_dataset[i], loaded_dataset[i]) for i in range(len(lp_solutions_dataset)) ]
        rounded_solution_fair_iter = []
        rounded_solution_inamdar = []
        all_rounds_fairiter = []
        all_rounds_inamdar = []
        for i in range(len(lp_solutions_dataset)):
            # Instance to test
            weight_rounded_solution_fair_iter = []
            weight_rounded_solution_inamdar = []
            for j in range(100): # Iterations of randomized rounding
                weight_rounded_solution_fair_iter.append(weight_solution(lp_rounding(1, lp_solutions_dataset[i], loaded_dataset[i]), loaded_dataset[i]))
                weight_rounded_solution_inamdar.append(weight_solution(inamdar_rounding( loaded_dataset[i],lp_solutions_dataset[i]), loaded_dataset[i]))
            all_rounds_fairiter.append(weight_rounded_solution_fair_iter)
            all_rounds_inamdar.append(weight_rounded_solution_inamdar)
            rounded_solution_fair_iter.append(np.mean(weight_rounded_solution_fair_iter)) # Expected_values.
            rounded_solution_inamdar.append(np.mean(weight_rounded_solution_inamdar)) # Expected_values.
        # SetCover/data/freq_data/final_dataset/all_rounds_rounded_sols_fairiter
        with open(f'../data/freq_data/final_dataset/all_rounds_rounded_sols_fairiter/all_rounds_fairiter_f_{f}.pkl', 'wb') as file:
            pickle.dump(all_rounds_fairiter, file)

        with open(f'../data/freq_data/final_dataset/all_rounds_rounded_sols_inamdar/all_rounds_inamdar_f_{f}.pkl', 'wb') as file:
            pickle.dump(all_rounds_inamdar, file)

        with open(f'../data/freq_data/final_dataset/lp_solutions/lp_solutions_small_dataset_f_{f}.pkl', 'wb') as file:
            pickle.dump(lp_solutions_dataset, file)

        with open(f'../data/freq_data/final_dataset/rounded_solutions_fairiter/rounded_solutions_fairiter_f_{f}.pkl', 'wb') as file:
            pickle.dump(rounded_solution_fair_iter, file)

        with open(f'../data/freq_data/final_dataset/rounded_solutions_inamdar/rounded_solutions_inamdar_f_{f}.pkl', 'wb') as file:
            pickle.dump(rounded_solution_inamdar, file)

        with open(f'../data/freq_data/final_dataset/max_solutions/max_weight_small_dataset_f_{f}.pkl', 'wb') as file:
            max_weights = [sum(loaded_dataset[i]['subset_weights']) for i in range(len(loaded_dataset))]
            pickle.dump(max_weights, file)