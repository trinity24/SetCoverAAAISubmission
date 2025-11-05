import numpy as np
import random
from typing import List, Tuple, Dict, Set
import json
import pickle

class SetCoverInstanceGenerator:
    """
    Generator for weighted set cover instances with colored elements.
    Problem Definition:
    - Universe P of n elements, each with a color from {1, 2, 3}
    - Collection of subsets, each containing at most d elements
    - Each subset has a real-valued weight
    - Color constraints: k_j <= |P \cap P_j| where P_j is set of elements with color j
    """
    def __init__(self, n, d,
                 k1, k2, k3, f,
                 num_subsets=None,  # Number of subsets to generate
                 weight_range=(0.1, 10.0),  # Weight range for subsets
                 color_distribution=(0.33, 0.33, 0.34),  # Distribution of colors
                 random_seed=None, element_colors=None, age_partitions=None):
        # Instance parameters
        self.n = n
        self.d = d
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.f = f
        self.random_seed = random_seed

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Subset generation parameters
        self.num_subsets = num_subsets if num_subsets is not None else int(n * 1.5)
        self.weight_range = weight_range
        self.color_distribution = color_distribution

        # Generated instance data
        self.universe = None
        self.element_colors = None
        self.subsets = None
        self.subset_weights = None

    def generate_universe_with_colors(self) -> Tuple[List[int], Dict[int, int]]:
        """
        Generate the universe of elements with coloring.
        """
        universe = list(range(self.n))
        element_colors = {}

        # First, ensure minimum color requirements are met
        elements_assigned = 0

        # Assign minimum required elements for each color - This makes sure we have a YES instance.
        for color, color_count in [(1, self.k1), (2, self.k2), (3, self.k3)]:
            for i in range(color_count):
                element_colors[elements_assigned] = color
                elements_assigned += 1

        # Assign remaining elements according to color distribution
        remaining_elements = self.n - elements_assigned
        remaining_colors = np.random.choice([1, 2, 3],
                                            size=remaining_elements,
                                            p=self.color_distribution)

        for i, color in enumerate(remaining_colors):
            element_colors[elements_assigned + i] = color

        # Shuffle to randomize element-color assignment
        elements = list(range(self.n))
        colors = list(element_colors.values())
        random.shuffle(colors)
        element_colors = {elem: color for elem, color in zip(elements, colors)}

        return universe, element_colors

    def generate_element_probs(self) -> Dict[int, float]:
        """
        Generate random probabilities for each element in the universe.
        """
        return {i: random.random() for i in range(self.n)}

    def generate_subsets(self) -> Tuple[List[Set[int]], List[float]]:
        """
        Generate subsets with their weights.

        Returns:
        --------
        subsets : List[Set[int]]
            List of subsets, each containing at most d elements
        weights : List[float]
            Weight for each subset
        """
        subsets = []
        weights = []

        # Ensure every element appears in at least one subset
        element_coverage = {elem: 0 for elem in self.universe}

        for subset_id in range(self.num_subsets):
            # Determine subset size (1 to d elements)
            subset_size = random.randint(1, self.d)

            # For first pass, prioritize uncovered elements
            if subset_id < self.n // self.d:
                # Select elements with lowest coverage first
                candidates = sorted(self.universe, key=lambda x: element_coverage[x])
                subset_elements = set(candidates[:subset_size])
            else:
                # Random selection for remaining subsets
                subset_elements = set(random.sample(self.universe, subset_size))

            # Generate weight for this subset
            weight = random.uniform(self.weight_range[0], self.weight_range[1])

            # Update coverage tracking
            for elem in subset_elements:
                element_coverage[elem] += 1

            subsets.append(subset_elements)
            weights.append(weight)

        # Ensure all elements are covered at least once
        uncovered = [elem for elem, count in element_coverage.items() if count == 0]
        while uncovered:
            # Create additional subsets for uncovered elements
            subset_size = min(len(uncovered), self.d)
            subset_elements = set(uncovered[:subset_size])
            weight = random.uniform(self.weight_range[0], self.weight_range[1])
        
            subsets.append(subset_elements)
            weights.append(weight)

            uncovered = uncovered[subset_size:]

        return subsets, weights
    def generate_subsets_with_max_frequency(self, max_frequency: int, agepartitions=None) -> Tuple[List[Set[int]], List[float]]:
        """
        Generate subsets such that no element appears in more than max_frequency subsets.
        If agepartitions is not None, each subset is a subset of some partition.
        """
        subsets = []
        weights = []
        element_coverage = {elem: 0 for elem in self.universe}

        # Prepare partitions if provided
        partitions = []
        if agepartitions is not None:
            # agepartitions should be a dict or list of sets/lists of elements
            if isinstance(agepartitions, dict):
                partitions = [set(v) for v in agepartitions.values()]
            else:
                partitions = [set(p) for p in agepartitions]
        else:
            partitions = [set(self.universe)]

        for subset_id in range(self.num_subsets):
            # Filter elements that have not reached max_frequency
            available_elements = [elem for elem in self.universe if element_coverage[elem] < max_frequency]
            if not available_elements:
                break  # No more elements can be added without exceeding max_frequency

            # Choose a partition to sample from
            partition = random.choice(partitions)
            partition_available = [elem for elem in partition if elem in available_elements]
            if not partition_available:
                continue

            subset_size = min(self.d, len(partition_available))
            if subset_size == 0:
                continue
            subset_size = random.randint(1, subset_size)  # Ensure at least one element

            # For first pass, prioritize uncovered elements
            if subset_id < self.n // self.d:
                candidates = sorted(partition_available, key=lambda x: element_coverage[x])
                subset_elements = set(candidates[:subset_size])
            else:
                subset_elements = set(random.sample(partition_available, subset_size))

            for elem in subset_elements:
                element_coverage[elem] += 1

            weight = random.uniform(self.weight_range[0], self.weight_range[1])
            subsets.append(subset_elements)
            weights.append(weight)

        # Ensure all elements are covered at least once
        uncovered = [elem for elem, count in element_coverage.items() if count == 0]
        while uncovered:
            available_uncovered = [elem for elem in uncovered if element_coverage[elem] < max_frequency]
            if not available_uncovered:
                break

            # Choose a partition containing uncovered elements
            partition = None
            for p in partitions:
                if any(elem in p for elem in available_uncovered):
                    partition = p
                    break
            if partition is None:
                break

            partition_uncovered = [elem for elem in available_uncovered if elem in partition]
            subset_size = min(len(partition_uncovered), self.d)
            if subset_size == 0:
                break
            subset_elements = set(partition_uncovered[:subset_size])
            for elem in subset_elements:
                element_coverage[elem] += 1
            weight = random.uniform(self.weight_range[0], self.weight_range[1])
            subsets.append(subset_elements)
            weights.append(weight)
            uncovered = [elem for elem in uncovered if element_coverage[elem] == 0]

        return subsets, weights
    def generate_instance(self, element_colors=None, age_partitions=None) -> Dict:
        """
        Generate a complete set cover instance.
            Complete instance data including all parameters and generated data
        """
        # Generate universe and colors
        if self.element_colors is None:
            self.universe, self.element_colors = self.generate_universe_with_colors()
        else:
            self.universe, _ = self.generate_universe_with_colors()
            self.element_colors = element_colors
        # Generate subsets and weights
        if age_partitions is None:
            self.subsets, self.subset_weights = self.generate_subsets_with_max_frequency(self.f)
        else:
            self.subsets, self.subset_weights = self.generate_subsets_with_max_frequency(self.f, age_partitions)
        self.element_to_subsets = { elem : set() for elem in self.universe }
        for subset_id, subset in enumerate(self.subsets):
            for elem in subset:
                self.element_to_subsets[elem].add(subset_id)
        self.element_frequencies = {elem: len(self.element_to_subsets[elem]) for elem in self.universe}
        self.partitions = {1: set(), 2: set(), 3: set()}
        for elem, color in self.element_colors.items():
            self.partitions[color].add(elem)
        # Generate element probabilities
        element_probs = self.generate_element_probs()

        # Compile instance data
        instance = {
            'parameters': {
                'n': self.n,
                'd': self.d,
                'k1': self.k1,
                'k2': self.k2,
                'k3': self.k3,
                'f': self.f,
                'num_subsets': len(self.subsets),
                'weight_range': self.weight_range,
                'color_distribution': self.color_distribution,
                'random_seed': self.random_seed
            },
            'element_to_subsets': self.element_to_subsets,
            'partitions': self.partitions,
            'universe': self.universe,
            'element_colors': self.element_colors,
            'element_probs': element_probs,
            'subsets': [list(s) for s in self.subsets],  # Convert sets to lists for JSON serialization
            'subset_weights': self.subset_weights,
            'element_frequencies': self.element_frequencies
        }

        return instance


    def save_instance(self, instance: Dict, filename: str, format: str = 'json'):
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(instance, f, indent=2)
        elif format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(instance, f)
        else:
            raise ValueError("Format must be 'json' or 'pickle'")

def generate_setcover_dataset(
    num_instances: int,
    n: int,
    d: int,
    k1: int,
    k2: int,
    k3: int,
    f: int,
    num_subsets: int = None,
    weight_range: tuple = (5.0, 10.0),
    color_distribution: tuple = (0.33, 0.33, 0.34),
    random_seed: int = None,
    element_colors: list = None,
    age_partitions: list = None,
    save_prefix: str = None
) -> list:
    """
    Generate a dataset of set cover instances.
    Returns a list of instances. 
    """
    instances = []
    for i in range(num_instances):
        seed = random_seed + i if random_seed is not None else None
        element_colors=element_colors if element_colors is not None else None
        age_partitions=age_partitions if age_partitions is not None else None
        generator = SetCoverInstanceGenerator(
            n=n,
            d=d,
            f=f,
            k1=k1,
            k2=k2,
            k3=k3,
            num_subsets=num_subsets,
            weight_range=weight_range,
            color_distribution=color_distribution,
            random_seed=seed,
            element_colors=element_colors,
            age_partitions=age_partitions
        )
        instance = generator.generate_instance(element_colors=element_colors, age_partitions=age_partitions)
        instances.append(instance)
        if save_prefix:
            filename = f"{save_prefix}_n{n}_d{d}_inst{i+1}.json"
            generator.save_instance(instance, filename)
    return instances
