""" Summary: python code for the normal simplicial complex Laplacian

    Author:
        Dong Chen
    Create:
        2023-04-07
    Modify:
        2024-06-02
    Dependencies:
        python                    3.7.4
        numpy                     1.21.5
"""


import numpy as np
import itertools
from functools import wraps
import copy
import argparse
import sys
import time
from scipy.spatial import distance
import scipy.sparse.linalg as spla
import scipy.sparse as sp


def timeit(func):
    """ Timer """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # print(f"{'='*5} Function {func.__name__}{args} {kwargs} Took {total_time:.3f} seconds {'='*5}")
        print(f"{'='*5} Function - {func.__name__} - took {total_time:.3f} seconds {'='*5}")
        return result
    return timeit_wrapper


class statistic_eigvalues(object):
    '''Input is 1-D array'''
    def __init__(self, eigvalues: np.array) -> None:
        digital = 5
        values = np.round(eigvalues, digital)
        self.all_values = sorted(values)
        self.second_small = self.all_values[1] if len(self.all_values) >=2 else 0
        self.nonzero_values = values[np.nonzero(values)]
        self.len_values = len(values)
        self.num_nonzero = np.count_nonzero(values)
        self.count_zero = self.len_values - self.num_nonzero
        self.max = np.max(values)
        self.sum = np.round(np.sum(values), digital)
        self.mean = np.round(np.mean(values), digital)
        self.std = np.round(np.std(values), digital)
        # 
        self.avg_degree = np.round(self.sum/self.len_values, digital)
        self.lap_energy = np.round(np.sum(np.abs(values - self.avg_degree)), digital)
        self.generalised_mean_graph_energy = np.round(self.lap_energy/self.len_values, digital)  # for the case alpha=1

        if self.num_nonzero > 0:
            self.nonzero_mean = np.round(np.mean(self.nonzero_values), digital)
            self.nonzero_std = np.round(np.std(self.nonzero_values), digital)
            self.nonzero_min = np.round(np.min(self.nonzero_values), digital)
            self.nonzero_var = np.round(np.var(self.nonzero_values), digital)
            # self.spanning_tree_number = np.round(np.prod(self.nonzero_values)/self.num_nonzero, digital)
        else:
            # if none nonzero min, set it as 0
            self.nonzero_mean = 0
            self.nonzero_std = 0
            self.nonzero_min = 0
            self.nonzero_var = 0

    def doc_for_feature(self):
        used_features = {
            'nonzero_min': self.nonzero_min, 'Fiedler_value': self.nonzero_min,
            'max': self.max,
            'mean': self.mean,
            'nonezero_mean': self.nonzero_mean,
            'laplacian_graph_energy': self.lap_energy,
            'generalised_mean_graph_energy': self.generalised_mean_graph_energy,
            'std': self.std,
            # 'spanning_tree_number': self.spanning_tree_number,
        }
        return used_features


class SimplicialComplexLaplacian(object):
    def __init__(self, eigenvalue_method='numpy_eigvalsh', eignv_num=100):
        self.distance_matrix = None
        if eigenvalue_method == 'numpy_eigvalsh':
            self.eigvalue_calculator = np.linalg.eigvalsh
        self.eignv_num = eignv_num

    @classmethod
    def adjacency_map_to_simplex(cls, adjacency_matrix: np.array, max_dim: int = 1) -> dict:
        """
            Given an adjacency matrix A for an undirected graph,
            construct the clique complex of the graph.
        """
        n = adjacency_matrix.shape[0]  # Number of nodes in the graph
        simplicial_complex = {dim: [] for dim in range(max_dim+1)}  # List of simplices in the clique complex

        # Add the 0-simplices (nodes)
        simplicial_complex[0] = [(i, ) for i in range(n)]

        # Add higher-dimensional simplices corresponding to cliques of size > 1
        target_dim = min(max_dim, n)
        for k in range(1, target_dim+1):
            for S in itertools.combinations(range(n), k+1):
                if all(adjacency_matrix[i,j] for i in S for j in S if i < j):
                    simplicial_complex[k].append(tuple(S))
        return simplicial_complex
    
    @classmethod
    def func_points_to_simplex_alpha(cls, xyz, max_distance, max_dim: int = 1):
        import gudhi
        max_dim = 2
        n = len(xyz)
        max_distance = 5  # local

        # complex 0
        data = xyz
        alpha_complex = defaultdict(list)
        alpha_complex_params = defaultdict(list)
        input_idx = np.arange(n)
        for sim_v_pair in gudhi.AlphaComplex(points=data).create_simplex_tree().get_filtration():
            dim = len(sim_v_pair[0]) - 1
            if dim > max_dim:  # filter higher dim simplex
                continue
            
            filter_p = np.sqrt(sim_v_pair[1])
            if filter_p > max_distance:  # filter the simplex that too far
                continue

            alpha_complex[dim].append(tuple(sim_v_pair[0]))
            alpha_complex_params[dim].append(filter_p)

        return alpha_complex, alpha_complex_params
    
    @classmethod
    def complex_to_boundary_matrix(cls, complex: dict) -> dict:
        # For dim_0, boundary matrix shape [len(node), 1]
        boundary_matrix_dict = {}
        for dim_n in sorted(complex.keys()):
            if dim_n == 0:
                boundary_matrix_dict[dim_n] = np.zeros([len(complex[0]), ])
                continue

            simplex_dim_n = sorted(complex[dim_n])
            simplex_dim_n_minus_1 = sorted(complex[dim_n-1])
            if len(simplex_dim_n) == 0:
                boundary_matrix_dict[dim_n] = None
                break

            boundary_matrix_dict[dim_n] = np.zeros([len(simplex_dim_n_minus_1), len(simplex_dim_n)])
            for idx_n, simplex_n in enumerate(simplex_dim_n):
                for omitted_n in range(len(simplex_n)):
                    omitted_simplex = tuple(np.delete(simplex_n, omitted_n))
                    omitted_simplex_idx = simplex_dim_n_minus_1.index(omitted_simplex)
                    boundary_matrix_dict[dim_n][omitted_simplex_idx, idx_n] = (-1)**omitted_n
        
        if boundary_matrix_dict[dim_n] is not None:
            boundary_matrix_dict[dim_n+1] = None

        return boundary_matrix_dict

    @classmethod
    def boundary_to_laplacian_matrix(cls, boundary_matrix_dict: dict, max_dim: int = None) -> dict:
        laplacian_matrix_dict = {}
        max_dim_boundary = max(boundary_matrix_dict.keys()) if max_dim is None else max_dim

        for dim_n in sorted(boundary_matrix_dict.keys()):
            boundary_matrix = boundary_matrix_dict[dim_n]
            if dim_n >= max_dim_boundary:
                break
            elif dim_n == 0:
                if boundary_matrix_dict[dim_n+1] is None:
                    laplacian_matrix_dict[dim_n] = np.zeros([len(boundary_matrix_dict[0])]*2)
                else:
                    laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T)
            elif dim_n > 0:
                if boundary_matrix_dict[dim_n+1] is None:
                    laplacian_matrix_dict[dim_n] = np.dot(boundary_matrix.T, boundary_matrix)
                    break
                else:
                    laplacian_matrix_dict[dim_n] = np.dot(
                        boundary_matrix_dict[dim_n+1], boundary_matrix_dict[dim_n+1].T) + np.dot(boundary_matrix.T, boundary_matrix)
        return laplacian_matrix_dict

    def simplicialcomplex_laplacian_from_simplicial_complex(self, simplicial_complex: dict, max_dim: int = None) -> dict:

        if max_dim is None:
            max_dim = max(simplicial_complex.keys())
        
        boundary_matrix_dict = self.complex_to_boundary_matrix(simplicial_complex)
        self.laplacian_matrix_dict = self.boundary_to_laplacian_matrix(boundary_matrix_dict)
        
        laplacian_eigenv = {}
        for dim_n in range(max_dim+1):
            if dim_n in self.laplacian_matrix_dict:
                eig_value = self.eigvalue_calculator(self.laplacian_matrix_dict[dim_n])
                eig_value = eig_value.real
                laplacian_eigenv[dim_n] = sorted(np.round(eig_value, 5))
            else:
                laplacian_eigenv[dim_n] = None
        return laplacian_eigenv

    def sparse_simplicialcomplex_laplacian_from_simplicial_complex(self, simplicial_complex: dict, max_dim: int = None) -> dict:

        if max_dim is None:
            max_dim = max(simplicial_complex.keys())
        
        boundary_matrix_dict = self.complex_to_boundary_matrix(simplicial_complex)
        self.laplacian_matrix_dict = self.boundary_to_laplacian_matrix(boundary_matrix_dict)
        
        laplacian_eigenv = {}
        for dim_n in range(max_dim+1):
            if dim_n in self.laplacian_matrix_dict:

                L_sparse = sp.csr_matrix(self.laplacian_matrix_dict[dim_n])
                L_d = L_sparse.shape[0]
                if L_d <= self.eignv_num:
                    eig_value = self.eigvalue_calculator(self.laplacian_matrix_dict[dim_n])
                else:
                    # Compute only k smallest eigenvalues (fastest)
                    ncv_n = min(L_d, 300)
                    eig_value = spla.eigsh(
                        L_sparse, k=self.eignv_num, which='SM', return_eigenvectors=False, ncv=ncv_n)

                laplacian_eigenv[dim_n] = sorted(np.round(eig_value, 5))
            else:
                laplacian_eigenv[dim_n] = None
        return laplacian_eigenv

    def simplicialcomplex_laplacian_from_adjacency_matrix(self, adjacency_matrix: np.array, max_dim: int = 1) -> dict:
        # max_dim + 1, used for boundary matrix
        simplicial_complex = self.adjacency_map_to_simplex(adjacency_matrix, max_dim+1)
        laplacian_eigenv = self.simplicialcomplex_laplacian_from_simplicial_complex(simplicial_complex, max_dim)

        return laplacian_eigenv

    def persistent_simplicialcomplex_laplacian(
        self, pointcloud=None, distance_matrix=None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        max_dim: int = 1, filtration: np.array = None,
        start_distance: float = 0, cutoff_distance: float = None, step_dis: float = None,
        print_by_step: bool = True,
    ) -> np.array:
        # initial setting
        if distance_matrix is None:
            distance_matrix = distance.cdist(pointcloud, pointcloud)
        points_num = distance_matrix.shape[0]

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(start_distance, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for step_n, threshold_dis in enumerate(filtration):
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if (adjacency_matrix == adjacency_matrix_temp).all():
                laplacian_eigenv = all_laplacian_features[-1]
            else:
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)
                laplacian_eigenv = self.simplicialcomplex_laplacian_from_adjacency_matrix(adjacency_matrix, max_dim)
            all_laplacian_features.append(laplacian_eigenv)
                
            if print_by_step:
                print(f"Filtration step: {step_n} | param: {threshold_dis:.3f} |")
                for dim_ii in range(max_dim):
                    print(f"dim_n: {dim_ii} | eigenvalues: {laplacian_eigenv[dim_ii]}")

        return all_laplacian_features


    def persistent_simplicialcomplex_laplacian_sparse(
        self, pointcloud=None, distance_matrix=None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        max_dim: int = 1, filtration: np.array = None,
        start_distance: float = 0, cutoff_distance: float = None, step_dis: float = None,
        print_by_step: bool = True,
    ) -> np.array:
        # initial setting
        if distance_matrix is None:
            distance_matrix = distance.cdist(pointcloud, pointcloud)
        points_num = distance_matrix.shape[0]

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(start_distance, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for step_n, threshold_dis in enumerate(filtration):
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if (adjacency_matrix == adjacency_matrix_temp).all():
                laplacian_eigenv = all_laplacian_features[-1]
            else:
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)

                simplicial_complex = self.adjacency_map_to_simplex(adjacency_matrix, max_dim+1)
                laplacian_eigenv = self.sparse_simplicialcomplex_laplacian_from_simplicial_complex(
                    simplicial_complex, max_dim)
            all_laplacian_features.append(laplacian_eigenv)
                
            if print_by_step:
                print(f"Filtration step: {step_n} | param: {threshold_dis:.3f} |")
                for dim_ii in range(max_dim+1):
                    print(f"dim_n: {dim_ii} | eigenvalues: {laplacian_eigenv[dim_ii]}")

        return all_laplacian_features


    def persistent_simplicialcomplex_laplacian_alpha(
        self, pointcloud=None, distance_matrix=None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        max_dim: int = 1, filtration: np.array = None,
        start_distance: float = 0, cutoff_distance: float = None, step_dis: float = None,
        print_by_step: bool = True,
    ) -> np.array:
        # initial setting
        if distance_matrix is None:
            distance_matrix = distance.cdist(pointcloud, pointcloud)
        points_num = distance_matrix.shape[0]

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(start_distance, cutoff_distance, step_dis)
        
        # get alpha complex from 0 to max_distancd
        assert pointcloud is not None, 'Point cloud required'
        assert cutoff_distance is not None, 'Cutoff distance required'
        import gudhi
        all_alpha_complex_distance2_pair = gudhi.AlphaComplex(points=pointcloud).create_simplex_tree().get_filtration()

        # get persistent laplacian
        all_laplacian_features = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for step_n, threshold_dis in enumerate(filtration):
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if (adjacency_matrix == adjacency_matrix_temp).all():
                laplacian_eigenv = all_laplacian_features[-1]
            else:
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)

                # get alpha complex
                simplicial_complex = {dim: [] for dim in range(max_dim+1+1)}
                for sim_v_pair in all_alpha_complex_distance2_pair:
                    dim_alpha = len(sim_v_pair[0]) - 1
                    if dim_alpha > max_dim + 1:
                        continue
                    filter_p = np.sqrt(sim_v_pair[1])
                    if filter_p > threshold_dis:  # filter the simplex that too far
                        continue
                    simplicial_complex[dim_alpha].append(tuple(sim_v_pair[0]))
                
                laplacian_eigenv = self.simplicialcomplex_laplacian_from_simplicial_complex(simplicial_complex, max_dim)
            all_laplacian_features.append(laplacian_eigenv)
                
            if print_by_step:
                print(f"Filtration step: {step_n} | param: {threshold_dis:.3f} |")
                for dim_ii in range(max_dim + 1):
                    print(f"dim_n: {dim_ii} | eigenvalues: {laplacian_eigenv[dim_ii]}")

        return all_laplacian_features

    def persistent_simplicialcomplex_laplacian_dim0(
        self, pointcloud=None, distance_matrix=None,
        max_adjacency_matrix: np.array = None, min_adjacency_matrix: np.array = None,
        filtration: np.array = None,
        start_distance: float = 0, cutoff_distance: float = None, step_dis: float = None,
        print_by_step: bool = True,
    ) -> dict:
        # initial setting
        if distance_matrix is None:
            distance_matrix = distance.cdist(pointcloud, pointcloud)
        points_num = distance_matrix.shape[0]

        if max_adjacency_matrix is None:
            max_adjacency_matrix = np.ones([points_num, points_num], dtype=int)
            np.fill_diagonal(max_adjacency_matrix, 0)
        
        if min_adjacency_matrix is None:
            min_adjacency_matrix = np.zeros([points_num, points_num], dtype=int)

        if filtration is None:
            filtration = np.arange(start_distance, cutoff_distance, step_dis)
        
        all_laplacian_features = []
        all_num_1_simplex = []
        adjacency_matrix_temp = np.ones([points_num]*2, dtype=int)
        for step_n, threshold_dis in enumerate(filtration):
            adjacency_matrix = (((distance_matrix <= threshold_dis) * max_adjacency_matrix + min_adjacency_matrix) > 0)

            if (adjacency_matrix == adjacency_matrix_temp).all():
                laplacian_eigenv = all_laplacian_features[-1]
                num_1_simplex = all_num_1_simplex[-1]
            else:
                adjacency_matrix_temp = copy.deepcopy(adjacency_matrix)
                laplacian_matrix_dim0 = np.diag(np.sum(adjacency_matrix, axis=0)) - adjacency_matrix
                eig_value = self.eigvalue_calculator(laplacian_matrix_dim0)
                eig_value = eig_value.real
                laplacian_eigenv = {0: sorted(np.round(eig_value, 5))}
                num_1_simplex = int(0.5*np.sum(adjacency_matrix_temp))
            all_laplacian_features.append(laplacian_eigenv)
            all_num_1_simplex.append(num_1_simplex)

            if print_by_step:
                print(f"Filtration step: {step_n} | param: {threshold_dis:.3f} | eigenvalues: {laplacian_eigenv[0]}")
        self.all_num_1_simplex = all_num_1_simplex
        return all_laplacian_features


def main():
    aa = SimplicialComplexLaplacian()
    adjacency_matrix = np.array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0],
    ])
    adjacency_matrix = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ])
    # adjacency_matrix = np.array([
    #     [0, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0],
    # ])
    ww = aa.adjacency_map_to_simplex(adjacency_matrix, max_dim=2)
    print('simplical complex')
    print(ww)
    feat = aa.simplicialcomplex_laplacian_from_adjacency_matrix(adjacency_matrix, max_dim=2)
    print(feat)
    print(aa.laplacian_matrix_dict[0])

    print(np.diag(np.sum(adjacency_matrix, axis=0)) - adjacency_matrix)
    return None


if __name__ == "__main__":
    main()
