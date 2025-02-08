import sys
import numpy as np
import os
import argparse
import linecache

# import code_pkg.simplicialcomplex_laplacian_sparse as plsparse
sys.path.append(os.path.join(os.path.dirname(__file__), "code_pkg"))
import simplicialcomplex_laplacian_sparse as plsparse


__author__ = "Dong Chen"
__date__ = "Jan. , 2025"


def example_for_one_structures():
    data_file = 'Li_cluster_data/TheDataOfClusters_4_40.data'

    # Get the point cloud
    data_line_num = 9324  # a random data from the Li cluster
    data_line = linecache.getline(data_file, data_line_num)
    line_ele = data_line.strip().split()
    n_atoms = int(float(line_ele[0]))
    cluster_energy = float(line_ele[-1])
    Li_coords = []
    for j in range(len(line_ele[1: -1])):
        if j % 3 == 0:
            Li_coords.append([float(coord) for coord in line_ele[1+j: 4+j]])

    # Get persistent laplacians
    p_lap = plsparse.SimplicialComplexLaplacian(eignv_num=40)  # the maximum eignvalues per Laplacian matrix, speed up the calculations
    laplacian_list = p_lap.persistent_simplicialcomplex_laplacian_sparse(
        pointcloud = Li_coords,
        max_dim = 2, # dim 0, 1, 2
        filtration = None,
        start_distance = 0.1,
        cutoff_distance = 10.1, 
        step_dis = 0.1,
        print_by_step = True,
    )  # return a list of eigenvalues

    # Get the fix-length vector, suitable for machine learning
    feature_for_one_structure = []
    for p, lap_dict in enumerate(laplacian_list):
        feature_for_one_filtration = []  # 6*3 value per scale
        for dim, eigvs in lap_dict.items():
            if eigvs is None:
                feature = [0] * 6
            else:
                stats_v = plsparse.statistic_eigvalues(eigvs)
                feature = [stats_v.count_zero, stats_v.nonzero_min, stats_v.max, stats_v.mean, stats_v.std, stats_v.generalised_mean_graph_energy]  # select the proper statistical information for eigenvalues
            feature_for_one_filtration.extend(feature)
        feature_for_one_structure.append(feature_for_one_filtration)
    flattened_feature = np.array(feature_for_one_structure, dtype=np.float32).reshape(-1, order='F')

    # Save vector as .txt, .npy and anyother format
    save_path = 'examples/ptl_features.txt'
    np.savetxt(save_path, flattened_feature)

    return None


def main():
    example_for_one_structures()
    return None


if __name__ == "__main__":
    main()
    print('End!')
