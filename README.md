
# Multiscale topological learning for Li cluster

## Introduction

The source data and code are derived from the work titled **"Enhancing Energy Predictions in Multi-Atom Systems with Multiscale Topological Learning"** by Dong Chen, Rui Wang, Guowei Wei, and Feng Pan.

---

## Features  
  
- Efficient construction of Persistent Topological Laplacians (PTLs) for given structures.  
- Generating prefixed-dimensional vector representation for given structures (point cloud).

## Installation  

### **Prerequisites**  

```bash
git clone https://github.com/ChenDdon/LiCluster.git
```

Suggested dependencies:  
- numpy                     1.26.4
- scipy                     1.13.1


## Usage  

### Basic Example
```bash
python main_example.py
```

## Code Structure

```
/project-folder
│── README.md
│── main_example.py
│── /code_pkg
│   │── simplicialcomplex_laplacian_sparse.py    # Functions for PTL computation
│── /Li_cluster_data
│   │── README.md                # Explaination for data
│   │── TheDataOfClusters_4_40.data         # Li cluster data
│── /examples                     # Example output directory
```

## Citation  
  
```
@article{cd2025mul,
  author = {Dong Chen, Rui Wang, Guowei Wei, and Feng Pan.},
  title = {Enhancing Energy Predictions in Multi-Atom Systems with Multiscale Topological Learning},
  journal = {Journal Name},
  year = {2025}
}
```

## **10. License**  

This project is licensed under the MIT License - see the LICENSE file for details.
