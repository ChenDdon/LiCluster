# Data Information

## Introduction

This dataset is derived from the research article titled *"Persistent topology based machine learning prediction of cluster binding energies."* It has been recollected and restructured for studying higher-order interactions in Lithium (Li) clusters. The dataset encompasses systems ranging from Li4 to Li40 atoms, with each structure computed using Density Functional Theory (DFT) calculations.

All structural and energetic data are compiled in a single file named `TheDataOfClusters_4_40.data`. This file includes the atomic coordinates and corresponding binding energies for each structure. The dataset provides a comprehensive view of the statistical properties of Li clusters of various sizes, facilitating further analysis of many-body interactions.

## File Description

- **Filename**: `TheDataOfClusters_4_40.data`
- **Data Format**:
  - Each row represents a unique atomic structure.
  - **First Column**: Number of atoms in the structure (denoted as `n`).
  - **Columns 2 to (3n+1)**: x, y, z coordinates for each of the `n` atoms.
  - **Last Column**: Binding energy of the structure in eV per atom.


## Statistic information of all Li cluster systems. Energy unit is eV/atom.

| Datasets | Structures | Maximum Energy | Minimum Energy | Mean Energy | Median Energy |
|----------|------------|----------------|----------------|-------------|---------------|
| Li₄      | 8326       | 1.7337         | -0.6567        | -0.5258     | -0.5734       |
| Li₅      | 20988      | 2.1347         | -0.7087        | -0.5354     | -0.6172       |
| Li₆      | 20977      | 2.0881         | -0.8346        | -0.6275     | -0.6962       |
| Li₇      | 20998      | 2.0502         | -0.9051        | -0.6406     | -0.7259       |
| Li₈      | 21000      | 2.1364         | -0.9462        | -0.6739     | -0.7552       |
| Li₉      | 20999      | 1.4381         | -0.9495        | -0.6841     | -0.7793       |
| Li₁₀     | 20999      | 1.0743         | -0.9927        | -0.7089     | -0.8059       |
| Li₂₀     | 1000       | -0.3215        | -1.1052        | -0.9084     | -0.9488       |
| Li₄₀     | 1000       | -0.3905        | -1.1832        | -0.9541     | -0.9899       |
