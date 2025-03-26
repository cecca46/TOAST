# Spatially aware Fused Gromov-Wasserstein Optimal Transport

<p align="left">
  <img src="mainfig.png" alt="Logo" width="500" align="left" style="margin-right: 20px;">

## Abstract
Optimal Transport (OT) has emerged as a foundational framework for analyzing and aligning distributions across various domains. Recent advancements in spatial omics technologies have provided unprecedented insights into the spatial organization of tissues, but challenges remain in aligning spatial slices and integrating complementary single-cell and spatial data. Here, we propose a novel OT-based framework that extends the traditional Fused Gromov-Wasserstein distance to more comprehensively model spatial microenvironments and molecular heterogeneity. By introducing _spatial coherence_, quantified through the entropy of local neighborhoods, and _neighborhood consistency_, which preserves the expression profiles of neighboring spots, our modified objective function significantly improves the alignment of spatially resolved tissue slices and the mapping between single-cell and spatial data. We demonstrate the efficacy of our framework on simulated and real-world datasets, showcasing its superior performance in preserving local biological context, and accurately mapping cellular states across modalities.

## Summary

Optimal Transport (OT) is a mathematical framework that originates from the problem of transferring one distribution of mass to another in the most cost-efficient manner. OT has become a foundational tool across fields, including economics, machine learning, and biology. By providing a geometry-based approach to realize couplings between two probability distributions, OT facilitates the analysis and alignment of datasets originating from different domains. In biology, the ability to align and compare distributions is critical for studying complex systems such as multicellular tissues, where different cell types dynamically interact to maintain tissue function. For example, changes in cell composition, structure, and spatial organization often underlie the transition from healthy to diseased states. OT’s capacity to match distributions in a principled manner makes it an ideal tool for uncovering these cellular relationships and dynamics.

In this work, we introduce a novel spatially aware Fused Gromov-Wasserstein (FGW) framework for spatial omics, which explicitly incorporates spatial constraints into the optimal transport objective. Our method extends the classical FGW formulation by introducing two additional terms, spatial coherence and neighborhood consistency, to account for spatial organization and molecular heterogeneity. 

## Installation

To install the necessary dependencies, clone the repository and run:
```bash
git clone https://github.com/tanevskilab/spatial_OT.git
cd spatial_OT
pip install -r requirements.txt
```

## Project Structure

- **`spatial_OT/OT`**: Compute the spatially aware Optimal Transport between two spatial slices. 
- **`spatial_OT/utils.py`**: Provides additional utility functions such as data preprocessing, and graph construction.

## Reproducing the results
We provide the following python notebooks to reproduce the results in the paper for the simulated datatset, the Human LIBD data, Mouse Atlas and Stereo-seq data: **`DLPFC.ipynb`**, **`1d-sim.ipynb`**, **`2d-sim.ipynb`**, **`MouseAtlas.ipynb`** and **`StereoSeq.ipynb`**.
