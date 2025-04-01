# Relaxed Embedded Wasserstein

This repository contains the code for the paper 'Joint Metric Space Embedding of Heterogeneous Data with Optimal Transport'.  Here, we employ Gromov-Wasserstein to perform joint embeddings of two datasets defined by a distance matrix and probability weights onto a reference space $(Z, d_Z)$. A preprint version of the paper is available on [Arxiv](https://arxiv.org/pdf/2502.07510).

The notebook 'Euclidean Example' gives a first intuition of our method for Euclidean embeddings and uses the REW function directly. The 'Spherical Examples' shows the use of the Wrapper_REW function and an embedding on a non-Euclidean space. We refer to the directories for visualizations. 
Finally, the notebook on latent space alignment provides a small example of domain adaptation. Here, we also use the barycentric projection to get a 'free-support' visualization. 

The REW_utils.py contains all the main functions, particularly the 'REW' and the 'Wrapper_REW' functions. The function 'Wrapper_REW' gives you some options for $(Z, d_Z)$ as an input to 'REW'. The main parameters of Wrapper_REW are:
- the GW regularizer $\lambda$: Higher values lead to more 'isometric' embeddings,
- the Sinkhorn parameter $\varepsilon$: Smaller values lead to better results but can cause numerical overflow,
- the embedding space $(Z, d_Z)$ defined by 'Z_name' (Geometry), 'n' (Grid Resolution) and 'max_len' (maximum distance $d_Z$): Choose 'Plane' (Euclidean Square), 'Sphere', 'Torus'
  
As is common with Gromov-Wasserstein can sometimes get stuck in local minima. To avoid this, it is recommended to vary these parameters. Additional details can be found in our paper.

## Requirements
The simulations have been performed with Python 3.12.2. Please take a look at the requirements.txt for our libraries. For the experiments in our paper, we additionally used these well-kept repositories:

* [JointMDS](https://github.com/borgwardtlab/jointmds>)
* [UnionCom](https://github.com/caokai1073/UnionCom)
* [SCOT](https://github.com/rsinghlab/SCOT)

Please cite the paper if you use the code.

## Citation
```bibtex
@article{beier2025joint,
  title={Joint Metric Space Embedding by Unbalanced OT with Gromov-Wasserstein Marginal Penalization},
  author={Beier*, Florian and Piening*, Moritz and Beinert, Robert and Steidl, Gabriele},
  journal={arXiv preprint arXiv:2502.07510},
  year={2025}
}

