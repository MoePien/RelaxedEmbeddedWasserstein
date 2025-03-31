# Relaxed Embedded Wasserstein

This repository contains the code for the paper 'Joint Metric Space Embedding of Heterogeneous Data with Optimal Transport'. 
A preprint version is available on [Arxiv](https://arxiv.org/pdf/2502.07510).

The notebook 'Euclidean Example' gives a first intuition for an Euclidean embedding. The 'Spherical Examples' shows the use of the Wrapper_REW function and an embedding on a non-Euclidean space. We refer to the directories for visualizations.

Finally, the notebook on latent space alignment provides a small example of domain adaptation.

Additional details can be found in our paper.

## Requirements
The simulations have been performed with Python 3.12.2. See the requirements.txt for our libraries. For the experiments in our paper, we additionally used these well-kept repositories:

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

