# real_option_models_rl

## About
This repository contains algorithms that solve optimal stopping problems motivated by real option models. The algorithms were proposed by the authors Jodi Dianetti, Giorgio Ferrari, and Renyuan Xu in their paper ''Reinforcement Learning in Real Option Models''. I have implemented their proposed algorithms into functioning Python code.


## Setup instructions:
```
git clone https://github.com/oskarru/real_option_models_rl.git
cd optimal-stopping-rl
pip install -r requirements.txt
```

**Computational Note**: Algorithm 4 is highly resource-intensive. Running it on a standard CPU might take several hours.


## References
If you find this work useful, please cite the original paper!:

> Dianetti, J., Ferrari, G., & Xu, R. (2026). *Reinforcement Learning in Real Option Models*

Or use the following BibTeX entry:
```
@misc{dianetti2026reinforcementlearningrealoption,
      title={Reinforcement Learning in Real Option Models}, 
      author={Jodi Dianetti and Giorgio Ferrari and Renyuan Xu},
      year={2026},
      eprint={2602.15643},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2602.15643}, 
}
```