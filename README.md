# Code for *Stochastic colonization and host-to-host transmission shape gut bacterial variability*

This repository contains the code and data accompanying the paper *Stochastic colonization and host-to-host transmission shape gut bacterial variability*. The analyses combine colony-forming unit (CFU) count data, probabilistic reconstruction of bacterial populations, grid-based likelihood calculations, and Markov chain Monte Carlo (MCMC) sampling.

The CFU count data used in the paper are stored in the `real_data` directory.

This repository also includes a local copy of `REPOP`, the probabilistic reconstruction framework used to infer bacterial population distributions from dilution-plate CFU counts. A maintained version of `REPOP` can be found in the original repository:

[https://github.com/PessoaP/REPOP](https://github.com/PessoaP/REPOP)

For more details on REPOP, see also our eLife paper:

[https://doi.org/10.7554/eLife.107122.1](https://doi.org/10.7554/eLife.107122.1)

The local copy is included here to make the analyses associated with this paper reproducible without requiring a separate installation.

## Analysis workflow

### Pre-calibrate carrying capacity

The first step is to pre-calibrate the carrying capacity parameter, as described in Methods Section X.Y of the paper. This calibration is performed with:

```bash
python calibrate_capacity.py
```

After the calibration is complete, run:

```bash
jupyter notebook fig_s7.ipynb
```

This notebook generates the results shown in Figure S7.

### Grid calculation

After the carrying-capacity pre-calibration step, the likelihood is evaluated over a parameter grid. The grid calculation is implemented in `run_grid.py` and launched using:

```bash
bash grid_preliminary.sh
```

The output of this step is used to initialize and guide the MCMC analysis.

after this is complete (will take a while and miht be a good idea to slpit) 

### MCMC sampling

The MCMC analysis uses the grid-calculation results and produces the posterior samples used in Figure 4, Figure S8, and Figure S10. To run the MCMC analysis, use:

```bash
bash mcmc_bash.sh
```



## Requirements

The computational environment is specified in:

```text
env_gpu.yml
```

Use this file to reproduce the package versions used for the analysis.

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
[add BibTeX citation here]
```

## License

```text
[add license information here]
```