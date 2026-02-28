# Can we understand the CMIP6 intermodel spread of the Eastern Pacific response from land-sea contrasts?

[Project pad](https://pad.gwdg.de/-gRNjrChQmqR8NeiddjMEg#)

[Günther et al. (2026)](https://doi.org/10.21203/rs.3.rs-7189653/v1) show that strong land–sea heating contrasts are associated with rapid Eastern Pacific cooling via atmospheric circulation changes under 4×CO₂ forcing.

This repository investigates whether similar relationships emerge across the CMIP6 multi-model ensemble and whether land–sea contrasts can explain part of the intermodel spread in the Eastern Pacific response.

### Repository Structure

The repository is organized around three physical hypotheses plus shared functionality:
```
.
├── itcz/
├── convection/
├── subtr-high/
└── common/
```

`itcz/` : Analysis related to the ITCZ-shift hypothesis.

`convection/` : Analysis related to the convection-shift mechanism.

`subtr-high/` : Analysis related to the subtropical-high hypothesis.

`common/` : Code shared across all hypotheses. Some useful functions are defined in tools.py within this folder (selecting SEP, averaging, ...)

Each of the four top-level folders contains:

```
<folder>/
├── src/
└── plots/
```

`src/` : source code for the respective hypothesis or shared utilities

`plots/` : generated figures

## Environment Setup

An environment specification file is provided at:

`common/src/environment.yaml`

Create the environment using either conda or mamba.

Using mamba (recommended)
```
mamba env create -f common/src/environment.yaml
mamba activate euclidyan-landheating
```
Using conda: replace `mamba` by `conda`

## Code Style

All code must be formatted with black before committing.

Run `black .` before committing.
