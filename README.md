

# MOT Particle Flow Simulation

This repository contains a simplified and preliminary implementation of the numerical simulations described in the paper: [**\[ Multi-Object Posterior Computation via Gibbs Sampling\]**](https://arxiv.org/pdf/2604.12449v1). 



## Implementation Details & Optimizations

To improve GPU performance, several computational shortcuts and optimizations have been implemented:
*   **Local PSF Calculation**: The Point Spread Function (PSF) is not computed across the entire image grid for every particle. Instead, it is restricted to a local window around each particle's position. (Note: This window is currently fixed and may require tuning).
*   **Gauss-Newton Hessian Approximation**: In the **Gromov Flow** implementation, the Hessian of the log-likelihood is approximated using the Gauss-Newton method. 

## Prerequisites

To run the simulations, you will need an **NVIDIA GPU** with CUDA support.

The following Julia packages are required:
*   `CUDA.jl`: For GPU-accelerated computing.
*   `StaticArrays.jl`: For efficient handling of state vectors.
*   `PlotlyJS.jl`: For visualizing results and trajectories.
*   `YAML.jl`: To load config.

## Usage

1.  Ensure all dependencies are installed:
    ```julia
    using Pkg; Pkg.add(["CUDA", "StaticArrays", "PlotlyJS", "YAML"])
    ```
2.  Execute the main simulation script:
    ```bash
    julia main.jl
    ```

---

*Note: This code is a research-grade, precarious implementation and is subject to further modifications.*
```