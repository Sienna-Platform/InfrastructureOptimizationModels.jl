# InfrastructureOptimizationModels.jl

[![Main - CI](https://github.com/Sienna-Platform/InfrastructureOptimizationModels.jl/actions/workflows/main-tests.yml/badge.svg)](https://github.com/Sienna-Platform/InfrastructureOptimizationModels.jl/actions/workflows/main-tests.yml)
[![codecov](https://codecov.io/gh/Sienna-Platform/InfrastructureOptimizationModels.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Sienna-Platform/InfrastructureOptimizationModels.jl)
[![Documentation Build](https://github.com/Sienna-Platform/InfrastructureOptimizationModels.jl/workflows/Documentation/badge.svg?)](https://sienna-platform.github.io/InfrastructureOptimizationModels.jl/stable)
[<img src="https://img.shields.io/badge/slack-@Sienna/InfrastructureOptimizationModels-sienna.svg?logo=slack">](https://join.slack.com/t/core-sienna/shared_invite/zt-glam9vdu-o8A9TwZTZqqNTKHa7q3BpQ)

InfrastructureOptimizationModels.jl provides core abstractions and optimization model structures for power systems operations modeling in the Sienna ecosystem. It defines `DecisionModel`, `EmulationModel`, `OptimizationContainer`, and related types used for formulating and solving power system optimization problems.

## Key Features

- Core abstractions: `DecisionModel`, `EmulationModel`, `OptimizationContainer`
- Device, service, and network formulation models
- Initial conditions management
- Time series parameter handling
- Optimization outputs processing and serialization

## Installation

```julia
julia> ] add InfrastructureOptimizationModels
```

For the latest development version:

```julia
julia> ] add InfrastructureOptimizationModels#main
```

## Quick Start

```julia
using InfrastructureOptimizationModels
using PowerSystems

# Create a decision model
template = ProblemTemplate(CopperPlatePowerModel)
sys = System("path/to/system.json")
model = DecisionModel(template, sys)

# Build and solve
build!(model; output_dir = "output")
solve!(model)
```

## Development

Contributions to the development and enhancement of InfrastructureOptimizationModels.jl are welcome. Please see [CONTRIBUTING.md](https://github.com/Sienna-Platform/InfrastructureOptimizationModels.jl/blob/main/CONTRIBUTING.md) for code contribution guidelines.

## License

InfrastructureOptimizationModels.jl is released under a BSD [license](https://github.com/Sienna-Platform/InfrastructureOptimizationModels.jl/blob/main/LICENSE). InfrastructureOptimizationModels.jl has been developed as part of the Sienna ecosystem at the U.S. Department of Energy's National Lab of the Rockies NLR (formerly known as NREL)
