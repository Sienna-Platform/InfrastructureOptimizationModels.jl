# InfrastructureOptimizationModels.jl

```@meta
CurrentModule = InfrastructureOptimizationModels
```

## Overview

`InfrastructureOptimizationModels.jl` is a [`Julia`](http://www.julialang.org) package that provides core abstractions and optimization model structures for power systems operations modeling. It defines [`DecisionModel`](@ref) and [`EmulationModel`](@ref) types along with their associated optimization containers, formulations, and output handling capabilities.

## About

`InfrastructureOptimizationModels` is part of the National Lab of the Rockies NLR (formerly known as NREL)
[Sienna ecosystem](https://sienna-platform.github.io/Sienna/), an open source framework for
scheduling problems and dynamic simulations for power systems. The Sienna ecosystem can be
[found on github](https://github.com/Sienna-Platform/Sienna). It contains three applications:

  - [Sienna\Data](https://sienna-platform.github.io/Sienna/pages/applications/sienna_data.html) enables
    efficient data input, analysis, and transformation
  - [Sienna\Ops](https://sienna-platform.github.io/Sienna/pages/applications/sienna_ops.html) enables
    enables system scheduling simulations by formulating and solving optimization problems
  - [Sienna\Dyn](https://sienna-platform.github.io/Sienna/pages/applications/sienna_dyn.html) enables
    system transient analysis including small signal stability and full system dynamic
    simulations

Each application uses multiple packages in the [`Julia`](http://www.julialang.org)
programming language.
