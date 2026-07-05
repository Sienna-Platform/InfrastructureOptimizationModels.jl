# 0001 — IOM network vocabulary neutralization

Status: Accepted

## Context

InfrastructureOptimizationModels (IOM) is the domain-neutral optimization layer of the
Sienna stack. It builds optimization models without knowing about power-system concepts;
downstream packages (chiefly PowerOperationsModels, POM) supply the power taxonomy.

That neutrality was leaking. IOM's `NetworkModel{T}` was parameterized on
`AbstractPowerModel` — a power-specific name that IOM did not even own. The type is defined
in `InfrastructureSystems.Optimization` (Layer 0) and IOM merely re-exported it. Worse, the
`NetworkModel` struct named its two matrix fields `PTDF_matrix` and `MODF_matrix`, both
PowerNetworkMatrices-specific concepts, with matching getters `get_PTDF_matrix` /
`get_MODF_matrix`. A domain-neutral layer should not hard-code PTDF/MODF vocabulary.

## Decision

Neutralize the network vocabulary inside IOM without touching InfrastructureSystems:

- Define a neutral supertype **in IOM**:
  `abstract type AbstractNetworkModel <: IS.Optimization.AbstractInfrastructureModel end`,
  placed near the top of `src/core/network_model.jl` (before `NetworkModel`). This anchors `NetworkModel{T}` on an
  IOM-owned neutral type rather than the re-exported power-specific one.
- Stop importing and re-exporting `AbstractPowerModel`; export `AbstractNetworkModel`
  instead. `AbstractPowerModel` remains defined in IS and is untouched there.
- Rename the `NetworkModel` matrix fields `PTDF_matrix` → `network_matrix` and
  `MODF_matrix` → `contingency_matrix`, with getters `get_network_matrix` /
  `get_contingency_matrix` and matching constructor kwargs.
- Rename the private formulation check `_check_pm_formulation` → `_check_network_formulation`.
- **No deprecation alias.** This is the psy6 breaking line; old names are removed outright.

## Consequences

- POM reparents its concrete network formulations (`DCPPowerModel`, `ACPPowerModel`, etc.)
  onto `IOM.AbstractNetworkModel` instead of the old `AbstractPowerModel`, and updates all
  `NetworkModel` matrix field/getter/kwarg references. This is a breaking change to the
  IOM↔POM surface; downstream callers using `PTDF_matrix`/`MODF_matrix`/`get_PTDF_matrix`/
  `get_MODF_matrix` must migrate.
- With the vocabulary neutral, a second non-power adapter on IOM (an optimization layer that
  is not power-system-shaped) becomes realistic: nothing in IOM's network model now presumes
  PTDF/MODF matrices or a "power model" formulation.
- IS stays clean: no IS symbol was renamed, so the Layer-0 curated surface is unchanged.
