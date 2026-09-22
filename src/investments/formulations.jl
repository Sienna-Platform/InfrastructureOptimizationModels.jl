# Should get moved to IS.Optimization and then re-exported by IOM?
abstract type AbstractTechnologyFormulation <: AbstractDeviceFormulation end

abstract type InvestmentTechnologyFormulation <: AbstractTechnologyFormulation end
abstract type OperationsTechnologyFormulation <: AbstractTechnologyFormulation end
abstract type FeasibilityTechnologyFormulation <: AbstractTechnologyFormulation end
abstract type RequirementFormulation <: AbstractServiceFormulation end