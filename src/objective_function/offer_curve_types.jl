#################################################################################
# Offer Curve Types
# Types for MarketBidCost / ImportExportCost piecewise linear offer curves.
# Abstract + concrete parameter, variable, and constraint types.
#################################################################################

#################################################################################
# Parameter types: cost at minimum power
#################################################################################

"Parameters to define the cost at the minimum available power"
abstract type AbstractCostAtMinParameter <: ObjectiveFunctionParameter end

"[`AbstractCostAtMinParameter`](@ref) for the incremental case (power source)"
struct IncrementalCostAtMinParameter <: AbstractCostAtMinParameter end

"[`AbstractCostAtMinParameter`](@ref) for the decremental case (power sink)"
struct DecrementalCostAtMinParameter <: AbstractCostAtMinParameter end

#################################################################################
# Parameter types: piecewise linear slopes
#################################################################################

"Parameters to define the slopes of a piecewise linear cost function"
abstract type AbstractPiecewiseLinearSlopeParameter <: ObjectiveFunctionParameter end

"[`AbstractPiecewiseLinearSlopeParameter`](@ref) for the incremental case (power source)"
struct IncrementalPiecewiseLinearSlopeParameter <: AbstractPiecewiseLinearSlopeParameter end

"[`AbstractPiecewiseLinearSlopeParameter`](@ref) for the decremental case (power sink)"
struct DecrementalPiecewiseLinearSlopeParameter <: AbstractPiecewiseLinearSlopeParameter end

#################################################################################
# Parameter types: piecewise linear breakpoints
#################################################################################

# ObjectiveFunctionParameter (not TimeSeriesParameter) so storage is Float64, matching the
# slope twin and every reader; under TimeSeriesParameter recurrent builds stored VariableRefs.
"Parameters to define the breakpoints of a piecewise linear function"
abstract type AbstractPiecewiseLinearBreakpointParameter <: ObjectiveFunctionParameter end

"[`AbstractPiecewiseLinearBreakpointParameter`](@ref) for the incremental case (power source)"
struct IncrementalPiecewiseLinearBreakpointParameter <:
       AbstractPiecewiseLinearBreakpointParameter end

"[`AbstractPiecewiseLinearBreakpointParameter`](@ref) for the decremental case (power sink)"
struct DecrementalPiecewiseLinearBreakpointParameter <:
       AbstractPiecewiseLinearBreakpointParameter end

#################################################################################
# Variable types: block offer
#################################################################################

"Abstract type for piecewise linear block offer variables"
abstract type AbstractPiecewiseLinearBlockOffer <: SparseVariableType end

"""
Struct to dispatch the creation of piecewise linear block incremental offer variables for objective function

Docs abbreviation: ``\\delta``
"""
struct PiecewiseLinearBlockIncrementalOffer <: AbstractPiecewiseLinearBlockOffer end

"""
Struct to dispatch the creation of piecewise linear block decremental offer variables for objective function

Docs abbreviation: ``\\delta_d``
"""
struct PiecewiseLinearBlockDecrementalOffer <: AbstractPiecewiseLinearBlockOffer end

#################################################################################
# Constraint types: block offer
#################################################################################

"Abstract type for piecewise linear block offer constraints"
abstract type AbstractPiecewiseLinearBlockOfferConstraint <: ConstraintType end

"""
Struct to create the PiecewiseLinearBlockIncrementalOfferConstraint associated with a specified variable.

See the piecewise linear cost functions section for more information.
"""
struct PiecewiseLinearBlockIncrementalOfferConstraint <:
       AbstractPiecewiseLinearBlockOfferConstraint end

"""
Struct to create the PiecewiseLinearBlockDecrementalOfferConstraint associated with a specified variable.

See the piecewise linear cost functions section for more information.
"""
struct PiecewiseLinearBlockDecrementalOfferConstraint <:
       AbstractPiecewiseLinearBlockOfferConstraint end
