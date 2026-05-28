#################################################################################
# Standard Variable Types
# Only types that IOM's own infrastructure code references belong here.
# Device-specific variable types are defined in PowerOperationsModels.jl.
#################################################################################

# Device Power Variables (used in objective_function/, rateofchange_constraints)
struct ActivePowerVariable <: VariableType end
struct ActivePowerInVariable <: VariableType end
struct ActivePowerOutVariable <: VariableType end
struct PowerAboveMinimumVariable <: VariableType end

# Device Status Variables (used in range_constraint, objective_function/)
struct OnVariable <: VariableType end
struct StartVariable <: VariableType end
struct StopVariable <: VariableType end

# Reservation Variable (used in range_constraint for reserve bounds)
struct ReservationVariable <: VariableType end

# ServiceRequirementVariable moved to POM — not needed by IOM objective formulations.

# Cost Variables (used in piecewise_linear)
"""
Lambda weight variables ``\\lambda_i \\in [0, 1]`` for the convex-combination PWL cost formulation.

Each breakpoint of a [`IS.PiecewisePointCurve`] gets one lambda variable per component per time step.
The operating point and cost are expressed as weighted averages of breakpoint values.
"""
struct PiecewiseLinearCostVariable <: SparseVariableType end

# Rate Constraint Slack Variables (used in rateofchange_constraints)
struct RateofChangeConstraintSlackUp <: VariableType end
struct RateofChangeConstraintSlackDown <: VariableType end

# HVDC Variables (used in add_pwl_methods)
struct DCVoltage <: VariableType end

#################################################################################
# Standard Expression Types
# These are the base expression types for aggregating terms
#################################################################################

# Abstract types for expression hierarchies (used in IOM infrastructure code)
abstract type SystemBalanceExpressions <: ExpressionType end
abstract type RangeConstraintLBExpressions <: ExpressionType end
abstract type RangeConstraintUBExpressions <: ExpressionType end
abstract type CostExpressions <: ExpressionType end
abstract type PostContingencyExpressions <: ExpressionType end

# Concrete expression types used in IOM code
struct ProductionCostExpression <: CostExpressions end
abstract type ConstituentCostExpression <: CostExpressions end
struct FuelCostExpression <: ConstituentCostExpression end
struct StartUpCostExpression <: ConstituentCostExpression end
struct ShutDownCostExpression <: ConstituentCostExpression end
struct FixedCostExpression <: ConstituentCostExpression end
struct VOMCostExpression <: ConstituentCostExpression end
struct CurtailmentCostExpression <: CostExpressions end
struct FuelConsumptionExpression <: ExpressionType end
struct ActivePowerRangeExpressionLB <: RangeConstraintLBExpressions end
struct ActivePowerRangeExpressionUB <: RangeConstraintUBExpressions end

# Concrete expression types defined here for POM (not used in IOM code directly,
# but IOM exports them and POM relies on getting them from IOM)
struct ActivePowerBalance <: SystemBalanceExpressions end
struct ReactivePowerBalance <: SystemBalanceExpressions end
struct EmergencyUp <: ExpressionType end
struct EmergencyDown <: ExpressionType end
struct RawACE <: ExpressionType end
struct PostContingencyBranchFlow <: PostContingencyExpressions end
struct PostContingencyActivePowerGeneration <: PostContingencyExpressions end
struct NetActivePower <: ExpressionType end
struct DCCurrentBalance <: ExpressionType end
struct HVDCPowerBalance <: ExpressionType end

# Output writing configuration for expression types
should_write_resulting_value(::Type{<:CostExpressions}) = true
should_write_resulting_value(::Type{FuelConsumptionExpression}) = true
should_write_resulting_value(::Type{RawACE}) = true
should_write_resulting_value(::Type{ActivePowerBalance}) = true
should_write_resulting_value(::Type{ReactivePowerBalance}) = true
should_write_resulting_value(::Type{DCCurrentBalance}) = true
should_write_resulting_value(::Type{PostContingencyBranchFlow}) = true

# CostExpressions container method (moved here from optimization_container.jl
# because it requires the cost expression types to be defined first). Covers
# ProductionCostExpression, ConstituentCostExpression subtypes, and
# CurtailmentCostExpression — all use a JuMP.QuadExpr container so quadratic
# fuel terms can be stored in the same expression.
function add_expression_container!(
    container::OptimizationContainer,
    ::Type{T},
    ::Type{U},
    axs...;
    sparse = false,
    meta = CONTAINER_KEY_EMPTY_META,
) where {
    T <: CostExpressions,
    U <: Union{IS.InfrastructureSystemsComponent, IS.InfrastructureSystemsContainer},
}
    expr_container =
        _add_container!(container, T, U, JuMP.QuadExpr, sparse, axs...; meta = meta)
    remove_undef!(expr_container)
    return expr_container
end

#################################################################################
# Base Methods
#################################################################################

"""
    requires_initialization(formulation::AbstractDeviceFormulation)

Check if a device formulation requires initial conditions.
Default implementation returns false. Override for formulations with state variables.
"""
function requires_initialization(::AbstractDeviceFormulation)
    return false
end

"""
    add_to_expression!(
        container::OptimizationContainer,
        expression_type::Type{<:ExpressionType},
        variable_type::Type{<:VariableType},
        devices,
        model::DeviceModel,
        network_model::NetworkModel,
    )

Add device variables to system-wide expression.
This is a generic fallback that errors - specific implementations should override.
"""
function add_to_expression!(
    container::OptimizationContainer,
    expression_type::Type{<:ExpressionType},
    variable_type::Type{<:VariableType},
    devices,
    model::DeviceModel,
    network_model::NetworkModel,
)
    error(
        "add_to_expression! not implemented for expression_type=$expression_type, variable_type=$variable_type, device_type=$(typeof(devices.values[1]))",
    )
end
