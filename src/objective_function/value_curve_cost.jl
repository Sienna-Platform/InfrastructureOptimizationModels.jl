#################################################################################
# Value Curve Objective Function: Delta PWL Formulation
#
# Objective function formulations for ValueCurve-based offer curves using the
# delta (incremental/block) PWL method. Maps ValueCurve types (static and
# time-series-backed) to slopes/breakpoints and routes to the delta formulation
# primitives in objective_function_pwl_delta.jl.
#
# IOM defines objective function formulations — the mathematical structure of
# JuMP objective terms. "Costs" (production cost, fuel cost, etc.) are a
# domain concept defined in POM. This file provides the formulation machinery
# that POM routes specific cost types into. PSY cost types appear in some
# function signatures for dispatch, but the formulations themselves are
# generic over IS.InfrastructureSystemsComponent and IS.ValueCurve types.
#
# Device-specific overloads (e.g., ThermalMultiStart, ControllableLoad) are
# in POM.
#################################################################################

#################################################################################
# Section 1: Offer Curve Accessor Wrappers
# Map PSY cost types (MarketBidCost, ImportExportCost) to a unified interface.
#################################################################################

####################### get_{output/input}_offer_curves #########################
# 1-argument getters: straight getfield calls (same PSY getter for static and TS variants)
get_output_offer_curves(cost::IEC_TYPES) = PSY.get_import_offer_curves(cost)
get_output_offer_curves(cost::MBC_TYPES) = PSY.get_incremental_offer_curves(cost)
get_input_offer_curves(cost::IEC_TYPES) = PSY.get_export_offer_curves(cost)
get_input_offer_curves(cost::MBC_TYPES) = PSY.get_decremental_offer_curves(cost)

# 2-argument getters: resolve time series if needed, return static curve(s).
# Static types: delegate to 1-arg getter (no resolution needed).
get_output_offer_curves(
    ::PSY.Component,
    cost::PSY.ImportExportCost;
    kwargs...,
) = PSY.get_import_offer_curves(cost)
get_output_offer_curves(
    ::PSY.Component,
    cost::PSY.MarketBidCost;
    kwargs...,
) = PSY.get_incremental_offer_curves(cost)
get_input_offer_curves(
    ::PSY.Component,
    cost::PSY.ImportExportCost;
    kwargs...,
) = PSY.get_export_offer_curves(cost)
get_input_offer_curves(
    ::PSY.Component,
    cost::PSY.MarketBidCost;
    kwargs...,
) = PSY.get_decremental_offer_curves(cost)
# TS types: resolve via PSY's 2-arg getters.
get_output_offer_curves(
    component::PSY.Component,
    cost::PSY.ImportExportTimeSeriesCost;
    kwargs...,
) = PSY.get_import_variable_cost(component, cost; kwargs...)
get_output_offer_curves(
    component::PSY.Component,
    cost::PSY.MarketBidTimeSeriesCost;
    kwargs...,
) = PSY.get_incremental_variable_cost(component, cost; kwargs...)
get_input_offer_curves(
    component::PSY.Component,
    cost::PSY.ImportExportTimeSeriesCost;
    kwargs...,
) = PSY.get_export_variable_cost(component, cost; kwargs...)
get_input_offer_curves(
    component::PSY.Component,
    cost::PSY.MarketBidTimeSeriesCost;
    kwargs...,
) = PSY.get_decremental_variable_cost(component, cost; kwargs...)

######################### get_offer_curves(direction, ...) ##############################

# direction and device:
get_offer_curves(::DecrementalOffer, device::PSY.StaticInjection) =
    get_input_offer_curves(PSY.get_operation_cost(device))
get_offer_curves(::IncrementalOffer, device::PSY.StaticInjection) =
    get_output_offer_curves(PSY.get_operation_cost(device))
get_initial_input(::DecrementalOffer, device::PSY.StaticInjection) =
    IS.get_initial_input(
        PSY.get_value_curve(get_input_offer_curves(PSY.get_operation_cost(device))),
    )
get_initial_input(::IncrementalOffer, device::PSY.StaticInjection) =
    IS.get_initial_input(
        PSY.get_value_curve(get_output_offer_curves(PSY.get_operation_cost(device))),
    )

# direction and cost curve (needed for VOM code path):
get_offer_curves(::DecrementalOffer, op_cost::PSY.OfferCurveCost) =
    get_input_offer_curves(op_cost)
get_offer_curves(::IncrementalOffer, op_cost::PSY.OfferCurveCost) =
    get_output_offer_curves(op_cost)

#################################################################################
# Section 2: OfferDirection Type Dispatch Table
# Maps OfferDirection to the appropriate parameter/variable/constraint types.
#################################################################################

_slope_param(::IncrementalOffer) = IncrementalPiecewiseLinearSlopeParameter
_slope_param(::DecrementalOffer) = DecrementalPiecewiseLinearSlopeParameter

_breakpoint_param(::IncrementalOffer) = IncrementalPiecewiseLinearBreakpointParameter
_breakpoint_param(::DecrementalOffer) = DecrementalPiecewiseLinearBreakpointParameter

_block_offer_var(::IncrementalOffer) = PiecewiseLinearBlockIncrementalOffer
_block_offer_var(::DecrementalOffer) = PiecewiseLinearBlockDecrementalOffer

_block_offer_constraint(::IncrementalOffer) = PiecewiseLinearBlockIncrementalOfferConstraint
_block_offer_constraint(::DecrementalOffer) = PiecewiseLinearBlockDecrementalOfferConstraint

_objective_sign(::IncrementalOffer) = OBJECTIVE_FUNCTION_POSITIVE
_objective_sign(::DecrementalOffer) = OBJECTIVE_FUNCTION_NEGATIVE

#################################################################################
# Section 3: _get_parameter_field Dispatch Table
# Maps parameter types to PSY getter functions.
#################################################################################

_get_parameter_field(::Type{<:StartupCostParameter}, op_cost) = PSY.get_start_up(op_cost)
_get_parameter_field(::Type{<:ShutdownCostParameter}, op_cost) = PSY.get_shut_down(op_cost)
_get_parameter_field(::Type{<:IncrementalCostAtMinParameter}, op_cost) =
    IS.get_initial_input(PSY.get_value_curve(get_output_offer_curves(op_cost)))
_get_parameter_field(::Type{<:DecrementalCostAtMinParameter}, op_cost) =
    IS.get_initial_input(PSY.get_value_curve(get_input_offer_curves(op_cost)))
_get_parameter_field(
    ::Type{
        <:Union{
            IncrementalPiecewiseLinearSlopeParameter,
            IncrementalPiecewiseLinearBreakpointParameter,
        },
    },
    op_cost,
) = get_output_offer_curves(op_cost)
_get_parameter_field(
    ::Type{
        <:Union{
            DecrementalPiecewiseLinearSlopeParameter,
            DecrementalPiecewiseLinearBreakpointParameter,
        },
    },
    op_cost,
) = get_input_offer_curves(op_cost)

#################################################################################
# Section 4: Device Cost Detection Predicates (generic)
# Device-specific overrides (RenewableNonDispatch, PowerLoad, etc.) are in POM.
#################################################################################

_has_market_bid_cost(device::PSY.StaticInjection) =
    _has_market_bid_cost(PSY.get_operation_cost(device))
_has_market_bid_cost(::MBC_TYPES) = true
_has_market_bid_cost(::PSY.OperationalCost) = false

_has_import_export_cost(::PSY.StaticInjection) = false
_has_import_export_cost(device::PSY.Source) =
    _has_import_export_cost(PSY.get_operation_cost(device))
_has_import_export_cost(::IEC_TYPES) = true
_has_import_export_cost(::PSY.OperationalCost) = false

_has_offer_curve_cost(device::PSY.Component) =
    _has_market_bid_cost(device) || _has_import_export_cost(device)

# With the static/TS type split, time-series parameters are determined by cost type:
# TS cost types always have time-series parameters; static types never do.
_has_parameter_time_series(device::PSY.StaticInjection) =
    _has_parameter_time_series(PSY.get_operation_cost(device))

_has_parameter_time_series(::TS_OFFER_CURVE_COST_TYPES) = true
_has_parameter_time_series(::PSY.OperationalCost) = false

#################################################################################
# Section 5: _consider_parameter (generic versions)
# Whether a parameter should be added based on what's in the container.
# POM overrides for ThermalMultiStart startup (MULTI_START_VARIABLES).
#################################################################################

_consider_parameter(
    ::Type{<:StartupCostParameter},
    container::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = has_container_key(container, StartVariable, T)

_consider_parameter(
    ::Type{<:ShutdownCostParameter},
    container::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = has_container_key(container, StopVariable, T)

_consider_parameter(
    ::Type{<:AbstractCostAtMinParameter},
    container::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = has_container_key(container, OnVariable, T)

_consider_parameter(
    ::Type{<:AbstractPiecewiseLinearSlopeParameter},
    ::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = true

_consider_parameter(
    ::Type{<:AbstractPiecewiseLinearBreakpointParameter},
    ::OptimizationContainer,
    ::DeviceModel{T, D},
) where {T, D} = true

#################################################################################
# Section 6: Validation
# Generic validation for offer curve costs. Device-specific overrides
# (ThermalMultiStart, RenewableDispatch, Storage) are in POM.
#################################################################################

curvity_check(::IncrementalOffer, x) = PSY.is_convex(x)
curvity_check(::DecrementalOffer, x) = PSY.is_concave(x)
expected_curvity(::IncrementalOffer) = "convex"
expected_curvity(::DecrementalOffer) = "concave"

function validate_occ_breakpoints_slopes(device::PSY.StaticInjection, dir::OfferDirection)
    offer_curves = get_offer_curves(dir, device)
    _validate_occ_curves(device, dir, offer_curves)
end

# Static: validate convexity/concavity and cost-type-specific constraints
function _validate_occ_curves(
    device::PSY.StaticInjection,
    dir::OfferDirection,
    cost_curve::PSY.CostCurve{PSY.PiecewiseIncrementalCurve},
)
    device_name = get_name(device)
    cost_curve_name = nameof(typeof(PSY.get_operation_cost(device)))
    curvity_check(dir, cost_curve) ||
        throw(
            ArgumentError(
                "$(uppercasefirst(string(dir))) $cost_curve_name for component $(device_name) is non-$(expected_curvity(dir))",
            ),
        )
    _validate_occ_subtype(PSY.get_operation_cost(device), dir, cost_curve, device_name)
end

# TS-backed: validated at parameter population time, not here
_validate_occ_curves(::PSY.StaticInjection, ::OfferDirection,
    ::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}) = nothing

_validate_occ_subtype(::PSY.MarketBidCost, ::OfferDirection, ::PSY.CostCurve, args...) =
    nothing

function _validate_occ_subtype(
    ::PSY.ImportExportCost,
    ::OfferDirection,
    curve::PSY.CostCurve,
    args...,
)
    !iszero(PSY.get_vom_cost(curve)) && throw(
        ArgumentError(
            "For ImportExportCost, VOM cost must be zero.",
        ),
    )
    !iszero(PSY.get_initial_input(curve)) && throw(
        ArgumentError(
            "For ImportExportCost, initial input must be zero.",
        ),
    )
    fd = PSY.get_function_data(PSY.get_value_curve(curve))
    if !iszero(first(PSY.get_x_coords(fd)))
        throw(
            ArgumentError(
                "For ImportExportCost, the first breakpoint must be zero.",
            ),
        )
    end
end

# Generic validate_occ_component overloads for PSY.StaticInjection.
# Device-specific overloads (ThermalMultiStart, RenewableDispatch, Storage) are in POM.

function validate_occ_component(::Type{<:StartupCostParameter}, device::PSY.StaticInjection)
    op_cost = PSY.get_operation_cost(device)
    # TS types are validated at parameter population time
    _is_time_series_cost(op_cost) && return
    startup = PSY.get_start_up(op_cost)
    if startup isa Union{NTuple{3, Float64}, StartUpStages}
        @warn "Multi-start costs detected for non-multi-start unit $(get_name(device)), will take the maximum"
    elseif !(startup isa Float64)
        throw(
            ArgumentError(
                "Expected Float64, NTuple{3, Float64}, or StartUpStages startup cost but got $(typeof(startup)) for $(get_name(device))",
            ),
        )
    end
    return
end

function validate_occ_component(
    ::Type{<:ShutdownCostParameter},
    device::PSY.StaticInjection,
)
    op_cost = PSY.get_operation_cost(device)
    # TS types are validated at parameter population time
    _is_time_series_cost(op_cost) && return
    # Static MBC: shut_down is LinearCurve; ThermalGenerationCost: shut_down is Float64
    shutdown = PSY.get_shut_down(op_cost)
    if shutdown isa IS.LinearCurve
        return  # valid
    elseif shutdown isa Float64
        return  # valid (e.g. ThermalGenerationCost)
    else
        throw(
            ArgumentError(
                "Expected Float64 or LinearCurve shutdown cost but got $(typeof(shutdown)) for $(get_name(device))",
            ),
        )
    end
end

# Consistency of initial_input vs offer curves is guaranteed by the static/TS type split
validate_occ_component(::Type{<:AbstractCostAtMinParameter}, ::PSY.StaticInjection) =
    nothing

validate_occ_component(
    ::Type{<:IncrementalPiecewiseLinearBreakpointParameter},
    device::PSY.StaticInjection,
) = validate_occ_breakpoints_slopes(device, IncrementalOffer())

validate_occ_component(
    ::Type{<:DecrementalPiecewiseLinearBreakpointParameter},
    device::PSY.StaticInjection,
) = validate_occ_breakpoints_slopes(device, DecrementalOffer())

# Slope and breakpoint validations are done together, nothing to do here
validate_occ_component(
    ::Type{<:AbstractPiecewiseLinearSlopeParameter},
    device::PSY.StaticInjection,
) = nothing

#################################################################################
# Section 7: Parameter Processing Orchestration
#################################################################################

function _process_occ_parameters_helper(
    ::Type{P},
    container::OptimizationContainer,
    model,
    devices,
) where {P <: ParameterType}
    for device in devices
        validate_occ_component(P, device)
    end
    if _consider_parameter(P, container, model)
        ts_devices =
            filter(device -> _has_parameter_time_series(device), devices)
        (length(ts_devices) > 0) && add_parameters!(container, P, ts_devices, model)
    end
end

"Validate ImportExportCosts and add the appropriate parameters"
function process_import_export_parameters!(
    container::OptimizationContainer,
    devices_in,
    model::DeviceModel,
)
    devices = [d for d in devices_in if _has_import_export_cost(d)]

    for param in (
        IncrementalPiecewiseLinearSlopeParameter,
        IncrementalPiecewiseLinearBreakpointParameter,
        DecrementalPiecewiseLinearSlopeParameter,
        DecrementalPiecewiseLinearBreakpointParameter,
    )
        _process_occ_parameters_helper(param, container, model, devices)
    end
end

"Validate MarketBidCosts and add the appropriate parameters"
function process_market_bid_parameters!(
    container::OptimizationContainer,
    devices_in,
    model::DeviceModel,
    incremental::Bool = true,
    decremental::Bool = false,
)
    devices = [d for d in devices_in if _has_market_bid_cost(d)]
    isempty(devices) && return

    for param in (
        StartupCostParameter,
        ShutdownCostParameter,
    )
        _process_occ_parameters_helper(param, container, model, devices)
    end
    if incremental
        for param in (
            IncrementalCostAtMinParameter,
            IncrementalPiecewiseLinearSlopeParameter,
            IncrementalPiecewiseLinearBreakpointParameter,
        )
            _process_occ_parameters_helper(param, container, model, devices)
        end
    end
    if decremental
        for param in (
            DecrementalCostAtMinParameter,
            DecrementalPiecewiseLinearSlopeParameter,
            DecrementalPiecewiseLinearBreakpointParameter,
        )
            _process_occ_parameters_helper(param, container, model, devices)
        end
    end
end

#################################################################################
# Section 10: PWL Data Retrieval
#################################################################################

function _get_pwl_data(
    dir::OfferDirection,
    container::OptimizationContainer,
    component::T,
    time::Int,
) where {T <: PSY.Component}
    name = PSY.get_name(component)
    cost_data = get_offer_curves(dir, component)
    breakpoint_cost_component, slope_cost_component, unit_system =
        _get_raw_pwl_data(dir, container, T, name, cost_data, time)

    breakpoints, slopes = get_piecewise_curve_per_system_unit(
        breakpoint_cost_component,
        slope_cost_component,
        unit_system,
        get_model_base_power(container),
        PSY.get_base_power(component),
    )
    return breakpoints, slopes
end

# static curve: read directly from the cost curve
function _get_raw_pwl_data(
    ::OfferDirection,
    ::OptimizationContainer,
    ::Type{<:PSY.Component},
    ::String,
    cost_data::PSY.CostCurve{PSY.PiecewiseIncrementalCurve},
    ::Int,
)
    cost_component = PSY.get_function_data(PSY.get_value_curve(cost_data))
    return PSY.get_x_coords(cost_component),
    PSY.get_y_coords(cost_component),
    PSY.get_power_units(cost_data)
end

# time-series curve: read from parameter arrays. Parameter containers for
# Slope/Breakpoint are allocated with axes `(names, segments|points, times)`, so the
# 3-index lookup mirrors `_fill_pwl_data_from_arrays!`. The parameter values carry the
# units declared on the `CostCurve`, so we forward those through (don't hardcode).
function _get_raw_pwl_data(
    dir::OfferDirection,
    container::OptimizationContainer,
    ::Type{T},
    name::String,
    cost_data::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve},
    time::Int,
) where {T <: PSY.Component}
    SlopeParam = _slope_param(dir)
    slope_arr = get_parameter_array(container, SlopeParam, T)
    slope_mult = get_parameter_multiplier_array(container, SlopeParam, T)
    @assert size(slope_arr) == size(slope_mult)
    seg_axis = axes(slope_arr)[2]
    slope_cost_component = Vector{Float64}(undef, length(seg_axis))
    for (i, seg) in enumerate(seg_axis)
        slope_cost_component[i] = slope_arr[name, seg, time] * slope_mult[name, seg, time]
    end

    BreakpointParam = _breakpoint_param(dir)
    bp_arr = get_parameter_array(container, BreakpointParam, T)
    bp_mult = get_parameter_multiplier_array(container, BreakpointParam, T)
    @assert size(bp_arr) == size(bp_mult)
    point_axis = axes(bp_arr)[2]
    breakpoint_cost_component = Vector{Float64}(undef, length(point_axis))
    for (i, pt) in enumerate(point_axis)
        breakpoint_cost_component[i] = bp_arr[name, pt, time] * bp_mult[name, pt, time]
    end

    @assert_op length(slope_cost_component) == length(breakpoint_cost_component) - 1
    return breakpoint_cost_component, slope_cost_component, PSY.get_power_units(cost_data)
end

#################################################################################
# Section 11: PWL Objective Terms + Variable Objective Formulation (generic)
# Load formulation overloads (AbstractControllablePowerLoadFormulation) are in POM.
#################################################################################

"""
Add PWL objective terms using the **delta (incremental/block-offer) formulation**.

Given an offer curve with breakpoints ``P_0, P_1, \\ldots, P_n`` and slopes
``m_1, m_2, \\ldots, m_n``, this function:

1. Creates delta variables ``\\delta_k \\geq 0`` for each segment via [`add_pwl_variables_delta!`](@ref),
   with no upper bound (block sizes are enforced by constraints).
2. Adds linking and block-size constraints via [`add_pwl_constraint_delta!`](@ref):
   ``p = \\sum_k \\delta_k`` and ``\\delta_k \\leq P_{k+1} - P_k``.
3. Builds the cost expression ``C = \\sum_k m_k \\, \\delta_k`` via [`get_pwl_cost_expression_delta`](@ref).

For convex offer curves (``m_1 \\leq m_2 \\leq \\cdots \\leq m_n``), no SOS2 or binary
variables are needed — the optimizer fills cheap segments first automatically.

Dispatches on `OfferDirection` (incremental or decremental) to select the appropriate
variable and constraint types.

See also: [`add_pwl_term_lambda!`](@ref) for the lambda (convex combination) formulation used by
`CostCurve{PiecewisePointCurve}`.
"""
function add_pwl_term_delta!(
    dir::OfferDirection,
    container::OptimizationContainer,
    component::T,
    ::PSY.OfferCurveCost,
    ::Type{U},
    ::Type{V},
) where {T <: PSY.Component, U <: VariableType, V <: AbstractDeviceFormulation}
    W = _block_offer_var(dir)
    X = _block_offer_constraint(dir)

    name = PSY.get_name(component)
    resolution = get_resolution(container)
    dt = Dates.value(resolution) / MILLISECONDS_IN_HOUR
    time_steps = get_time_steps(container)
    is_variant = is_time_variant(get_offer_curves(dir, component))
    # Static offer curves are time-invariant: compute breakpoints/slopes once.
    static_breakpoints, static_slopes = if is_variant
        (Float64[], Float64[])
    else
        _get_pwl_data(dir, container, component, first(time_steps))
    end
    for t in time_steps
        breakpoints, slopes = if is_variant
            _get_pwl_data(dir, container, component, t)
        else
            (static_breakpoints, static_slopes)
        end
        pwl_vars =
            add_pwl_variables_delta!(
                container,
                W,
                T,
                name,
                t,
                length(slopes);
                upper_bound = Inf,
            )
        add_pwl_constraint_delta!(
            container,
            component,
            U,
            V,
            breakpoints,
            pwl_vars,
            t,
            X,
        )
        pwl_cost =
            get_pwl_cost_expression_delta(pwl_vars, slopes, _objective_sign(dir) * dt)

        add_cost_to_expression!(
            container,
            ProductionCostExpression,
            pwl_cost,
            T,
            name,
            t,
        )

        if is_variant
            add_to_objective_variant_expression!(container, pwl_cost)
        else
            add_to_objective_invariant_expression!(container, pwl_cost)
        end
    end
end

# FIXME better validation: for static, != ZERO_OFFER_CURVE would be clearer 
# and for time series, actually check.
"""
Is this offer curve carrying meaningful data, as opposed to the default `ZERO_OFFER_CURVE`
placeholder that PSY assigns to unused sides of a `MarketBidCost` / `ImportExportCost`?
Only used for load formulations, to decide whether to throw an error about a non-trivial 
supply offer curve.
"""
function is_nontrivial_offer(curve::PSY.CostCurve{PSY.PiecewiseIncrementalCurve})
    xs = PSY.get_x_coords(PSY.get_function_data(PSY.get_value_curve(curve)))
    return last(xs) > first(xs)
end
is_nontrivial_offer(::PSY.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}) = false

function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::PSY.Component,
    cost_function::PSY.OfferCurveCost,
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    component_name = PSY.get_name(component)
    @debug "Market Bid" _group = LOG_GROUP_COST_FUNCTIONS component_name
    if is_nontrivial_offer(get_input_offer_curves(cost_function))
        throw(
            ArgumentError(
                "Component $(component_name) is not allowed to participate as a demand.",
            ),
        )
    end
    add_pwl_term_delta!(
        IncrementalOffer(),
        container,
        component,
        cost_function,
        T,
        U,
    )
    return
end

# Default: most formulations use incremental offers
_vom_offer_direction(::Type{<:AbstractDeviceFormulation}) = IncrementalOffer()

function _add_vom_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::PSY.Component,
    op_cost::PSY.OfferCurveCost,
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    dir = _vom_offer_direction(U)
    cost_curves = get_offer_curves(dir, op_cost)
    if is_time_variant(cost_curves)
        @warn "$(typeof(dir)) curves are time variant, there is no VOM cost source. Skipping VOM cost."
        return
    end
    _add_vom_cost_to_objective_helper!(
        container, T, component, op_cost, cost_curves, U)
    return
end

function _add_vom_cost_to_objective_helper!(
    container::OptimizationContainer,
    ::Type{T},
    component::PSY.Component,
    ::PSY.OfferCurveCost,
    cost_data::PSY.CostCurve{PSY.PiecewiseIncrementalCurve},
    ::Type{U},
) where {T <: VariableType, U <: AbstractDeviceFormulation}
    power_units = PSY.get_power_units(cost_data)
    cost_term = PSY.get_proportional_term(PSY.get_vom_cost(cost_data))
    add_proportional_cost_invariant!(container, T, component, cost_term, power_units)
    return
end

#################################################################################
# Section 12: TimeSeriesValueCurve Objective Formulation
# PSY-free delta PWL objective for CostCurve{TimeSeriesPiecewiseIncrementalCurve}.
# Reads slopes/breakpoints from pre-populated parameter containers.
#################################################################################

"""
    add_variable_cost_to_objective!(container, ::T, component, cost_function, ::U; dir)

Objective function dispatch for `CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}`.
Routes to the PSY-free delta PWL formulation that reads from parameter containers.
"""
function add_variable_cost_to_objective!(
    container::OptimizationContainer,
    ::Type{T},
    component::C,
    cost_function::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve},
    ::Type{U};
    dir::OfferDirection = IncrementalOffer(),
) where {
    T <: VariableType,
    C <: IS.InfrastructureSystemsComponent,
    U <: AbstractDeviceFormulation,
}
    power_units = IS.get_power_units(cost_function)
    device_base_power = get_base_power(component)
    _add_ts_incremental_pwl_cost!(dir, container, component, T, U,
        power_units, device_base_power)
    return
end

"""
PSY-free delta PWL objective formulation for time-series-backed incremental
value curves. Reads slopes/breakpoints from parameter containers populated
externally. All parameter array lookups and buffer allocations are hoisted
before the time loop to avoid repeated dictionary lookups and allocations.
"""
function _add_ts_incremental_pwl_cost!(
    dir::D,
    container::OptimizationContainer,
    component::C,
    ::Type{T},
    ::Type{U},
    power_units::IS.UnitSystem,
    device_base_power::Float64,
) where {
    D <: OfferDirection,
    C <: IS.InfrastructureSystemsComponent,
    T <: VariableType,
    U <: AbstractDeviceFormulation,
}
    W = _block_offer_var(dir)
    X = _block_offer_constraint(dir)
    name = get_name(component)
    dt = Dates.value(get_resolution(container)) / MILLISECONDS_IN_HOUR
    sign_dt = _objective_sign(dir) * dt
    model_base_power = get_model_base_power(container)

    # Hoist parameter array lookups out of the time loop (4 dict lookups total, not 4*T)
    SlopeParam = _slope_param(dir)
    BPParam = _breakpoint_param(dir)
    slope_arr = get_parameter_array(container, SlopeParam, C)
    slope_mult = get_parameter_multiplier_array(container, SlopeParam, C)
    bp_arr = get_parameter_array(container, BPParam, C)
    bp_mult = get_parameter_multiplier_array(container, BPParam, C)

    # Pre-allocate buffers sized from the parameter array axes
    seg_axis = axes(slope_arr)[2]
    point_axis = axes(bp_arr)[2]
    n_segments = length(seg_axis)
    n_points = length(point_axis)
    @assert_op n_segments == n_points - 1
    slopes = Vector{Float64}(undef, n_segments)
    breakpoints = Vector{Float64}(undef, n_points)

    for t in get_time_steps(container)
        _fill_pwl_data_from_arrays!(
            slopes, breakpoints, slope_arr, slope_mult, bp_arr, bp_mult,
            seg_axis, point_axis, name, t,
            power_units, model_base_power, device_base_power)
        pwl_vars = add_pwl_variables_delta!(
            container, W, C, name, t, n_segments; upper_bound = Inf)
        add_pwl_constraint_delta!(
            container,
            component,
            T,
            U,
            breakpoints,
            pwl_vars,
            t,
            X,
        )
        pwl_cost = get_pwl_cost_expression_delta(pwl_vars, slopes, sign_dt)
        add_cost_to_expression!(container, ProductionCostExpression, pwl_cost, C, name, t)
        add_to_objective_variant_expression!(container, pwl_cost)
    end
    return
end

"""
Fill pre-allocated slope and breakpoint buffers from parameter arrays for a single
time step. Reads raw values from parameter arrays, then applies unit conversion
via `get_piecewise_curve_per_system_unit`.
"""
function _fill_pwl_data_from_arrays!(
    slopes::Vector{Float64},
    breakpoints::Vector{Float64},
    slope_arr::DenseAxisArray{Float64},
    slope_mult::DenseAxisArray{Float64},
    bp_arr::DenseAxisArray{Float64},
    bp_mult::DenseAxisArray{Float64},
    seg_axis::UnitRange{Int64},
    point_axis::UnitRange{Int64},
    name::String,
    time::Int,
    power_units::IS.UnitSystem,
    model_base_power::Float64,
    device_base_power::Float64,
)
    # Read raw values from parameter arrays
    for (i, seg) in enumerate(seg_axis)
        slopes[i] = slope_arr[name, seg, time] * slope_mult[name, seg, time]
    end
    for (i, pt) in enumerate(point_axis)
        breakpoints[i] = bp_arr[name, pt, time] * bp_mult[name, pt, time]
    end
    # Convert to system per-unit
    converted_bp, converted_slopes = get_piecewise_curve_per_system_unit(
        breakpoints, slopes, power_units, model_base_power, device_base_power)
    copyto!(slopes, converted_slopes)
    copyto!(breakpoints, converted_bp)
    return
end
