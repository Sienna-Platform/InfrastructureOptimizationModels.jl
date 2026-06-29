#################################################################################
# Value Curve Objective Function: Delta PWL Formulation
#
# Objective function formulations for ValueCurve-based offer curves using the
# delta (incremental/block) PWL method. IOM owns the generic pieces:
#   * `OfferDirection` dispatch tables (parameter / variable / constraint types)
#   * `_consider_parameter` generics
#   * IS-only PWL predicates (`is_nontrivial_offer`, `curvity_check`)
#   * The PSY-free time-series delta PWL path
#     (`add_variable_cost_to_objective!` for `IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve}`
#      and `_add_ts_incremental_pwl_cost!`)
#   * Abstract extension points (stubs) for the PSY orchestration that lives in
#     downstream packages.
#
# PSY-specific orchestration (accessor wrappers for MBC / IEC, validation,
# parameter processing, the static `add_pwl_term_delta!` entry point) lives in
# POM's `common_models/market_bid_plumbing.jl`.
#################################################################################

#################################################################################
# Section 1: Extension points
# Declared here so IOM can call them generically; downstream packages (POM)
# add methods for PSY-specific cost types.
#################################################################################

"""
    get_offer_curves(direction, device_or_cost)

Return the output/input offer curve(s) for the given direction. POM provides
methods dispatching on `PSY.StaticInjection` and `PSY.OfferCurveCost`.
"""
function get_offer_curves end

"""
    get_initial_input(direction, device)

Return the `initial_input` scalar (cost at minimum) from the direction's side of
the offer curve. POM provides methods dispatching on `PSY.StaticInjection`.
"""
function get_initial_input end

"""
    validate_occ_component(::Type{<:ParameterType}, device)

Validate that `device` can be processed by the given offer-curve-cost parameter.
POM provides overloads per device type.
"""
function validate_occ_component end

"""
    validate_occ_breakpoints_slopes(device, direction)

Validate breakpoints/slopes on the given direction's offer curve.
"""
function validate_occ_breakpoints_slopes end

"""
    _get_parameter_field(::Type{<:ParameterType}, op_cost)

Extract the raw field corresponding to this parameter type from an operation
cost object. POM provides overloads.
"""
function _get_parameter_field end

"""
    _get_pwl_data(direction, container, component, time)

Return `(breakpoints, slopes)` for the given component at the given time step,
ready for unit conversion. POM's static-curve overload forwards to
`_get_raw_pwl_data`.
"""
function _get_pwl_data end

"""
    _get_raw_pwl_data(direction, container, ComponentType, name, cost_data, time)

Return `(breakpoint_cost, slope_cost, unit_system)`. The IS-backed TS method
lives here in IOM; the static `CostCurve{PiecewiseIncrementalCurve}` method is
provided by POM.
"""
function _get_raw_pwl_data end

"""
    add_pwl_term_delta!(direction, container, component, cost_function, ::Type{VariableType}, ::Type{Formulation})

Add the delta PWL objective term for a static cost function. POM provides the
method dispatching on `PSY.OfferCurveCost`.
"""
function add_pwl_term_delta! end

"""
    _add_vom_cost_to_objective!(container, ::Type{VariableType}, component, op_cost, ::Type{Formulation})

Add the variable operations & maintenance (VOM) cost term. POM provides the
method dispatching on `PSY.OfferCurveCost`.
"""
function _add_vom_cost_to_objective! end

"Default: most formulations use incremental offers. POM overrides for loads."
function _vom_offer_direction end

"Predicate: is this op_cost time-series-backed? POM extends for TS types."
_is_time_series_cost(::Any) = false

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
# Section 3: _consider_parameter (generic versions)
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
# Section 4: Curvity + IS-only predicates
#################################################################################

curvity_check(::IncrementalOffer, x) = IS.is_convex(x)
curvity_check(::DecrementalOffer, x) = IS.is_concave(x)
expected_curvity(::IncrementalOffer) = "convex"
expected_curvity(::DecrementalOffer) = "concave"

"""
Is this offer curve carrying meaningful data, as opposed to the default
`ZERO_OFFER_CURVE` placeholder that PSY assigns to unused sides of a
`MarketBidCost` / `ImportExportCost`? Only used for load formulations.
"""
function is_nontrivial_offer(curve::IS.CostCurve{IS.PiecewiseIncrementalCurve})
    lo, hi = IS.get_domain(IS.get_function_data(IS.get_value_curve(curve)))
    return hi > lo
end
# A TS-backed offer side is the absent/placeholder side of a one-sided participant
# (a load with no supply offer, or a generator with no demand offer) when it carries a
# reserved empty time-series key name. Any non-empty key references a real forecast, so
# it is a genuine offer. This mirrors the static `ZERO_OFFER_CURVE` placeholder check
# above (which inspects the curve's x-range) for the time-series-backed case, where the
# curve data lives in a forecast and cannot be inspected at build time.
function is_nontrivial_offer(curve::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve})
    return !isempty(IS.get_name(IS.get_time_series_key(curve)))
end

#################################################################################
# Section 5: TimeSeriesValueCurve Objective Formulation (PSY-free)
# Delta PWL objective for CostCurve{TimeSeriesPiecewiseIncrementalCurve}.
# Reads slopes/breakpoints from pre-populated parameter containers.
#################################################################################

# TS-backed PWL data retrieval. Parameter containers for Slope/Breakpoint are
# allocated with axes `(names, segments|points, times)`, so the 3-index lookup
# mirrors `_fill_pwl_data_from_arrays!`. The parameter values carry the units
# declared on the `CostCurve`, so we forward those through (don't hardcode).
function _get_raw_pwl_data(
    dir::OfferDirection,
    container::OptimizationContainer,
    ::Type{T},
    name::String,
    cost_data::IS.CostCurve{IS.TimeSeriesPiecewiseIncrementalCurve},
    time::Int;
    meta = CONTAINER_KEY_EMPTY_META,
) where {T <: IS.InfrastructureSystemsComponent}
    SlopeParam = _slope_param(dir)
    slope_arr = get_parameter_array(container, SlopeParam, T, meta)
    slope_mult = get_parameter_multiplier_array(container, SlopeParam, T, meta)
    @assert size(slope_arr) == size(slope_mult)
    seg_axis = axes(slope_arr)[2]
    slope_cost_component = Vector{Float64}(undef, length(seg_axis))
    for (i, seg) in enumerate(seg_axis)
        slope_cost_component[i] = slope_arr[name, seg, time] * slope_mult[name, seg, time]
    end

    BreakpointParam = _breakpoint_param(dir)
    bp_arr = get_parameter_array(container, BreakpointParam, T, meta)
    bp_mult = get_parameter_multiplier_array(container, BreakpointParam, T, meta)
    @assert size(bp_arr) == size(bp_mult)
    point_axis = axes(bp_arr)[2]
    breakpoint_cost_component = Vector{Float64}(undef, length(point_axis))
    for (i, pt) in enumerate(point_axis)
        breakpoint_cost_component[i] = bp_arr[name, pt, time] * bp_mult[name, pt, time]
    end

    @assert_op length(slope_cost_component) == length(breakpoint_cost_component) - 1
    return breakpoint_cost_component, slope_cost_component, IS.get_power_units(cost_data)
end

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
    power_units::IS.AbstractUnitSystem,
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
    power_units::IS.AbstractUnitSystem,
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

#################################################################################
# Section 6: Tranche-axis helpers for time-varying PWL parameters (PSY-free)
#
# Time-varying piecewise-linear cost curves are stored in 3-D parameter arrays
# `(component, tranche, time)`. These helpers size the tranche axis to the global
# maximum across all time steps and unwrap each time-series element (an
# `IS.PiecewiseStepData`) into the per-tranche slope/breakpoint vector that fills
# one column of the array. The PSY-touching orchestration (resolving a service's
# time series into raw data, `calc_additional_axes`) lives in the downstream
# package (POM) and calls down into these data-level helpers.
#################################################################################

# It's nice for debugging to have meaningful labels on the tranche axis. These
# labels are never relied upon numerically.
make_tranche_axis(n_tranches) = "tranche_" .* string.(1:n_tranches)

"""
Given a parameter array, return any additional axes, i.e. those that aren't the
first (component) or the last (time).
"""
lookup_additional_axes(parameter_array) = axes(parameter_array)[2:(end - 1)]

# Maximum number of tranches (segments) across a piecewise time series.
get_max_tranches(data::Vector{IS.PiecewiseStepData}) = maximum(length, data)
get_max_tranches(data::TS.TimeArray) = get_max_tranches(TS.values(data))
get_max_tranches(data::AbstractDict) = maximum(get_max_tranches.(values(data)))

# Layer of indirection so that parameters whose time series represents multiple
# things (e.g. both slopes and breakpoints come from the same `PiecewiseStepData`
# series) can unwrap each element into the per-tranche vector that fills the
# parameter array. The default is the identity used by scalar time-series
# parameters.
unwrap_for_param(::ParameterType, ts_elem, expected_axs) = ts_elem

# Scalar-linear cost time series (shut-down / no-load and other scalar cost fields
# of a time-series offer cost, incremental or decremental): the resolved per-hour
# element is an `IS.LinearFunctionData` whose proportional term carries the scalar
# cost. This mirrors the static path's `_shutdown_cost_value(::IS.LinearCurve)`
# extraction. Return the scalar so the parameter container is filled with a
# `Float64` (its `size` is `()`, matching the empty additional-axes of a scalar
# parameter). Without this method the element falls through the identity above and
# `_size_wrapper` then calls `size(::LinearFunctionData)`, which has no method.
unwrap_for_param(::ParameterType, ts_elem::IS.LinearFunctionData, expected_axs) =
    IS.get_proportional_term(ts_elem)

# For piecewise data the number of tranches can vary over time, so the parameter
# container is sized for the maximum number of tranches and shorter curves are
# padded. We pad with "degenerate" tranches at the top end of the curve with
# dx = 0 so their dispatch variables are constrained to 0. The slope of those
# segments shouldn't matter; we use slope = 0 so the term drops trivially from
# the objective.
function unwrap_for_param(
    ::AbstractPiecewiseLinearSlopeParameter,
    ts_elem::IS.PiecewiseStepData,
    expected_axs::Tuple{AbstractVector},
)
    max_len = length(only(expected_axs))
    y_coords = IS.get_y_coords(ts_elem)
    length(y_coords) <= max_len || error(
        "PiecewiseStepData y-coords ($(length(y_coords))) exceed expected axis length " *
        "($max_len) for slope parameter",
    )
    fill_value = 0.0  # pad with slope = 0 if necessary (see above)
    out = Vector{Float64}(undef, max_len)
    copyto!(out, y_coords)
    fill!(view(out, (length(y_coords) + 1):max_len), fill_value)
    return out
end

function unwrap_for_param(
    ::AbstractPiecewiseLinearBreakpointParameter,
    ts_elem::IS.PiecewiseStepData,
    expected_axs::Tuple{AbstractVector},
)
    max_len = length(only(expected_axs))
    x_coords = IS.get_x_coords(ts_elem)
    length(x_coords) <= max_len || error(
        "PiecewiseStepData x-coords ($(length(x_coords))) exceed expected axis length " *
        "($max_len) for breakpoint parameter",
    )
    fill_value = x_coords[end]  # repeat the last breakpoint so dx = 0 (see above)
    out = Vector{Float64}(undef, max_len)
    copyto!(out, x_coords)
    fill!(view(out, (length(x_coords) + 1):max_len), fill_value)
    return out
end
