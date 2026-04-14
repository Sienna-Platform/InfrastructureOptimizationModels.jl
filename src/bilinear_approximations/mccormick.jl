# McCormick envelope for bilinear products z = x·y.
# Adds 4 linear inequalities that bound z given variable bounds on x and y.

"Standard McCormick envelope constraints bounding the bilinear product z = x·y."
struct McCormickConstraint <: ConstraintType end

"Reformulated McCormick constraints on Bin2 separable variables."
struct ReformulatedMcCormickConstraint <: ConstraintType end

"""
    _mc_setindex!(cons, index, n, constraint)

Helper function for setting constraints by-index in a McCormick constraint container.

Supports 2- and 3-length tuples.
"""
@inline function _mc_setindex!(cons, index::Tuple{A, B}, n::Int, constraint) where {A, B}
    cons[index[1], n, index[2]] = constraint
end

@inline function _mc_setindex!(
    cons,
    index::Tuple{A, B, C},
    n::Int,
    constraint,
) where {A, B, C}
    cons[index[1], index[2], n, index[3]] = constraint
end

function _add_mccormick_envelope!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    z::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64;
    lower_bounds::Bool = true,
)
    if lower_bounds
        _mc_setindex!(
            cons,
            index,
            1,
            JuMP.@constraint(
                jump_model,
                z >= x_min * y + x * y_min - x_min * y_min,
            )
        )
        _mc_setindex!(
            cons,
            index,
            2,
            JuMP.@constraint(
                jump_model,
                z >= x_max * y + x * y_max - x_max * y_max,
            )
        )
    end
    _mc_setindex!(
        cons,
        index,
        3,
        JuMP.@constraint(
            jump_model,
            z <= x_max * y + x * y_min - x_max * y_min,
        )
    )
    _mc_setindex!(
        cons,
        index,
        4,
        JuMP.@constraint(
            jump_model,
            z <= x_min * y + x * y_max - x_min * y_max,
        )
    )
end

"""
    _add_mccormick_envelope!(container, C, names, time_steps, x_var, y_var, z_var, x_min, x_max, y_min, y_max, meta)

Add McCormick envelope constraints for the bilinear product z ≈ x·y.

For each (name, t), adds 4 linear inequalities:
```
z ≥ x_min·y + x·y_min − x_min·y_min
z ≥ x_max·y + x·y_max − x_max·y_max
z ≤ x_max·y + x·y_min − x_max·y_min
z ≤ x_min·y + x·y_max − x_min·y_max
```

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `z_var`: container of z variables indexed by (name, t)
- `x_min::Float64`: lower bound of x
- `x_max::Float64`: upper bound of x
- `y_min::Float64`: lower bound of y
- `y_max::Float64`: upper bound of y
- `meta::String`: identifier for container keys

# Returns
- Nothing. Constraints are added in-place.
"""
function _add_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    z_var,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
    meta::String;
    lower_bounds::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    IS.@assert_op x_max > x_min
    IS.@assert_op y_max > y_min
    jump_model = get_jump_model(container)

    mc_cons = add_constraints_container!(
        container,
        McCormickConstraint,
        C,
        names,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )

    for name in names, t in time_steps
        _add_mccormick_envelope!(
            jump_model, mc_cons, (name, t),
            x_var[name, t], y_var[name, t], z_var[name, t],
            x_min, x_max, y_min, y_max;
            lower_bounds,
        )
    end

    return
end

function _add_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    z_var,
    x_min::Float64,
    x_max::Float64,
    meta::String;
    lower_bounds::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    _add_mccormick_envelope!(
        container, C, names, time_steps,
        x_var, x_var, z_var,
        x_min, x_max, x_min, x_max,
        meta; lower_bounds,
    )
    return
end

"""
    _add_mccormick_envelope!(container, C, names, time_steps, x_var, y_var, z_var, x_bounds, y_bounds, meta)

Add McCormick envelope constraints for the bilinear product z ≈ x·y with per-name bounds.

For each (name, t), adds 4 linear inequalities using bounds looked up by name index.

# Arguments
- `container::OptimizationContainer`: the optimization container
- `::Type{C}`: component type
- `names::Vector{String}`: component names
- `time_steps::UnitRange{Int}`: time periods
- `x_var`: container of x variables indexed by (name, t)
- `y_var`: container of y variables indexed by (name, t)
- `z_var`: container of z variables indexed by (name, t)
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
- `meta::String`: identifier for container keys

# Returns
- Nothing. Constraints are added in-place.
"""
function _add_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    z_var,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String;
    lower_bounds::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    mc_cons = add_constraints_container!(
        container,
        McCormickConstraint(),
        C,
        names,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        xb = x_bounds[i]
        yb = y_bounds[i]
        IS.@assert_op xb.max > xb.min
        IS.@assert_op yb.max > yb.min
        _add_mccormick_envelope!(
            jump_model, mc_cons, (name, t),
            x_var[name, t], y_var[name, t], z_var[name, t],
            xb.min, xb.max, yb.min, yb.max;
            lower_bounds,
        )
    end

    return
end

function _add_mccormick_envelope!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    z_var,
    bounds::Vector{MinMax},
    meta::String;
    lower_bounds::Bool = true,
) where {C <: IS.InfrastructureSystemsComponent}
    _add_mccormick_envelope!(
        container, C, names, time_steps,
        x_var, x_var, z_var,
        bounds, bounds,
        meta; lower_bounds,
    )
    return
end

function _add_mccormick_envelope!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.VariableRef,
    z::JuMP.VariableRef,
    x_min::Float64,
    x_max::Float64;
    lower_bounds::Bool = true,
)
    _add_mccormick_envelope!(
        jump_model, cons, index,
        x, x, z,
        x_min, x_max, x_min, x_max;
        lower_bounds,
    )
end

# Lower McCormick bounds on (z_p1 − z_x − z_y) for the Bin2 reformulation.
function _add_reformulated_lower_mccormick!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    zp1::JuMP.AbstractJuMPScalar,
    zx::JuMP.AbstractJuMPScalar,
    zy::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    _mc_setindex!(
        cons,
        index,
        1,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy >= 2.0 * (x_min * y + x * y_min - x_min * y_min),
        )
    )
    _mc_setindex!(
        cons,
        index,
        2,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy >= 2.0 * (x_max * y + x * y_max - x_max * y_max),
        )
    )
end

function _add_reformulated_mccormick_bin2!(
    jump_model::JuMP.Model,
    cons,
    index,
    x::JuMP.AbstractJuMPScalar,
    y::JuMP.AbstractJuMPScalar,
    zp1::JuMP.AbstractJuMPScalar,
    zx::JuMP.AbstractJuMPScalar,
    zy::JuMP.AbstractJuMPScalar,
    x_min::Float64,
    x_max::Float64,
    y_min::Float64,
    y_max::Float64,
)
    _add_reformulated_lower_mccormick!(
        jump_model, cons, index, x, y, zp1, zx, zy, x_min, x_max, y_min, y_max,
    )
    # Upper bounds also on (z_p1 − z_x − z_y) since Bin2 has no z_p2
    _mc_setindex!(
        cons,
        index,
        3,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy <= 2.0 * (x_max * y + x * y_min - x_max * y_min),
        )
    )
    _mc_setindex!(
        cons,
        index,
        4,
        JuMP.@constraint(
            jump_model,
            zp1 - zx - zy <= 2.0 * (x_min * y + x * y_max - x_min * y_max),
        )
    )
end

"""
    _add_reformulated_mccormick!(container, C, names, time_steps, x_var, y_var, psq, xsq, ysq, x_bounds, y_bounds, meta)

Add 4 reformulated McCormick cuts for Bin2 separable bilinear approximation.
Substitutes z = ½(z_p1 − z_x − z_y) into the standard McCormick envelope.

`psq`, `xsq`, `ysq` are expression containers for (x+y)², x², y² approximations.

# Arguments
- `x_bounds::Vector{MinMax}`: per-name lower and upper bounds of x
- `y_bounds::Vector{MinMax}`: per-name lower and upper bounds of y
"""
function _add_reformulated_mccormick!(
    container::OptimizationContainer,
    ::Type{C},
    names::Vector{String},
    time_steps::UnitRange{Int},
    x_var,
    y_var,
    psq,
    xsq,
    ysq,
    x_bounds::Vector{MinMax},
    y_bounds::Vector{MinMax},
    meta::String,
) where {C <: IS.InfrastructureSystemsComponent}
    jump_model = get_jump_model(container)

    mc_cons = add_constraints_container!(
        container,
        ReformulatedMcCormickConstraint,
        C,
        names,
        1:4,
        time_steps;
        sparse = true,
        meta,
    )

    for (i, name) in enumerate(names), t in time_steps
        xb = x_bounds[i]
        yb = y_bounds[i]
        IS.@assert_op xb.max > xb.min
        IS.@assert_op yb.max > yb.min
        _add_reformulated_mccormick_bin2!(
            jump_model, mc_cons, (name, t),
            x_var[name, t], y_var[name, t],
            psq[name, t], xsq[name, t], ysq[name, t],
            xb.min, xb.max, yb.min, yb.max,
        )
    end

    return
end
