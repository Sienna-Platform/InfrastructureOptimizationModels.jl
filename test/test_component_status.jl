"""
Unit tests for per-step component statuses (`get_component_statuses`,
`get_component_status`), the component-entry iterator (`each_component_entry`) and
`fix_inactive_variables!`.
"""

struct InactiveFixedVariable <: IOM.VariableType end
struct InactiveFreeVariable <: IOM.VariableType end
# A sparse variable keyed (service, component, outage, t): the component is not the
# second-to-last index, so the type declares its axis.
struct OutageDeploymentVariable <: IOM.VariableType end

IOM.fixed_when_inactive(::Type{InactiveFixedVariable}) = true
IOM.fixed_when_inactive(::Type{OutageDeploymentVariable}) = true
IOM.component_axis(::Type{OutageDeploymentVariable}, ::SparseAxisArray) = 2

function _status_container(time_steps)
    container = make_test_container(time_steps)
    return container, IOM.get_jump_model(container)
end

function _dense_variable!(container, ::Type{T}, ::Type{C}, names, time_steps) where {T, C}
    var = IOM.add_variable_container!(container, T, C, names, time_steps)
    jump_model = IOM.get_jump_model(container)
    for name in names, t in time_steps
        var[name, t] = JuMP.@variable(jump_model, lower_bound = 0.0, upper_bound = 5.0)
    end
    return var
end

@testset "Component statuses" begin
    time_steps = 1:3

    @testset "default: every component is active" begin
        container, _ = _status_container(time_steps)
        @test IOM.get_component_statuses(container, IS.TestComponent) isa
              IOM.AllComponentsActive
        @test IOM.get_component_statuses(container, MockThermalGen) isa
              IOM.AllComponentsActive
        @test IOM.get_component_status(container, IS.TestComponent, "x", 1) === 1.0
        @test IOM.get_component_status(IOM.AllComponentsActive(), "x", 2) === 1.0
    end

    @testset "ComponentStatusArray" begin
        values = DenseAxisArray([1.0 0.0 0.5], ["A"], time_steps)
        statuses = IOM.ComponentStatusArray(values)
        @test IOM.get_component_status(statuses, "A", 1) === 1.0
        @test IOM.get_component_status(statuses, "A", 2) === 0.0
        @test IOM.get_component_status(statuses, "A", 3) === 0.5
        @test IOM.get_component_status(statuses, "not-scheduled", 2) === 1.0
        container, _ = _status_container(time_steps)
        set_test_statuses!(container, MockThermalGen, ["A"], [1.0 0.0 0.5])
        @test IOM.get_component_status(container, MockThermalGen, "A", 2) === 0.0
        @test IOM.get_component_status(container, MockRenewableGen, "A", 2) === 1.0
    end

    @testset "component_axis_type" begin
        @test IOM.component_axis_type(MockThermalGen) === MockThermalGen
        @test IOM.component_axis_type(
            IOM.ComponentPairKey{MockThermalGen, MockReserve{MockUp}},
        ) === MockThermalGen
    end

    @testset "public names are documented" begin
        for name in (
            :get_component_statuses, :get_component_status, :AbstractComponentStatuses,
            :AllComponentsActive, :ComponentStatusArray, :is_always_committed,
            :get_on_status, :component_axis, :component_axis_type,
            :each_component_entry, :fixed_when_inactive, :fix_inactive_variables!,
            :get_component_status_array, :scaled_by_component_status,
        )
            @test Base.Docs.hasdoc(IOM, name)
            @test Base.isexported(IOM, name)
        end
    end
end

@testset "each_component_entry" begin
    time_steps = 1:2

    @testset "dense (name, t)" begin
        container, _ = _status_container(time_steps)
        var = _dense_variable!(
            container, InactiveFreeVariable, MockThermalGen, ["A", "B"], time_steps)
        key = IOM.VariableKey(InactiveFreeVariable, MockThermalGen)
        entries = collect(IOM.each_component_entry(key, var))
        @test length(entries) == 4
        @test Set((n, t) for (n, t, _) in entries) ==
              Set([("A", 1), ("A", 2), ("B", 1), ("B", 2)])
        for (name, t, ref) in entries
            @test ref === var[name, t]
        end
    end

    @testset "dense (name, block, t) skips unassigned entries" begin
        container, jump_model = _status_container(time_steps)
        var = IOM.add_variable_container!(
            container, InactiveFreeVariable, MockThermalGen, ["A"], 1:2, time_steps)
        var["A", 1, 1] = JuMP.@variable(jump_model)
        var["A", 2, 2] = JuMP.@variable(jump_model)
        key = IOM.VariableKey(InactiveFreeVariable, MockThermalGen)
        entries = collect(IOM.each_component_entry(key, var))
        @test Set((n, t) for (n, t, _) in entries) == Set([("A", 1), ("A", 2)])
        @test length(entries) == 2
    end

    @testset "sparse (service, name, t) on a component pair key" begin
        container, jump_model = _status_container(time_steps)
        C = IOM.ComponentPairKey{MockThermalGen, MockReserve{MockUp}}
        var = IOM.add_variable_container!(
            container, InactiveFixedVariable, C, String[], String[], Int[];
            sparse = true)
        for t in time_steps
            var["reserve", "A", t] = JuMP.@variable(jump_model)
        end
        key = IOM.VariableKey(InactiveFixedVariable, C)
        entries = collect(IOM.each_component_entry(key, var))
        @test Set((n, t) for (n, t, _) in entries) == Set([("A", 1), ("A", 2)])
    end

    @testset "sparse with a declared component axis" begin
        container, jump_model = _status_container(time_steps)
        C = IOM.ComponentPairKey{MockThermalGen, MockReserve{MockUp}}
        var = IOM.add_variable_container!(
            container, OutageDeploymentVariable, C, String[], String[], String[], Int[];
            sparse = true)
        var["reserve", "A", "outage", 1] = JuMP.@variable(jump_model)
        key = IOM.VariableKey(OutageDeploymentVariable, C)
        entries = collect(IOM.each_component_entry(key, var))
        @test only(entries)[1:2] == ("A", 1)
    end

    @testset "sparse constraint entries holding nothing are skipped" begin
        container, jump_model = _status_container(time_steps)
        x = JuMP.@variable(jump_model)
        contents = Dict{Tuple{String, Int}, Union{Nothing, JuMP.ConstraintRef}}(
            ("A", 1) => JuMP.@constraint(jump_model, x >= 0),
            ("A", 2) => nothing,
        )
        rows = SparseAxisArray(contents)
        key = IOM.ConstraintKey(TestConstraintType, MockThermalGen)
        entries = collect(IOM.each_component_entry(key, rows))
        @test only(entries)[1:2] == ("A", 1)
    end
end

@testset "fix_inactive_variables!" begin
    time_steps = 1:3
    statuses = [1.0 0.0 1.0; 0.0 0.5 1.0]

    container, jump_model = _status_container(time_steps)
    fixed = _dense_variable!(
        container, InactiveFixedVariable, MockThermalGen, ["A", "B"], time_steps)
    free = _dense_variable!(
        container, InactiveFreeVariable, MockThermalGen, ["A", "B"], time_steps)
    unscheduled = _dense_variable!(
        container, InactiveFixedVariable, MockRenewableGen, ["A"], time_steps)
    C = IOM.ComponentPairKey{MockThermalGen, MockReserve{MockUp}}
    service = IOM.add_variable_container!(
        container, InactiveFixedVariable, C, String[], String[], Int[]; sparse = true)
    for t in time_steps
        service["reserve", "B", t] = JuMP.@variable(jump_model, lower_bound = 0.0)
    end
    set_test_statuses!(container, MockThermalGen, ["A", "B"], statuses)

    IOM.fix_inactive_variables!(container)

    for (i, name) in enumerate(["A", "B"]), t in time_steps
        inactive = iszero(statuses[i, t])
        @test JuMP.is_fixed(fixed[name, t]) == inactive
        inactive && @test JuMP.fix_value(fixed[name, t]) == 0.0
        @test !JuMP.is_fixed(free[name, t])
    end
    for t in time_steps
        @test !JuMP.is_fixed(unscheduled["A", t])
        @test JuMP.is_fixed(service["reserve", "B", t]) == iszero(statuses[2, t])
    end
    @test !IOM.fixed_when_inactive(InactiveFreeVariable)
end

struct StatusTestParameter <: IOM.ParameterType end
struct ScaledTestParameter <: IOM.ParameterType end
struct UnscaledTestParameter <: IOM.ParameterType end
IOM.scaled_by_component_status(::Type{ScaledTestParameter}) = true

function _float_parameter!(
    container,
    ::Type{P},
    ::Type{C},
    names,
    time_steps,
    value,
) where {P, C}
    param_container = IOM.add_param_container_shared_axes!(
        container, IOM.ParameterKey(P, C), IOM.NoAttributes(), Float64, names,
        time_steps)
    fill!(IOM.get_parameter_array(param_container).data, value)
    fill!(IOM.get_multiplier_array(param_container).data, 2.0)
    return param_container
end

function _sum_statuses(statuses, names, time_steps)
    total = 0.0
    for name in names, t in time_steps
        total += IOM.get_component_status(statuses, name, t)
    end
    return total
end

@testset "Status arrays are built once per container" begin
    time_steps = 1:48
    names = ["g$i" for i in 1:5000]
    container = make_test_container(time_steps)
    param_container =
        _float_parameter!(
            container,
            StatusTestParameter,
            MockThermalGen,
            names,
            time_steps,
            1.0,
        )
    IOM.get_parameter_array(param_container)["g17", 3] = 0.0
    key = IOM.ParameterKey(StatusTestParameter, MockThermalGen)
    statuses = IOM.get_component_status_array(container, key)
    @test IOM.get_component_status_array(container, key) === statuses
    @test IOM.get_component_status(statuses, "g17", 3) === 0.0
    @test IOM.get_component_status(statuses, "unscheduled", 3) === 1.0
    # Values are read in place, so a later write is seen without rebuilding the lookup.
    IOM.get_parameter_array(param_container)["g18", 4] = 0.0
    @test IOM.get_component_status(statuses, "g18", 4) === 0.0
    # A cached fetch and every one of the 240 000 lookups run without allocating, so no
    # lookup rebuilds the name set.
    IOM.get_component_status_array(container, key)
    _sum_statuses(statuses, names, time_steps)
    @test (@allocated IOM.get_component_status_array(container, key)) == 0
    @test (@allocated _sum_statuses(statuses, names, time_steps)) == 0
    @test _sum_statuses(statuses, names, time_steps) == length(names) * 48 - 2
    IOM.reset_optimization_model!(container)
    @test IOM.get_component_status_array(container, key) !== statuses
end

@testset "Parameter values follow the status of the parameters it scales" begin
    time_steps = 1:3
    names = ["A", "B"]
    statuses = [1.0 0.0 1.0; 1.0 1.0 0.0]
    container = make_test_container(time_steps)
    set_test_statuses!(container, MockThermalGen, names, statuses)
    _float_parameter!(
        container,
        ScaledTestParameter,
        MockThermalGen,
        names,
        time_steps,
        3.0,
    )
    _float_parameter!(
        container,
        UnscaledTestParameter,
        MockThermalGen,
        names,
        time_steps,
        3.0,
    )
    scaled = IOM.ParameterKey(ScaledTestParameter, MockThermalGen)
    unscaled = IOM.ParameterKey(UnscaledTestParameter, MockThermalGen)
    @test !IOM.scaled_by_component_status(UnscaledTestParameter)
    read = IOM.read_parameters(container)
    for (i, name) in enumerate(names), t in time_steps
        @test IOM.lookup_value(container, scaled)[name, t] == 6.0 * statuses[i, t]
        @test IOM.calculate_parameter_values(container, scaled)[name, t] ==
              6.0 * statuses[i, t]
        @test read[scaled][name, t] == 6.0 * statuses[i, t]
        @test IOM.lookup_value(container, unscaled)[name, t] == 6.0
        @test read[unscaled][name, t] == 6.0
    end
    # The parameter itself is untouched: only the reported value follows the status.
    @test all(==(3.0), IOM.get_parameter_array(container, scaled).data)
end

@testset "Status-scaled values of a sparse parameter container" begin
    time_steps = 1:3
    names = ["A", "B"]
    statuses = [1.0 0.0 1.0; 1.0 1.0 0.0]
    container = make_test_container(time_steps)
    set_test_statuses!(container, MockThermalGen, names, statuses)
    key = IOM.ParameterKey(ScaledTestParameter, MockThermalGen)
    param_container = IOM.add_param_container_shared_axes!(
        container, key, IOM.NoAttributes(), Float64, names, time_steps; sparse = true)
    params = IOM.get_parameter_array(param_container)
    multipliers = IOM.get_multiplier_array(param_container)
    for name in names, t in time_steps
        params[(name, t)] = 3.0
        multipliers[(name, t)] = 2.0
    end
    expected = JuMP.Containers.SparseAxisArray(
        Dict(
            (name, t) => 6.0 * statuses[i, t] for (i, name) in enumerate(names) for
            t in time_steps
        ),
    )
    @test IOM.lookup_value(container, key) == IOM.to_matrix(expected)
    @test IOM.calculate_parameter_values(container, key) == IOM.to_matrix(expected)
    # The container keeps its own values and multipliers.
    @test all(==(2.0), values(multipliers.data))
    @test all(==(3.0), values(params.data))
end
