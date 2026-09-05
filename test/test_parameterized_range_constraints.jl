"""
Unit tests for the parameterized range constraints
(`add_parameterized_lower_bound_range_constraints` /
`add_parameterized_upper_bound_range_constraints`).

The real branching is the parameter family: generic, `EventParameter`, and
`TimeSeriesParameter` dispatch to different `_bound_range_with_parameter!` methods, and the
time series one additionally filters devices, so those carry the bulk of the coverage.
"""

struct TestValueParameter <: IOM.VariableValueParameter end
struct TestTimeSeriesParameter <: IOM.TimeSeriesParameter end
struct TestEventParameter <: IOM.EventParameter end

const PARAM_RANGE_TS_NAME = "max_active_power"
const PARAM_RANGE_MULTIPLIER = 0.5

param_range_value(name, t) = name == "A" ? 1.0 * t : 10.0 + t

function _make_parameterized_range_container(devices, time_steps)
    mock_sys = MockSystem(100.0)
    settings = IOM.Settings(
        mock_sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(mock_sys, settings, nothing, MockDeterministic)
    IOM.set_time_steps!(container, time_steps)
    jump_model = IOM.get_jump_model(container)
    names = [get_name(d) for d in devices]
    D = eltype(devices)
    var = IOM.add_variable_container!(container, TestVariableType, D, names, time_steps)
    expr =
        IOM.add_expression_container!(container, TestExpressionType, D, names, time_steps)
    for name in names, t in time_steps
        v = JuMP.@variable(jump_model)
        var[name, t] = v
        # Scaled so a test can tell the expression path from the variable path.
        expr[name, t] = 2.0 * v
    end
    return container
end

"Add a `TestValueParameter` container with `param[name, t] = param_range_value(name, t)`."
function _add_value_parameter!(container, names, time_steps)
    param_container = IOM.add_param_container!(
        container,
        TestValueParameter,
        MockThermalGen,
        IOM.VariableKey(TestVariableType, MockThermalGen),
        names,
        time_steps,
    )
    param_array = IOM.get_parameter_array(param_container)
    mult_array = IOM.get_multiplier_array(param_container)
    for name in names, t in time_steps
        param_array[name, t] = param_range_value(name, t)
        mult_array[name, t] = PARAM_RANGE_MULTIPLIER
    end
    return param_container
end

@testset "Parameterized range constraints" begin
    time_steps = 1:3
    devices = [make_mock_thermal("A"), make_mock_thermal("B")]
    names = ["A", "B"]
    model = IOM.DeviceModel(
        MockThermalGen,
        TestFormulation;
        time_series_names = Dict{Type{<:IOM.ParameterType}, String}(
            TestTimeSeriesParameter => PARAM_RANGE_TS_NAME,
        ),
    )

    @testset "Bounds from a variable value parameter" begin
        container = _make_parameterized_range_container(devices, time_steps)
        _add_value_parameter!(container, names, time_steps)
        for f in (
            add_parameterized_lower_bound_range_constraints,
            add_parameterized_upper_bound_range_constraints,
        )
            f(
                container,
                TestConstraintType,
                TestVariableType,
                TestValueParameter,
                devices,
                model,
                TestPowerModel,
            )
        end
        con_lb = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        con_ub = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "ub")
        var = IOM.get_variable(container, TestVariableType, MockThermalGen)
        # The two directions land in distinct containers under the "lb"/"ub" metas. Sense is
        # compile-time dispatch, invariant across names and time steps, so spot-check it
        # once; the RHS arithmetic is what varies, so check that elementwise.
        @test JuMP.constraint_object(con_lb["A", 1]).set isa MOI.GreaterThan
        @test JuMP.constraint_object(con_ub["A", 1]).set isa MOI.LessThan
        for name in names, t in time_steps
            @test JuMP.normalized_coefficient(con_lb[name, t], var[name, t]) ≈ 1.0
            @test JuMP.normalized_rhs(con_lb[name, t]) ≈
                  PARAM_RANGE_MULTIPLIER * param_range_value(name, t)
        end
    end

    @testset "Exercise expression codepath" begin
        container = _make_parameterized_range_container(devices, time_steps)
        _add_value_parameter!(container, names, time_steps)
        add_parameterized_lower_bound_range_constraints(
            container,
            TestConstraintType,
            TestExpressionType,
            TestValueParameter,
            devices,
            model,
            TestPowerModel,
        )
        con = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        var = IOM.get_variable(container, TestVariableType, MockThermalGen)
        # The expression is 2 * var, so the LHS coefficient must be 2, not 1.
        for name in names, t in time_steps
            @test JuMP.normalized_coefficient(con[name, t], var[name, t]) ≈ 2.0
        end
    end

    @testset "Time series parameter constrains only devices owning the time series" begin
        mock_clear_time_series!()
        ts_devices =
            [make_mock_thermal("A"), make_mock_thermal("B"), make_mock_thermal("C")]
        ts_names = ["A", "C"]  # B has no time series
        for device in ts_devices
            get_name(device) in ts_names &&
                mock_add_time_series!(device, PARAM_RANGE_TS_NAME)
        end
        uuids = Dict(name => "uuid-$name" for name in ts_names)
        container = _make_parameterized_range_container(ts_devices, time_steps)
        param_container = IOM.add_param_container!(
            container,
            TestTimeSeriesParameter,
            MockThermalGen,
            MockDeterministic,
            PARAM_RANGE_TS_NAME,
            collect(values(uuids)),
            ts_names,
            (),
            time_steps,
        )
        attributes = IOM.get_attributes(param_container)
        param_array = IOM.get_parameter_array(param_container)
        mult_array = IOM.get_multiplier_array(param_container)
        for name in ts_names
            IOM.add_component_name!(attributes, name, uuids[name])
            for t in time_steps
                param_array[uuids[name], t] = param_range_value(name, t)
                mult_array[name, t] = PARAM_RANGE_MULTIPLIER
            end
        end

        add_parameterized_lower_bound_range_constraints(
            container,
            TestConstraintType,
            TestVariableType,
            TestTimeSeriesParameter,
            ts_devices,
            model,
            TestPowerModel,
        )
        con = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        @test axes(con)[1] == ts_names
        # RHS comes from the parameter's column refs rather than the parameter array.
        for name in ts_names, t in time_steps
            @test JuMP.normalized_rhs(con[name, t]) ≈
                  PARAM_RANGE_MULTIPLIER * param_range_value(name, t)
        end
        mock_clear_time_series!()
    end

    @testset "Event parameter scales by the device's max active power" begin
        container = _make_parameterized_range_container(devices, time_steps)
        param_container = IOM.add_param_container!(
            container,
            TestEventParameter,
            MockThermalGen,
            MockOutageAttribute,
            names,
            time_steps,
        )
        param_array = IOM.get_parameter_array(param_container)
        for name in names, t in time_steps
            param_array[name, t] = param_range_value(name, t)
        end
        add_parameterized_lower_bound_range_constraints(
            container,
            TestConstraintType,
            TestVariableType,
            TestEventParameter,
            devices,
            model,
            TestPowerModel,
        )
        con = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "lb")
        for (device, name) in zip(devices, names), t in time_steps
            # The event path scales by the device's max active power, not the multiplier
            # array (which is left as NaN here).
            expected = IOM.get_max_active_power(device) * param_range_value(name, t)
            @test JuMP.normalized_rhs(con[name, t]) ≈ expected
        end
    end
end
