function _make_directional_container(devices, time_steps)
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
    var = IOM.add_variable_container!(
        container, TestVariableType, eltype(devices), names, time_steps,
    )
    for name in names, t in time_steps
        var[name, t] = JuMP.@variable(jump_model)
    end
    return container
end

_test_flow_sign(::IOM.FromTo) = 1.0
_test_flow_sign(::IOM.ToFrom) = -1.0

@testset "Directional limits: accessors" begin
    limits = (from_to = (min = 0.0, max = 2.0), to_from = (min = 0.0, max = 3.0))
    @test IOM.get_directional_value(limits, IOM.FromTo()) == (min = 0.0, max = 2.0)
    @test IOM.get_directional_value(limits, IOM.ToFrom()) == (min = 0.0, max = 3.0)
    @test IOM.reverse_directions(limits) ==
          (from_to = (min = 0.0, max = 3.0), to_from = (min = 0.0, max = 2.0))
    @test IOM.constraint_meta(IOM.FromTo()) == "ft"
    @test IOM.constraint_meta(IOM.ToFrom()) == "tf"
end

@testset "Directional limits: effective_limit" begin
    limits = (from_to = (min = 0.0, max = 2.0), to_from = (min = 0.0, max = 9.0))
    @test IOM.effective_limit(limits, 5.0, IOM.FromTo()) == 2.0
    @test IOM.effective_limit(limits, 5.0, IOM.ToFrom()) == 5.0
end

@testset "Directional limits: constraint builder" begin
    time_steps = 1:2
    bus_a = MockBus("a", 1, :PV)
    bus_b = MockBus("b", 2, :PQ)
    devices = [MockBranch("L1", true, bus_a, bus_b, 5.0)]
    container = _make_directional_container(devices, time_steps)
    var = IOM.get_variable(container, TestVariableType, MockBranch)
    for dir in (IOM.FromTo(), IOM.ToFrom())
        sign = _test_flow_sign(dir)
        IOM.add_directional_limit_constraints!(
            container, TestConstraintType, MockBranch, dir, ["L1"],
            (name, t) -> sign * var[name, t],
            (name, t) -> 2.0 * t,
        )
    end
    con_ft = IOM.get_constraint(container, TestConstraintType, MockBranch, "ft")
    con_tf = IOM.get_constraint(container, TestConstraintType, MockBranch, "tf")
    for t in time_steps
        obj_ft = JuMP.constraint_object(con_ft["L1", t])
        obj_tf = JuMP.constraint_object(con_tf["L1", t])
        @test JuMP.coefficient(obj_ft.func, var["L1", t]) == 1.0
        @test JuMP.coefficient(obj_tf.func, var["L1", t]) == -1.0
        @test obj_ft.set == MOI.LessThan(2.0 * t)
        @test obj_tf.set == MOI.LessThan(2.0 * t)
    end
end

_dl(ft_max, tf_max; ft_min = 0.0, tf_min = 0.0) =
    (from_to = (min = ft_min, max = ft_max), to_from = (min = tf_min, max = tf_max))

@testset "Directional limits: validation errors" begin
    @test_throws IS.InvalidValue IOM.validate_directional_limits(
        MockBranch, [("L1", _dl(NaN, 1.0), 5.0)],
    )
    @test_throws IS.InvalidValue IOM.validate_directional_limits(
        MockBranch, [("L1", _dl(1.0, -1.0), 5.0)],
    )
    @test_throws IS.InvalidValue IOM.validate_directional_limits(
        MockBranch, [("L1", _dl(Inf, 1.0), 5.0)],
    )
    err = try
        IOM.validate_directional_limits(MockBranch, [("L1", _dl(NaN, 1.0), 5.0)])
    catch e
        e
    end
    @test occursin("L1", sprint(showerror, err))
    @test occursin("from_to", sprint(showerror, err))
end

@testset "Directional limits: validation warnings" begin
    entries = [("L$i", _dl(9.0, 1.0), 5.0) for i in 1:7]
    @test_logs (
        :warn,
        r"^7 MockBranch components have a directional limit at or above the rating: L1, L2, L3, L4, L5$",
    ) IOM.validate_directional_limits(MockBranch, entries)
    @test_logs (
        :warn,
        r"^1 MockBranch components have a directional min that is not zero; min is ignored: L1$",
    ) IOM.validate_directional_limits(
        MockBranch, [("L1", _dl(1.0, 1.0; ft_min = 0.5), 5.0)],
    )
    @test_logs IOM.validate_directional_limits(MockBranch, [("L1", _dl(1.0, 1.0), 5.0)])
end

@testset "Directional limits: time series values" begin
    @test_throws IS.InvalidValue IOM.validate_directional_limit_values(
        MockBranch, "L1", IOM.FromTo(), [1.0, 1.0, -0.5],
    )
    err = try
        IOM.validate_directional_limit_values(
            MockBranch, "L1", IOM.FromTo(), [1.0, 1.0, -0.5],
        )
    catch e
        e
    end
    @test occursin("time step 3", sprint(showerror, err))
    @test IOM.validate_directional_limit_values(
        MockBranch, "L1", IOM.ToFrom(), [1.0, 0.0, 2.0],
    ) === nothing
end
