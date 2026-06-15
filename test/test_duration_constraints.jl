"""
Unit tests for duration constraints: each constraint container must be keyed by
the devices that have an initial condition in its own column (Task 2.14).
Previously both containers shared the up-column axis, causing KeyErrors or
permanently-#undef rows for asymmetric up/down ICs.
"""

function _make_duration_test_container(names, time_steps)
    mock_sys = MockSystem(100.0)
    settings = PSI.Settings(
        mock_sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = PSI.OptimizationContainer(mock_sys, settings, nothing, MockDeterministic)
    PSI.set_time_steps!(container, time_steps)
    jump_model = PSI.get_jump_model(container)
    for T in (PSI.OnVariable, PSI.StartVariable, PSI.StopVariable)
        var = PSI.add_variable_container!(container, T, MockThermalGen, names, time_steps)
        for name in names, t in time_steps
            var[name, t] = JuMP.@variable(jump_model, binary = true)
        end
    end
    return container
end

@testset "Duration constraints with asymmetric up/down ICs (Task 2.14)" begin
    time_steps = 1:4
    names = ["A", "B"]
    dev_a = make_mock_thermal("A")
    dev_b = make_mock_thermal("B")
    # A has only an up IC; B has only a down IC. The element type must be the
    # abstract InitialCondition to match the builders' Matrix{InitialCondition} signature.
    ics = PSI.InitialCondition[
        PSI.InitialCondition(MockInitialCondition, dev_a, 2.0) PSI.InitialCondition{
        MockInitialCondition,
        Nothing
    }(
        dev_a,
        nothing
    )
        PSI.InitialCondition{MockInitialCondition, Nothing}(dev_b, nothing) PSI.InitialCondition(
        MockInitialCondition,
        dev_b,
        3.0
    )
    ]
    duration_data = [(up = 2.0, down = 2.0), (up = 2.0, down = 2.0)]

    container = _make_duration_test_container(names, time_steps)
    PSI.device_duration_retrospective!(
        container,
        duration_data,
        ics,
        TestConstraintType,
        MockThermalGen,
    )
    con_up = PSI.get_constraint(container, TestConstraintType, MockThermalGen, "up")
    con_down = PSI.get_constraint(container, TestConstraintType, MockThermalGen, "dn")
    @test axes(con_up)[1] == ["A"]
    @test axes(con_down)[1] == ["B"]
    # Every container slot must be written — no #undef rows.
    @test all(i -> isassigned(con_up.data, i), eachindex(con_up.data))
    @test all(i -> isassigned(con_down.data, i), eachindex(con_down.data))

    container = _make_duration_test_container(names, time_steps)
    PSI.device_duration_look_ahead!(
        container,
        duration_data,
        ics,
        TestConstraintType,
        TestCostConstraint,
        MockThermalGen,
    )
    con_up = PSI.get_constraint(container, TestConstraintType, MockThermalGen)
    con_down = PSI.get_constraint(container, TestCostConstraint, MockThermalGen)
    @test axes(con_up)[1] == ["A"]
    @test axes(con_down)[1] == ["B"]
    @test all(i -> isassigned(con_up.data, i), eachindex(con_up.data))
    @test all(i -> isassigned(con_down.data, i), eachindex(con_down.data))

    container = _make_duration_test_container(names, time_steps)
    PSI.device_duration_parameters!(
        container,
        duration_data,
        ics,
        TestConstraintType,
        MockThermalGen,
    )
    con_up = PSI.get_constraint(container, TestConstraintType, MockThermalGen, "up")
    con_down = PSI.get_constraint(container, TestConstraintType, MockThermalGen, "dn")
    @test axes(con_up)[1] == ["A"]
    @test axes(con_down)[1] == ["B"]
    @test all(i -> isassigned(con_up.data, i), eachindex(con_up.data))
    @test all(i -> isassigned(con_down.data, i), eachindex(con_down.data))
end
