if !isdefined(InfrastructureOptimizationModelsTests, :RampTestFormulation)
    struct RampTestFormulation <: IOM.AbstractDeviceFormulation end
end

IOM._get_initial_condition_type(
    ::Type{TestConstraintType},
    ::Type{MockThermalGen},
    ::Type{RampTestFormulation},
) = MockInitialCondition

IOM.get_ramp_limits(::MockThermalGen) = (up = 0.1, down = 0.2)

@testset "Ramp constraints" begin
    @testset "Initial on/off status for t=1 big-M" begin
        # `_initial_on_status` derives the prior on/off status that gates the t=1 ramp
        # big-M, mirroring the `2 - yprev - ycur` relaxation used for t ≥ 2. Without it
        # the old `1 - ycur` bound made a unit started at t=1 from an off initial state
        # infeasible whenever Pmin > ramp_up·dt.
        # Non-recurrent build: power IC is a Float64, status is exact.
        @test IOM._initial_on_status(0.0) == 0.0          # off
        @test IOM._initial_on_status(0.5) == 1.0          # on
        @test IOM._initial_on_status(IOM.ABSOLUTE_TOLERANCE / 2) == 0.0  # below tolerance -> off
        @test IOM._initial_on_status(2.0 * IOM.ABSOLUTE_TOLERANCE) == 1.0

        # Recurrent build: power IC is a parameter VariableRef whose value is unknown at
        # build time. Conservatively reported as off (0.0) so the t=1 big-M fully relaxes
        # — never reintroducing the start-up infeasibility.
        m = JuMP.Model()
        JuMP.@variable(m, p)
        @test IOM._initial_on_status(p) == 0.0
    end

    @testset "Expression-path ramp down limits the decrease in a solved model" begin
        time_steps = 1:2
        dev = make_mock_thermal("A")
        mock_sys = MockSystem(100.0)
        settings = IOM.Settings(
            mock_sys;
            horizon = Dates.Hour(length(time_steps)),
            resolution = Dates.Hour(1),
            time_series_cache_size = 0,
        )
        container =
            IOM.OptimizationContainer(mock_sys, settings, nothing, MockDeterministic)
        IOM.set_time_steps!(container, time_steps)
        jump_model = IOM.get_jump_model(container)
        names = ["A"]

        p = IOM.add_variable_container!(
            container, IOM.ActivePowerVariable, MockThermalGen, names, time_steps)
        r = IOM.add_variable_container!(
            container, TestVariableType, MockThermalGen, names, time_steps)
        lb = IOM.add_expression_container!(
            container, IOM.ActivePowerRangeExpressionLB, MockThermalGen, names,
            time_steps)
        ub = IOM.add_expression_container!(
            container, IOM.ActivePowerRangeExpressionUB, MockThermalGen, names,
            time_steps)
        for t in time_steps
            p["A", t] = JuMP.@variable(jump_model)
            r["A", t] = JuMP.@variable(jump_model)
            lb["A", t] = JuMP.@expression(jump_model, p["A", t] - r["A", t])
            ub["A", t] = JuMP.@expression(jump_model, 1.0 * p["A", t])
        end
        ics = IOM.add_initial_condition_container!(
            container, MockInitialCondition(), MockThermalGen, names)
        ics[1] = IOM.InitialCondition(MockInitialCondition, dev, 50.0)

        model = IOM.DeviceModel(MockThermalGen, RampTestFormulation)
        IOM.add_linear_ramp_constraints!(
            container, TestConstraintType, IOM.ActivePowerVariable, [dev], model,
            TestPowerModel)

        # Minimizing power must stop at the ramp-down limit from the initial value:
        # p1 >= 50 - 12 = 38 and p2 >= p1 - 12 = 26. A row that bounds the increase
        # instead would let both go to zero.
        JuMP.@constraint(jump_model, [t in time_steps], p["A", t] >= 0.0)
        JuMP.fix.(r["A", :], 0.0; force = true)
        JuMP.@objective(jump_model, Min, sum(p["A", t] for t in time_steps))
        JuMP.set_optimizer(jump_model, HiGHS.Optimizer)
        JuMP.set_silent(jump_model)
        JuMP.optimize!(jump_model)

        @test JuMP.termination_status(jump_model) == MOI.OPTIMAL
        @test JuMP.value(p["A", 1]) ≈ 38.0
        @test JuMP.value(p["A", 2]) ≈ 26.0
    end
end

# One unit (P_min 30, P_max 100), two hourly steps, the ramp limits defined above.
function _semicontinuous_ramp_rows(::Type{U}) where {U <: IOM.VariableType}
    device = make_mock_thermal("gen1"; limits = (min = 30.0, max = 100.0))
    time_steps = 1:2
    names = ["gen1"]
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
    power = IOM.add_variable_container!(container, U, MockThermalGen, names, time_steps)
    start = IOM.add_variable_container!(
        container, IOM.StartVariable, MockThermalGen, names, time_steps)
    stop = IOM.add_variable_container!(
        container, IOM.StopVariable, MockThermalGen, names, time_steps)
    lb = IOM.add_expression_container!(
        container, IOM.ActivePowerRangeExpressionLB, MockThermalGen, names, time_steps)
    ub = IOM.add_expression_container!(
        container, IOM.ActivePowerRangeExpressionUB, MockThermalGen, names, time_steps)
    for t in time_steps
        power["gen1", t] = JuMP.@variable(jump_model)
        start["gen1", t] = JuMP.@variable(jump_model)
        stop["gen1", t] = JuMP.@variable(jump_model)
        lb["gen1", t] = JuMP.@expression(jump_model, 1.0 * power["gen1", t])
        ub["gen1", t] = JuMP.@expression(jump_model, 1.0 * power["gen1", t])
    end
    ics = IOM.add_initial_condition_container!(
        container, MockInitialCondition(), MockThermalGen, names)
    ics[1] = IOM.InitialCondition(MockInitialCondition, device, 0.0)
    model = IOM.DeviceModel(MockThermalGen, RampTestFormulation)
    IOM.add_semicontinuous_ramp_constraints!(
        container, TestConstraintType, U, [device], model, TestPowerModel)
    up = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "up")
    dn = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "dn")
    return up, dn, start, stop
end

@testset "Semicontinuous ramp rows on PowerAboveMinimumVariable have no start/stop term" begin
    up, dn, start, stop = _semicontinuous_ramp_rows(IOM.PowerAboveMinimumVariable)
    for t in 1:2
        @test iszero(JuMP.normalized_coefficient(up["gen1", t], start["gen1", t]))
        @test iszero(JuMP.normalized_coefficient(dn["gen1", t], stop["gen1", t]))
    end
    # Control: a row on total power keeps the P_min allowance at start-up and shut-down.
    up, dn, start, stop = _semicontinuous_ramp_rows(IOM.ActivePowerVariable)
    @test JuMP.normalized_coefficient(up["gen1", 1], start["gen1", 1]) ≈ -30.0
    @test JuMP.normalized_coefficient(dn["gen1", 1], stop["gen1", 1]) ≈ -30.0
end
