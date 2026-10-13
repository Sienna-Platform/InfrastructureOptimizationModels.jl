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

    @testset "Expression-path ramp down row bounds the decrease" begin
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

        con_dn = IOM.get_constraint(container, TestConstraintType, MockThermalGen, "dn")
        down_limit = 0.2 * 60

        # t = 1: ic - (p1 - r1) <= down * dt
        obj = JuMP.constraint_object(con_dn["A", 1])
        @test obj.set == MOI.LessThan(down_limit - 50.0)
        @test JuMP.coefficient(obj.func, p["A", 1]) == -1.0
        @test JuMP.coefficient(obj.func, r["A", 1]) == 1.0

        # t = 2: p1 - (p2 - r2) <= down * dt
        obj = JuMP.constraint_object(con_dn["A", 2])
        @test obj.set == MOI.LessThan(down_limit)
        @test JuMP.coefficient(obj.func, p["A", 1]) == 1.0
        @test JuMP.coefficient(obj.func, p["A", 2]) == -1.0
        @test JuMP.coefficient(obj.func, r["A", 2]) == 1.0
    end
end
