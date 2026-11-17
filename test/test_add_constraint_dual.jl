"""
Unit tests for `add_constraint_dual!` and the `dual_assigned_later` trait.
"""

struct LaterDualConstraint <: IOM.ConstraintType end
IOM.dual_assigned_later(::Type{LaterDualConstraint}) = true

function _make_dual_container(sys, time_steps)
    settings = IOM.Settings(
        sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(sys, settings, nothing, MockDeterministic)
    IOM.set_time_steps!(container, time_steps)
    return container
end

function _fill_constraints!(container, ::Type{C}, names, time_steps) where {C}
    jump_model = IOM.get_jump_model(container)
    cons = IOM.add_constraints_container!(container, C, MockThermalGen, names, time_steps)
    for name in names, t in time_steps
        x = JuMP.@variable(jump_model)
        cons[name, t] = JuMP.@constraint(jump_model, x <= 1.0)
    end
    return
end

@testset "add_constraint_dual! with dual_assigned_later" begin
    time_steps = 1:2
    bus = MockBus("bus1", 1, :PV)
    gen = MockThermalGen("A", true, bus, (min = 0.0, max = 1.0))
    sys = MockSystem(100.0)
    add_component!(sys, gen)
    names = [get_name(gen)]

    @testset "The device hook skips a later-assigned type" begin
        container = _make_dual_container(sys, time_steps)
        _fill_constraints!(container, TestConstraintType, names, time_steps)
        model = IOM.DeviceModel(
            MockThermalGen,
            TestDeviceFormulation;
            duals = [TestConstraintType, LaterDualConstraint],
        )
        IOM.add_constraint_dual!(container, sys, model)
        duals = IOM.get_duals(container)
        @test haskey(duals, IOM.ConstraintKey(TestConstraintType, MockThermalGen))
        @test !haskey(duals, IOM.ConstraintKey(LaterDualConstraint, MockThermalGen))
        @test (LaterDualConstraint, MockThermalGen) in IOM.get_deferred_duals(container)

        # The build check errors while no pass assigned the deferred dual.
        @test_throws ErrorException IOM.check_optimization_container(container)

        # The later pass assigns the dual once the rows exist.
        _fill_constraints!(container, LaterDualConstraint, names, time_steps)
        IOM.assign_dual_variable!(
            container,
            LaterDualConstraint,
            [gen],
            TestDeviceFormulation,
        )
        @test haskey(duals, IOM.ConstraintKey(LaterDualConstraint, MockThermalGen))
        IOM.check_optimization_container(container)
        @test true
    end

    @testset "The default trait still fails on a missing container" begin
        container = _make_dual_container(sys, time_steps)
        model = IOM.DeviceModel(
            MockThermalGen,
            TestDeviceFormulation;
            duals = [TestConstraintType],
        )
        @test_throws IS.InvalidValue IOM.add_constraint_dual!(container, sys, model)
    end
end
