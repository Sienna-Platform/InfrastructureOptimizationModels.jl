"""
Unit tests for left-hand-side (coefficient) parameters.

An LHS parameter's value multiplies a decision variable. It reaches the model only through a
product row `y - v * x == 0` whose coefficient on `x` IOM records as a binding and rewrites
in place, so its arrays hold numbers in every build mode.
"""

struct MockLHSParameter <: IOM.TimeSeriesLHSParameter end
struct MockRHSParameter <: IOM.TimeSeriesParameter end

function _make_lhs_container(time_steps)
    mock_sys = MockSystem(100.0)
    settings = IOM.Settings(
        mock_sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(mock_sys, settings, nothing, MockDeterministic)
    IOM.set_time_steps!(container, time_steps)
    return container
end

# One component "c1" whose profile lives in parameter row "hash1", plus a scalar multiplier.
function _lhs_fixture(values::Vector{Float64}; multiplier = 0.5, recurrent = false)
    time_steps = 1:length(values)
    container = _make_lhs_container(time_steps)
    container.built_for_recurrent_solves = recurrent
    param_container = IOM.add_time_series_parameter_container!(
        container,
        MockLHSParameter,
        MockThermalGen,
        MockDeterministic,
        "profile",
        ["hash1"],
        ["c1"],
        (),
        time_steps,
    )
    for t in time_steps
        IOM.get_parameter_array(param_container)["hash1", t] = values[t]
        IOM.get_multiplier_array(param_container)["c1", t] = multiplier
    end
    IOM.add_component_name!(IOM.get_attributes(param_container), "c1", "hash1")
    jump_model = IOM.get_jump_model(container)
    x = JuMP.@variable(jump_model, [time_steps])
    y = JuMP.@variable(jump_model, [time_steps])
    key = IOM.ParameterKey(MockLHSParameter, MockThermalGen)
    rows = [
        IOM.add_parameterized_product_constraint!(container, key, "c1", t, y[t], x[t])
        for t in time_steps
    ]
    return container, param_container, x, y, rows
end

@testset "LHS parameters" begin
    @testset "LHS types sit under LeftHandSideParameter" begin
        @test IOM.TimeSeriesLHSParameter <: IOM.LeftHandSideParameter
        @test !(IOM.TimeSeriesLHSParameter <: IOM.TimeSeriesParameter)
    end

    @testset "LHS containers hold Float64 even for recurrent solves" begin
        container, param_container, _, _, _ = _lhs_fixture([0.2, 0.4]; recurrent = true)
        @test IOM.get_param_eltype(container) == JuMP.VariableRef
        @test eltype(IOM.get_parameter_array(param_container)) == Float64
        @test IOM.get_attributes(param_container) isa IOM.TimeSeriesAttributes
    end

    @testset "RHS time series containers still follow get_param_eltype" begin
        container = _make_lhs_container(1:2)
        container.built_for_recurrent_solves = true
        param_container = IOM.add_time_series_parameter_container!(
            container,
            MockRHSParameter,
            MockThermalGen,
            MockDeterministic,
            "load",
            ["hash1"],
            ["c1"],
            (),
            1:2,
        )
        @test eltype(IOM.get_parameter_array(param_container)) == JuMP.VariableRef
    end

    @testset "Product rows carry v = parameter * multiplier" begin
        _, _, x, y, rows = _lhs_fixture([0.2, 0.4])
        @test JuMP.normalized_coefficient(rows[1], x[1]) ≈ -0.1
        @test JuMP.normalized_coefficient(rows[2], x[2]) ≈ -0.2
        @test JuMP.normalized_coefficient(rows[1], y[1]) ≈ 1.0
        @test JuMP.normalized_rhs(rows[1]) == 0.0
    end

    @testset "apply_coefficient_bindings! refreshes coefficients, including from zero" begin
        container, param_container, x, _, rows = _lhs_fixture([0.0, 0.4])
        IOM.get_parameter_array(param_container)["hash1", 1] = 0.6
        IOM.get_parameter_array(param_container)["hash1", 2] = 0.0
        IOM.apply_coefficient_bindings!(container)
        @test JuMP.normalized_coefficient(rows[1], x[1]) ≈ -0.3
        @test JuMP.normalized_coefficient(rows[2], x[2]) ≈ 0.0
    end

    @testset "apply_coefficient_bindings! overhead does not grow with the binding count" begin
        # JuMP's vectorized update allocates per entry; the wrapper must add only a constant.
        function overhead(n)
            container, _, x, _, rows = _lhs_fixture(collect(range(0.1, 0.9; length = n)))
            coefficients = fill(-0.25, n)
            IOM.apply_coefficient_bindings!(container)
            JuMP.set_normalized_coefficient(rows, x.data, coefficients)
            jump_bytes =
                @allocated JuMP.set_normalized_coefficient(rows, x.data, coefficients)
            return (@allocated IOM.apply_coefficient_bindings!(container)) - jump_bytes
        end
        overhead(10)
        @test overhead(500) - overhead(50) < 256
    end

    @testset "Multiplier changes are applied too" begin
        container, param_container, x, _, rows = _lhs_fixture([0.2, 0.4])
        IOM.get_multiplier_array(param_container)["c1", 2] = 1.0
        IOM.apply_coefficient_bindings!(container)
        @test JuMP.normalized_coefficient(rows[2], x[2]) ≈ -0.4
    end

    @testset "has_lhs_parameter_component" begin
        container, _, _, _, _ = _lhs_fixture([0.2, 0.4])
        key = IOM.ParameterKey(MockLHSParameter, MockThermalGen)
        @test IOM.has_lhs_parameter_component(container, key, "c1")
        @test !IOM.has_lhs_parameter_component(container, key, "c2")
    end

    @testset "A component without a series row cannot be bound" begin
        container, _, _, _, _ = _lhs_fixture([0.2, 0.4])
        key = IOM.ParameterKey(MockLHSParameter, MockThermalGen)
        jump_model = IOM.get_jump_model(container)
        x = JuMP.@variable(jump_model)
        y = JuMP.@variable(jump_model)
        @test_throws ArgumentError IOM.add_parameterized_product_constraint!(
            container, key, "c2", 1, y, x)
    end

    @testset "reset_optimization_model! drops bindings" begin
        container, _, _, _, _ = _lhs_fixture([0.2, 0.4])
        @test !isempty(container.coefficient_bindings)
        IOM.reset_optimization_model!(container)
        @test isempty(container.coefficient_bindings)
        IOM.apply_coefficient_bindings!(container)
    end

    @testset "apply_coefficient_bindings! is a no-op without bindings" begin
        container = _make_lhs_container(1:2)
        IOM.apply_coefficient_bindings!(container)
        @test isempty(container.coefficient_bindings)
    end

    @testset "LHS entry points are documented" begin
        @test Base.Docs.hasdoc(IOM, :add_time_series_parameter_container!)
        @test Base.Docs.hasdoc(IOM, :CoefficientBindings)
        @test Base.Docs.hasdoc(IOM, :add_parameterized_product_constraint!)
        @test Base.Docs.hasdoc(IOM, :apply_coefficient_bindings!)
    end

    @testset "Product variables are internal and not written to outputs" begin
        @test !IOM.should_write_resulting_value(IOM.ParameterizedProductVariable)
    end

    @testset "ServiceModel accepts LHS keys in time_series_names" begin
        names = Dict{Type{<:IOM.ParameterType}, String}(MockLHSParameter => "profile")
        model = IOM.ServiceModel(
            MockReserve{MockUp},
            MockReserveFormulation;
            time_series_names = names,
        )
        @test IOM.get_time_series_names(model)[MockLHSParameter] == "profile"
    end
end
