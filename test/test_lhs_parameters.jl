"""
Unit tests for left-hand-side (coefficient) parameters.

An LHS parameter is a time series whose value multiplies a decision variable, so formulations
write it into constraints as a fixed number. Its containers hold `Float64` in every build mode,
and a model holding one is rebuilt every simulation step to pick up refreshed values.
"""

struct MockLHSParameter <: IOM.LeftHandSideTimeSeriesParameter end
struct MockRHSParameter <: IOM.TimeSeriesParameter end

function _make_lhs_container(time_steps; rebuild_model = false)
    mock_sys = MockSystem(100.0)
    settings = IOM.Settings(
        mock_sys;
        horizon = Dates.Hour(length(time_steps)),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
        rebuild_model = rebuild_model,
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
    key = IOM.ParameterKey(MockLHSParameter, MockThermalGen)
    return container, param_container, key
end

@testset "LHS parameters" begin
    @testset "LHS time series are time series parameters" begin
        @test IOM.LeftHandSideTimeSeriesParameter <: IOM.TimeSeriesParameter
        @test supertype(IOM.VariableValueParameter) === IOM.ParameterType
    end

    @testset "LHS containers hold Float64 even for recurrent solves" begin
        container, param_container, _ = _lhs_fixture([0.2, 0.4]; recurrent = true)
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

    @testset "get_lhs_parameter_values returns parameter * multiplier" begin
        container, param_container, key = _lhs_fixture([0.2, 0.4])
        @test IOM.get_lhs_parameter_values(container, key, "c1") ≈ [0.1, 0.2]
        IOM.get_multiplier_array(param_container)["c1", 2] = 1.0
        @test IOM.get_lhs_parameter_values(container, key, "c1") ≈ [0.1, 0.4]
        @test IOM.get_lhs_parameter_values(container, key, "c1") isa Vector{Float64}
    end

    @testset "has_lhs_parameter_component" begin
        container, _, key = _lhs_fixture([0.2, 0.4])
        @test IOM.has_lhs_parameter_component(container, key, "c1")
        @test !IOM.has_lhs_parameter_component(container, key, "c2")
        @test_throws ArgumentError IOM.get_lhs_parameter_values(container, key, "c2")
    end

    @testset "rebuild_model defaults to false and can be switched on" begin
        settings = IOM.get_settings(_make_lhs_container(1:2))
        @test IOM.get_rebuild_model(settings) === false
        IOM.set_rebuild_model!(settings, true)
        @test IOM.get_rebuild_model(settings) === true
        explicit = IOM.get_settings(_make_lhs_container(1:2; rebuild_model = true))
        @test IOM.get_rebuild_model(explicit) === true
    end

    @testset "Recurrent eltype follows the rebuild setting" begin
        container = _make_lhs_container(1:2)
        container.built_for_recurrent_solves = true
        @test IOM.get_param_eltype(container) == JuMP.VariableRef
        IOM.set_rebuild_model!(IOM.get_settings(container), true)
        @test IOM.get_param_eltype(container) == Float64
    end

    @testset "ServiceModel accepts LHS keys in time_series_names" begin
        names = Dict{Type{<:IOM.TimeSeriesParameter}, String}(MockLHSParameter => "profile")
        model = IOM.ServiceModel(
            MockReserve{MockUp},
            MockReserveFormulation;
            time_series_names = names,
        )
        @test IOM.get_time_series_names(model)[MockLHSParameter] == "profile"
    end

    @testset "LHS entry points are documented" begin
        @test Base.Docs.hasdoc(IOM, :add_time_series_parameter_container!)
        @test Base.Docs.hasdoc(IOM, :get_lhs_parameter_values)
        @test Base.Docs.hasdoc(IOM, :LeftHandSideTimeSeriesParameter)
    end
end

# A real IS store: the mock time series registry holds no data.
function _axes_fixture()
    data = IS.SystemData()
    component = IS.TestComponent("c1", 1)
    IS.add_component!(data, component)
    t0 = DateTime("2024-01-01T00:00:00")
    links = reshape(collect(Int64, 1:(48 * 2 * 3)), 48, 2, 3)
    IS.add_time_series!(
        data,
        component,
        IS.SingleTimeSeries("links", t0, Hour(1), links;
            value_axes = [IS.TimeSeriesAxis("block", [1, 2]),
                IS.TimeSeriesAxis("product", ["a", "b", "c"])]),
    )
    shares = reshape(collect(Float64, 1:(24 * 3)), 24, 3)
    IS.add_time_series!(
        data,
        component,
        IS.Deterministic("shares",
            Dict(t0 => shares), Hour(1), Hour(24);
            value_axes = [IS.TimeSeriesAxis("bus", [10, 20, 30])]),
    )
    IS.add_time_series!(data, component,
        IS.Deterministic("plain", Dict(t0 => collect(1.0:24.0)), Hour(1), Hour(24)))
    return component, t0, links, shares
end

function _axes_container(t0; steps = 24)
    container = _make_lhs_container(1:steps)
    IOM.set_initial_time!(IOM.get_settings(container), t0)
    return container
end

@testset "Series with value axes read as flattened steps" begin
    component, t0, links, shares = _axes_fixture()

    @testset "a Deterministic matrix gives one vector per step" begin
        container = _axes_container(t0)
        got = IOM.get_time_series_initial_values!(
            container, IS.Deterministic, component, "shares")
        @test length(got) == 24
        @test all(got[t] == shares[t, :] for t in 1:24)
    end

    @testset "a rank-3 SingleTimeSeries is flattened column-major and keeps Int64" begin
        container = _axes_container(t0 + Hour(24))
        got = IOM.get_time_series_initial_values!(
            container, IS.SingleTimeSeries, component, "links")
        @test length(got) == 24
        @test eltype(first(got)) == Int64
        @test got[1] == vec(links[25, :, :])
        @test got[24] == vec(links[48, :, :])
    end

    @testset "unaxed series are unchanged" begin
        container = _axes_container(t0)
        got = IOM.get_time_series_initial_values!(
            container, IS.Deterministic, component, "plain")
        @test collect(got) == collect(1.0:24.0)
    end

    @testset "only Deterministic and SingleTimeSeries carry value axes" begin
        @test IOM._value_axes(
            MockDeterministic("x", Float64[], Hour(1), DateTime("2024-01-01T00:00:00")),
        ) === nothing
    end
end

@testset "LHS steps are padded onto the extra axis" begin
    @test IOM.unwrap_for_param(MockLHSParameter(), [1, 2], (1:4,)) == [1.0, 2.0, 0.0, 0.0]
    @test IOM.unwrap_for_param(MockLHSParameter(), [1, 2], (1:4,)) isa Vector{Float64}
    @test_throws ArgumentError IOM.unwrap_for_param(MockLHSParameter(), [1, 2, 3], (1:2,))
    # A parameter that is not left-hand-side keeps the identity.
    step = [1.0, 2.0]
    @test IOM.unwrap_for_param(MockRHSParameter(), step, (1:4,)) === step
    @test IOM.get_value_length([IS.TimeSeriesAxis("block", [1, 2]),
        IS.TimeSeriesAxis("product", ["a", "b", "c"])]) == 6
end
