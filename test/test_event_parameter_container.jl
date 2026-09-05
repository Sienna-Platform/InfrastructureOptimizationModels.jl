"""
Unit tests for the event parameter container (`EventParametersAttributes`).

The type parameter describing the event is a *supplemental attribute*, not a component:
downstream event models key parameters off types such as `PSY.FixedForcedOutage`, which
subtype `IS.SupplementalAttribute`. Bounding that slot on `IS.InfrastructureSystemsComponent`
makes every real call a `MethodError`, so the bound is what these tests pin down.
"""

struct EventContainerParameter <: IOM.EventParameter end

function _make_event_parameter_container(time_steps)
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

@testset "Event parameter container" begin
    time_steps = 1:3
    names = ["A", "B"]

    @testset "Allocation with a supplemental-attribute event type" begin
        container = _make_event_parameter_container(time_steps)
        param_container = IOM.add_event_parameter_container!(
            container,
            EventContainerParameter,
            MockThermalGen,
            MockOutageAttribute,
            names,
            time_steps,
        )
        param_array = IOM.get_parameter_array(param_container)
        @test axes(param_array)[1] == names
        @test axes(param_array)[2] == time_steps
        @test size(IOM.get_multiplier_array(param_container)) == (2, 3)

        attributes = IOM.get_attributes(param_container)
        @test attributes isa
              IOM.EventParametersAttributes{MockOutageAttribute, EventContainerParameter}
        @test IOM.get_param_type(attributes) === EventContainerParameter
        @test IOM.get_attribute_type(attributes) === MockOutageAttribute

        @test IOM.has_container_key(
            container,
            EventContainerParameter,
            MockThermalGen,
        )
    end

    @testset "The add_param_container! shim routes to the event overload" begin
        container = _make_event_parameter_container(time_steps)
        param_container = IOM.add_param_container!(
            container,
            EventContainerParameter,
            MockThermalGen,
            MockOutageAttribute,
            names,
            time_steps,
        )
        @test IOM.get_attributes(param_container) isa
              IOM.EventParametersAttributes{MockOutageAttribute, EventContainerParameter}
    end
end
