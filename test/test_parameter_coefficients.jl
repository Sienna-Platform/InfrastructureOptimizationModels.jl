struct CoefficientParameter <: IOM.EventParameter end

# min x  s.t.  x + coef·y ≥ 6,  0 ≤ y ≤ 1,  x ≥ 0   ⇒   optimum = 6 - coef  (coef ∈ [0, 6])
function _hardcoded_optimum(coef::Float64)
    jm = JuMP.Model(HiGHS.Optimizer)
    JuMP.set_silent(jm)
    JuMP.@variable(jm, x >= 0)
    JuMP.@variable(jm, 0 <= y <= 1)
    JuMP.@constraint(jm, x + coef * y >= 6)
    JuMP.@objective(jm, Min, x)
    JuMP.optimize!(jm)
    return JuMP.objective_value(jm)
end

@testset "Parameter coefficients updated in place" begin
    mock_sys = MockSystem(100.0)
    settings = IOM.Settings(
        mock_sys;
        horizon = Dates.Hour(1),
        resolution = Dates.Hour(1),
        time_series_cache_size = 0,
    )
    container = IOM.OptimizationContainer(mock_sys, settings, nothing, MockDeterministic)
    IOM.set_time_steps!(container, 1:1)
    container.built_for_recurrent_solves = true
    param_container = IOM.add_event_parameter_container!(
        container,
        CoefficientParameter,
        MockThermalGen,
        MockOutageAttribute,
        ["c1"],
        1:1,
    )
    jm = IOM.get_jump_model(container)
    IOM._set_parameter_at!(IOM.get_parameter_array_data(param_container), jm, 1.0, 1, 1)
    status = IOM.get_parameter_array(param_container)["c1", 1]

    JuMP.@variable(jm, x >= 0)
    JuMP.@variable(jm, 0 <= y <= 1)
    row = JuMP.@constraint(jm, x + 3.0 * y >= 6)
    JuMP.@objective(jm, Min, x)
    key = IOM.ParameterKey(CoefficientParameter, MockThermalGen)
    IOM.add_parameter_coefficient!(container, key, "c1", 1, row, y)

    JuMP.set_optimizer(jm, HiGHS.Optimizer)
    JuMP.set_silent(jm)
    for value in (1.0, 0.0, 0.5, 1.0)
        IOM.fix_parameter_value(status, value)
        IOM.update_parameter_coefficients!(container, key)
        JuMP.optimize!(jm)
        @test JuMP.objective_value(jm) ≈ _hardcoded_optimum(3.0 * value)
    end
end
