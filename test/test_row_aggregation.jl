struct MockAggregatedParameter <: IOM.TimeSeriesParameter end

# Four components on three rows: "a", "b" and "d" share series "h1"; "c" uses "h2".
# Rows 10 and 30 use only "h1", so they share its parameter.
function _row_aggregation_fixture(; recurrent = true)
    time_steps = 1:2
    container = _make_lhs_container(time_steps)
    container.built_for_recurrent_solves = recurrent
    row_of = Dict("a" => 10, "b" => 20, "c" => 20, "d" => 30)
    pc = IOM.add_aggregated_time_series_parameter_container!(
        container,
        MockAggregatedParameter,
        MockThermalGen,
        MockDeterministic,
        "load",
        ["h1", "h2"],
        ["a", "b", "c", "d"],
        row_of,
        time_steps,
    )
    jump_model = IOM.get_jump_model(container)
    parent_param = IOM.get_parameter_array_data(pc)
    parent_mult = IOM.get_multiplier_array_data(pc)
    for (i, series) in enumerate(([1.0, 2.0], [3.0, 4.0])), t in time_steps
        IOM._set_parameter_at!(parent_param, jump_model, series[t], i, t)
    end
    for (i, (name, hash, mult)) in
        enumerate((("a", "h1", 1.0), ("b", "h1", 2.0), ("c", "h2", -1.0), ("d", "h1", 3.0)))
        IOM._set_multiplier_at!(parent_mult, mult, i)
        IOM.add_component_name!(IOM.get_attributes(pc), name, hash)
    end
    IOM.build_row_aggregates!(container, pc)
    return container, pc
end

@testset "Row aggregation of time-series parameters" begin
    @testset "recurrent build: one series parameter or one sum per row" begin
        container, pc = _row_aggregation_fixture()
        @test eltype(IOM.get_parameter_array(pc)) == Float64
        refs10, coefficient10 = IOM.get_row_aggregate_refs(pc, 10)
        refs20, coefficient20 = IOM.get_row_aggregate_refs(pc, 20)
        refs30, coefficient30 = IOM.get_row_aggregate_refs(pc, 30)
        @test eltype(refs20) == JuMP.VariableRef
        @test JuMP.fix_value.(collect(refs10)) == [1.0, 2.0]
        @test refs30 == refs10
        @test (coefficient10, coefficient30) == (1.0, 3.0)
        # 2 * h1 - h2
        @test JuMP.fix_value.(collect(refs20)) == [-1.0, 0.0]
        @test coefficient20 == 1.0
        @test JuMP.num_variables(IOM.get_jump_model(container)) == 4
    end

    @testset "refresh recomputes and fixes the sums" begin
        _, pc = _row_aggregation_fixture()
        parent_param = IOM.get_parameter_array_data(pc)
        parent_param[1, 2] = 5.0
        parent_param[2, 1] = 0.5
        IOM.refresh_row_aggregates!(IOM.get_attributes(pc), parent_param)
        refs10, _ = IOM.get_row_aggregate_refs(pc, 10)
        refs20, _ = IOM.get_row_aggregate_refs(pc, 20)
        @test JuMP.fix_value.(collect(refs10)) == [1.0, 5.0]
        @test JuMP.fix_value.(collect(refs20)) == [1.5, 6.0]
    end

    @testset "single-solve build stores Float64 sums" begin
        _, pc = _row_aggregation_fixture(; recurrent = false)
        @test collect(first(IOM.get_row_aggregate_refs(pc, 20))) == [-1.0, 0.0]
        @test collect(first(IOM.get_row_aggregate_refs(pc, 30))) == [1.0, 2.0]
    end

    @testset "per-component values stay available; refs error" begin
        _, pc = _row_aggregation_fixture()
        @test IOM.get_parameter_column_values(pc, "b") == [1.0, 2.0]
        @test_throws ErrorException IOM.get_parameter_column_refs(pc, "b")
        @test IOM.calculate_parameter_values(pc)["b", :] == [2.0, 4.0]
    end

    @testset "time-varying multiplier errors" begin
        container = _make_lhs_container(1:2)
        pc = IOM.add_aggregated_time_series_parameter_container!(
            container, MockAggregatedParameter, MockThermalGen, MockDeterministic,
            "load", ["h1"], ["a"], Dict("a" => 1), 1:2)
        IOM.get_parameter_array(pc)["h1", :] .= 1.0
        IOM.get_multiplier_array(pc)["a", 1] = 1.0
        IOM.get_multiplier_array(pc)["a", 2] = 2.0
        IOM.add_component_name!(IOM.get_attributes(pc), "a", "h1")
        @test_throws ErrorException IOM.build_row_aggregates!(container, pc)
    end
end
