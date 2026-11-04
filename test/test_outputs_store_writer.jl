function _test_stats(objective, gap)
    return IOM.OptimizerStats(
        true, objective, 1, 1, 1, 0.5, 1, true, false, missing, gap, missing,
        0.7, missing, missing, missing, 0.6, 0.0, 0.0, missing, missing,
    )
end

function _test_params(num_executions::Int; horizon::Int = 3)
    return IOM.ModelStoreParams(
        num_executions,
        horizon,
        Millisecond(Hour(24)),
        Millisecond(Hour(1)),
        100.0,
        Base.UUID(UInt128(1)),
        IOM.OptimizationContainerMetadata(),
    )
end

function _test_decision_store()
    store = IOM.DecisionModelStore()
    key = IOM.VariableKey(TestVariableType, MockComponentType)
    key2 = IOM.VariableKey(TestVariableType2, MockComponentType)
    data = IOM.OrderedDict{DateTime, DenseAxisArray{Float64, 2}}()
    data2 = IOM.OrderedDict{DateTime, DenseAxisArray{Float64, 2}}()
    for (i, t) in enumerate((DateTime(2024, 1, 1), DateTime(2024, 1, 2)))
        data[t] = DenseAxisArray(
            i * 10.0 .+ reshape(1.0:6.0, 2, 3), ["g1", "g2"], 1:3,
        )
        data2[t] = DenseAxisArray(
            i * 10.0 .+ reshape(1.0:3.0, 1, 3), [IOM.encode_key_as_string(key2)], 1:3,
        )
    end
    store.variables[key] = data
    store.variables[key2] = data2
    store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.5, 0.1)
    store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.5, missing)
    return store
end

function _test_emulation_store()
    store = IOM.EmulationModelStore()
    key = IOM.VariableKey(TestVariableType, MockComponentType)
    values = DenseAxisArray(reshape(1.0:8.0, 2, 4), ["g1", "g2"], 1:4)
    IOM.set_dataset!(store.data_container, key, IOM.InMemoryDataset(values))
    for i in 1:4
        store.optimizer_stats[i] = _test_stats(Float64(i), 0.0)
    end
    return store
end

const _TEST_PRODUCER = IS.ProducerInfo("Test", "0.0.0", string(VERSION))

function _test_write(dir, store, num_executions; horizon::Int = 3)
    return IOM.write_store_outputs!(
        dir,
        "m",
        store,
        _test_params(num_executions; horizon = horizon),
        MockSystem(),
        IOM.Settings(
            MockSystem();
            horizon = Hour(horizon),
            interval = Hour(24),
            resolution = Hour(1),
        ),
        _TEST_PRODUCER,
        DateTime(2024, 1, 1),
    )
end

function _test_executions(labels, steps::Int)
    return IOM.OrderedDict(
        t => DenseAxisArray(
            e * 10.0 .+
            reshape(collect(1.0:(length(labels) * steps)), length(labels), steps),
            labels,
            1:steps,
        ) for (e, t) in enumerate((DateTime(2024, 1, 1), DateTime(2024, 1, 2)))
    )
end

function _test_executions(first_labels, second_labels, steps::Int)
    n = length(first_labels) * length(second_labels) * steps
    return IOM.OrderedDict(
        t => DenseAxisArray(
            e * 10.0 .+
            reshape(collect(1.0:n), length(first_labels), length(second_labels), steps),
            first_labels,
            second_labels,
            1:steps,
        ) for (e, t) in enumerate((DateTime(2024, 1, 1), DateTime(2024, 1, 2)))
    )
end

function _trait_error_message(store)
    try
        _test_write(mktempdir(), store, 2)
    catch e
        return sprint(showerror, e)
    end
    return ""
end

@testset "Outputs store writer" begin
    @testset "stats table" begin
        (table, columns) = IOM.optimizer_stats_table(
            _test_decision_store().optimizer_stats, "m",
        )
        @test table.length == 2
        @test table.group == "/m/tables/optimizer_stats"
        @test [c.name for c in table.columns] == [
            "execution", "objective_value", "termination_status", "primal_status",
            "dual_status", "solve_time", "solver_solve_time", "objective_bound",
            "relative_gap",
        ]
        @test columns.execution == [1, 2]
        @test columns.objective_value == [1.5, 2.5]
        @test isnan(columns.objective_bound[1])
        @test columns.relative_gap[1] == 0.1
        @test isnan(columns.relative_gap[2])
    end

    @testset "decision round trip" begin
        dir = _test_write(mktempdir(), _test_decision_store(), 2)
        a = IS.open_outputs(dir)
        (vals, axes) = IS.read_dataset(a, "m", "TestVariableType", "MockComponentType")
        @test size(vals) == (3, 2, 2)
        @test vals[:, :, 2] == permutedims(20.0 .+ reshape(1.0:6.0, 2, 3))
        @test axes == ["horizon", "MockComponentType", "execution"]
        (vals2, axes2) = IS.read_dataset(a, "m", "TestVariableType2", "")
        @test size(vals2) == (3, 2)
        @test vals2[:, 1] == [11.0, 12.0, 13.0]
        @test axes2 == ["horizon", "execution"]
        @test IS.read_table(a, "m", "optimizer_stats").execution == [1, 2]
        @test IS.read_outputs_metadata(joinpath(dir, "outputs.json")).status ==
              "SUCCESSFULLY_FINALIZED"
    end

    @testset "emulation round trip" begin
        dir = _test_write(mktempdir(), _test_emulation_store(), 4)
        a = IS.open_outputs(dir)
        (vals, axes) = IS.read_dataset(a, "m", "TestVariableType", "MockComponentType")
        @test size(vals) == (2, 4)
        @test vals == reshape(1.0:8.0, 2, 4)
        @test axes == ["MockComponentType", "time"]
        @test length(IS.read_table(a, "m", "optimizer_stats").execution) == 4
    end

    @testset "emulation 3-D round trip" begin
        store = IOM.EmulationModelStore()
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
        )
        data = reshape(collect(1.0:24.0), 2, 3, 4)
        IOM.set_dataset!(
            store.data_container,
            key,
            IOM.InMemoryDataset(
                DenseAxisArray(data, ["g1", "g2"], ["h1", "h2", "h3"], 1:4),
            ),
        )
        for i in 1:4
            store.optimizer_stats[i] = _test_stats(Float64(i), 0.0)
        end
        a = IS.open_outputs(_test_write(mktempdir(), store, 4))
        (vals, axes) = IS.read_dataset(
            a, "m", "TestVariableType", "MockComponentType"; meta = "MockComponentType2",
        )
        @test axes == ["MockComponentType2", "MockComponentType", "time"]
        @test size(vals) == (3, 2, 4)
        @test vals == permutedims(data, (2, 1, 3))
    end

    @testset "decision 3-D round trip" begin
        store = IOM.DecisionModelStore()
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
        )
        arrays = [
            DenseAxisArray(
                e * 100.0 .+ reshape(collect(1.0:12.0), 2, 3, 2),
                ["g1", "g2"],
                ["h1", "h2", "h3"],
                1:2,
            ) for e in 1:2
        ]
        store.variables[key] = IOM.OrderedDict(
            DateTime(2024, 1, 1) => arrays[1], DateTime(2024, 1, 2) => arrays[2],
        )
        store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.0, 0.0)
        store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.0, 0.0)
        dir = _test_write(mktempdir(), store, 2; horizon = 2)
        (vals, axes) = IS.read_dataset(
            IS.open_outputs(dir), "m", "TestVariableType", "MockComponentType";
            meta = "MockComponentType2",
        )
        @test axes == ["horizon", "MockComponentType2", "MockComponentType", "execution"]
        @test size(vals) == (2, 3, 2, 2)
        @test vals[:, :, :, 2] == permutedims(arrays[2].data, (3, 2, 1))
    end

    @testset "label only axis round trip" begin
        store = IOM.DecisionModelStore()
        key = IOM.VariableKey(TestVariableType, MockLabelOnlyType)
        store.variables[key] = _test_executions(["x", "y"], 3)
        store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.0, 0.0)
        store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.0, 0.0)
        a = IS.open_outputs(_test_write(mktempdir(), store, 2))
        data = IS.read_axis_data(a, "m", "MockLabelOnlyType")
        @test data.labels == ["x" "y"]
        @test isempty(data.ids)
        @test !IS.get_axis(a, "m", "MockLabelOnlyType").columns[1].has_ids
    end

    @testset "per-execution scalar has no horizon axis" begin
        store = _test_decision_store()
        key = IOM.VariableKey(TestVariableType2, MockComponentType)
        store.variables[key] = _test_executions(["g1", "g2"], 1)
        a = IS.open_outputs(_test_write(mktempdir(), store, 2))
        (vals, axes) = IS.read_dataset(a, "m", "TestVariableType2", "MockComponentType")
        @test axes == ["MockComponentType", "execution"]
        @test vals == [11.0 21.0; 12.0 22.0]
        (normal, normal_axes) =
            IS.read_dataset(a, "m", "TestVariableType", "MockComponentType")
        @test normal_axes == ["horizon", "MockComponentType", "execution"]
        @test size(normal) == (3, 2, 2)
    end

    @testset "time step count mismatch fails" begin
        store = _test_decision_store()
        key = IOM.VariableKey(TestVariableType2, MockComponentType)
        store.variables[key] = _test_executions(["g1", "g2"], 2)
        message = _trait_error_message(store)
        @test occursin("TestVariableType2", message)
        @test occursin("2 time steps", message)
        @test occursin("expected 3", message)
    end

    @testset "keys with different meta" begin
        store = IOM.DecisionModelStore()
        for meta in ("a", "b")
            key = IOM.VariableKey(TestVariableType, MockComponentType, meta)
            store.variables[key] = _test_executions(["g1", "g2"], 3)
        end
        store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.0, 0.0)
        store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.0, 0.0)
        a = IS.open_outputs(_test_write(mktempdir(), store, 2))
        for meta in ("a", "b")
            (vals, _) = IS.read_dataset(
                a, "m", "TestVariableType", "MockComponentType"; meta = meta,
            )
            @test size(vals) == (3, 2, 2)
        end
    end

    @testset "pair key with meta passes validation" begin
        store = IOM.DecisionModelStore()
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
            "x",
        )
        store.variables[key] = _test_executions(["g1", "g2"], ["h1", "h2", "h3"], 3)
        store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.0, 0.0)
        store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.0, 0.0)
        (metadata, _, datasets) = IOM.outputs_metadata(
            "m", store, _test_params(2), MockSystem(),
            IOM.Settings(
                MockSystem(); horizon = Hour(3), interval = Hour(24),
                resolution = Hour(1),
            ),
            _TEST_PRODUCER, DateTime(2024, 1, 1),
            first(IOM.optimizer_stats_table(store.optimizer_stats, "m")),
        )
        @test only(datasets)[2].meta == "MockComponentType2_x"
        @test isnothing(IS.validate_metadata(metadata))
    end

    @testset "type parameters keep datasets apart" begin
        function metas(keys...)
            store = IOM.DecisionModelStore()
            for key in keys
                store.variables[key] = _test_executions(["g1", "g2"], 3)
            end
            store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.0, 0.0)
            store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.0, 0.0)
            table = first(IOM.optimizer_stats_table(store.optimizer_stats, "m"))
            (metadata, _, datasets) = IOM.outputs_metadata(
                "m", store, _test_params(2), MockSystem(),
                IOM.Settings(
                    MockSystem(); horizon = Hour(3), interval = Hour(24),
                    resolution = Hour(1),
                ),
                _TEST_PRODUCER, DateTime(2024, 1, 1), table,
            )
            return sort!([d.meta for (_, d) in datasets])
        end
        @test metas(
            IOM.VariableKey(ParamVar{ParamMarkerA}, MockComponentType),
            IOM.VariableKey(ParamVar{ParamMarkerB}, MockComponentType),
        ) == ["ParamMarkerA", "ParamMarkerB"]
        @test metas(
            IOM.VariableKey(TestVariableType, ParamComponent{ParamMarkerA}),
            IOM.VariableKey(TestVariableType, ParamComponent{ParamMarkerB}),
        ) == ["ParamMarkerA", "ParamMarkerB"]
        @test metas(
            IOM.VariableKey(
                TestVariableType,
                IOM.ComponentPairKey{ParamComponent{ParamMarkerA}, MockComponentType2},
            ),
        ) == ["ParamMarkerA_MockComponentType2"]
        @test metas(IOM.VariableKey(TestVariableType, MockComponentType, "x")) == ["x"]
        @test metas(IOM.VariableKey(TestVariableType, MockComponentType)) == [""]
    end

    @testset "pair keys with the same first type" begin
        store = IOM.DecisionModelStore()
        for B in (MockComponentType, MockComponentType2)
            key = IOM.VariableKey(
                TestVariableType,
                IOM.ComponentPairKey{MockComponentType, B},
            )
            store.variables[key] = _test_executions(["g1", "g2"], ["h1", "h2", "h3"], 3)
        end
        store.optimizer_stats[DateTime(2024, 1, 1)] = _test_stats(1.0, 0.0)
        store.optimizer_stats[DateTime(2024, 1, 2)] = _test_stats(2.0, 0.0)
        a = IS.open_outputs(_test_write(mktempdir(), store, 2))
        for B in ("MockComponentType", "MockComponentType2")
            (vals, _) = IS.read_dataset(
                a, "m", "TestVariableType", "MockComponentType"; meta = B,
            )
            @test size(vals) == (3, 3, 2, 2)
        end
    end

    @testset "rerun overwrites" begin
        dir = mktempdir()
        store = _test_decision_store()
        _test_write(dir, store, 2)
        _test_write(dir, store, 2)
        a = IS.open_outputs(dir)
        @test IS.get_metadata(a).status == "SUCCESSFULLY_FINALIZED"
    end

    @testset "missing trait fails" begin
        store = _test_decision_store()
        key = IOM.VariableKey(TestCostVariable, MockComponentType)
        store.variables[key] =
            store.variables[IOM.VariableKey(TestVariableType, MockComponentType)]
        @test_throws ArgumentError _test_write(mktempdir(), store, 2)
        @test occursin("TestCostVariable", _trait_error_message(store))
    end
end
