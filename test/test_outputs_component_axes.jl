function _test_registry()
    return IOM.AxisRegistry(
        Dict{String, IS.AbstractAxis}(),
        Dict{Tuple{String, String}, IS.AxisData}(),
        "m",
    )
end

@testset "Outputs component axes" begin
    sys = MockSystem()

    @testset "single axis with ids" begin
        reg = _test_registry()
        key = IOM.register_component_axis!(reg, sys, MockComponentType, ["g1", "g2"], "d")
        @test key == "MockComponentType"
        axis = reg.axes[key]
        @test axis.columns[1].has_ids
        @test axis.length == 2
        @test axis.group == "/m/axes/MockComponentType"
        data = reg.data[("m", key)]
        @test data.ids == [1 2]
        @test data.labels == ["g1" "g2"]
    end

    @testset "dedup identical" begin
        reg = _test_registry()
        k1 = IOM.register_component_axis!(reg, sys, MockComponentType, ["g1", "g2"], "a")
        k2 = IOM.register_component_axis!(reg, sys, MockComponentType, ["g1", "g2"], "b")
        @test k1 == k2
        @test length(reg.axes) == 1
    end

    @testset "subset on conflict" begin
        reg = _test_registry()
        IOM.register_component_axis!(reg, sys, MockComponentType, ["g1", "g2"], "a")
        k = IOM.register_component_axis!(
            reg, sys, MockComponentType, ["g1"],
            "TestVariableType__MockComponentType__x",
        )
        @test k == "MockComponentType__testvariabletype__mockcomponenttype__x"
        @test length(reg.axes) == 2
        @test_throws ArgumentError IOM.register_component_axis!(
            reg, sys, MockComponentType, ["g2"],
            "TestVariableType__MockComponentType__x",
        )
    end

    @testset "canonical labels replace bus numbers" begin
        reg = _test_registry()
        k = IOM.register_component_axis!(reg, sys, MockBusType, ["1", "2"], "d")
        @test reg.data[("m", k)].labels == ["bus1" "bus2"]
        @test reg.data[("m", k)].ids == [1 2]
        k2 = IOM.register_component_axis!(reg, sys, MockBusType, ["1", "2"], "e")
        @test k2 == k
        @test length(reg.axes) == 1
    end

    @testset "label only axis" begin
        reg = _test_registry()
        k = IOM.register_component_axis!(reg, sys, MockLabelOnlyType, ["x", "y"], "d")
        @test !reg.axes[k].columns[1].has_ids
        @test isempty(reg.data[("m", k)].ids)
        @test reg.data[("m", k)].labels == ["x" "y"]
    end

    @testset "encoded labels no ids" begin
        reg = _test_registry()
        k = IOM.register_component_axis!(
            reg, sys, MockComponentType, ["s1__g1", "s1__g2"], "d",
        )
        @test !reg.axes[k].columns[1].has_ids
        @test isempty(reg.data[("m", k)].ids)
    end

    @testset "mixed encoded and plain labels fail" begin
        @test IOM.is_encoded("a", ["s1__g1", "s1__g2"])
        @test !IOM.is_encoded("a", ["g1", "g2"])
        err = try
            IOM.is_encoded("a", ["s1__g1", "g2"])
        catch e
            sprint(showerror, e)
        end
        @test occursin("Axis a", err)
        @test occursin("s1__g1", err)
        @test !occursin("g2", err)
        @test_throws ArgumentError IOM.register_component_axis!(
            _test_registry(), sys, MockComponentType, ["s1__g1", "g2"], "d",
        )
    end

    @testset "resolver result length mismatch fails" begin
        for T in (MockShortIdsType, MockShortNamesType)
            err = try
                IOM.register_component_axis!(_test_registry(), sys, T, ["x", "y"], "d")
            catch e
                sprint(showerror, e)
            end
            @test occursin(string(nameof(T)), err)
            @test occursin("for 2 labels", err)
        end
    end

    @testset "componentless" begin
        key = IOM.VariableKey(TestVariableType, MockComponentType)
        @test IOM.is_componentless(key, [[IOM.encode_key_as_string(key)]])
        @test !IOM.is_componentless(key, [["g1"]])
    end

    @testset "extra columns are label axes" begin
        reg = _test_registry()
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
        )
        keys = IOM.dataset_axes!(
            reg, sys, key, [["a1", "a2"], ["b1"], ["t1", "t2", "t3"]], "Name__X",
        )
        @test keys == ["MockComponentType", "MockComponentType2", "name__x"]
        @test reg.axes["name__x"] == IS.LabelAxis("label", ["t1", "t2", "t3"])
        keys = IOM.dataset_axes!(
            reg, sys, key, [["a1", "a2"], ["b1"], ["t1"], ["u1", "u2"]], "Other",
        )
        @test keys[3:4] == ["other_3", "other_4"]
        @test_throws ArgumentError IOM.dataset_axes!(
            reg, sys, key, [["a1", "a2"], ["b1"], ["t9"]], "Name__X",
        )
    end

    @testset "three d two axes" begin
        reg = _test_registry()
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
        )
        array = DenseAxisArray(zeros(2, 3, 4), ["s1", "s2"], ["g1", "g2", "g3"], 1:4)
        labels = IOM.column_labels(key, array)
        @test labels == [["s1", "s2"], ["g1", "g2", "g3"]]
        keys = IOM.dataset_axes!(reg, sys, key, labels, "d")
        @test keys == ["MockComponentType", "MockComponentType2"]
        @test length(reg.axes) == 2
    end
end
