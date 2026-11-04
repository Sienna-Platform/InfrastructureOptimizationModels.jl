
@testset "Outputs descriptors" begin
    @testset "dataset descriptor from key" begin
        d = IOM.dataset_descriptor(
            IOM.VariableKey(TestVariableType, MockComponentType);
            model = "m",
            axes = ["execution", "MockComponentType", "horizon"],
            axis_lengths = [3, 2, 4],
        )
        @test d.entry_type == "TestVariableType"
        @test d.component_type == "MockComponentType"
        @test d.container == "variables"
        @test d.layout.chunks == [1, 2, 4]
        @test d.path == "/m/datasets/TestVariableType__MockComponentType"
    end

    @testset "pair key component type" begin
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
        )
        d = IOM.dataset_descriptor(
            key;
            model = "m",
            axes = ["execution", "horizon"],
            axis_lengths = [3, 4],
        )
        @test d.component_type == "MockComponentType"
    end

    params = IOM.ModelStoreParams(
        3,
        4,
        Millisecond(Hour(24)),
        Millisecond(Hour(1)),
        100.0,
        Base.UUID(UInt128(1)),
        IOM.OptimizationContainerMetadata(),
    )

    @testset "model axes decision" begin
        (axes, data) =
            IOM.model_axes(IOM.DecisionModelStore(), params, DateTime(2024, 1, 1), "m")
        @test Set(keys(axes)) == Set(["execution", "horizon"])
        @test axes["execution"].interval == "P1D"
        @test axes["execution"].count == 3
        @test axes["horizon"].horizon_count == 4
        @test data[("m", "execution")].timestamps ==
              [DateTime(2024, 1, 1), DateTime(2024, 1, 2), DateTime(2024, 1, 3)]
        @test IOM.model_type_name(IOM.DecisionModelStore()) == "decision"
    end

    @testset "model axes emulation" begin
        (axes, data) =
            IOM.model_axes(IOM.EmulationModelStore(), params, DateTime(2024, 1, 1), "m")
        @test Set(keys(axes)) == Set(["time"])
        @test axes["time"].length == 3
        @test axes["time"].resolution == "PT1H"
        @test data[("m", "time")].timestamps ==
              [DateTime(2024, 1, 1, h) for h in 0:2]
        @test IOM.model_type_name(IOM.EmulationModelStore()) == "emulation"
    end

    @testset "settings dict" begin
        sys = MockSystem(100.0, false)
        s = IOM.Settings(sys; horizon = Hour(48), interval = Hour(24), resolution = Hour(1))
        d = IOM.settings_dict(s)
        @test d["horizon"] == "P2D"
        @test d["interval"] == "P1D"
        @test d["resolution"] == "PT1H"
        @test !haskey(d, "optimizer")
        s2 = IOM.Settings(
            sys;
            horizon = Hour(48),
            interval = Hour(24),
            resolution = Hour(1),
            optimizer = HiGHS.Optimizer,
        )
        @test occursin("HiGHS", IOM.settings_dict(s2)["optimizer"])
    end

    @testset "component ids stub" begin
        @test_throws ArgumentError IOM.component_ids(
            MockSystem(),
            MockComponentType,
            ["a"],
        )
    end

    @testset "lower snake" begin
        @test IOM._lower_snake("ThermalStandard") == "thermalstandard"
        @test IOM._lower_snake("a-b c") == "a_b_c"
        @test IOM._lower_snake("1x") == "k1x"
        @test IOM._lower_snake("_x") == "k_x"
    end
end
