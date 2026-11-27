
@testset "Outputs descriptors" begin
    @testset "dataset descriptor from key" begin
        d = IOM.dataset_descriptor(
            IOM.VariableKey(TestVariableType, MockComponentType);
            axes = IS.OutputAxis[
                IS.ExecutionAxis(DateTime(2024, 1, 1), Day(1), 3),
                IS.ComponentAxis([IS.ComponentAxisColumn("MockComponentType", true)], 2),
                IS.HorizonAxis(Hour(1), 4),
            ],
        )
        @test IS.get_entry_type(d) == "TestVariableType"
        @test IS.get_component_type(d) == "MockComponentType"
        @test IS.get_container(d) == "variables"
        @test IS.get_chunks(d) == [1, 2, 4]
        @test IS.dataset_key(d) == ("TestVariableType", "MockComponentType", "")
        @test IS.get_axis_names(d) == ["execution", "MockComponentType", "horizon"]
    end

    @testset "pair key component type" begin
        key = IOM.VariableKey(
            TestVariableType,
            IOM.ComponentPairKey{MockComponentType, MockComponentType2},
        )
        d = IOM.dataset_descriptor(
            key;
            axes = IS.OutputAxis[
                IS.ExecutionAxis(DateTime(2024, 1, 1), Day(1), 3),
                IS.HorizonAxis(Hour(1), 4),
            ],
        )
        @test IS.get_component_type(d) == "MockComponentType"
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
        axes = IOM.model_axes(IOM.DecisionModelStore(), params, DateTime(2024, 1, 1))
        @test [IS.axis_name(a) for a in axes] == ["execution", "horizon"]
        @test IS.get_interval(axes[1]) == Millisecond(Day(1))
        @test IS.axis_length(axes[1]) == 3
        @test IS.axis_length(axes[2]) == 4
        @test IS.get_initial_timestamp(axes[1]) == DateTime(2024, 1, 1)
        @test IOM.model_type_name(IOM.DecisionModelStore()) == "decision"
    end

    @testset "model axes emulation" begin
        axes = IOM.model_axes(IOM.EmulationModelStore(), params, DateTime(2024, 1, 1))
        @test [IS.axis_name(a) for a in axes] == ["time"]
        @test IS.axis_length(axes[1]) == 3
        @test IS.get_resolution(axes[1]) == Millisecond(Hour(1))
        @test IS.get_initial_timestamp(axes[1]) == DateTime(2024, 1, 1)
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

    @testset "resolve component axis stub" begin
        @test_throws ArgumentError IOM.resolve_component_axis(
            nothing,
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
