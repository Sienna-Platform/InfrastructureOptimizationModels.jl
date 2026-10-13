@testset "DeviceModel Tests" begin
    @test_throws ArgumentError DeviceModel(ThermalGen, ThermalStandardUnitCommitment)
    @test_throws ArgumentError DeviceModel(ThermalStandard, IOM.AbstractThermalFormulation)
    @test_throws ArgumentError NetworkModel(AbstractNetworkModel)
end

@testset "NetworkModel Tests" begin
    @test_throws ArgumentError NetworkModel(AbstractNetworkModel)
    @test NetworkModel(PTDFPowerModel; use_slacks = true) isa NetworkModel

    ec_multi = EvaluationContainer()
    add_evaluator!(ec_multi, DCPowerFlow, DCPowerFlow())
    add_evaluator!(ec_multi, PSSEExportPowerFlow, PSSEExportPowerFlow(:v33, "exports"))
    @test length(ec_multi) == 2
    @test haskey(ec_multi, PSSEExportPowerFlow)
end

@testset "Feedforward Struct Tests" begin
    ffs = [
        UpperBoundFeedforward(;
            component_type = RenewableDispatch,
            source = ActivePowerVariable,
            affected_values = [ActivePowerVariable],
            add_slacks = true,
        ),
        LowerBoundFeedforward(;
            component_type = RenewableDispatch,
            source = ActivePowerVariable,
            affected_values = [ActivePowerVariable],
            add_slacks = true,
        ),
        SemiContinuousFeedforward(;
            component_type = ThermalMultiStart,
            source = OnVariable,
            affected_values = [ActivePowerVariable, ReactivePowerVariable],
        ),
    ]

    for ff in ffs
        for av in IOM.get_affected_values(ff)
            @test isa(av, IOM.VariableKey)
        end
    end

    ff = FixValueFeedforward(;
        component_type = HydroDispatch,
        source = OnVariable,
        affected_values = [OnStatusParameter],
    )

    for av in IOM.get_affected_values(ff)
        @test isa(av, IOM.ParameterKey)
    end

    @test_throws ErrorException UpperBoundFeedforward(
        component_type = RenewableDispatch,
        source = ActivePowerVariable,
        affected_values = [OnStatusParameter],
        add_slacks = true,
    )

    @test_throws ErrorException LowerBoundFeedforward(
        component_type = RenewableDispatch,
        source = ActivePowerVariable,
        affected_values = [OnStatusParameter],
        add_slacks = true,
    )

    @test_throws ErrorException SemiContinuousFeedforward(
        component_type = ThermalMultiStart,
        source = OnVariable,
        affected_values = [ActivePowerVariable, OnStatusParameter],
    )
end
