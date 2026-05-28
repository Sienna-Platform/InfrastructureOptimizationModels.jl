@testset "FlowSign trait" begin
    @testset "multiplier_from_sign maps to expected ±1.0" begin
        @test IOM.multiplier_from_sign(IOM.FlowInjection) == 1.0
        @test IOM.multiplier_from_sign(IOM.FlowWithdrawal) == -1.0
        @test IOM.multiplier_from_sign(IOM.FlowUndirected) == 1.0
    end

    @testset "flow_sign defaults to FlowUndirected" begin
        # Any VariableType without an explicit override falls back to FlowUndirected.
        # OnVariable / StartVariable / etc. have no flow semantics by design.
        @test IOM.flow_sign(OnVariable) === IOM.FlowUndirected
        @test IOM.flow_sign(StartVariable) === IOM.FlowUndirected
        @test IOM.flow_sign(StopVariable) === IOM.FlowUndirected
    end

    @testset "Standard active-power variables carry the correct sign" begin
        @test IOM.flow_sign(ActivePowerVariable) === IOM.FlowInjection
        @test IOM.flow_sign(ActivePowerInVariable) === IOM.FlowWithdrawal
        @test IOM.flow_sign(ActivePowerOutVariable) === IOM.FlowInjection
    end

    @testset "Roundtrip: multiplier_from_sign ∘ flow_sign" begin
        @test IOM.multiplier_from_sign(IOM.flow_sign(ActivePowerVariable)) == 1.0
        @test IOM.multiplier_from_sign(IOM.flow_sign(ActivePowerInVariable)) == -1.0
        @test IOM.multiplier_from_sign(IOM.flow_sign(ActivePowerOutVariable)) == 1.0
        @test IOM.multiplier_from_sign(IOM.flow_sign(OnVariable)) == 1.0
    end
end
