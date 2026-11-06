import InfrastructureOptimizationModels: ModelInternal
@testset "Test Model Internal" begin
    internal = ModelInternal(
        MockContainer(),
    )
    @test IOM.get_status(internal) == IOM.ModelBuildStatus.EMPTY
    IOM.set_initial_conditions_model_container!(
        internal,
        MockContainer(),
    )
    @test isa(
        IOM.get_initial_conditions_model_container(internal),
        MockContainer,
    )
    IOM.add_recorder!(internal, :MockRecorder)
    @test IOM.get_recorders(internal)[1] == :MockRecorder
    IOM.set_status!(internal, IOM.ModelBuildStatus.BUILT)
    @test IOM.get_status(internal) == IOM.ModelBuildStatus.BUILT
    IOM.set_output_dir!(internal, mktempdir())
    log_config = IOM.configure_logging(internal, "test_log.log", "a")
    @test !isempty(log_config.loggers)
end

@testset "Released flag" begin
    internal = ModelInternal(MockContainer())
    @test IOM.is_released(internal) == false
    IOM.set_released!(internal, true)
    @test IOM.is_released(internal) == true
    IOM.set_released!(internal, false)
    @test IOM.is_released(internal) == false
end

struct ReleaseTestTemplate <: IOM.AbstractProblemTemplate end
IOM.finalize_template!(::ReleaseTestTemplate, args...) = nothing
IOM.get_network_model(::ReleaseTestTemplate) = nothing
struct ReleaseTestProblem <: IOM.AbstractOptimizationProblem end
IOM.validate_time_series!(::IOM.DecisionModel{ReleaseTestProblem}) = nothing

@testset "A released model refuses its JuMP model" begin
    model = IOM.DecisionModel{ReleaseTestProblem}(
        ReleaseTestTemplate(),
        MockSystem(100.0, false);
        name = "Released",
    )
    @test !IOM.is_released(model)
    @test IOM.get_jump_model(model) isa JuMP.Model
    IOM.set_released!(model, true)
    @test IOM.is_released(model)
    err = try
        IOM.get_jump_model(model)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Released", err.msg)
    @test occursin("release_model = true", err.msg)
    @test IOM.get_jump_model(IOM.get_optimization_container(model)) isa JuMP.Model
    @test_throws ArgumentError IOM.solve_model!(model)
    IOM.set_released!(model, false)
    @test IOM.get_jump_model(model) isa JuMP.Model
end
