# Interface conformance for the approximation method families.
#
# `add_quadratic_approx!` / `add_bilinear_approx!` are uniform-arity interfaces:
# every concrete config implements exactly one method dispatching only on the config
# type. A config that fails to implement it falls through to the error fallback defined
# on the abstract supertype. These tests assert each concrete config provides a real
# method (i.e. `which` resolves to something other than the fallback) and that the
# fallback fires for an unimplemented config.

struct _DummyQuadConfig <: IOM.QuadraticApproxConfig end
struct _DummyBilinearConfig <: IOM.BilinearApproxConfig end

# The fallback methods, captured by resolving a config-only call (no concrete method
# matches a 1-arg call, so `which` returns the abstract-typed fallback).
const _QUAD_FALLBACK = which(IOM.add_quadratic_approx!, Tuple{_DummyQuadConfig})
const _BILINEAR_FALLBACK = which(IOM.add_bilinear_approx!, Tuple{_DummyBilinearConfig})

# Canonical interface signatures (`x_var`/`y_var` are duck-typed, hence `Any`).
function _implements_quadratic(cfg)
    tup = Tuple{
        typeof(cfg), IOM.OptimizationContainer, Type{MockThermalGen},
        Vector{String}, UnitRange{Int}, Any, Vector{IOM.MinMax}, String,
    }
    return which(IOM.add_quadratic_approx!, tup) !== _QUAD_FALLBACK
end

function _implements_bilinear(cfg)
    tup = Tuple{
        typeof(cfg), IOM.OptimizationContainer, Type{MockThermalGen},
        Vector{String}, UnitRange{Int}, Any, Any,
        Vector{IOM.MinMax}, Vector{IOM.MinMax}, String,
    }
    return which(IOM.add_bilinear_approx!, tup) !== _BILINEAR_FALLBACK
end

@testset "Approximation interface conformance" begin
    quad_configs = [
        IOM.NoQuadApproxConfig(),
        IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 2),
        IOM.SOS2QuadConfig{IOM.ManualBackend}(; depth = 2),
        IOM.SawtoothQuadConfig(; depth = 2),
        IOM.EpigraphQuadConfig(; depth = 2),
        IOM.NMDTQuadConfig{IOM.SingleNMDT}(; depth = 2),
        IOM.NMDTQuadConfig{IOM.DoubleNMDT}(; depth = 2),
    ]
    @testset "quadratic configs implement add_quadratic_approx!" begin
        for cfg in quad_configs
            @test _implements_quadratic(cfg)
        end
    end

    bilinear_configs = [
        IOM.NoBilinearApproxConfig(),
        IOM.Bin2Config(IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 2)),
        IOM.HybSConfig(
            IOM.SOS2QuadConfig{IOM.SolverBackend}(; depth = 2);
            cross_term_depth = 2,
        ),
        IOM.NMDTBilinearConfig{IOM.SingleNMDT}(; depth = 2),
        IOM.NMDTBilinearConfig{IOM.DoubleNMDT}(; depth = 2),
    ]
    @testset "bilinear configs implement add_bilinear_approx!" begin
        for cfg in bilinear_configs
            @test _implements_bilinear(cfg)
        end
    end

    @testset "fallback errors for unimplemented configs" begin
        @test_throws ErrorException IOM.add_quadratic_approx!(_DummyQuadConfig())
        @test_throws ErrorException IOM.add_bilinear_approx!(_DummyBilinearConfig())
    end
end
