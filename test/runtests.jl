using Test
using InfrastructureOptimizationModels
using Logging

const BASE_DIR = joinpath(@__DIR__, "..")

# Import InfrastructureSystems for logging utilities
using InfrastructureSystems
const IS = InfrastructureSystems

# Code Quality Tests
import Aqua
@testset "Code Quality (Aqua.jl)" begin
    Aqua.test_all(InfrastructureOptimizationModels; persistent_tasks = false)
end

# Load the test module
include("InfrastructureOptimizationModelsTests.jl")

# Run the test suite
logger = global_logger()

try
    InfrastructureOptimizationModelsTests.run_tests()
finally
    # Guarantee that the global logger is reset
    global_logger(logger)
end
