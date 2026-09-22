#!/bin/zsh
#
# Run tests with coverage and generate lcov.info for Coverage Gutters.
# Usage: ./scripts/test_with_coverage.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Running tests with coverage..."
julia --project=. --code-coverage -e '
    using TestEnv; TestEnv.activate()
    include("test/load_tests.jl")
    InfrastructureOptimizationModelsTests.run_tests()
'

echo "==> Generating lcov.info..."
# Coverage.jl caps HTTP at 1.x while the OpenAPI packages need 2.x, so it has its own env.
julia --project=scripts/coverage -e 'using Pkg; Pkg.instantiate(); include("scripts/generate_lcov.jl")'

echo "==> Done. lcov.info written to $(pwd)/lcov.info"
