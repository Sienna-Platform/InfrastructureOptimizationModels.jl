"""
Sequential (single-process) entry point for the bilinear delta benchmark.

Runs every approximation method × refinement in one Julia process. Used locally and
in CI/CD. MIP solves get a 1-hour time limit; NLP references (Ipopt/Uno) have none.

The solver is chosen by environment: HiGHS on CI/CD (so Xpress need not be
installed), Xpress when run locally.

Usage:
    julia --project=test test/performance/bilinear_delta_local.jl [options]

Options:
    -N, --nodes INT          number of nodes (default 10, must be even)
    -K, --cost INT           number of PWL cost segments per generator (default 3)
    -S, --seed INT           random seed for network generation (default 42)
    -R, --refinements INT... refinement levels (default 4 6 8)

For the parallel (Kestrel/SLURM) runner, see `bilinear_delta_kestrel.jl`.
"""

include("bilinear_delta_common.jl")

using ArgParse
using Ipopt
using UnoSolver

const IS_CI = get(ENV, "CI", "") == "true" || haskey(ENV, "GITHUB_ACTIONS")

# We use Pkg.add for Xpress because precompilation will fail on CI/CD if Xpress is
# in the manifest, so CI must run on HiGHS instead.
const LP_OPT = if IS_CI
    @eval using HiGHS
    HiGHS.Optimizer
else
    @eval import Pkg
    Pkg.add("Xpress")
    Pkg.add("Xpress_jll")
    @eval import Xpress_jll
    ENV["XPRESS_JL_LIBRARY"] = Xpress_jll.libxprs
    @eval using Xpress
    Xpress.Optimizer
end

# ─── Sequential benchmark ─────────────────────────────────────────────────────

"""
    run_benchmark(; N, K, seed, refinements)

Run the full benchmark sequentially. MIP solves get a 1-hour time limit.
NLP references (Ipopt/Uno) have no time limit.
"""
function run_benchmark(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    refinements::Vector{Int} = [4, 6, 8],
)
    net, logfile = setup_run(; N, K, seed)

    all_methods = [
        ("NLP (Ipopt)", exact),
        ("NLP (Uno)", exact),
        bilinear_methods...,
    ]

    print_table_header()

    nlp_obj = NaN

    try
        for (label, method) in all_methods
            is_exact = method === exact
            refs = is_exact ? [0] : refinements

            for ref in refs
                bilin_config, quad_config = method(ref)
                opt = if is_exact && contains(label, "Ipopt")
                    Ipopt.Optimizer
                elseif is_exact && contains(label, "Uno")
                    optimizer_with_attributes(
                        UnoSolver.Optimizer,
                        "preset" => "filtersqp",
                    )
                else
                    LP_OPT
                end

                r = run_single_case(;
                    optimizer = opt,
                    net,
                    bilinear_config = bilin_config,
                    quad_config,
                    refinement = ref,
                    label,
                    is_exact,
                    nlp_obj,
                    logfile,
                    time_limit = MIP_TIME_LIMIT_SEC,
                    threads = 0,
                )

                if r.solved && is_exact
                    nlp_obj = r.obj
                end

                print_result_row(r)
            end
            println()
        end
        println("="^RULE_WIDTH)
    finally
        isopen(logfile) && (flush(logfile); close(logfile))
        flush(stdout)
    end
end

# ─── Entry point ──────────────────────────────────────────────────────────────

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--nodes", "-N"
        arg_type = Int
        default = 10
        help = "number of nodes (must be even)"
        "--cost", "-K"
        arg_type = Int
        default = 3
        help = "number of PWL cost segments per generator"
        "--seed", "-S"
        arg_type = Int
        default = 42
        help = "random seed for network generation"
        "--refinements", "-R"
        arg_type = Int
        nargs = '+'
        default = [4, 6, 8]
        help = "refinement levels (list of integers)"
    end
    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        parsed = parse_commandline()
        N = parsed["nodes"]
        K = parsed["cost"]
        seed = parsed["seed"]
        refinements = parsed["refinements"]

        # Warm up compilation on a tiny network so the real run is faster.
        redirect_stdout(devnull) do
            run_benchmark(; N = 2, K = 1, seed = 0, refinements = [1])
        end
        run_benchmark(; N, K, seed, refinements)
    catch e
        @error "Benchmark failed" exception = (e, catch_backtrace())
    finally
        flush(stdout)
        flush(stderr)
    end
end
