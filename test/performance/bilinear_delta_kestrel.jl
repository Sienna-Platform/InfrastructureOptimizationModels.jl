"""
Parallel (Kestrel/SLURM) entry point for the bilinear delta benchmark.

NLP references run sequentially first to establish the reference objective, then MIP
method × refinement combinations are farmed out to separate Julia worker processes
(up to KESTREL_MAX_WORKERS concurrent). Each worker writes JSON results that the
coordinator aggregates into the summary table. Always uses Xpress for MIP solves.

Workers are spawned by re-executing THIS file with `--worker`; that path is why the
worker subprocess gets all the shared model code (this file `include`s the common
module, same as the coordinator).

Usage (on Kestrel):
    julia --project=test test/performance/bilinear_delta_kestrel.jl [options]

Options:
    -N, --nodes INT          number of nodes (default 10, must be even)
    -K, --cost INT           number of PWL cost segments per generator (default 3)
    -S, --seed INT           random seed for network generation (default 42)
    -R, --refinements INT... refinement levels (default 4 6 8)

For the sequential (local + CI/CD) runner, see `bilinear_delta_local.jl`.
"""

include("bilinear_delta_common.jl")

using ArgParse
using JSON3
using Ipopt
using UnoSolver

import Pkg
Pkg.add("Xpress")
using Xpress

const LP_OPT = Xpress.Optimizer
const KESTREL_XPRESS_THREADS = 20
const KESTREL_MAX_WORKERS = 5

# ─── Parallel benchmark ───────────────────────────────────────────────────────

"""Return the results directory path, creating it if needed."""
function results_dir()
    dir = joinpath(@__DIR__, "results")
    mkpath(dir)
    return dir
end

"""
    run_benchmark_parallel(; N, K, seed, refinements)

Run the benchmark with parallel MIP solves on Kestrel. NLP references run
sequentially first to establish nlp_obj, then MIP method×refinement
combinations are farmed out to separate Julia worker processes (capped at
KESTREL_MAX_WORKERS concurrent).
"""
function run_benchmark_parallel(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    refinements::Vector{Int} = [4, 6, 8],
)
    net, logfile = setup_run(; N, K, seed)

    mip_methods = bilinear_methods

    print_table_header()

    # ── Phase 1: NLP references (sequential, no time limit) ──
    nlp_obj = NaN
    nlp_results = []
    for (label, opt_fn) in [
        ("NLP (Ipopt)", Ipopt.Optimizer),
        (
            "NLP (Uno)",
            optimizer_with_attributes(UnoSolver.Optimizer, "preset" => "filtersqp"),
        ),
    ]
        r = run_single_case(;
            optimizer = opt_fn,
            net,
            bilinear_config = IOM.NoBilinearApproxConfig(),
            quad_config = IOM.NoQuadApproxConfig(),
            refinement = 0,
            label,
            is_exact = true,
            nlp_obj,
            logfile,
            time_limit = NaN,
            threads = 0,
        )
        if r.solved
            nlp_obj = r.obj
        end
        print_result_row(r)
        push!(nlp_results, r)
        println()
    end
    flush(logfile)

    # ── Phase 2: MIP methods in parallel via worker processes ──
    res_dir = results_dir()

    # Build the list of (method_index, refinement) jobs
    jobs = Tuple{Int, Int}[]
    for (mi, _) in enumerate(mip_methods)
        for ref in refinements
            push!(jobs, (mi, ref))
        end
    end

    println(
        "Launching $(length(jobs)) MIP workers (max $KESTREL_MAX_WORKERS concurrent)...",
    )
    flush(stdout)

    script_path = @__FILE__
    project_dir = joinpath(@__DIR__, "..", "..")
    project_arg = joinpath(project_dir, "test")

    # Use a semaphore to cap concurrency
    sem = Base.Semaphore(KESTREL_MAX_WORKERS)
    tasks = Task[]

    for (mi, ref) in jobs
        label = mip_methods[mi][1]
        outfile =
            joinpath(res_dir, "$(SLURM_JOB_ID)_$(replace(label, " " => "_"))_R$(ref).json")
        cmd = Cmd(`julia --project=$project_arg $script_path
            --worker
            --method-index $mi
            --refinement-single $ref
            --nlp-obj $nlp_obj
            --output-file $outfile
            --nodes $N
            --cost $K
            --seed $seed`)

        t = @task begin
            Base.acquire(sem)
            try
                proc = Base.run(pipeline(cmd; stdout = logfile, stderr = stderr); wait = false)
                wait(proc)
                if !success(proc)
                    @warn "Worker $label R=$ref exited with non-zero status"
                end
            catch e
                @warn "Worker $label R=$ref failed" exception = (e, catch_backtrace())
            finally
                Base.release(sem)
            end
        end
        schedule(t)
        push!(tasks, t)
    end

    # Wait for all workers
    for t in tasks
        wait(t)
    end

    isopen(logfile) && (flush(logfile); close(logfile))

    # ── Phase 3: Aggregate results ──
    mip_results = BenchmarkResult[]
    for (mi, ref) in jobs
        label = mip_methods[mi][1]
        outfile =
            joinpath(res_dir, "$(SLURM_JOB_ID)_$(replace(label, " " => "_"))_R$(ref).json")
        if isfile(outfile)
            d = open(outfile) do io
                JSON3.read(io, Dict{String, Any})
            end
            push!(mip_results, from_dict(d))
        else
            @warn "Missing results for $label R=$ref"
            push!(
                mip_results,
                BenchmarkResult(;
                    label,
                    refinement = ref,
                    status = "WORKER_FAILED",
                    nlp_obj,
                ),
            )
        end
    end

    # Print MIP results grouped by method
    for (mi, (label, _)) in enumerate(mip_methods)
        for r in mip_results
            if r.label == label
                print_result_row(r)
            end
        end
        println()
    end
    println("="^RULE_WIDTH)
    flush(stdout)
end

# ─── Worker entry point ──────────────────────────────────────────────────────

"""
Run a single MIP benchmark case as a worker process and write JSON results.
Called when the script is invoked with --worker.
"""
function run_worker(parsed)
    mi = parsed["method-index"]
    ref = parsed["refinement-single"]
    nlp_obj = parsed["nlp-obj"]
    outfile = parsed["output-file"]
    N = parsed["nodes"]
    K = parsed["cost"]
    seed = parsed["seed"]

    net = generate_network(; N, K, seed)

    label, method = bilinear_methods[mi]
    bilin_config, quad_config = method(ref)

    log_dir = joinpath(@__DIR__, "logs")
    mkpath(log_dir)
    safe_label = replace(label, " " => "_")
    worker_log_path = joinpath(log_dir, "solver_$(log_tag())_$(safe_label)_R$(ref).log")
    logfile = open(worker_log_path, "a")

    try
        r = run_single_case(;
            optimizer = LP_OPT,
            net,
            bilinear_config = bilin_config,
            quad_config,
            refinement = ref,
            label,
            is_exact = false,
            nlp_obj,
            logfile,
            time_limit = MIP_TIME_LIMIT_SEC,
            threads = KESTREL_XPRESS_THREADS,
        )

        mkpath(dirname(outfile))
        open(outfile, "w") do io
            JSON3.write(io, to_dict(r))
        end

        @info "Worker done: $label R=$ref status=$(r.status) obj=$(r.obj) log=$worker_log_path"
    finally
        flush(logfile)
        close(logfile)
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
        "--worker"
        action = :store_true
        help = "run as a parallel worker process (Kestrel)"
        "--method-index"
        arg_type = Int
        default = 0
        help = "1-based index into bilinear_methods (worker mode)"
        "--refinement-single"
        arg_type = Int
        default = 0
        help = "single refinement level (worker mode)"
        "--nlp-obj"
        arg_type = Float64
        default = NaN
        help = "NLP reference objective for gap calculation (worker mode)"
        "--output-file"
        arg_type = String
        default = ""
        help = "path to write JSON results (worker mode)"
    end
    return parse_args(s)
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        parsed = parse_commandline()

        if parsed["worker"]
            run_worker(parsed)
        else
            N = parsed["nodes"]
            K = parsed["cost"]
            seed = parsed["seed"]
            refinements = parsed["refinements"]
            run_benchmark_parallel(; N, K, seed, refinements)
        end
    catch e
        @error "Benchmark failed" exception = (e, catch_backtrace())
    finally
        flush(stdout)
        flush(stderr)
    end
end
