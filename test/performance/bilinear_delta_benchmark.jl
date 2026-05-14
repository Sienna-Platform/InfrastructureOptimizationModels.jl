"""
Benchmark script for the bilinear delta model described in `bilinear_delta_model.tex`.

Builds a lossy network OPF with delta (incremental) PWL costs and bilinear V·I power
balance constraints. Compares bilinear approximation methods from
InfrastructureOptimizationModels against an exact NLP reference (Ipopt).

The lossy generator model is P = V·I − a·I² − b·I − c.

Usage:
    julia --project=test test/performance/bilinear_delta_benchmark.jl [options]

Options:
    -N, --nodes INT          number of nodes (default 10, must be even)
    -K, --cost INT           number of PWL cost segments per generator (default 3)
    -S, --seed INT           random seed for network generation (default 0)
    -R, --refinements INT... refinement levels (default 4 6 8)
    -B, --build-only         don't solve, only build the model

On Kestrel (SLURM), MIP solves run in parallel across separate Julia processes
(up to KESTREL_MAX_WORKERS concurrent). Each worker writes JSON results that the
coordinator aggregates into the summary table.
"""

using InfrastructureOptimizationModels
using ArgParse
using JuMP
using Dates
using Random
using Printf
using JSON3

ENVIRONMENT = if get(ENV, "CI", "") == "true" || haskey(ENV, "GITHUB_ACTIONS")
    :github
elseif get(ENV, "HOSTNAME", "") == "kl1" || haskey(ENV, "SLURM_NODEID")
    :kestrel
else
    :local
end

# We use Pkg.add here because precompilation will fail on CI/CD if Xpress is
# in the manifest.
LP_OPT = if ENVIRONMENT == :github
    @eval using HiGHS
    HiGHS.Optimizer
elseif ENVIRONMENT == :kestrel
    @eval import Pkg
    Pkg.add("Xpress")
    @eval using Xpress
    Xpress.Optimizer
else
    @eval import Pkg
    Pkg.add("Xpress")
    Pkg.add("Xpress_jll")
    @eval import Xpress_jll
    ENV["XPRESS_JL_LIBRARY"] = Xpress_jll.libxprs
    @eval using Xpress
    Xpress.Optimizer
end

using Ipopt
using UnoSolver

const IOM = InfrastructureOptimizationModels
const IS = IOM.IS

const MIP_TIME_LIMIT_SEC = 600.0
const KESTREL_XPRESS_THREADS = 20
const KESTREL_MAX_WORKERS = 5
const SLURM_JOB_ID = get(ENV, "SLURM_JOB_ID", "")

include("../mocks/mock_system.jl")
include("../mocks/mock_components.jl")

struct MockPowerEqualityConstraint <: IOM.ConstraintType end
struct MockKCLConstraint <: IOM.ConstraintType end
struct MockLineLossConstraint <: IOM.ConstraintType end

struct MockKCLExpression <: IOM.ExpressionType end
struct MockLineLossExpression <: IOM.ExpressionType end

struct MockLineLossAuxVariable <: IOM.AuxVariableType end
function IOM.calculate_aux_variable_value!(
    container::OptimizationContainer,
    key::AuxVarKey{MockLineLossAuxVariable, MockNetworkNode},
    system::MockSystem,
)
    cont = get_aux_variable(container, key)
    names = axes(cont, 1)
    time_steps = get_time_steps(container)

    Isq_container =
        get_expression(container, IOM.BilinearProductExpression, MockNetworkNode, "gen")
    I_container =
        get_expression(container, IOM.QuadraticExpression, MockNetworkNode, "gen_I_sq")

    for name in names, t in time_steps
        Isq = JuMP.value(Isq_container[name, t])
        I = JuMP.value(I_container[name, t])
        a, b, c = get_component(MockNetworkNode, system, name).loss
        cont[name, t] = -a * Isq - b * I - c
    end

    return
end

struct MockNetworkProblem
    N::Int
    K::Int
    gen_nodes::Vector{String}
    dem_nodes::Vector{String}
    all_nodes::Vector{String}
    edges::Vector{Tuple{String, String}}
    conductances::Dict{Tuple{String, String}, Float64}
    demands::Dict{String, Float64}
    marginal_costs::Dict{String, Vector{Float64}}
    segment_widths::Dict{String, Vector{Float64}}
    loss::Dict{String, Vector{Float64}}
end

# ─── Network generation ──────────────────────────────────────────────────────

"""
Generate a lossy network with per-generator loss coefficients a, b, c ∈ [0, 0.01].
"""
function generate_network(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
)
    @assert iseven(N) "N must be even"
    rng = MersenneTwister(seed)
    all_nodes = ["n$(i)" for i in 1:N]
    gen_nodes = all_nodes[1:(N ÷ 2)]
    dem_nodes = all_nodes[(N ÷ 2 + 1):N]

    edges = Tuple{String, String}[]
    conductances = Dict{Tuple{String, String}, Float64}()

    # Random spanning tree
    perm = shuffle(rng, 1:N)
    for idx in 1:(N - 1)
        a, b = all_nodes[perm[idx]], all_nodes[perm[idx + 1]]
        e = a < b ? (a, b) : (b, a)
        if e ∉ edges
            push!(edges, e)
            conductances[e] = 1.0 + 4.0 * rand(rng)
        end
    end

    # Extra edges for density
    for _ in 1:(N ÷ 2)
        i, j = rand(rng, 1:N), rand(rng, 1:N)
        i == j && continue
        a, b = all_nodes[i], all_nodes[j]
        e = a < b ? (a, b) : (b, a)
        if e ∉ edges
            push!(edges, e)
            conductances[e] = 1.0 + 4.0 * rand(rng)
        end
    end

    demands = Dict(d => 0.05 + 0.1 * rand(rng) for d in dem_nodes)

    marginal_costs = Dict{String, Vector{Float64}}()
    segment_widths = Dict{String, Vector{Float64}}()
    for g in gen_nodes
        mc = sort(rand(rng, K) .* 10.0 .+ 1.0)
        marginal_costs[g] = mc
        widths = rand(rng, K) .+ 0.1
        widths .*= 1.5 / sum(widths)
        segment_widths[g] = widths
    end

    loss = Dict(g => 0.01 * rand(rng, 3) for g in gen_nodes)

    return MockNetworkProblem(
        N, K, gen_nodes, dem_nodes, all_nodes,
        edges, conductances, demands,
        marginal_costs, segment_widths,
        loss,
    )
end

"""Build adjacency list from edge set."""
function adjacency_list(net::MockNetworkProblem)
    adj = Dict{String, Vector{Tuple{String, Float64}}}()
    for n in net.all_nodes
        adj[n] = Tuple{String, Float64}[]
    end
    for (a, b) in net.edges
        g = net.conductances[(a, b)]
        push!(adj[a], (b, g))
        push!(adj[b], (a, g))
    end
    return adj
end

# ─── Model constants ─────────────────────────────────────────────────────────

const V_MIN = 0.8
const V_MAX = 1.2
const I_GEN_MIN = 0.0
const I_GEN_MAX = 1.0
const I_DEM_MIN = -1.0
const I_DEM_MAX = 0.0
const P_MAX = 1.5

# ─── IOM container setup ─────────────────────────────────────────────────────

function make_container(optimizer)
    system = MockSystem(100.0)
    settings = IOM.Settings(
        system;
        horizon = Dates.Hour(1),
        resolution = Dates.Hour(1),
        warm_start = false,
        optimizer,
        optimizer_solve_log_print = true,
    )
    container = IOM.OptimizationContainer(system, settings, JuMP.Model(), IS.Deterministic)
    IOM.set_time_steps!(container, 1:1)
    IOM.init_optimization_container!(container, NetworkModel(TestPowerModel), system)
    return container, system
end

# ─── Dispatched gen bilinear construction ─────────────────────────────────────

"""
Separable methods: precompute V² and I², call wrapper's precomputed overload.
I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    bilinear_config::Union{IOM.Bin2Config, IOM.HybSConfig}, quad_config,
)
    V_sq = IOM.add_quadratic_approx!(
        quad_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        V_container,
        V_MIN,
        V_MAX,
        "gen_V_sq",
    )
    I_sq = IOM.add_quadratic_approx!(
        quad_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        I_container,
        I_GEN_MIN,
        I_GEN_MAX,
        "gen_I_sq",
    )
    z_gen = IOM.add_bilinear_approx!(
        bilinear_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        V_sq,
        I_sq,
        V_container,
        I_container,
        V_MIN,
        V_MAX,
        I_GEN_MIN,
        I_GEN_MAX,
        "gen",
    )
    return z_gen, I_sq
end

"""
DNMDT: discretize V and I, compute I² from I's discretization,
call DNMDT bilinear with pre-built discretizations. I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    bilinear_config::IOM.DNMDTBilinearConfig, quad_config::IOM.DNMDTQuadConfig,
)
    V_disc = IOM._discretize!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, V_MIN, V_MAX, quad_config.depth, "gen_V",
    )
    I_disc = IOM._discretize!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_GEN_MIN, I_GEN_MAX, quad_config.depth, "gen_I",
    )
    I_sq = IOM.add_quadratic_approx!(
        quad_config, container, MockNetworkNode, net.gen_nodes, time_steps,
        I_disc, I_GEN_MIN, I_GEN_MAX, "gen_I_sq",
    )
    z_gen = IOM.add_bilinear_approx!(
        bilinear_config, container, MockNetworkNode, net.gen_nodes, time_steps,
        V_disc, I_disc, V_MIN, V_MAX, I_GEN_MIN, I_GEN_MAX, "gen",
    )
    return z_gen, I_sq
end

"""
Exact (NLP): exact bilinear V·I and exact quadratic I².
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    bilinear_config::IOM.NoBilinearApproxConfig, quad_config::IOM.NoQuadApproxConfig,
)
    z_gen = IOM.add_bilinear_approx!(
        bilinear_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        V_container,
        I_container,
        V_MIN,
        V_MAX,
        I_GEN_MIN,
        I_GEN_MAX,
        "gen",
    )
    I_sq = IOM.add_quadratic_approx!(
        quad_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        I_container,
        I_GEN_MIN,
        I_GEN_MAX,
        "gen_I_sq",
    )
    return z_gen, I_sq
end

# ─── MIP model  ──────────────────────────────────────────────────────────────

"""
    build_mip_model(optimizer, net, bilinear_config, quad_config, refinement) -> NamedTuple

Build the MIP (or NLP) model for a `MockNetworkProblem`. Bilinear
precomputation is dispatched on the config type.
"""
function build_mip_model(
    optimizer, net::MockNetworkProblem, bilinear_config, quad_config, refinement::Int,
)
    container, system = make_container(optimizer)
    tdf = TestDeviceFormulation
    jump_model = IOM.get_jump_model(container)
    time_steps = 1:1
    adj = adjacency_list(net)

    gen_devices = [
        MockNetworkNode(g, net.loss[g], I_GEN_MIN, I_GEN_MAX, V_MIN, V_MAX) for
        g in net.gen_nodes
    ]
    dem_devices = [
        MockNetworkNode(d, [0.0], I_DEM_MIN, I_DEM_MAX, V_MIN, V_MAX) for
        d in net.dem_nodes
    ]
    all_devices = [gen_devices; dem_devices]
    for device in all_devices
        add_component!(system, device)
    end

    IOM.add_variables!(container, ActivePowerVariable, gen_devices, tdf)
    IOM.add_variables!(container, MockVoltageVariable, all_devices, tdf)
    IOM.add_variables!(container, MockCurrentVariable, all_devices, tdf)
    IOM.add_variables!(container, MockLineLossAuxVariable, gen_devices, tdf)

    dm = DeviceModel(MockNetworkNode, TestDeviceFormulation)
    add_range_constraints!(
        container,
        MockPowerRangeConstraint,
        ActivePowerVariable,
        gen_devices,
        dm,
        TestPowerModel,
    )

    V_container = IOM.get_variable(container, MockVoltageVariable, MockNetworkNode)
    I_container = IOM.get_variable(container, MockCurrentVariable, MockNetworkNode)
    Pg = IOM.get_variable(container, ActivePowerVariable, MockNetworkNode)

    # --- Bilinear gen: dispatched on config type ---
    z_gen, I_sq = build_gen_bilinear(
        container, net, V_container, I_container, time_steps,
        bilinear_config, quad_config,
    )

    # --- Bilinear dem: always uses the config-based dispatch ---
    z_dem = IOM.add_bilinear_approx!(
        bilinear_config, container, MockNetworkNode, net.dem_nodes, time_steps,
        V_container, I_container,
        V_MIN, V_MAX, I_DEM_MIN, I_DEM_MAX, "dem",
    )

    pwl_link_constraints = IOM.add_constraints_container!(
        container,
        IOM.PiecewiseLinearBlockIncrementalOfferConstraint,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
    )
    for g in net.gen_nodes
        breakpoints = vcat(0.0, cumsum(net.segment_widths[g]))
        pwl_vars = IOM.add_pwl_variables_delta!(
            container,
            IOM.PiecewiseLinearBlockIncrementalOffer,
            MockNetworkNode,
            g,
            1,
            net.K;
            upper_bound = Inf,
        )
        IOM.add_pwl_block_offer_constraints!(
            jump_model,
            pwl_link_constraints,
            g,
            1,
            Pg[g, 1],
            pwl_vars,
            breakpoints,
        )

        pwl_cost = IOM.get_pwl_cost_expression_delta(
            pwl_vars,
            net.marginal_costs[g],
            1.0,
        )
        IOM.add_to_objective_invariant_expression!(container, pwl_cost)
    end

    # Objective: min Σ m_{i,k} · δ_{i,k}, assembled via IOM's delta-PWL helper.
    # With a 1-hour benchmark resolution, the formulation multiplier is dt = 1.0.
    IOM.update_objective_function!(container)

    # --- Generator power: P = V·I − loss ---
    gen_pwr_constraints = IOM.add_constraints_container!(
        container,
        MockPowerEqualityConstraint,
        MockNetworkNode,
        net.gen_nodes,
        time_steps;
        meta = "Pg",
    )
    for g in net.gen_nodes
        a, b, c = net.loss[g]
        gen_pwr_constraints[g, 1] =
            JuMP.@constraint(
                jump_model,
                Pg[g, 1] ==
                z_gen[g, 1] + a * I_sq[g, 1] + b * I_container[g, 1] + c
            )
    end

    # --- Demand: V·I == -d ---
    dem_pwr_constraints = IOM.add_constraints_container!(
        container,
        MockPowerEqualityConstraint,
        MockNetworkNode,
        net.dem_nodes,
        time_steps;
        meta = "Pd",
    )
    for d in net.dem_nodes
        dem_pwr_constraints[d, 1] = JuMP.@constraint(
            jump_model, z_dem[d, 1] == -net.demands[d]
        )
    end

    # --- KCL: I_i = Σ g_{ij}(V_i - V_j) ---
    kcl_expressions = IOM.add_expression_container!(
        container,
        MockKCLExpression,
        MockNetworkNode,
        net.all_nodes,
        time_steps,
    )
    kcl_constraints = IOM.add_constraints_container!(
        container,
        MockKCLConstraint,
        MockNetworkNode,
        net.all_nodes,
        time_steps,
    )
    for n in net.all_nodes
        expr = kcl_expressions[n, 1] = JuMP.AffExpr(0.0)
        for (j, c) in adj[n]
            IOM.add_proportional_to_jump_expression!(
                expr, V_container[n, 1], c,
            )
            IOM.add_proportional_to_jump_expression!(
                expr, V_container[j, 1], -c,
            )
        end
        kcl_constraints[n, 1] = JuMP.@constraint(
            jump_model, I_container[n, 1] == expr
        )
    end

    return (; container, system, jump_model, V_container, I_container, z_gen, z_dem, I_sq)
end

# ─── Metrics ──────────────────────────────────────────────────────────────────

@inline function residual(actual, measured, eps = 1e-10)
    return abs(actual - measured) / max(abs(actual), eps)
end

@inline function rmse(sequence)
    return sqrt(sum(x -> x^2, sequence) / length(sequence))
end

"""
Compute per-node relative residuals |true − approx| / |true| for the bilinear
and quadratic products. Returns (rmse_bi, max_bi, rmse_q, max_q).
"""
function compute_bilinear_residuals(result, net::MockNetworkProblem)
    V = result.V_container
    I = result.I_container
    bilin_residuals = Float64[]
    quad_residuals = Float64[]

    for g in net.gen_nodes
        v = JuMP.value(V[g, 1])
        i = JuMP.value(I[g, 1])
        product = v * i
        approx = JuMP.value(result.z_gen[g, 1])
        push!(bilin_residuals, residual(product, approx))
        quad = i * i
        approx = JuMP.value(result.I_sq[g, 1])
        push!(quad_residuals, residual(quad, approx))
    end

    for d in net.dem_nodes
        product = JuMP.value(V[d, 1]) * JuMP.value(I[d, 1])
        approx = JuMP.value(result.z_dem[d, 1])
        push!(bilin_residuals, residual(product, approx))
    end

    return (
        rmse(bilin_residuals),
        maximum(bilin_residuals),
        rmse(quad_residuals),
        maximum(quad_residuals),
    )
end

function model_size(jump_model)
    nv = JuMP.num_variables(jump_model)
    nc = sum(
        JuMP.num_constraints(jump_model, f, s)
        for (f, s) in JuMP.list_of_constraint_types(jump_model)
    )
    nb = JuMP.num_constraints(jump_model, JuMP.VariableRef, JuMP.MOI.ZeroOne)
    return (; variables = nv, constraints = nc, binaries = nb)
end

# ─── Benchmark output helpers ─────────────────────────────────────────────────

function print_network_info(net::MockNetworkProblem)
    println(
        "Network: $(net.N) nodes, $(length(net.edges)) edges, $(net.K) cost segments",
    )
    println("Generators: $(length(net.gen_nodes)), Demands: $(length(net.dem_nodes))")
    println("Loss coefficients (a, b, c) per generator:")
    for g in net.gen_nodes
        @printf("  %s: a=%.6f  b=%.6f  c=%.6f\n",
            g, net.loss[g][1], net.loss[g][2], net.loss[g][3])
    end
end

"""Return the path for the solver log file, creating the logs directory if needed."""
function solver_log_path()
    log_dir = joinpath(@__DIR__, "logs")
    mkpath(log_dir)
    tag = if isempty(SLURM_JOB_ID)
        Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
    else
        SLURM_JOB_ID
    end
    return joinpath(log_dir, "solver_$(tag).log")
end

# ─── Single case runner ───────────────────────────────────────────────────────

"""
    run_single_case(; ...) -> NamedTuple

Build and solve a single (method, refinement) benchmark case. Returns all metrics
needed for the summary table row. Uses `JuMP.optimize!` directly.
"""
function run_single_case(;
    optimizer,
    net::MockNetworkProblem,
    bilinear_config,
    quad_config,
    refinement::Int,
    label::String,
    is_exact::Bool,
    nlp_obj::Float64 = NaN,
    logfile::IO = devnull,
    build_only::Bool = false,
    time_limit::Float64 = NaN,
    threads::Int = 0,
)
    build_t = @elapsed result =
        build_mip_model(optimizer, net, bilinear_config, quad_config, refinement)

    if build_only
        solve_t = 0.0
        status = nothing
    else
        jump_model = result.jump_model
        if !isnan(time_limit) && !is_exact
            JuMP.set_time_limit_sec(jump_model, time_limit)
        end
        if threads > 0 && !is_exact
            JuMP.set_attribute(jump_model, JuMP.MOI.NumberOfThreads(), threads)
        end

        println(logfile, "\n", "="^80)
        println(logfile, "$label  R=$refinement  $(Dates.now())")
        println(logfile, "="^80)
        flush(logfile)
        solve_t = @elapsed redirect_stdout(logfile) do
            # We use this directly because we need to cut the model off after 
            # a time limit. If we use the optimization container, it will
            # attempt to re-solve.
            JuMP.optimize!(jump_model)
        end
        flush(logfile)
        status = JuMP.termination_status(jump_model)
    end

    sz = model_size(result.jump_model)

    solved = if is_exact
        status == JuMP.LOCALLY_SOLVED
    else
        status in (JuMP.OPTIMAL, JuMP.TIME_LIMIT, JuMP.SOLUTION_LIMIT) &&
            JuMP.has_values(result.jump_model)
    end

    obj = NaN
    gap = NaN
    mip_gap = NaN
    lower_bound = NaN
    mn_bi = NaN
    mx_bi = NaN
    mn_q = NaN
    mx_q = NaN

    if solved
        obj = JuMP.objective_value(result.jump_model)
        gap = residual(nlp_obj, obj) * 100.0
        mn_bi, mx_bi, mn_q, mx_q = compute_bilinear_residuals(result, net)
        if !is_exact
            try
                mip_gap = JuMP.relative_gap(result.jump_model) * 100.0
            catch
            end
            try
                lower_bound = JuMP.objective_bound(result.jump_model)
            catch
            end
        end
    end

    return (;
        label,
        refinement,
        is_exact,
        solved,
        status,
        obj,
        nlp_obj,
        gap,
        mip_gap,
        lower_bound,
        mn_bi, mx_bi, mn_q, mx_q,
        variables = sz.variables,
        constraints = sz.constraints,
        binaries = sz.binaries,
        build_t,
        solve_t,
    )
end

# ─── Result formatting ────────────────────────────────────────────────────────

function print_table_header()
    println("="^145)
    println("Bilinear Approximation Benchmarks")
    println("  Refinement = depth for all methods")
    println("="^145)
    @printf("%-12s %4s %6s %7s %6s %12s %8s %9s %12s %9s %9s %9s %9s %8s %8s\n",
        "Method", "R", "Vars", "Constrs", "Bins", "Objective",
        "Gap(%)", "MIPGap(%)", "LowerBnd",
        "rmse δbi", "max δbi", "rmse δq", "max δq", "build_t", "solve_t")
    println("-"^145)
    flush(stdout)
end

function print_result_row(r)
    if r.solved
        gap_str = isnan(r.gap) ? "    -" : @sprintf("%8.4f", r.gap)
        ref_str = r.is_exact ? "   -" : @sprintf("%4d", r.refinement)
        mip_gap_str =
            (r.is_exact || isnan(r.mip_gap)) ? "        -" : @sprintf("%9.4f", r.mip_gap)
        lb_str = if (r.is_exact || isnan(r.lower_bound))
            "           -"
        else
            @sprintf("%12.6f", r.lower_bound)
        end
        @printf(
            "%-12s %2s %6d %7d %6d %12.6f %6s %9s %12s %9.2e %9.2e %9.2e %9.2e %8.4f %8.4f\n",
            r.label, ref_str,
            r.variables, r.constraints, r.binaries,
            r.obj, gap_str, mip_gap_str, lb_str,
            r.mn_bi, r.mx_bi, r.mn_q, r.mx_q, r.build_t, r.solve_t)
    else
        ref_str = r.is_exact ? "  -" : @sprintf("%4d", r.refinement)
        @printf("%-12s %2s %6d %7d %6d %12s %6s %9s %12s %9s %9s %9s %9s %8.4f %8.4f\n",
            r.label, ref_str,
            r.variables, r.constraints, r.binaries,
            string(r.status), "-", "-", "-", "-", "-", "-", "-", r.build_t, r.solve_t)
    end
    flush(stdout)
end

"""Serialize a result NamedTuple to a JSON-writable Dict (NaN/Inf → nothing)."""
function result_to_dict(r)
    nan_safe(x) = (x isa AbstractFloat && !isfinite(x)) ? nothing : x
    return Dict{String, Any}(
        "label" => r.label,
        "refinement" => r.refinement,
        "is_exact" => r.is_exact,
        "solved" => r.solved,
        "status" => string(r.status),
        "obj" => nan_safe(r.obj),
        "nlp_obj" => nan_safe(r.nlp_obj),
        "gap" => nan_safe(r.gap),
        "mip_gap" => nan_safe(r.mip_gap),
        "lower_bound" => nan_safe(r.lower_bound),
        "mn_bi" => nan_safe(r.mn_bi),
        "mx_bi" => nan_safe(r.mx_bi),
        "mn_q" => nan_safe(r.mn_q),
        "mx_q" => nan_safe(r.mx_q),
        "variables" => r.variables,
        "constraints" => r.constraints,
        "binaries" => r.binaries,
        "build_t" => r.build_t,
        "solve_t" => r.solve_t,
    )
end

"""Deserialize a Dict from JSON back to a NamedTuple (nothing → NaN)."""
function dict_to_result(d)
    as_float(x) = x === nothing ? NaN : Float64(x)
    return (;
        label = d["label"],
        refinement = Int(d["refinement"]),
        is_exact = d["is_exact"],
        solved = d["solved"],
        status = d["status"],
        obj = as_float(d["obj"]),
        nlp_obj = as_float(d["nlp_obj"]),
        gap = as_float(d["gap"]),
        mip_gap = as_float(d["mip_gap"]),
        lower_bound = as_float(d["lower_bound"]),
        mn_bi = as_float(d["mn_bi"]),
        mx_bi = as_float(d["mx_bi"]),
        mn_q = as_float(d["mn_q"]),
        mx_q = as_float(d["mx_q"]),
        variables = Int(d["variables"]),
        constraints = Int(d["constraints"]),
        binaries = Int(d["binaries"]),
        build_t = Float64(d["build_t"]),
        solve_t = Float64(d["solve_t"]),
    )
end

# ─── Sequential benchmark (all environments) ─────────────────────────────────

"""
    run_benchmark(; N, K, seed, refinements, build_only)

Run the full benchmark sequentially. MIP solves get a 1-hour time limit.
NLP references (Ipopt/Uno) have no time limit.
"""
function run_benchmark(;
    N::Int = 10,
    K::Int = 3,
    seed::Int = 42,
    refinements::Vector{Int} = [4, 6, 8],
    build_only::Bool = false,
)
    net = generate_network(; N, K, seed)
    print_network_info(net)
    println()

    log_path = solver_log_path()
    println("Solver logs: $log_path")
    println()

    logfile = open(log_path, "a")
    atexit(() -> (flush(stdout); isopen(logfile) && (flush(logfile); close(logfile))))

    threads = ENVIRONMENT == :kestrel ? KESTREL_XPRESS_THREADS : 0

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
                    build_only,
                    time_limit = MIP_TIME_LIMIT_SEC,
                    threads,
                )

                if r.solved && is_exact
                    nlp_obj = r.obj
                end

                print_result_row(r)
            end
            println()
        end
        println("="^120)
    finally
        isopen(logfile) && (flush(logfile); close(logfile))
        flush(stdout)
    end
end

# ─── Parallel benchmark (Kestrel only) ───────────────────────────────────────

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
    net = generate_network(; N, K, seed)
    print_network_info(net)
    println()

    log_path = solver_log_path()
    println("Solver logs: $log_path")
    println()

    logfile = open(log_path, "a")
    atexit(() -> (flush(stdout); isopen(logfile) && (flush(logfile); close(logfile))))

    mip_methods = bilinear_methods

    print_table_header()

    # ── Phase 1: NLP references (sequential, no time limit) ──
    nlp_obj = NaN
    nlp_results = []
    for (label, opt_fn) in [
        ("NLP (Ipopt)", Ipopt.Optimizer),
        ("NLP (Uno)", () -> UnoSolver.Optimizer(; preset = "filtersqp")),
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
            build_only = false,
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
    mip_results = []
    for (mi, ref) in jobs
        label = mip_methods[mi][1]
        outfile =
            joinpath(res_dir, "$(SLURM_JOB_ID)_$(replace(label, " " => "_"))_R$(ref).json")
        if isfile(outfile)
            d = open(outfile) do io
                JSON3.read(io, Dict{String, Any})
            end
            push!(mip_results, dict_to_result(d))
        else
            @warn "Missing results for $label R=$ref"
            push!(
                mip_results,
                (;
                    label,
                    refinement = ref,
                    is_exact = false,
                    solved = false,
                    status = "WORKER_FAILED",
                    obj = NaN, nlp_obj, gap = NaN,
                    mip_gap = NaN, lower_bound = NaN,
                    mn_bi = NaN, mx_bi = NaN, mn_q = NaN, mx_q = NaN,
                    variables = 0, constraints = 0, binaries = 0,
                    build_t = 0.0, solve_t = 0.0,
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
    println("="^120)
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

    mip_methods_list = bilinear_methods
    label, method = mip_methods_list[mi]
    bilin_config, quad_config = method(ref)

    log_dir = joinpath(@__DIR__, "logs")
    mkpath(log_dir)
    tag = if isempty(SLURM_JOB_ID)
        Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
    else
        SLURM_JOB_ID
    end
    safe_label = replace(label, " " => "_")
    worker_log_path = joinpath(log_dir, "solver_$(tag)_$(safe_label)_R$(ref).log")
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
            build_only = false,
            time_limit = MIP_TIME_LIMIT_SEC,
            threads = KESTREL_XPRESS_THREADS,
        )

        mkpath(dirname(outfile))
        open(outfile, "w") do io
            JSON3.write(io, result_to_dict(r))
        end

        @info "Worker done: $label R=$ref status=$(r.status) obj=$(r.obj) log=$worker_log_path"
    finally
        flush(logfile)
        close(logfile)
    end
end

# ─── Entry point ──────────────────────────────────────────────────────────────

epi_C = 1.5

function Bin2_(R, quad_config)
    q = quad_config(R)
    IOM.Bin2Config(q), q
end
Bin2_sSOS(R) = Bin2_(R, IOM.SolverSOS2QuadConfig)
Bin2_mSOS(R) = Bin2_(R, IOM.ManualSOS2QuadConfig)
Bin2_Saw(R) = Bin2_(R, IOM.SawtoothQuadConfig)

function HybS_(R, quad_config)
    q = quad_config(R)
    IOM.HybSConfig(q, ceil(Int, epi_C * R)), q
end
HybS_sSOS(R) = HybS_(R, IOM.SolverSOS2QuadConfig)
HybS_mSOS(R) = HybS_(R, IOM.ManualSOS2QuadConfig)
HybS_Saw(R) = HybS_(R, IOM.SawtoothQuadConfig)

function DNMDT_DNMDT(R)
    IOM.DNMDTBilinearConfig(R), IOM.DNMDTQuadConfig(R, ceil(Int, epi_C * R))
end

exact(_) = (IOM.NoBilinearApproxConfig(), IOM.NoQuadApproxConfig())

bilinear_methods = (
    ("Bin2+sSOS", Bin2_sSOS),
    ("Bin2+mSOS", Bin2_mSOS),
    ("Bin2+Saw", Bin2_Saw),
    ("HybS+sSOS", HybS_sSOS),
    ("HybS+mSOS", HybS_mSOS),
    ("HybS+Saw", HybS_Saw),
    ("DNMDT", DNMDT_DNMDT),
)

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
        "--build-only", "-B"
        action = :store_true
        help = "don't solve, only build the model"
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
            build_only = parsed["build-only"]
            refinements = parsed["refinements"]

            if ENVIRONMENT == :kestrel && !build_only
                run_benchmark_parallel(; N, K, seed, refinements)
            else
                # Run small network so second run is faster.
                redirect_stdout(devnull) do
                    run_benchmark(;
                        N = 2,
                        K = 1,
                        seed = 0,
                        build_only = false,
                        refinements = [1],
                    )
                end
                run_benchmark(; N, K, seed, build_only, refinements)
            end
        end
    catch e
        @error "Benchmark failed" exception = (e, catch_backtrace())
    finally
        flush(stdout)
        flush(stderr)
    end
end
