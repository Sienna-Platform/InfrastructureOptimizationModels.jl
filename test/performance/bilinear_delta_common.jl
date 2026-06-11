"""
Shared logic for the bilinear delta benchmark (see `bilinear_delta_model.tex`).

Builds a lossy network OPF with delta (incremental) PWL costs and bilinear V·I power
balance constraints, and compares bilinear approximation methods from
InfrastructureOptimizationModels against an exact NLP reference.

The lossy generator model is P = V·I − a·I² − b·I − c.

This file is solver-agnostic: it defines the model, metrics, formatting, and the
per-case runner, but does NOT pick an optimizer or define an entry point. The
optimizer is passed into `run_single_case`. It is `include`d by the entry-point
scripts:

  - `bilinear_delta_local.jl`   — sequential runner (local + CI/CD)
  - `bilinear_delta_kestrel.jl` — parallel runner + worker (Kestrel/SLURM)
"""

using InfrastructureOptimizationModels
using JuMP
using Dates
using Random
using Printf

const IOM = InfrastructureOptimizationModels
const IS = IOM.IS

const MIP_TIME_LIMIT_SEC = 600.0
const SLURM_JOB_ID = get(ENV, "SLURM_JOB_ID", "")
const LOG_BANNER_WIDTH = 80

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

"""Per-generator V and I bound vectors, shared by all `build_gen_bilinear` methods."""
function gen_bounds(net::MockNetworkProblem)
    ng = length(net.gen_nodes)
    V_bounds = fill((min = V_MIN, max = V_MAX), ng)
    I_bounds = fill((min = I_GEN_MIN, max = I_GEN_MAX), ng)
    return V_bounds, I_bounds
end

"""
Separable methods: precompute V² and I², call wrapper's precomputed overload.
I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    bilinear_config::Union{IOM.Bin2Config, IOM.HybSConfig}, quad_config,
)
    V_bounds, I_bounds = gen_bounds(net)
    V_sq = IOM.add_quadratic_approx!(
        quad_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        V_container,
        V_bounds,
        "gen_V_sq",
    )
    I_sq = IOM.add_quadratic_approx!(
        quad_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        I_container,
        I_bounds,
        "gen_I_sq",
    )
    z_gen = if bilinear_config isa IOM.Bin2Config
        IOM._assemble_bin2!(
            bilinear_config, container, MockNetworkNode, net.gen_nodes, time_steps,
            V_sq, I_sq, V_container, I_container, V_bounds, I_bounds, "gen",
        )
    else
        IOM._assemble_hybs!(
            bilinear_config, container, MockNetworkNode, net.gen_nodes, time_steps,
            V_sq, I_sq, V_container, I_container, V_bounds, I_bounds, "gen",
        )
    end
    return z_gen, I_sq
end

"""
DNMDT: discretize V and I, compute I² from I's discretization,
call DNMDT bilinear with pre-built discretizations. I² is reused in the loss constraint.
"""
function build_gen_bilinear(
    container, net::MockNetworkProblem, V_container, I_container, time_steps,
    bilinear_config::IOM.NMDTBilinearConfig{IOM.DoubleNMDT}, quad_config::IOM.NMDTQuadConfig{IOM.DoubleNMDT},
)
    V_bounds, I_bounds = gen_bounds(net)
    V_disc = IOM._discretize!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        V_container, V_bounds, quad_config.depth, "gen_V",
    )
    I_disc = IOM._discretize!(
        container, MockNetworkNode, net.gen_nodes, time_steps,
        I_container, I_bounds, quad_config.depth, "gen_I",
    )
    I_sq = IOM._quadratic_from_discretization!(
        quad_config, container, MockNetworkNode, net.gen_nodes, time_steps,
        I_disc, I_bounds, "gen_I_sq",
    )
    z_gen = IOM._bilinear_from_discretization!(
        bilinear_config, container, MockNetworkNode, net.gen_nodes, time_steps,
        V_disc, I_disc, V_bounds, I_bounds, "gen",
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
    V_bounds, I_bounds = gen_bounds(net)
    z_gen = IOM.add_bilinear_approx!(
        bilinear_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        V_container,
        I_container,
        V_bounds,
        I_bounds,
        "gen",
    )
    I_sq = IOM.add_quadratic_approx!(
        quad_config,
        container,
        MockNetworkNode,
        net.gen_nodes,
        time_steps,
        I_container,
        I_bounds,
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
    nd = length(net.dem_nodes)
    z_dem = IOM.add_bilinear_approx!(
        bilinear_config, container, MockNetworkNode, net.dem_nodes, time_steps,
        V_container, I_container,
        fill((min = V_MIN, max = V_MAX), nd),
        fill((min = I_DEM_MIN, max = I_DEM_MAX), nd),
        "dem",
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

# ─── Result record ────────────────────────────────────────────────────────────

"""
    BenchmarkResult

All metrics for one (method, refinement) case. Single source of truth for the
result schema — serialization (`to_dict`/`from_dict`) and the summary table
(`COLUMNS`) both derive from these fields, so adding a metric only touches here.
"""
Base.@kwdef struct BenchmarkResult
    label::String
    refinement::Int
    is_exact::Bool = false
    solved::Bool = false
    status = nothing            # JuMP termination status, or String after a round-trip
    obj::Float64 = NaN
    nlp_obj::Float64 = NaN
    gap::Float64 = NaN
    mip_gap::Float64 = NaN
    lower_bound::Float64 = NaN
    mn_bi::Float64 = NaN
    mx_bi::Float64 = NaN
    mn_q::Float64 = NaN
    mx_q::Float64 = NaN
    variables::Int = 0
    constraints::Int = 0
    binaries::Int = 0
    build_t::Float64 = 0.0
    solve_t::Float64 = 0.0
end

"""Serialize to a JSON-writable Dict (status → String, NaN/Inf → nothing)."""
function to_dict(r::BenchmarkResult)
    d = Dict{String, Any}()
    for f in fieldnames(BenchmarkResult)
        v = getfield(r, f)
        if f === :status
            v = string(v)
        elseif v isa AbstractFloat && !isfinite(v)
            v = nothing
        end
        d[String(f)] = v
    end
    return d
end

"""Reconstruct a `BenchmarkResult` from a Dict (nothing → NaN, per field type)."""
function from_dict(d::AbstractDict)
    kw = Dict{Symbol, Any}()
    for f in fieldnames(BenchmarkResult)
        T = fieldtype(BenchmarkResult, f)
        v = d[String(f)]
        if T === Float64
            v = v === nothing ? NaN : Float64(v)
        elseif T === Int
            v = Int(v)
        end
        kw[f] = v
    end
    return BenchmarkResult(; kw...)
end

# ─── Result formatting ────────────────────────────────────────────────────────

"""One column of the summary table: header, field width, alignment, and a cell renderer."""
struct Col
    header::String
    width::Int
    align::Symbol           # :left or :right
    render::Function        # (r::BenchmarkResult) -> String
end

_fmt(fstr, x) = Printf.format(Printf.Format(fstr), x)
_pad(s, w, align) = align === :left ? rpad(s, w) : lpad(s, w)

const COLUMNS = Col[
    Col("Method", 12, :left, r -> r.label),
    Col("R", 4, :right, r -> r.is_exact ? "-" : string(r.refinement)),
    Col("Vars", 6, :right, r -> string(r.variables)),
    Col("Constrs", 7, :right, r -> string(r.constraints)),
    Col("Bins", 6, :right, r -> string(r.binaries)),
    Col("Objective", 12, :right, r -> r.solved ? _fmt("%.6f", r.obj) : string(r.status)),
    Col("Gap(%)", 8, :right, r -> (r.solved && !isnan(r.gap)) ? _fmt("%.4f", r.gap) : "-"),
    Col("MIPGap(%)", 9, :right,
        r -> if (r.solved && !r.is_exact && !isnan(r.mip_gap))
            _fmt("%.4f", r.mip_gap)
        else
            "-"
        end),
    Col("LowerBnd", 12, :right,
        r -> if (r.solved && !r.is_exact && !isnan(r.lower_bound))
            _fmt("%.6f", r.lower_bound)
        else
            "-"
        end),
    Col("rmse δbi", 9, :right, r -> r.solved ? _fmt("%.2e", r.mn_bi) : "-"),
    Col("max δbi", 9, :right, r -> r.solved ? _fmt("%.2e", r.mx_bi) : "-"),
    Col("rmse δq", 9, :right, r -> r.solved ? _fmt("%.2e", r.mn_q) : "-"),
    Col("max δq", 9, :right, r -> r.solved ? _fmt("%.2e", r.mx_q) : "-"),
    Col("build_t", 8, :right, r -> _fmt("%.4f", r.build_t)),
    Col("solve_t", 8, :right, r -> _fmt("%.4f", r.solve_t)),
]

# Table rule width: column widths plus the single spaces between them.
const RULE_WIDTH = sum(c.width for c in COLUMNS) + (length(COLUMNS) - 1)

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

function print_table_header()
    println("="^RULE_WIDTH)
    println("Bilinear Approximation Benchmarks")
    println("  Refinement = depth for all methods")
    println("="^RULE_WIDTH)
    println(join((_pad(c.header, c.width, c.align) for c in COLUMNS), " "))
    println("-"^RULE_WIDTH)
    flush(stdout)
end

function print_result_row(r::BenchmarkResult)
    println(join((_pad(c.render(r), c.width, c.align) for c in COLUMNS), " "))
    flush(stdout)
end

# ─── Logging helpers ──────────────────────────────────────────────────────────

"""SLURM job id when running under SLURM, otherwise a timestamp — used in log filenames."""
function log_tag()
    return if isempty(SLURM_JOB_ID)
        Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS")
    else
        SLURM_JOB_ID
    end
end

"""Return the path for the solver log file, creating the logs directory if needed."""
function solver_log_path()
    log_dir = joinpath(@__DIR__, "logs")
    mkpath(log_dir)
    return joinpath(log_dir, "solver_$(log_tag()).log")
end

"""
Shared runner preamble: generate the network, print its info, open the solver log,
and register cleanup. Returns `(net, logfile)`.
"""
function setup_run(; N::Int, K::Int, seed::Int)
    net = generate_network(; N, K, seed)
    print_network_info(net)
    println()

    log_path = solver_log_path()
    println("Solver logs: $log_path")
    println()

    logfile = open(log_path, "a")
    atexit(() -> (flush(stdout); isopen(logfile) && (flush(logfile); close(logfile))))
    return net, logfile
end

# ─── Single case runner ───────────────────────────────────────────────────────

"""
    run_single_case(; ...) -> BenchmarkResult

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
    time_limit::Float64 = NaN,
    threads::Int = 0,
)
    build_t = @elapsed result =
        build_mip_model(optimizer, net, bilinear_config, quad_config, refinement)

    jump_model = result.jump_model
    if !isnan(time_limit) && !is_exact
        JuMP.set_time_limit_sec(jump_model, time_limit)
    end
    if threads > 0 && !is_exact
        JuMP.set_attribute(jump_model, JuMP.MOI.NumberOfThreads(), threads)
    end

    println(logfile, "\n", "="^LOG_BANNER_WIDTH)
    println(logfile, "$label  R=$refinement  $(Dates.now())")
    println(logfile, "="^LOG_BANNER_WIDTH)
    flush(logfile)
    solve_t = @elapsed redirect_stdout(logfile) do
        # We use this directly because we need to cut the model off after
        # a time limit. If we use the optimization container, it will
        # attempt to re-solve.
        JuMP.optimize!(jump_model)
    end
    flush(logfile)
    status = JuMP.termination_status(jump_model)

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

    return BenchmarkResult(;
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

# ─── Method configurations ────────────────────────────────────────────────────

epi_C = 1.5

function Bin2_(R, quad_config)
    q = quad_config(; depth = R)
    IOM.Bin2Config(q), q
end
Bin2_sSOS(R) = Bin2_(R, IOM.SOS2QuadConfig{IOM.SolverBackend})
Bin2_mSOS(R) = Bin2_(R, IOM.SOS2QuadConfig{IOM.ManualBackend})
Bin2_Saw(R) = Bin2_(R, IOM.SawtoothQuadConfig)

function HybS_(R, quad_config)
    q = quad_config(; depth = R)
    IOM.HybSConfig(q; cross_term_depth = ceil(Int, epi_C * R)), q
end
HybS_sSOS(R) = HybS_(R, IOM.SOS2QuadConfig{IOM.SolverBackend})
HybS_mSOS(R) = HybS_(R, IOM.SOS2QuadConfig{IOM.ManualBackend})
HybS_Saw(R) = HybS_(R, IOM.SawtoothQuadConfig)

function DNMDT_DNMDT(R)
    IOM.NMDTBilinearConfig{IOM.DoubleNMDT}(; depth = R),
    IOM.NMDTQuadConfig{IOM.DoubleNMDT}(;
        depth = R, tightener = IOM.EpigraphTightener(ceil(Int, epi_C * R)),
    )
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
