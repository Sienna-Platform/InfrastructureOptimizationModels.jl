"""
Relates an upper time scale (periods) to the model's time steps. Each period contains one or
more blocks, and each block is a contiguous range of time steps.

Subtypes implement:

  - `get_period_count(tm)`: number of periods
  - `get_block_ranges(tm)`: the time steps of each block, ordered and contiguous
  - `get_block_period_index(tm)`: the period each block belongs to
  - `get_timestamps(tm)`: the time stamp of each time step

Optional accessors, implemented by mappings whose consumers need them (e.g. investment
time mappings):

  - `get_base_date(tm)`: the date the model's outputs are indexed by
  - `get_total_operation_period_count(tm)`: the number of operational time steps

`NoTimeMapping` is the placeholder for "no mapping set". It implements none of this interface.
"""
abstract type AbstractTimeMapping end

"""
Placeholder held by an `OptimizationContainer` until `set_time_mapping!` attaches a mapping.
"""
struct NoTimeMapping <: AbstractTimeMapping end

function get_period_count end
function get_block_ranges end
function get_block_period_index end
function get_base_date end
function get_total_operation_period_count end

get_time_steps(tm::AbstractTimeMapping) = 1:length(get_timestamps(tm))
get_periods(tm::AbstractTimeMapping) = 1:get_period_count(tm)
get_blocks(tm::AbstractTimeMapping, period::Int) =
    findall(==(period), get_block_period_index(tm))

function get_period_time_steps(tm::AbstractTimeMapping, period::Int)
    blocks = get_blocks(tm, period)
    if isempty(blocks)
        error("Period $period has no block in the time mapping.")
    end
    return reduce(vcat, get_block_ranges(tm)[blocks])
end

function get_period_index(tm::AbstractTimeMapping, t::Int)
    ranges = get_block_ranges(tm)
    block = searchsortedlast(ranges, t; by = first)
    if iszero(block) || t > last(ranges[block])
        error("Time step $t is not in any block of the time mapping.")
    end
    return get_block_period_index(tm)[block]
end

get_last_time_step(tm::AbstractTimeMapping, period::Int) =
    last(get_block_ranges(tm)[last(get_blocks(tm, period))])

get_block_weights(tm::AbstractTimeMapping) = ones(length(get_block_ranges(tm)))

"""
Check that the blocks are contiguous and cover the time steps in order, without gaps or
overlaps, that each block belongs to a valid period, and that each period has a block.
"""
function validate_time_mapping(tm::AbstractTimeMapping)
    ranges = get_block_ranges(tm)
    period_index = get_block_period_index(tm)
    if length(period_index) != length(ranges)
        error(
            "The time mapping has $(length(ranges)) blocks but $(length(period_index)) " *
            "block period indexes.",
        )
    end
    next_step = 1
    for (b, r) in enumerate(ranges)
        if isempty(r) || first(r) != next_step
            error(
                "Block $b covers time steps $r; blocks must be non-empty and start at " *
                "time step $next_step.",
            )
        end
        if r != first(r):last(r)
            error("Block $b covers time steps $r; blocks must be contiguous.")
        end
        next_step = last(r) + 1
    end
    step_count = length(get_timestamps(tm))
    if next_step != step_count + 1
        error(
            "The blocks cover $(next_step - 1) time steps but the time mapping has " *
            "$step_count time stamps.",
        )
    end
    period_count = get_period_count(tm)
    for (b, k) in enumerate(period_index)
        if !(1 <= k <= period_count)
            error("Block $b belongs to period $k, outside 1:$period_count.")
        end
    end
    has_block = falses(period_count)
    for k in period_index
        has_block[k] = true
    end
    for k in 1:period_count
        if !has_block[k]
            error("The time mapping has no block in period $k.")
        end
    end
    weight_count = length(get_block_weights(tm))
    if weight_count != length(ranges)
        error(
            "The time mapping has $(length(ranges)) blocks but $weight_count block weights.",
        )
    end
    return
end

"""
Time mapping with one block per period. Calendar periods such as `Dates.Month(1)` or
`Dates.Year(1)` give blocks of unequal length.

    NestedTimeMapping(initial_time, resolution, time_steps, period; allow_partial = false)

groups `time_steps` of length `resolution` starting at `initial_time` into periods of length
`period`, e.g. `Dates.Day(1)` for daily windows in an hourly model. `resolution` must be a
fixed period and `time_steps` must start at 1. A last period that ends before its calendar
boundary is an error unless `allow_partial` is set.
"""
struct NestedTimeMapping <: AbstractTimeMapping
    timestamps::Vector{Dates.DateTime}
    block_ranges::Vector{UnitRange{Int}}
    block_period_index::Vector{Int}
end

_fixed_period_ms(p::Dates.FixedPeriod) = Dates.value(Dates.Millisecond(p))

function _fixed_period_ms(p::Dates.OtherPeriod)
    error("The resolution $p must be a fixed period such as Hour or Minute.")
end

function NestedTimeMapping(
    initial_time::Dates.DateTime,
    resolution::Dates.Period,
    time_steps::UnitRange{Int},
    period::Dates.Period;
    allow_partial::Bool = false,
)
    resolution_ms = _fixed_period_ms(resolution)
    if resolution_ms <= 0
        error("The resolution $resolution must be positive.")
    end
    if first(time_steps) != 1
        error("The time steps $time_steps must start at 1.")
    end
    if !(initial_time + period > initial_time)
        error("The period $period must be positive.")
    end
    step_count = length(time_steps)
    timestamps =
        [initial_time + Dates.Millisecond((t - 1) * resolution_ms) for t in 1:step_count]
    end_time = initial_time + Dates.Millisecond(step_count * resolution_ms)
    block_ranges = UnitRange{Int}[]
    start, k = 1, 1
    while start <= step_count
        boundary = initial_time + k * period
        stop = searchsortedfirst(timestamps, boundary) - 1
        if (stop < step_count && timestamps[stop + 1] != boundary) ||
           (stop == step_count && end_time > boundary)
            error(
                "The period boundary $boundary does not align with the resolution " *
                "$resolution.",
            )
        end
        if stop == step_count && boundary != end_time && !allow_partial
            error(
                "The last period of $period ends at $end_time, before $boundary; " *
                "set allow_partial = true to keep a shorter last period.",
            )
        end
        push!(block_ranges, start:stop)
        start, k = stop + 1, k + 1
    end
    return NestedTimeMapping(timestamps, block_ranges, collect(1:length(block_ranges)))
end

get_period_count(tm::NestedTimeMapping) = length(tm.block_ranges)
get_block_ranges(tm::NestedTimeMapping) = tm.block_ranges
get_block_period_index(tm::NestedTimeMapping) = tm.block_period_index
get_timestamps(tm::NestedTimeMapping) = tm.timestamps
