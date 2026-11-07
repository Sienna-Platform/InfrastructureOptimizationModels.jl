"""
Relates an upper time scale (periods) to the model's time steps. Each period contains one or
more blocks, and each block is a contiguous range of time steps.

Subtypes implement:

  - `get_period_count(tm)`: number of periods
  - `get_block_ranges(tm)`: the time steps of each block, ordered and contiguous
  - `get_block_period_index(tm)`: the period each block belongs to
  - `get_timestamps(tm)`: the time stamp of each time step
"""
abstract type AbstractTimeMapping end

function get_period_count end
function get_block_ranges end
function get_block_period_index end

get_time_steps(tm::AbstractTimeMapping) = 1:length(get_timestamps(tm))
get_periods(tm::AbstractTimeMapping) = 1:get_period_count(tm)
get_blocks(tm::AbstractTimeMapping, period::Int) =
    findall(==(period), get_block_period_index(tm))

function get_period_time_steps(tm::AbstractTimeMapping, period::Int)
    ranges = get_block_ranges(tm)
    return mapreduce(b -> collect(ranges[b]), vcat, get_blocks(tm, period); init = Int[])
end

function get_period_index(tm::AbstractTimeMapping, t::Int)
    block = findfirst(r -> t in r, get_block_ranges(tm))
    if block === nothing
        error("Time step $t is not in any block of the time mapping.")
    end
    return get_block_period_index(tm)[block]
end

get_last_time_step(tm::AbstractTimeMapping, period::Int) =
    last(get_block_ranges(tm)[last(get_blocks(tm, period))])

get_block_weights(tm::AbstractTimeMapping) = ones(length(get_block_ranges(tm)))

"""
Check that the blocks cover the time steps in order, without gaps or overlaps, and that each
block belongs to a valid period.
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
    weight_count = length(get_block_weights(tm))
    if weight_count != length(ranges)
        error(
            "The time mapping has $(length(ranges)) blocks but $weight_count block weights.",
        )
    end
    return
end

"""
Time mapping with one block per period, all periods of equal length.

    NestedTimeMapping(initial_time, resolution, time_steps, period; allow_partial = false)

groups `time_steps` of length `resolution` starting at `initial_time` into periods of length
`period`, e.g. `Dates.Day(1)` for daily windows in an hourly model. A last period shorter than
the others is an error unless `allow_partial` is set.
"""
struct NestedTimeMapping <: AbstractTimeMapping
    timestamps::Vector{Dates.DateTime}
    block_ranges::Vector{UnitRange{Int}}
end

function NestedTimeMapping(
    initial_time::Dates.DateTime,
    resolution::Dates.Period,
    time_steps::UnitRange{Int},
    period::Dates.Period;
    allow_partial::Bool = false,
)
    resolution_ms = Dates.value(Dates.Millisecond(resolution))
    period_ms = Dates.value(Dates.Millisecond(period))
    if resolution_ms <= 0 || period_ms % resolution_ms != 0
        error(
            "The period $period is not a positive multiple of the resolution $resolution.",
        )
    end
    steps_per_period = period_ms ÷ resolution_ms
    step_count = length(time_steps)
    if step_count % steps_per_period != 0 && !allow_partial
        error(
            "$step_count time steps do not divide into periods of $steps_per_period " *
            "time steps; set allow_partial = true to keep a shorter last period.",
        )
    end
    timestamps =
        [initial_time + Dates.Millisecond((t - 1) * resolution_ms) for t in 1:step_count]
    block_ranges = [
        s:min(s + steps_per_period - 1, step_count) for
        s in 1:steps_per_period:step_count
    ]
    return NestedTimeMapping(timestamps, block_ranges)
end

get_period_count(tm::NestedTimeMapping) = length(tm.block_ranges)
get_block_ranges(tm::NestedTimeMapping) = tm.block_ranges
get_block_period_index(tm::NestedTimeMapping) = collect(1:length(tm.block_ranges))
get_timestamps(tm::NestedTimeMapping) = tm.timestamps
