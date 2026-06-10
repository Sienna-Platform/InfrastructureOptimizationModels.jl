"""
calculates the index in the time series corresponding to the data. Assumes that the dates vector is sorted.
"""
function find_timestamp_index(
    dates::Union{Vector{Dates.DateTime}, StepRange{Dates.DateTime, Dates.Millisecond}},
    date::Dates.DateTime,
)
    if date == first(dates)
        return 1
    elseif date == last(dates)
        return length(dates)
    elseif length(dates) < 2
        # No stride can be derived from a single element; the equality checks
        # above already covered the only valid case.
        error("Requested timestamp $date not in the provided dates $dates")
    end
    dates_resolution = dates[2] - dates[1]
    index = 1 + ((date - first(dates)) ÷ dates_resolution)
    # Uncomment for debugging. The method below is fool proof but slower
    # s_index = findlast(dates .<= date)
    # IS.@assert_op index == s_index
    if index < 1 || index > length(dates)
        error("Requested timestamp $date not in the provided dates $dates")
    end
    # The integer division above floors off-grid timestamps onto the previous
    # row; reject them instead of silently returning the wrong index.
    if first(dates) + (index - 1) * dates_resolution != date
        error("Requested timestamp $date not aligned to the provided dates $dates")
    end
    return index
end
