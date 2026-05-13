# Pure helpers for piecewise linear approximation breakpoint generation.

"""
    _get_breakpoints_for_pwl_function(min_val, max_val, f; num_segments = DEFAULT_INTERPOLATION_LENGTH)

Generate `num_segments + 1` equally-spaced breakpoints over `[min_val, max_val]`
and evaluate `f` at each. Returns `(x_bkpts, y_bkpts)`.
"""
function _get_breakpoints_for_pwl_function(
    min_val::Float64,
    max_val::Float64,
    f;
    num_segments = DEFAULT_INTERPOLATION_LENGTH,
)
    num_bkpts = num_segments + 1
    step = (max_val - min_val) / num_segments
    x_bkpts = Vector{Float64}(undef, num_bkpts)
    y_bkpts = Vector{Float64}(undef, num_bkpts)
    x_bkpts[1] = min_val
    y_bkpts[1] = f(min_val)
    for i in 1:num_segments
        x = min_val + step * i
        x_bkpts[i + 1] = x
        y_bkpts[i + 1] = f(x)
    end
    return x_bkpts, y_bkpts
end

"Returns x² (used as the default function for PWL breakpoint generation)."
_square(x::Float64) = x * x
