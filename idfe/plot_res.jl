using CairoMakie
using DataFrames
using CSV


df = CSV.read("fits_summaries.csv", DataFrame)
# Extract the band number
df.band = string.(first.(getindex.(split.(df.fitsfiles, "_b"),2)))
df.τe =

dfb = groupby(df, :band)

function make_fit(dfb)
    fig = Figure(;resolution=(700,300))
    ax1 = Axis(fig[1,1], xlabel="diameter √ab (μas)", yticklabelsvisible=false)
    ax2 = Axis(fig[1,2], xlabel="ellipticity 1-b/a (μas)", yticklabelsvisible=false)
    ax3 = Axis(fig[1,3], xlabel="ellipticity PA (deg)", yticklabelsvisible=false)

    density!(ax1, 2 .*dfb[1].r0, label="Band 1")
    density!(ax1, 2 .*dfb[2].r0, label="Band 3")
    vlines!(ax1, [40.7], color=:black, linestyle=:dash, label="Truth")

    density!(ax2, dfb[1].τ, label="Band 1")
    density!(ax2, dfb[2].τ, label="Band 3")
    vlines!(ax2, [0.187], color=:black, linestyle=:dash)

    density!(ax3, rad2deg.(dfb[1].ξτ), label="Band 1")
    density!(ax3, rad2deg.(dfb[2].ξτ), label="Band 3")
    vlines!(ax3, [0.0], color=:black, linestyle=:dash)


    xlims!(ax1, 15.0, 48.0)
    xlims!(ax2, 0.0, 0.5)
    ylims!(ax1, low=0.0)
    ylims!(ax2, low=0.0)

    axislegend(ax1, framevisible=false, position=:lt)

    return fig
end

fig = make_fit(dfb)
