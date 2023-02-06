using VIDA
using CSV
using DataFrames
using Statistics
using ArgParse
using NamedTupleTools



function row2template(row::DataFrameRow, model)
    k = String.(keys(row))
    kn = Tuple(Symbol.(filter(x->!(occursin("s", x)||occursin("σ", x)), k)))
    sorder = length(filter(x->occursin("ξs",x), k))
    σorder = length(filter(x->occursin("ξσ", x), k))

    skeys = push!(["s"], ("s_$i" for i in 1:(sorder-1))...)
    ξskeys = push!(["ξs"], ("ξs_$i" for i in 1:(sorder-1))...)
    σkeys = push!(["σ"], ("σ_$i" for i in 1:σorder)...)
    ξσkeys = push!(["ξσ"], ("ξσ_$i" for i in 1:(σorder-1))...)

    v1 = getproperty.(Ref(row), kn)
    t1 =  NamedTuple{kn}(Tuple(v1))
    ts = (  s=getproperty.(Ref(row), skeys),
            ξs = getproperty.(Ref(row), ξskeys),
         )
    if σorder > 1
        tσ = (
                σ = getproperty.(Ref(row), σkeys),
                ξσ = getproperty.(Ref(row), ξσkeys),
            )
    else
        tσ = (σ = Float64[row[:σ]], ξσ=Float64[])
    end

    modelarg = merge(merge(t1, ts), tσ)
    modelarg = delete(modelarg, :divmin, :fitsfiles, :Irel)
    if :τ ∈ kn
        modelarg = delete(modelarg, :τ, :ξτ)
        return stretchrotate(model{σorder, sorder}(;modelarg...), row[:τ], row[:ξτ])
    else
        return model{σorder, sorder}(;modelarg...)
    end
end

const RSCosine = VIDA.RotateMod{<:VIDA.StretchMod{<:SymCosineRingwFloor}}
getcenter(m::SymCosineRingwFloor) = (m.x0, m.y0)
getcenter(m::RSCosine) = VIDA.imagecenter(m)
getradius(m::SymCosineRingwFloor) = m.r0
getradius(m::RSCosine) = VIDA.basetemplate(VIDA.basetemplate(m)).r0

function get_fc(m::Union{SymCosineRingwFloor, VIDA.RotateMod{<:VIDA.StretchMod{<:SymCosineRingwFloor}}})
    x0,y0 = getcenter(m)
    r0 = getradius(m)
    θitr = range(0.0, 2π, length=512)[1:end-1]
    ringbrightness = 0.0
    for th in θitr
        x,y = r0*cos(th)+x0, r0*sin(th)+y0
	ringbrightness += m(x,y)
    end
    ringbrightness = ringbrightness*step(θitr)/(2*pi)
    diskbrightness = 0.0
    ritr = range(0.0, 5.0, length=512)
    for r in ritr
        for θ in θitr
            x,y = r*cos(θ)+x0, r*sin(θ)+y0
            diskbrightness += m(x,y)*r
        end
    end
    diskbrightness = diskbrightness*step(ritr)*step(θitr)/(pi*25.0)
    diskbrightness/ringbrightness
end

function main(args)
    s = ArgParseSettings(description="Convert the VIDA's central flux floor to REx's brightness ratio")

    @add_arg_table! s begin
	"arg1"
	  help="List of VIDA csv files to read in"
	  arg_type = String
	  required = true
    end

    parsed_args = parse_args(args, s)
    filelist = parsed_args["arg1"]

    #Read in the file list
    #the last line is the termination of the file
    files = split(read(filelist,String),"\n")

    #check if the last entry of files is an empty string
    if files[end] == ""
    	files = files[1:end-1]
    end

    #mstring = parsed_args["model"]
    #if mstring == "FLOOR"
    #	println("Using SymCosineRingwFloor")
	model = SymCosineRingwFloor
    # elseif mstring == "GFLOOR"
    	# println("Using SymCosineRingwGFloor")
	# model = SymCosineRingwGFloor
    # else
    #    throw("$(mstring) not found")
    # end


    for f in files
        @info "On file $f"
	    df = CSV.read(f, DataFrame)
	    ms = row2template.(eachrow(df), model)
	    df[!,:fc] = get_fc.(ms)
	    CSV.write(replace(basename(f), ".csv" => "_fc2.csv"), df)
    end


end

main(ARGS)
