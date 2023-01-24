using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Distributed
@everywhere begin
    using Pkg; Pkg.activate(@__DIR__)
end

@everywhere using VIDA

using ArgParse
using CSV
using DataFrames
using Random


function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "arg1"
            help = "file list of fits images to read"
            arg_type = String
            required = true
        "--stride"
             help = "Checkpointing stride, i.e. number of steps."
             arg_type = Int
             default = 200
        "--out"
            help = "name of output files with extractions"
            default = "fit_summaries.csv"
        "--restart"
            help = "Tells the sampler to read in the old files and restart the run."
            action = :store_true
        "--template"
            help = "Parses sting with models separated by spaces. For instance to\n"*
                   "run a model with 1 m=(1,4) m-ring, 2 gaussian and a stretched disk\n"*
                   "one could do `--template mring_1_4 gauss_2 disk_1`. The current model\n"*
                   "options are: \n"*
                   "  - `[stretch]mring_n_m`: adds a [stretched] m-ring of order `n` `m` thickenss and azimuth resp.\n"*
                   "  - `gauss_n`           : add `n` asymmetric gaussian components to the template\n"*
                   "  - `[stretch]disk_n    : add `n` [stretched] flat top disks to the template\n"
            action = :store_arg
            nargs = '*'
            arg_type = String
            required = true
        "--seed"
            help = "Random seed for initial positions in extract"
            arg_type = Int
            default = 42
    end
    return parse_args(s)
end

function main()
    #Parse the command line arguments
    parsed_args = parse_commandline()
    #Assign the arguments to specific variables
    fitsfiles = parsed_args["arg1"]
    out_name = parsed_args["out"]
    seed = parsed_args["seed"]
    stride = parsed_args["stride"]
    templates = parsed_args["template"]
    @info "Template types $templates"
    restart = parsed_args["restart"]
    println("Using options: ")
    println("list of files: $fitsfiles, ")
    println("output name: $out_name, ")
    println("random seed $seed")
    println("Checkpoint stride $stride")

    #Read in a file and create list of images to template
    #the last line is the termination of the file
    files = split(read(fitsfiles,String),"\n")

    #check if the last entry of files is an empty string
    if files[end] == ""
        files = files[1:end-1]
    end

    println("Starting fit")


    #Now run on the files for real
    main_sub(files, out_name,
             templates,
             seed,
             restart, stride)
    println("Done! Check $out_name for summary")
    return 0
end

function make_initial_templates(templates...)
    res = make_initial_template.(templates)
    templates = reduce(vcat, getindex.(res, 1))
    lowers    = reduce(vcat, getindex.(res, 2))
    uppers    = reduce(vcat, getindex.(res, 3))

    println(typeof(templates))

    templates isa VIDA.AbstractTemplate && return (templates + 0.1*Constant(),
                                      lowers + 1e-8*Constant(),
                                      uppers + 5.0*Constant())

    template = templates[1]
    lower    = lowers[1]
    upper    = uppers[1]

    tcat = mapreduce(+, templates[2:end]; init=template) do l
        return 1.0*l
    end

    lcat = mapreduce(+, lowers[2:end]; init=lower) do l
        return 1e-8*l
    end

    ucat = mapreduce(+, uppers[2:end]; init=upper) do l
        return 10.0*l
    end

    return tcat + 1.0*Constant(), lcat + 1e-10*Constant(), ucat + 2.0*Constant()
end



function make_initial_template(template)
    if occursin("mring", template)
        stretch = occursin("stretch", template)
        type = parse.(Int, split(template, "_")[2:3])
        @info "Template includes a m-ring of order $(type) with stretch $stretch"
        return make_initial_template_mring(type, stretch)
    elseif occursin("gauss", template)
        ngauss = parse.(Int, split(template, "_")[end])
        @info "Template includes $ngauss gaussians"
        return make_initial_template_gauss(ngauss)
    elseif occursin("disk", template)
        ndisk = parse.(Int, split(template, "_")[end])
        stretch = occursin("stretch", template)
        @info "Template includes $ndisk disks with stretch $stretch"
        return make_initial_template_disk(ndisk, stretch)
    else
        @error "Template $template not available"
    end
end

function make_keyname(template)
    if occursin("mring", template)
        stretch = occursin("stretch", template)
        type = parse.(Int, split(template, "_")[2:3])
        return make_keynames_mring(type, stretch)
    elseif occursin("gauss", template)
        ngauss = parse.(Int, split(template, "_")[end])
        return make_keynames_gauss(ngauss)
    elseif occursin("disk", template)
        ndisk = parse.(Int, split(template, "_")[end])
        stretch = occursin("stretch", template)
        return make_keynames_disk(ndisk, stretch)
    else
        @error "Template $template not available"
    end
end

function make_keynames_mring(orders, stretch)
    println(orders)
    n = orders[1]+1
    m = orders[2]
    key_names = [:r0,
                [:σ for i in 1:n]...,
                [:ξσ for i in 2:n]...,
                [:s for i in 1:m]...,
                [:ξs for i in 1:m]...,
                :floor,
                :x0,
                :y0,
                ]
    if stretch
        push!(key_names, :τ, :ξτ)
    end
    return [key_names]
end

function make_keynames_gauss(n::Int)
    key_names = Vector{Symbol}[]
    for i in 1:n
        push!(key_names,
                    [
                     Symbol(:σ, "_g$i"),
                     Symbol(:τ, "_g$i"),
                     Symbol(:ξ, "_g$i"),
                     Symbol(:x, "_g$i"),
                     Symbol(:y, "_g$i"),
                    ])
    end
    return key_names
end

function make_keynames_disk(n::Int, stretch)
    key_names = Vector{Symbol}[]
    for i in 1:n
        push!(key_names,
                    [
                     Symbol(:r0, "_d$i"),
                     Symbol(:α, "_d$i"),
                     Symbol(:x, "_d$i"),
                     Symbol(:y, "_d$i"),
                    ])
        if stretch
            append!(key_names[i], [Symbol(:τ, "_d$i"), Symbol(:ξ, "_d$i")])
        end
    end
    return key_names
end

function make_keynames(templates...)
    tkeys = mapreduce(make_keyname, vcat, templates)
    for i in eachindex(tkeys)[begin+1:end]
        push!(tkeys[i], :Irel)
    end
    return push!(reduce(vcat, tkeys), :Irel)
end

function make_initial_template_gauss(n::Int)
    lower    = AsymGaussian(0.5, 0.001, 0.0, -60.0,-60.0)
    upper    = AsymGaussian(35.0, 0.999, π, 60.0, 60.0)
    template = AsymGaussian(5.0,0.001, 0.001, 0.0,0.0)
    return fill(template, n), fill(lower, n), fill(upper, n)
end

function make_initial_template_disk(n::Int, stretch)
    lower    = Disk(0.5, 0.01, -60.0,-60.0)
    upper    = Disk(35.0, 20.0, 60.0, 60.0)
    template = Disk(20.0, 1.0, 0.0,0.0)
    if stretch
        lower    = stretchrotate(lower, 0.0001, -π/2)
        upper    = stretchrotate(upper, 0.999, π/2)
        template = stretchrotate(template, 0.1, 0.0)
    end
    return fill(template, n), fill(lower, n), fill(upper, n)
end


function make_initial_template_mring(template_type, stretch)
    lower_σ = [0.01, [ -2.0 for i in 1:template_type[1]]... ]
    upper_σ = [30.0, [ 2.0 for i in 1:template_type[1]]... ]
    lower_ξσ = Float64[]
    upper_ξσ = Float64[]
    if template_type[1] > 0
        lower_ξσ = Float64[-π for i in 1:template_type[1] ]
        upper_ξσ = Float64[ π for i in 1:template_type[1] ]
    end
    lower_s = [0.001, [-0.99 for i in 2:template_type[2]]...]
    upper_s = [0.999, [0.99 for i in 2:template_type[2]]...]
    lower_ξs = [-π for i in 1:template_type[2] ]
    upper_ξs = [π for i in 1:template_type[2] ]

    lower = [10.0 ,
             lower_σ..., lower_ξσ...,
             lower_s..., lower_ξs...,
             0.001,
             -80.0, -80.0
            ]
    upper = [ 45.0 ,
              upper_σ..., upper_ξσ...,
              upper_s..., upper_ξs...,
              1.0,
              80.0, 80.0,
            ]
    t = SymCosineRingwFloor{template_type[1],template_type[2]}(lower.+0.1)
    if stretch
      push!(lower, 0.001, -π/2)
      push!(upper, 0.5, π/2)
      template = stretchrotate(t, 0.1, 0.0)
    else
      template = t
    end
    return template, typeof(template)(lower), typeof(template)(upper)
end

function create_initial_df(fitsfiles, templates, restart, outname)
    start_indx = 1
    nfiles = length(fitsfiles)
    df = DataFrame()
    if !restart
      #we want the keynames to match the model parameters
      key_names = make_keynames(templates...)
      for i in 1:length(key_names)
        insertcols!(df, ncol(df)+1, Symbol(key_names[i]) => zeros(nfiles); makeunique=true)
      end
      #fill the data frame with some likely pertinent information
      df[:, :divmin] = zeros(nfiles)
      df[:, :fitsfiles] =  fitsfiles
    else
      df = DataFrame(CSV.File(outname))
      start_indx = findfirst(isequal(0.0), df[:,1])
      println("Restarting run for $outname at index $start_indx")
    end
    return df, start_indx
end

@everywhere function fit_template(file, template,lower, upper)
    println("Extracting $file")
    image = load_fits(string(file))
    rimage = VIDA.rescale(image, 64, (-100.0, 100.0), (-100.0, 100.0))
    cimage = VIDA.clipimage(0.0,rimage)
    div = VIDA.LeastSquares(cimage)
    t = @elapsed begin
        prob = ExtractProblem(div, template, lower, upper)
        θ,divmin = extractor(prob, BBO(tracemode=:silent, maxevals=90_000))
        prob_new = ExtractProblem(div, θ, lower, upper)
        θ,divmin = extractor(prob_new, CMAES(cov_scale=0.01, verbosity=0))
    end
    println("This took $t seconds")
    return VIDA.unpack(θ), divmin
end


function main_sub(fitsfiles, out_name,
                  templates,
                  seed,
                  restart, stride)

    #"Define the template I want to use and the var bounds"
    model, lower, upper = make_initial_templates(templates...)

    #Set up the data frame to hold the optimizer output that
    #will be saved
    start_indx = 1
    df,start_indx = create_initial_df(fitsfiles, templates, restart, out_name)

    #Now fit the files!
    indexpart = Iterators.partition(start_indx:length(fitsfiles), stride)
    for ii in indexpart
      results = pmap(fitsfiles[ii]) do f
                fit_template(f, model, lower, upper)
      end
      df[ii,1:VIDA.size(typeof(lower))] = hcat(first.(results)...)'
      df[ii,VIDA.size(typeof(lower))+1] = last.(results)
      df[ii,end] = fitsfiles[ii]
      #save the file
      println("Checkpointing $(ii)")
      CSV.write(out_name, df)
    end
    CSV.write(out_name, df)

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
