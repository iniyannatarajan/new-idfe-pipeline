using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
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
            help = "Cosine template to use for feature extraction. Expects two numbers"
            action = :store_arg
            nargs = 2
            default=["1", "4"]
        "--seed"
            help = "Random seed for initial positions in extract"
            arg_type = Int
            default = 42
        "--stretch"
            help = "Includes a ellipticity in the ring"
            action = :store_true
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
    template_type = parse.(Int, parsed_args["template"])
    println("Template type $template_type")
    restart = parsed_args["restart"]
    stretch = parsed_args["stretch"]
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
             template_type, stretch,
             seed,
             restart, stride)
    println("Done! Check $out_name for summary")
    return 0
end

function make_initial_template(template_type, stretch)
    @show template_type
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
    return template, lower, upper
end

function create_initial_df!(df, fitsfiles, template, restart, outname, stretch)
    start_indx = 1
    nfiles = length(fitsfiles)
    df = DataFrame()
    if !restart
      #we want the keynames to match the model parameters
      n = first(template)+1
      m = last(template)
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
        push!(key_names, :τ, :ξτ, :Irel)
      else
        push!(key_names, :Irel)
      end
      for i in 1:length(key_names)
        insertcols!(df, ncol(df)+1, Symbol(key_names[i]) => zeros(nfiles); makeunique=true)
      end
      @show key_names
      #fill the data frame with some likely pertinent information
      setproperty!(df, :divmin, zeros(nfiles))
      setproperty!(df, :fitsfiles,  fitsfiles)
    else
      df = DataFrame(CSV.File(outname, delim=";"))
      start_indx = findfirst(isequal(0.0), df[:,:r0])
      println("Restarting run for $outname at index $start_indx")
    end
    return df, start_indx
end

@everywhere function fit_func(template,lower, upper)
    function (file)
        println("Extracting $file")
        image = load_fits(string(file))
        rimage = VIDA.rescale(image, 64, (-100.0, 100.0), (-100.0, 100.0))
        cimage = VIDA.clipimage(0.0,rimage)
        div = VIDA.LeastSquares(cimage)
        t = @elapsed begin prob = ExtractProblem(div, template, lower, upper)
            θ,divmin = extractor(prob, BBO(tracemode=:silent, maxevals=50_000))
            prob_new = ExtractProblem(div, θ, lower, upper)
            θ,divmin = extractor(prob_new, CMAES(cov_scale=0.01, verbosity=0))
        end
        println("This took $t seconds")
        return VIDA.unpack(θ), divmin
    end
end


function main_sub(fitsfiles, out_name,
                  template_type, stretch,
                  seed,
                  restart, stride)

    #"Define the template I want to use and the var bounds"
    matom, l, u = make_initial_template(template_type, stretch)
    model = matom + 0.1*Constant()
    lower = typeof(matom)(l) + 1e-8*Constant()
    upper = typeof(matom)(u) + 1.0*Constant()


    #Need to make sure all the procs know this information
    @everywhere model = $(model)
    @everywhere lower = $(lower)
    @everywhere upper = $(upper)

    #Set up the data frame to hold the optimizer output that
    #will be saved
    start_indx = 1
    df,start_indx = create_initial_df!(start_indx,fitsfiles, template_type, restart, out_name, stretch)
    rng = MersenneTwister(seed) #set the rng

    #Now fit the files!
    @everywhere fit = fit_func(model,lower,upper)
    indexpart = Iterators.partition(start_indx:length(fitsfiles), stride)
    for ii in indexpart
      results = pmap(fit, fitsfiles[ii])
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
