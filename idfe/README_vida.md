# Running the Script

The main script to run is `vida_LS_stretched_mring.jl`. This can be run from the command line as follows

```
> julia -p NCORES vida_LS_stretched_mring.jl image_list.txt --stride 50 --out output_name.csv --template 0 4 --stretch
```
where:
    - *-p NCORES*: is the number physical CPU you want to use analyze (do not pick more physical cores than what you have on the machine)
    - *image_list.txt* is the list of absolute paths to the FITS image files you want to analyze with VIDA
    - *--stride* This is the checkpoint stride to use when analyzing
    - *--out* is the name of the CSV file where the output will be saved
    - *--template* is the degree of m-ring you want to fit. `--template 0 4` uses a m-ring with equal width and a fourth order m-ring expansion in brightness
    - *--stretch* Whether the m-ring you fit to the image include stretch/ellipticity (for M87 this should always be set)

