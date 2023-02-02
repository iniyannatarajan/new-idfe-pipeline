# Running the Script

The main script to run is `vida_LS_general.jl`. This can be run from the command line as follows

```
> julia -p NCORES vida_LS_general.jl image_list.txt --stride 200 --out output_name.csv --template stretchmring_1_4 --stretch
```
where:
    - *-p NCORES*: is the number physical CPU you want to use analyze (do not pick more physical cores than what you have on the machine)
    - *image_list.txt* is the list of absolute paths to the FITS image files you want to analyze with VIDA
    - *--stride* This is the checkpoint stride to use when analyzing
    - *--out* is the name of the CSV file where the output will be saved
    - *--template* is the predefined name of the model; for instance, 'stretchmring_1_4' to fit an m-ring with equal width and a fourth order m-ring expansion in brightness
    - *--stretch* Whether the m-ring you fit to the image include stretch/ellipticity (for M87 this should always be set)

