# sed_vis
Plots best-fit SEDs from Naveen Reddy's SED Fitting software.

    Parameters:
          phot_in (str): Photometric input file provided for SED fits
          filt_file (str): File that lists filters used in the above fit
          res_dir (str): Results directory (contains summary*.dat and bestfit/bestfit*.dat)
          filt_dir (str): Filter directory (contains *.bp)

    Optional parameters:
          sfh_ages (list or str):  Specific SFH/ages to plot (i.e "csf_allage"). If none
                                   supplied, defaults to plotting all found in res_dir.

This script will place figure files in res_dir/plots/*.pdf.
    
## Example
```python
from sed_vis import *

phot_in = "./EXAMPLE/westphal_inphot_refit.dat"
filt_file = "./EXAMPLE/filters_standard.westphal.v2"
res_dir = "./EXAMPLE/results/"
filt_dir = "./EXAMPLE/filters/"

sed_vis(phot_in, filt_file, res_dir, filt_dir, sfh_ages="csf_agegt50")
```
Generates plots in ./EXAMPLE/results/plots/*.pdf.
    
Here's one:
![alt text](https://github.com/tonypahl/sed_vis/blob/main/example_output.png?sanitize=true)
    
    
