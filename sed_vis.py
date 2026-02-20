# Written by Anthony Pahl, 7/28/21
# Visualize best fit SEDs from Naveen Reddy's fitting code. Displays
# best-fit parameters of the fit.

import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from astropy.io import ascii
import astropy.units as u
import math
from scipy import interpolate
import glob

SFH_LABELS = {
    "csf_": "Constant SFH",
    "tau_": "Tau model",
    "taur_": "Tau rising model",
}

def read_bestfit_params(sum_file, ids):

    psum = ascii.read(sum_file).to_pandas()
    sum_cols = ["id", "tau", "ebmv", "age", "sfr", "mass", "chisq"]
    psum.columns = sum_cols

    # Mass in logspace
    with np.errstate(divide="ignore"):
        psum.mass = np.log10(psum.mass.astype("float"))

    # Ensure one-to-one match for IDs
    pbst = DataFrame(index=ids, columns=sum_cols[1:])
    for i, r in psum.iterrows():
        if r.id in pbst.index:
            pbst.loc[r.id, sum_cols[1:]] = r.values[1:]

    return pbst


def read_phot_in(photfile, filtfile):

    pfilt = ascii.read(filtfile).to_pandas()
    pin = ascii.read(photfile).to_pandas()

    # Replace index with first column (ID)
    pin.index = pin.col1.values
    pin.drop("col1", axis=1, inplace=True)

    # Replace column names with filters used
    rep_dir = {}
    for fi in pfilt.col1:
        iif = pfilt.loc[pfilt.col1 == fi].index[0]
        mcol = "col{}".format(iif * 2 + 2)
        mecol = "col{}".format(iif * 2 + 3)

        rep_dir[mcol] = "{}_mag".format(fi)
        rep_dir[mecol] = "{}_emag".format(fi)

    rep_dir[pin.columns[-1]] = "zsys"
    pin.rename(columns=rep_dir, inplace=True)

    # Replace missing values
    pin.replace(9999.0, np.nan, inplace=True)
    pin.replace(99.0, np.nan, inplace=True)

    return pin


def reformat_obs_phot(phot_series, filt_dir):
    fis = []
    ws = []
    ms = []
    ems = []
    for fn, vn in phot_series.items():
        if "_mag" in fn:
            ms.append(vn)
            ws.append(
                (find_central_wave(fn.replace("_mag", ""), filt_dir) * u.AA)
                .to(u.micron)
                .value
            )
            fis.append(fn.replace("_mag", ""))
        elif "_emag" in fn:
            ems.append(vn)
        else:
            pass

    return DataFrame(
        index=fis, columns=["wave", "mag", "emag"], data=np.array([ws, ms, ems]).T
    )


def find_central_wave(filt, filt_dir):
    bp_file = "{}/{}.bp".format(filt_dir, filt)
    pbp = ascii.read(bp_file).to_pandas()
    pbp.columns = ["wave", "trans"]

    pbp_sl = pbp.loc[pbp.trans > 0.5]
    cen_wave = pbp_sl.iloc[int(pbp_sl.shape[0] / 2)].wave

    return cen_wave


def convolve_filter_sed(filt, wave, flam, filt_dir):
    bp_file = "{}/{}.bp".format(filt_dir, filt)
    pbp = ascii.read(bp_file).to_pandas()
    pbp.columns = ["wave", "trans"]

    tck = interpolate.splrep(wave, flam, s=0)
    flam_new = interpolate.splev(pbp.wave, tck, der=0)

    # Calculate delta wavelength
    dlams = np.append(np.diff(pbp.wave), 0.0)
    flux = flam_new * pbp.trans.values * dlams

    # Calculate total integrated filter response
    bw = pbp.trans.values * dlams

    # Convert to AB mag
    return -2.5 * np.log10(flux.sum() / bw.sum()) - 48.6


def add_modelmag_pd(pobs2, wave, fnu, filt_dir):
    # for each filter ..
    pobs = pobs2.copy(deep=True)

    for i, r in pobs.iterrows():
        mag_model = convolve_filter_sed(i, wave, fnu, filt_dir)
        pobs.loc[i, "bmag"] = mag_model

    return pobs


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx


def get_bestfit_sed(res_dir, obj_id, sfh_age):
    path = f"{res_dir}/bestfit/bestfit.{obj_id}.{sfh_age}.dat"
    if not os.path.isfile(path):
        return None
    psed = ascii.read(path).to_pandas()
    psed.columns = ["wave", "flam"]
    flam = psed.flam.values * u.erg / u.s / u.cm ** 2 / u.AA
    lam = psed.wave.values * u.AA
    fnu = flam.to(u.erg / u.s / u.cm ** 2 / u.Hz, u.spectral_density(lam))
    abmag = fnu.to(u.ABmag)
    psed["fnu"] = fnu.value
    psed["ABmag"] = abmag.value
    # convert wavelength to microns
    mlam = lam.to(u.micron)
    psed["wave_um"] = mlam.value
    return psed


def compute_label_position(ax, fr):
    if np.isnan(fr.mag):
        return None, None
    # Define upper and lower possible positions
    if fr.mag > fr.bmag:
        if np.isnan(fr.emag):
            lpos = fr.mag + 0.1
        else:
            lpos = fr.mag + fr.emag + 0.1
        upos = fr.bmag - 0.1
    else:
        if np.isnan(fr.emag):
            upos = fr.mag - 0.1
        else:
            upos = fr.mag - fr.emag - 0.1
        lpos = fr.bmag + 0.1
    upos_ax = ax.transLimits.transform([fr.wave, upos])
    lpos_ax = ax.transLimits.transform([fr.wave, lpos])
    # Check if going above or below the figure
    if upos_ax[1] > 0.87:
        pos = lpos
        va = "top"
    elif lpos_ax[1] < 0.13:
        pos = upos
        va = "bottom"
    else:
        if fr.mag > fr.bmag:
            pos = lpos
            va = "top"
        else:
            pos = upos
            va = "bottom"
    return pos, va

def decode_sfh_age(sfh_age):
    sfh = next(
        (label for key, label in SFH_LABELS.items() if key in sfh_age),
        sfh_age.split("_")[0]
        )
    age = (
        "All ages" if "allage" in sfh_age
        else r"Age $>$ 50Myr" if "agegt50" in sfh_age
        else sfh_age.split("_")[1]
        )
    return [sfh, age]


def sed_vis(phot_in, filt_file, res_dir, filt_dir, sfh_ages=None):
    """
    Plots best-fit SEDs from Naveen Reddy's SED Fitting software.

    Parameters:
          phot_in (str): Photometric input file provided for SED fits
          filt_file (str): File that lists filters used in the above fit
          res_dir (str): Results directory (contains summary*.dat and bestfit/bestfit*.dat)
          filt_dir (str): Filter directory (contains *.bp)

    Optional parameters:
          sfh_ages (list or str):  Specific SFH/age fits to plot (i.e "csf_allage"). If none
                                   supplied, defaults to plotting all found in res_dir.

    Generates plots in res_dir/plots/*.pdf
    """

    colors = np.array(
        [
            ["#9E3549", "#C98B97", "#B45C6D", "#89152C", "#740017"],
            ["#256F5B", "#628D81", "#417E6D", "#0F604A", "#00523B"],
            ["#89A236", "#BFCD8F", "#A3B85E", "#708C15", "#5B7700"],
        ],
        dtype="<U7",
    )

    # Check inputs
    if not os.path.isfile(phot_in):
        raise Exception("File not found: {}".format(phot_in))
    if not os.path.isfile(filt_file):
        raise Exception("File not found: {}".format(filt_file))
    if not os.path.isdir(res_dir):
        raise Exception("Not a directory: {}".format(res_dir))
    else:
        res_dir = os.path.abspath(res_dir)
    if not os.path.isdir(filt_dir):
        raise Exception("Not a directory: {}".format(filt_dir))
    else:
        filt_dir = os.path.abspath(filt_dir)

    # Construct array of input summary files
    if sfh_ages is None:
        sum_files = glob.glob("{}/summary_*.dat".format(res_dir))
    elif type(sfh_ages) is list:
        sum_files = [
            "{}/summary_{}.dat".format(res_dir, sfh_age) for sfh_age in sfh_ages
        ]
    else:
        sum_files = ["{}/summary_{}.dat".format(res_dir, sfh_ages)]

    # Create plotting directory
    plot_dir = "{}/plots/".format(res_dir)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # Generate one set of figures per SFH/age combination
    for sum_file in sum_files:
        # Check if file is empty
        if os.stat(sum_file).st_size == 0:
            print("{} is empty".format(os.path.basename(sum_file)))
            continue

        sfh_age = (
            sum_file.replace(res_dir, "")
            .replace("summary_", "")
            .replace("/", "")
            .replace(".dat", "")
        )
        pobsr = read_phot_in(phot_in, filt_file)
        pbst = read_bestfit_params(sum_file, pobsr.index.values)

        # Generate one PDF per SED fit
        for i, r in pbst.iterrows():
            # Read in best fit SED
            psed = get_bestfit_sed(res_dir, i, sfh_age)
            if psed is None:
                print(f"Not found: bestfit.{i}.{sfh_age}.dat")
                continue

            # Instantiate figure
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)

            # Plot best fit SED
            ax.plot(
                psed.wave_um,
                psed.ABmag,
                color=colors[0][2],
                ds="steps-mid",
                marker="",
                linestyle="-",
                label="Best fit SED",
                zorder=2.0,
            )

            # Plot observed photometry
            pobsc = pobsr.loc[i]
            pobs = reformat_obs_phot(pobsc, filt_dir)
            # Defined errorbars
            pdef = pobs.loc[~np.isnan(pobs.emag)]
            ax.errorbar(
                pdef.wave,
                pdef.mag,
                yerr=pdef.emag,
                color="k",
                linestyle="",
                marker="o",
                ms=10,
                mfc="none",
                mew=3,
                label="Observed photometry",
                zorder=2.5,
            )
            # Undefined errorbars
            pudef = pobs.loc[np.isnan(pobs.emag)]
            ax.errorbar(
                pudef.wave,
                pudef.mag,
                yerr=pudef.emag,
                color="k",
                linestyle="",
                marker="x",
                ms=10,
                mfc="none",
                mew=3,
                label="Undef mag error",
                zorder=2.5,
            )

            # Plot predicted photometry from best-fit model
            pobs = add_modelmag_pd(pobs, psed.wave.values, psed.fnu, filt_dir)
            pobs_magdef = pobs.loc[~np.isnan(pobs.mag)]
            ax.scatter(
                pobs_magdef.wave,
                pobs_magdef.bmag,
                marker="s",
                s=100,
                color=colors[0][4],
                fc="none",
                linewidths=2,
                alpha=1.0,
                label="Model prediction",
                zorder=2.2,
            )

            # Axes limits and config
            # x axis: [min(filter_wavelengths) - 0.2, max(filter_wavelengths) + 0.5]
            xmin = pobs.wave.min() - 0.2
            xmax = pobs.wave.max() + 0.5
            xlim = [xmin, xmax]
            # y axis: [max(observed_photometry) + 1.0, min(observed_photometry, best_SED) - 0.5]
            ymin = pobs.mag.max() + 1
            ymax = min([psed.ABmag.min(), pobs.mag.min()]) - 0.5
            ylim = [ymin, ymax]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.semilogx()
            # Set up tick labels for microns
            xt = np.array([0.1, 0.5, 1, 2, 4, 8, 24, 160, 500]) * 1.0e4
            valid_ticks = (xt > xlim[0] * 1.0e4) & (xt < xlim[1] * 1.0e4)
            if valid_ticks.sum() > 0:
                xt = xt[valid_ticks]
                ax.set_xticks(xt / 1.0e4)
                ax.set_xticklabels(xt / 1.0e4)

            # Label observed photometry
            for fi, fr in pobs.iterrows():
                pos, va = compute_label_position(ax, fr)
                if pos is None:
                    continue
                ax.text(
                    fr.wave,
                    pos,
                    fi,
                    ha="center",
                    va=va,
                    color="k",
                    size=14,
                    rotation=90,
                )

            # Axes labels
            ax.set_xlabel(r"$\lambda_{obs}$ ($\mu$m)")
            ax.set_ylabel("AB Mag")

            # Legend
            ax.legend(loc="upper left", fontsize=14)

            # Best-fit parameters
            sfh_age_str = decode_sfh_age(sfh_age)
            fit_info = [
                *sfh_age_str,
                r"$\tau$/Myr:    {}".format(r.tau),
                r"EBMV:    {}".format(r.ebmv),
                "Age:   {} Myr".format(r.age),
                r"SFR:   {} M$_{{\odot}}$/yr".format(r.sfr),
                r"log(M$_*$/M$_{{\odot}}$):   {:.3f}".format(r.mass),
                r"$\chi^2$:   {}".format(r.chisq),
            ]
            ax.annotate(
                "\n".join(fit_info),
                [0.78, 0.03],
                xycoords="axes fraction",
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round", fc="w"),
                fontsize=14,
                usetex=True,
                family="serif",
            )

            # Title
            ax.set_title(r"{} at $z$={:.3f}".format(i, pobsc.zsys), fontsize=15)

            fig.tight_layout()
            fig.savefig("{}/{}_{}.pdf".format(plot_dir, i, sfh_age))
            plt.close(fig)


if __name__ == "__main__":

    phot_in = "./EXAMPLE/westphal_inphot_refit.dat"
    filt_file = "./EXAMPLE/filters_standard.westphal.v2"
    res_dir = "./EXAMPLE/results/"
    filt_dir = "./EXAMPLE/filters/"

    sed_vis(phot_in, filt_file, res_dir, filt_dir, sfh_ages="csf_agegt50")
