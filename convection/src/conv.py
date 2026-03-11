#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#SBATCH --job-name=conv
#SBATCH --partition=compute
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=mh1421
#SBATCH --output=conv.o%j
#SBATCH --error=conv.o%j

# %% imports, settings
import sys

sys.path.append("../../common/src")
from intake_demo import load_cmip6_datasets
from intake_demo import *
import tools
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pickle


# %% functions
# center of convection = centroid of wap_500 on the equator in the Indo-West Pacific
def conv_center(da_wap, da_sftlf):
    da_wap_eq = da_wap.sel(lat=slice(-5, 5), lon=slice(50, 200)).mean("lat").compute()
    da_wap_eq = da_wap_eq.where(da_wap_eq < 0, drop=True)
    # weighted mean longitude of convection
    conv_center = (da_wap_eq * da_wap_eq.lon).sum("lon") / da_wap_eq.sum("lon")
    return conv_center


def land_ocean_contrast(da_tas, da_sftlf, da_areacella):
    # determine range of land sea mask: 0 to 1 or 0 to 100?
    if da_sftlf.max() > 10:
        da_sftlf = da_sftlf / 100
    da_land = da_tas.where(da_sftlf > 0.5)
    da_ocean = da_tas.where(da_sftlf <= 0.5)

    # area-weighted mean
    land_mean = tools.fldmean(da_land, da_areacella)
    ocean_mean = tools.fldmean(da_ocean, da_areacella)

    return land_mean - ocean_mean


def trade_wind_strength(da_wind, da_sftlf, da_areacella):
    # determine range of land sea mask: 0 to 1 or 0 to 100?
    if da_sftlf.max() > 10:
        da_sftlf = da_sftlf / 100
    da_ocean = da_wind.where(da_sftlf <= 0.5, drop=True)

    # area-weighted mean
    ocean_mean = tools.fldmean(da_ocean, da_areacella)

    return ocean_mean


def process_model(model):
    try:
        print(f"Processing model: {model}")
        dset_key_piControl = f"CMIP.{model}.piControl.Amon.gn"
        dset_key_4xco2 = f"CMIP.{model}.abrupt-4xCO2.Amon.gn"
        dset_key_fx = f"CMIP.{model}.piControl.fx.gn"

        if not dset_key_piControl in dset_dict_piControl:
            dset_key_piControl = f"CMIP.{model}.piControl.Amon.gr"
            dset_key_4xco2 = f"CMIP.{model}.abrupt-4xCO2.Amon.gr"
            dset_key_fx = f"CMIP.{model}.piControl.fx.gr"

        da_sftlf = dset_dict_piControl[dset_key_fx]["sftlf"]
        da_areacella = dset_dict_piControl[dset_key_fx]["areacella"]

        # fix that some models have slightly different horizontal grids for Amon and fx
        da_sftlf["lon"] = dset_dict_piControl[dset_key_piControl]["lon"]
        da_areacella["lon"] = dset_dict_piControl[dset_key_piControl]["lon"]
        da_sftlf["lat"] = dset_dict_piControl[dset_key_piControl]["lat"]
        da_areacella["lat"] = dset_dict_piControl[dset_key_piControl]["lat"]

        da_tas_piControl = (
            tools.annual_mean_from_monthly(dset_dict_piControl[dset_key_piControl])
            .tas.isel(year=slice(-100, None))
            .mean("year")
            .squeeze()
        )
        da_wap_piControl = (
            tools.annual_mean_from_monthly(dset_dict_piControl[dset_key_piControl])
            .wap.isel(year=slice(-100, None))
            .mean("year")
            .squeeze()
            .compute()
        )
        da_sfcWind_piControl = (
            tools.annual_mean_from_monthly(dset_dict_piControl[dset_key_piControl])
            .sfcWind.isel(year=slice(-100, None))
            .mean("year")
            .squeeze()
            .compute()
        )
        da_tas_4xco2 = (
            tools.annual_mean_from_monthly(dset_dict_4xco2[dset_key_4xco2])
            .tas.squeeze()
            .isel(year=1)
        )
        da_wap_4xco2 = (
            tools.annual_mean_from_monthly(dset_dict_4xco2[dset_key_4xco2])
            .wap.isel(year=1)
            .squeeze()
            .compute()
        )
        da_sfcWind_4xco2 = (
            tools.annual_mean_from_monthly(dset_dict_4xco2[dset_key_4xco2])
            .sfcWind.isel(year=1)
            .squeeze()
            .compute()
        )

        da_delta_tas = (da_tas_4xco2 - da_tas_piControl).compute()
        delta_contrast = land_ocean_contrast(
            da_delta_tas, da_sftlf, da_areacella
        ).compute()
        delta_contrast_tr = land_ocean_contrast(
            da_delta_tas.sel(lat=slice(-8, 8)), da_sftlf, da_areacella
        ).compute()
        delta_conv = (
            conv_center(da_wap_4xco2, da_sftlf)
            - conv_center(da_wap_piControl, da_sftlf)
        ).compute()
        delta_sfcWind = tools.fldmean(
            tools.select_sep(
                da_sfcWind_4xco2 - da_sfcWind_piControl, da_sftlf
            ).compute(),
            da_areacella,
        )

        return (
            delta_contrast.values,
            delta_contrast_tr.values,
            delta_conv.values,
            delta_sfcWind.values,
        )
    except Exception as e:
        print(f"Error processing model {model}: {e}")
        return (None, None, None, None)


def is_interactive_session():
    return "ipykernel" in sys.modules or hasattr(sys, "ps1")


# %% load data
if __name__ == "__main__":
    table_ids = ["Amon", "fx"]
    variable_ids = [
        "tas",
        "wap",
        "uas",
        "vas",
        "sfcWind",
        "sftlf",
        "areacella",
    ]
    source_ids = [
        "MPI-ESM1-2-LR",
        "CESM2",
        "MIROC6",
    ]
    query_piControl = dict(
        experiment_id="piControl",
        table_id=table_ids,
        variable_id=variable_ids,
        # source_id=source_ids,
        grid_label=["gn", "gr"],
        member_id="r1i1p1f1",
    )
    query_4xco2 = dict(
        experiment_id="abrupt-4xCO2",
        table_id=table_ids,
        variable_id=variable_ids,
        # source_id=source_ids,
        grid_label=["gn", "gr"],
        member_id="r1i1p1f1",
    )
    dset_dict_piControl = load_cmip6_datasets(
        query_piControl,
        catalog_name="cat_piControl_conv",
        force_reload=False,
    )
    dset_dict_4xco2 = load_cmip6_datasets(
        query_4xco2,
        catalog_name="cat_4xco2_conv",
        force_reload=False,
    )
    for key, value in dset_dict_piControl.items():
        try:
            if "Amon" in key:
                # only select 500 hPa level for vertical velocity
                dset_dict_piControl[key]["wap"] = value["wap"].sel(
                    plev=500e2, method="nearest"
                )
                # only select last 100 years to speed up processing
                dset_dict_piControl[key] = value.isel(time=slice(-100 * 12, None))
        except KeyError:
            pass
    for key, value in dset_dict_4xco2.items():
        try:
            if "Amon" in key:
                # only select 500 hPa level for vertical velocity
                dset_dict_4xco2[key]["wap"] = value["wap"].sel(
                    plev=500e2, method="nearest"
                )
        except KeyError:
            pass
# %% computations
if __name__ == "__main__":
    # Extract model names from keys (word between first and second dot)
    models_piControl = set(key.split(".")[1] for key in dset_dict_piControl.keys())
    models_4xco2 = set(key.split(".")[1] for key in dset_dict_4xco2.keys())
    common_models = models_piControl & models_4xco2

    model_list = [k for k in common_models]
    # run in parallel if not in an interactive session, otherwise sequentially
    if is_interactive_session():
        results = [process_model(model) for model in model_list]
    else:
        # Python 3.14 may default to a non-fork start method, which does not
        # inherit globals defined in __main__ (e.g., dset_dict_*).
        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=max(1, mp.cpu_count() // 2),
            mp_context=mp_context,
        ) as executor:
            results = list(executor.map(process_model, model_list))

    (
        delta_land_ocean_contrasts,
        delta_land_ocean_contrasts_tr,
        delta_conv_centers,
        delta_sfcWinds,
    ) = zip(*results)

# pickle results
if __name__ == "__main__":
    with open("../data/conv_results.pkl", "wb") as f:
        pickle.dump(
            {
                "delta_land_ocean_contrasts": delta_land_ocean_contrasts,
                "delta_land_ocean_contrasts_tr": delta_land_ocean_contrasts_tr,
                "delta_conv_centers": delta_conv_centers,
                "delta_sfcWinds": delta_sfcWinds,
            },
            f,
        )

# %% plots
# load pickled results
with open("../data/conv_results.pkl", "rb") as f:
    results = pickle.load(f)
    delta_land_ocean_contrasts = results["delta_land_ocean_contrasts"]
    delta_land_ocean_contrasts_tr = results["delta_land_ocean_contrasts_tr"]
    delta_conv_centers = results["delta_conv_centers"]
    delta_sfcWinds = results["delta_sfcWinds"]

if __name__ == "__main__":
    for ii, (x, y, xname, yname, savename) in enumerate(
        zip(
            (
                delta_land_ocean_contrasts,
                delta_land_ocean_contrasts_tr,
                delta_conv_centers,
            ),
            (delta_conv_centers, delta_conv_centers, delta_sfcWinds),
            (
                "Land-Ocean $\Delta T$ contrast / K",
                "Tr. Land-Ocean $\Delta T$ contrast / K",
                "Center of convection shift / °",
            ),
            (
                "Center of convection shift / °",
                "Center of convection shift / °",
                "SEP wind speed change / ms$^{-1}$",
            ),
            (
                "scatter_contrast_conv.png",
                "scatter_contrast_tr_conv.png",
                "scatter_conv_sfcWind.png",
            ),
        )
    ):
        # unpack the lists
        x = [a for a in x if a is not None]
        y = [a for a in y if a is not None]
        for i in range(len(x)):
            if np.isnan(x[i]).any() or np.isnan(y[i]).any():
                x[i] = None
                y[i] = None
        x = [a for a in x if a is not None]
        y = [a for a in y if a is not None]
        try:
            x = [a[0][0] for a in x]
        except (IndexError, TypeError):
            pass
        try:
            y = [a[0][0] for a in y]
        except (IndexError, TypeError):
            pass
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(x, y, s=100)
        r, p = pearsonr(x, y)
        ax.text(
            0.05,
            0.95,
            f"r$^2$={r**2:.2f}, p={p:.2e}",
            transform=ax.transAxes,
            verticalalignment="top",
        )

        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        plt.savefig(f"../plots/{savename}", dpi=300, bbox_inches="tight")
        plt.close()
