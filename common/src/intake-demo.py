"""
How to use intake to load CMIP6 data on levante.
"""

# %% imports
import intake
from dask.distributed import LocalCluster, Client
from xmip.preprocessing import combined_preprocessing
import traceback
import pandas as pd
from dask import delayed
import re
import xarray as xr
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import tools
import warnings

warnings.simplefilter("ignore", FutureWarning)
cdf_kwargs = {
    "decode_times": True,
    "use_cftime": True,
    "chunks": {"time": 120},
}

combine_kwargs = {
    "coords": "minimal",
    "data_vars": "minimal",
    "compat": "override",
    "join": "outer",
}

# %% load catalog
force_reload = False
if os.path.exists("cat_piControl.json") and not force_reload:
    cat = intake.open_esm_datastore("cat_piControl.json")
else:
    col_url = "/work/ik1017/Catalogs/dkrz_cmip6_disk.json"
    col = intake.open_esm_datastore(col_url)
    query = dict(
        experiment_id="piControl",
        table_id=["Amon", "fx"],
        variable_id=["tas", "sftlf", "areacella"],
        grid_label=["gn", "gr"],
        member_id="r1i1p1f1",
    )
    cat = col.search(**query)
    print(cat.df["source_id"].unique())
    cat.serialize(name="cat_piControl", catalog_type="file")


# %% load datasets
def _sort_uri_by_timerange(uri: str):
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", uri)
    return m.group(1) if m else uri


def load_key_fallback(cat, key, cdf_kwargs):
    uris = cat[key].df["uri"].drop_duplicates().tolist()
    uris = sorted(uris, key=_sort_uri_by_timerange)

    # explicit concat when combine_by_coords cannot infer dimension coords
    ds = xr.open_mfdataset(
        uris,
        combine="nested",
        concat_dim="time",
        decode_times=cdf_kwargs.get("decode_times", False),
        use_cftime=cdf_kwargs.get("use_cftime", True),
        chunks=cdf_kwargs.get("chunks", None),
        engine=cdf_kwargs.get("engine", None),
        coords="minimal",
        compat="override",
        join="outer",
    )
    return ds


def load_from_cat(cat, cdf_kwargs, combine_kwargs):
    out = {}
    failed = []
    for key in cat.keys():
        print(key)
        try:
            sub = cat.search(
                activity_id=key.split(".")[0],
                source_id=key.split(".")[1],
                experiment_id=key.split(".")[2],
                table_id=key.split(".")[3],
                grid_label=key.split(".")[4],
            )
            d = sub.to_dataset_dict(
                parallel=False,
                threaded=False,
                skip_on_error=False,
                cdf_kwargs=cdf_kwargs,
                xarray_combine_by_coords_kwargs=combine_kwargs,
                progressbar=False,
            )
            out.update(d)
        except Exception:
            print("Failed to load key:", key)
            failed.append(key)

    for key in failed:
        print("Fallback attempt for key:", key)
        try:
            out[key] = load_key_fallback(cat, key, cdf_kwargs)
            print(f"fallback loaded: {key}")
        except Exception:
            print(f"Failed to load fallback for key: {key}")

    return out


dset_dict = load_from_cat(cat, cdf_kwargs, combine_kwargs)

# %% list of all models
models = set()
for key in dset_dict.keys():
    model = key.split(".")[1]
    models.add(model)
print(models)

# %% access a dataset
model = "MPI-ESM1-2-LR"
da = (
    tools.annual_mean_from_monthly(dset_dict[f"CMIP.{model}.piControl.Amon.gn"])
    .mean("year")
    .tas.squeeze()
    .compute()
)
da_area = dset_dict[f"CMIP.{model}.piControl.fx.gn"].areacella.squeeze().compute()
lsm = dset_dict[f"CMIP.{model}.piControl.fx.gn"].sftlf.squeeze().compute()
fig, ax = plt.subplots(
    subplot_kw={
        "projection": ccrs.Robinson(central_longitude=230),
        "transform": ccrs.PlateCarree(),
        "frameon": True,
    }
)
da.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis")
ax.coastlines()
tools.plot_sep_outlines(ax)
ax.set_title(None)
plt.show()
# Mean temperature in SEP
tas_sep = tools.select_sep(da, lsm)
print(f"Mean temperature in SEP: {tools.fldmean(tas_sep, da_area):.2f} K")
