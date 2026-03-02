"""
Utilities for regridding CMIP datasets to a common 2.5° grid using xESMF.
Includes caching of regridding weights to speed up repeated remapping.
"""

# %% imports
import hashlib
from pathlib import Path
import numpy as np
import xarray as xr
import xesmf as xe
import intake
import re
import xarray as xr
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

# %% load data
# load catalog
force_reload = False
cat = intake.open_esm_datastore("cat_piControl.json")


# load datasets
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

# %% regridding utilities
TARGET_GRID_2P5 = xe.util.grid_global(2.5, 2.5)


# helpers
def lon_to_0_360(ds: xr.Dataset) -> xr.Dataset:
    """Normalize lon to [0, 360) (common CMIP convention)."""
    if "lon" in ds.coords:
        lon = ds["lon"]
        return ds.assign_coords(lon=(lon % 360).astype(lon.dtype))
    return ds


def grid_signature(ds: xr.Dataset) -> str:
    """
    Hash the source grid based on lon/lat arrays.
    Works for 1D or 2D lon/lat.
    """
    if "lon" not in ds.coords or "lat" not in ds.coords:
        raise ValueError("Dataset must have lon/lat coordinates for xESMF regridding.")

    lon = np.asarray(ds["lon"].values)
    lat = np.asarray(ds["lat"].values)

    h = hashlib.sha256()
    h.update(str(lon.shape).encode())
    h.update(str(lat.shape).encode())
    h.update(str(lon.dtype).encode())
    h.update(str(lat.dtype).encode())

    def sample(a: np.ndarray) -> bytes:
        a = a.ravel()
        if a.size == 0:
            return b""
        idx = np.linspace(0, a.size - 1, num=min(2048, a.size), dtype=int)
        return np.ascontiguousarray(a[idx]).tobytes()

    h.update(sample(lon))
    h.update(sample(lat))
    return h.hexdigest()[:16]


def weights_path(weights_dir: str, model: str, ghash: str, method: str) -> str:
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    safe_model = "".join(c if c.isalnum() or c in "-_." else "_" for c in model)
    return str(Path(weights_dir) / f"{safe_model}__{ghash}__to_2p5x2p5__{method}.nc")


def build_or_load_regridder(ds_src, ds_tgt, wpath, method="bilinear", periodic=True):
    wpath = str(Path(wpath).resolve())
    wfile = Path(wpath)
    wfile.parent.mkdir(parents=True, exist_ok=True)

    periodic_ok = bool(
        periodic
        and "lon" in ds_src.coords
        and "lat" in ds_src.coords
        and ds_src["lon"].ndim == 1
        and ds_src["lat"].ndim == 1
    )

    if wfile.exists():
        return xe.Regridder(
            ds_src,
            ds_tgt,
            method=method,
            periodic=periodic_ok,
            filename=wpath,
            reuse_weights=True,
            ignore_degenerate=True,
        )

    rg = xe.Regridder(
        ds_src,
        ds_tgt,
        method=method,
        periodic=periodic_ok,
        reuse_weights=False,
        ignore_degenerate=True,
    )
    rg.to_netcdf(wpath)
    if not wfile.exists():
        raise RuntimeError(f"Weight file was not written: {wpath}")
    return rg


# 1) precompute weights for all models
def precompute_weights_for_models(
    model_to_ds: dict[str, xr.Dataset],
    weights_dir: str,
    method: str = "bilinear",
    periodic: bool = True,
) -> dict[str, dict]:
    """
    model_to_ds: {model_name: xr.Dataset}
      Each dataset must expose lon/lat coords (1D or 2D). Variables inside can be anything.

    Returns:
      {model_name: {"grid_hash": "...", "weights_path": "..."}}
    """
    index: dict[str, dict] = {}

    for model, ds in model_to_ds.items():
        ds0 = lon_to_0_360(ds)

        ghash = grid_signature(ds0)
        wpath = weights_path(weights_dir, model, ghash, method)

        # Build or load weights. This is cheap if weights already exist.
        _ = build_or_load_regridder(
            ds0, TARGET_GRID_2P5, wpath, method=method, periodic=periodic
        )

        index[model] = {"grid_hash": ghash, "weights_path": wpath}

    return index


# 2) later: remap using the saved weights
def remap_var(
    model: str,
    ds: xr.Dataset,
    varname: str,
    weights_dir: str,
    method: str = "bilinear",
    periodic: bool = True,
) -> xr.DataArray:
    """
    Uses saved weights if present; otherwise creates them (and saves).
    Returns a DataArray on the 2.5° target grid.
    """
    ds0 = lon_to_0_360(ds)
    ghash = grid_signature(ds0)
    wpath = weights_path(weights_dir, model, ghash, method)

    regridder = build_or_load_regridder(
        ds0, TARGET_GRID_2P5, wpath, method=method, periodic=periodic
    )
    return regridder(ds0[varname])


# %% precompute weights for all models
# drop ICON and MCM-UA models because they have very different grids
dset_dict_filtered = {
    k: v
    for k, v in dset_dict.items()
    if not any(m in k for m in ["ICON-ESM-LR", "MCM-UA-1-0"])
}
weights_dir = str((Path(__file__).resolve().parents[1] / "weights" / "2p5"))
weights_index = precompute_weights_for_models(
    dset_dict_filtered,
    weights_dir,
    # "/work/mh1421/m300849/euclidyan-landheating/weights/2p5",
    method="bilinear",
)

# %% example remapping on the fly
tas_2p5 = remap_var(
    "CMIP.MPI-ESM1-2-LR.piControl.Amon.gn",
    dset_dict["CMIP.MPI-ESM1-2-LR.piControl.Amon.gn"],
    "tas",
    weights_dir,
    method="bilinear",
)

da = tas_2p5
