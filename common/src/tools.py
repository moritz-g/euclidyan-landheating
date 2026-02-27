"""
Commonly used functions for the project.
"""

import numpy as np
import shapely
import xarray as xr
import cartopy.crs as ccrs


def select_sep(ds, lsm, xy1=(235, 0), xy2=(290, 0), xy3=(290, -50)):
    """
    Select South-East Pacific.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to select from.
    lsm : xarray.Dataset
        The land-sea mask.
    xy1, xy2, xy3 : tuple
        Coordinates of the bounding box (x1, y1), (x2, y2), (x3, y3).

    Returns
    -------
    xarray.Dataset
        The selected dataset.
    """
    x1, y1 = xy1
    x2, y2 = xy2
    x3, y3 = xy3

    # Create polygon from the 3 coordinates
    polygon = shapely.geometry.Polygon([xy1, xy2, xy3])

    # Get latitude and longitude coordinates from dataset
    lats = ds.coords["lat"].values
    lons = ds.coords["lon"].values

    # Create a mask for points inside the polygon
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            if polygon.contains(shapely.geometry.Point(lon, lat)):
                mask[i, j] = True

    # Select data where mask is True and only ocean
    return ds.where(
        xr.DataArray(mask, coords=[("lat", lats), ("lon", lons)]) & (lsm == 0),
        drop=True,
    )


def plot_sep_outlines(ax, xy1=(235, 0), xy2=(290, 0), xy3=(290, -50), lw=1):
    """
    Plot South-East Pacific outlines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    xy1, xy2, xy3 : tuple
        Coordinates of the bounding box (x1, y1), (x2, y2), (x3, y3).
    """
    x1, y1 = xy1
    x2, y2 = xy2
    x3, y3 = xy3

    # Create polygon from the 3 coordinates
    polygon = shapely.geometry.Polygon([xy1, xy2, xy3])

    # Get the exterior coordinates of the polygon
    x, y = polygon.exterior.xy

    # Plot the outline
    ax.plot(x, y, color="black", linewidth=lw, transform=ccrs.PlateCarree())


def fldmean(ds, da_area):
    """
    Compute area-weighted field mean.

    Works also if some grid points are NaN, and for subsets of the globe.
    If lon (or x) and lat (or y) are present, computes the weighted field mean.
    If only lat (or y) is present, computes the weighted mean over all latitudes.

    Parameters
    ----------
    ds : xr.DataArray or xr.Dataset
        Field that contains at least the coordinates 'lat' and 'lon'.
    da_area : xr.DataArray
        Field that contains the surface area of each grid cell.

    Returns
    -------
    Spatially averaged field of same type as input.

    """
    # make sure we're dealing with a DataSet or DataArray
    assert isinstance(ds, xr.core.dataset.Dataset) or isinstance(
        ds, xr.core.dataarray.DataArray
    )
    # find out if dimension coordinates are called lon/lat or x/y
    if "lat" in ds.dims:
        xdim = "lon"
        ydim = "lat"
    elif "y" in ds.dims:
        xdim = "x"
        ydim = "y"
    else:
        raise AttributeError(
            "Dataset must contain at least coordinates called 'lat' or 'y'."
        )
    # check that latitude and longitude data are coherent
    assert set(ds[ydim].values).issubset(set(da_area[ydim].values))
    if xdim in ds.coords and ydim in ds.coords:
        assert set(ds[xdim].values).issubset(set(da_area[xdim].values))
        if isinstance(ds, xr.core.dataset.Dataset):
            ds_weighted = ds.weighted(da_area)
        elif isinstance(ds, xr.core.dataarray.DataArray):
            ds_weighted = ds.weighted(da_area)
        return ds_weighted.mean([ydim, xdim], skipna=True)
    elif ydim in ds.coords:
        if isinstance(ds, xr.core.dataset.Dataset):
            ds_weighted = ds.weighted(da_area.mean(xdim))
        elif isinstance(ds, xr.core.dataarray.DataArray):
            ds_weighted = ds.weighted(da_area.mean(xdim))
        return ds_weighted.mean([ydim], skipna=True)


def annual_mean_from_monthly(
    ds: xr.Dataset, require_complete_years: bool = False
) -> xr.Dataset:
    """
    Compute proper annual means from monthly-mean data using day weights.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a monthly `time` coordinate.
    require_complete_years : bool, default False
        If True, only years with all 12 months are kept.

    Returns
    -------
    xr.Dataset
        Annual-mean dataset with dimension `year`.
    """
    if "time" not in ds.coords:
        raise ValueError("Dataset must contain a 'time' coordinate.")

    year = ds.time.dt.year
    days = ds.time.dt.days_in_month

    if require_complete_years:
        # assumes monthly data has one timestamp per month
        n_per_year = ds.time.groupby("time.year").count()
        full_years = n_per_year["year"].where(n_per_year == 12, drop=True)
        ds = ds.sel(time=ds.time.dt.year.isin(full_years))
        year = ds.time.dt.year
        days = ds.time.dt.days_in_month

    w = days.groupby(year) / days.groupby(year).sum()
    annual = (ds * w).groupby("time.year").sum("time", skipna=True)

    return annual


def centroid(da, da_area, lat_min=-30, lat_max=30):
    """Compute the centroid between lat_min and lat_max.

    Can be used, for example, as ITCZ position.

        Parameters
    ----------
    da : xarray.DataArray
        Field to compute the centroid of
    da_area : xarray.DataArray
        Field that contains the surface area of each grid cell.
    lat_min, lat_max : float
        Minimum and maximum latitude to consider for the centroid calculation.

    Returns
    -------
    xarray.DataArray
        Centroid (one latitude value for each longitude)
    """
    if ("lat" in da.dims) and ("lon" in da.dims):
        xdim = "lon"
        ydim = "lat"
    elif ("y" in da.dims) and ("x" in da.dims):
        xdim = "x"
        ydim = "y"
    else:
        raise AttributeError(
            "Dataset must contain at least coordinates called 'lat'/'lon' or 'y'/'x'."
        )
    assert set(da[ydim].values).issubset(set(da_area[ydim].values))
    assert set(da[xdim].values).issubset(set(da_area[xdim].values))

    da = da.sel(**{ydim: slice(lat_max, lat_min)})
    centroid = (da * da[ydim] * da_area).sum(ydim) / (da * da_area).sum(ydim)
    return centroid
