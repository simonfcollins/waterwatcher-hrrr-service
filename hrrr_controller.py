"""
FastAPI service for retrieving HRRR forecast data.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import xarray as xr
from pandas.errors import InvalidIndexError
from pydantic import BaseModel
import os
from datetime import datetime, timedelta
import threading
from typing import Dict, Optional, List
import math
import cartopy.crs as ccrs
import json

# -------------------------------------
# Configuration and Data Paths
# -------------------------------------
zarr_root = "/data/hrrrzarr/conus/sfc"
grid_root = "/data/hrrrzarr/conus/grid"
projparams_path = f"{grid_root}/projparams.json"
hrrr_latlon_path = f"{grid_root}/HRRR_latlon.h5"
os.makedirs(zarr_root, exist_ok=True)

# -------------------------------------
# Global Locks and State
# -------------------------------------
dataset_lock = threading.Lock()
current_datasets: Dict[str, xr.Dataset] = {}

# -------------------------------------
# Load Grid Metadata for bounding box
# -------------------------------------
hrrr_latlon = xr.open_dataset(hrrr_latlon_path, engine="h5netcdf", phony_dims="access")
lat_arr = hrrr_latlon["latitude"].values
lon_arr = hrrr_latlon["longitude"].values

# Bounding box for validation
lat_min, lat_max = float(lat_arr.min()), float(lat_arr.max())
lon_min, lon_max = float(lon_arr.min()), float(lon_arr.max())

# -------------------------------------
# Load Projection
# -------------------------------------
with open(projparams_path, "r") as projparamsf:
    pp = json.load(projparamsf)
projection = ccrs.LambertConformal(central_longitude=pp["lon_0"],
                                   central_latitude=pp["lat_0"],
                                   standard_parallels=(pp["lat_1"], pp["lat_2"]),
                                   globe=ccrs.Globe(semimajor_axis=pp["a"],
                                                    semiminor_axis=pp["b"]))

# -------------------------------------
# Utility Functions
# -------------------------------------

def to_safe_float(v: float) -> Optional[float]:
    """
    Converts floats to JavaScript-safe floats.
    NaN values are returned as None.

    :param v: The value to convert.
    :return: A JavaScrip-safe float.
    """
    f = float(v)
    return None if math.isnan(f) else f


def pa_to_inhg(pa: float) -> float:
    """
    Convert pressure from Pascals to inches of mercury (inHg)

    :param pa: Pressure in Pascals.
    :return: Pressure in inHg.
    """
    return pa * 0.0002952998

def load_zarr(group_path: str, subgroup_path: str) -> xr.Dataset:
    """
    Load a zarr group as a xarray Dataset.

    :param group_path: The complete path to a zarr group.
    :param subgroup_path: The complete path to a zarr subgroup within the specified group_path.
    :return: The specified zarr group represented as a xarray dataset
    """
    ds = xr.open_mfdataset([group_path, subgroup_path], engine="zarr", decode_timedelta=True)
    ds = ds.rename(projection_x_coordinate="x", projection_y_coordinate="y")
    ds = ds.metpy.assign_crs(projection.to_cf())
    ds = ds.metpy.assign_latitude_longitude()
    return ds


def load_all_zarrs() -> Dict[str, xr.Dataset]:
    """
    Load all Zarr datasets from the configured root directory.

    :return: Mapping of filename to open dataset.
    """
    datasets = {}
    for f in os.listdir(zarr_root):
        if f.endswith(".zarr"):
            group_path = os.path.join(zarr_root, f"{f}/mean_sea_level/MSLMA")
            subgroup_path = os.path.join(group_path, "mean_sea_level")
            datasets[f] = load_zarr(group_path, subgroup_path)
    return datasets


def refresh_datasets() -> None:
    """
    Refresh the global datasets by reloading all Zarr files.
    Closes any datasets that are no longer present.

    :return: Void
    """
    global current_datasets
    new_datasets = load_all_zarrs()
    with dataset_lock:
        # Close datasets that were removed
        for f, ds in current_datasets.items():
            if f not in new_datasets:
                ds.close()
        current_datasets = new_datasets

# -------------------------------------
# FastAPI Lifespan
# -------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context to load datasets on startup
    and start the refresher background task.
    """
    refresh_datasets()
    yield
    # Close all datasets on shutdown
    for ds in current_datasets.values():
        ds.close()


app = FastAPI(lifespan=lifespan)

# -------------------------------------
# Pydantic Models
# -------------------------------------
class Variable(BaseModel):
    name: str
    units: str
    values: dict

class Point(BaseModel):
    type: str = "Point"
    coordinates: List[float]

class Feature(BaseModel):
    type: str = "Feature"
    geometry: Optional[Point]
    properties: dict

class FeatureCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[Feature]

# -------------------------------------
# API Endpoints
# -------------------------------------
@app.get("/forecast/{lat},{lon}")
def get_forecast(lat: float, lon: float):

    """
    Retrieve historical and forecasted weather data for a given lat/lon.

    :param lat: latitude value (21.138 <= lat <= 52.61565).
    :param lon: longitude value (-134.09613 <= lon <= -60.91784).
    :return: Historical and forecasted weather data at the given coordinates.
    """
    # Validate coordinates
    if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coordinates. {lat_min} <= lat <= {lat_max}, {lon_min} <= lon <= {lon_max} required"
        )

    with dataset_lock:
        datasets = current_datasets.copy()
        # Separate analysis and forecast files
        anl_list = [f for f in datasets if f.endswith("anl.zarr")]
        fcst_list = [f for f in datasets if f.endswith("fcst.zarr")]

        if not anl_list or not fcst_list:
            raise HTTPException(status_code=500, detail="We've misplaced something")

    x, y = projection.transform_point(lon, lat, src_crs=ccrs.PlateCarree())
    pressures = {}

    # Parse forecasted values from forecast file
    fcst_file = fcst_list[-1]
    time_str = fcst_file.split("z_")[0]
    start_dt = datetime.strptime(time_str, "%Y%m%d_%H")
    ds = datasets[fcst_file]
    pressure_fcst = ds.MSLMA.metpy.sel(x=x, y=y, method="nearest").values

    for p in pressure_fcst:
        pressures[start_dt.isoformat()] = to_safe_float(pa_to_inhg(p))
        start_dt += timedelta(hours=1)

    # Parse observed values from analysis files
    for f in anl_list:
        time_str = f.split("z_")[0]
        dt = datetime.strptime(time_str, "%Y%m%d_%H")
        ds = datasets[f]
        try:
            value = to_safe_float(pa_to_inhg(ds.MSLMA.metpy.sel(y=y, x=x, method="nearest").values.item()))
        except (ValueError, InvalidIndexError, KeyError):
            value = None

        pressures[dt.isoformat()] = value

            # Sort by timestamp
    sorted_pressures = {k: pressures[k] for k in sorted(pressures)}

    return Feature(
        geometry = Point(coordinates=[lon, lat]),
        properties={
            "MSLMA": Variable(name="mean_sea_level_pressure", units="inHg", values=sorted_pressures),
        }
    )

@app.post("/refresh")
async def refresh():
    """
    Post endpoint to reload HRRR datasets.

    :return: Message signifying a successful refresh.
    """
    refresh_datasets()
    return {"status": "datasets refreshed"}