from typing import List
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from geopy.geocoders import Nominatim


def compute_geo_features(
    df: pd.DataFrame,
    lat_label: str = "Latitude",
    lon_label: str = "Longitude",
    features: List = None,
    cache: str | Path = None,
) -> pd.DataFrame:
    """Compute the geo feautures for the latitude & longitude data.

    If features is None, all feautures are computed ("road", "neighbourhood", "town",
    "county", "city", "state_district", "postcode").
    cache can be used to store data and not compute it if cached before.
    """
    if cache is not None and Path(cache).exists():
        return pd.read_csv(cache, index_col=0)

    if features is None:
        features = [
            "road",
            "neighbourhood",
            "town",
            "county",
            "city",
            "state_district",
            "postcode",
        ]

    geolocator = Nominatim(user_agent="geo-features")
    locations = list(zip(df[lat_label], df[lon_label]))
    results = []
    for loc in tqdm(locations):
        results.append(_extract_geo_features(geolocator, loc, features))

    df_geo = pd.DataFrame(results, index=df.index)
    # df_geo = pd.concat((df_test, df_geo), axis=1)
    if cache is not None:
        df_geo.to_csv(cache)  # , index=None)

    return df_geo


def _extract_geo_features(geolocator: Nominatim, lat_lon, keys: List):
    location = geolocator.reverse(lat_lon)
    return {k: _extract_key(location, k) for k in keys}


def _extract_key(location, k: str):
    # # if not location or not location.raw or not location.raw.get("address"):
    # #     print(location, "failed")
    # #     return None

    return location.raw["address"].get(k)
