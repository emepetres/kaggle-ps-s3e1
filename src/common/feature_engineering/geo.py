from typing import Dict, List
from pathlib import Path
from tqdm import tqdm

"""
Functions to create features from latitude and longitud data

Based on the work at https://www.kaggle.com/code/phongnguyen1/distance-to-key-locations
"""

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import KMeans

# # from sklearn.decomposition import PCA
from math import radians


def add_rotation_features(df: pd.DataFrame, rotations: List[int]):
    # The formula seems wrong but produces better results?!
    for rot in tqdm(rotations):
        df["rot_{rot}_x"] = (np.cos(np.radians(rot)) * df["Longitude"]) + (
            np.sin(np.radians(rot)) * df["Latitude"]
        )


def add_cluster_locations(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_clusters: int = 20,
    lat_label: str = "Latitude",
    lon_label: str = "Longitude",
):
    """Adds distance to each point of a line, and the min distance"""
    # TODO not tested
    coords = df_train[[lat_label, lon_label]].values
    clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)
    df_train["cluster"] = clustering.labels_

    if df_test is not None:
        coords = df_test[[lat_label, lon_label]].values
        df_test["cluster"] = clustering.transform(coords)


def add_distance_to_cluster_centroids(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    n_clusters: int = 20,
    lat_label: str = "Latitude",
    lon_label: str = "Longitude",
):
    coords = df_train[[lat_label, lon_label]].values
    clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(coords)

    _add_distance_to_cluster_centroids(df_train, clustering)
    _add_distance_to_cluster_centroids(df_test, clustering)


def _add_distance_to_cluster_centroids(
    df: pd.DataFrame,
    clustering: KMeans,
):
    centers = clustering.cluster_centers_
    for i in range(len(centers)):
        df[f"cluster_{i}"] = df.apply(
            lambda x: _compute_distance((x.Latitude, x.Longitude), centers[i]), axis=1
        )

    df["cluster_min"] = np.min(df[[f"cluster_{i}" for i in range(len(centers))]])


def add_distance_to_line(
    name: str,
    line_points: List,
    df: pd.DataFrame,
    lat_label: str = "Latitude",
    lon_label: str = "Longitude",
):
    """Adds distance to each point of a line, and the min distance"""
    for idx, loc in enumerate(tqdm(line_points)):
        df[f"to_{name}_{idx}"] = df.apply(
            lambda t: _compute_distance((t[lat_label], t[lon_label]), loc), axis=1
        )

    df[f"to_{name}_min"] = np.min(
        df[[f"to_{name}_{i}" for i in range(len(line_points))]]
    )


def add_distance_to_locations(
    locations: Dict,
    df: pd.DataFrame,
    lat_label: str = "Latitude",
    lon_label: str = "Longitude",
):
    """Adds distance to each location"""
    for name, loc in tqdm(locations.items()):
        df[f"to_{name}"] = df.apply(
            lambda t: _compute_distance((t[lat_label], t[lon_label]), loc), axis=1
        )


def _compute_distance(loc1, loc2):
    # Haversine distance
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.haversine_distances.html
    loc1 = [radians(x) for x in loc1]
    loc2 = [radians(x) for x in loc2]
    result = haversine_distances([loc1, loc2])
    return (result * 6371000 / 1000)[0][1]


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
