# Predicting armed conflict using protest data - transforms

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree  # type: ignore

def divide_by_pop(df, varname, multiplier):
    
    var_out = df[f'{varname}']/df['pgd_pop_gpw_sum']
    var_out = var_out.replace(np.inf, np.nan)
    var_out = var_out.fillna(0) # due to pgd_pop_gpw_sum == 0 
    
    var_out = var_out*multiplier
    
    return var_out

def moving_sum(s: pd.Series, time: int):
    "Moving sum"
    
    # Group by pg_id
    y = s.groupby(level=1)
    y = y.rolling(time, min_periods=0) # min_period = n option simply means that you require at least n valid observations to compute your rolling stats
    #  Get max value. 
    y = y.sum()
    # Groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index().values
    return y

def distance_to_event(
    df,
    col, 
    k,
    fill_value,
):
    """Get spatial distance to event

    Args:
        gdf: GeoDataFrame with a multiindex like [time, group]  and
            cols for centroid and col.
        col: Name of col to count as event if == 1
        k: Number of neighbors to consider
        fill_value: When no events are found fill with this value
    Returns:
        dist: pd.Series of distance to event
    """
    
    # Set geometry.
    gdf = df.set_geometry("geometry")

    # Make centroids.
    gdf["x"] = gdf.geometry.centroid.x
    gdf["y"] = gdf.geometry.centroid.y

    # Index-only gdf to hold results
    gdf_results = gdf[[]].copy()
    gdf_results["distance"] = np.nan

    times = sorted(list(set(gdf_results.index.get_level_values(0))))

    # (x,y) coord pairs for all grids
    points_canvas = np.array(
        list(zip(gdf.loc[times[0]].centroid.x, gdf.loc[times[0]].centroid.y))
    )

    for t in times:
        gdf_events_t = gdf.loc[t][gdf.loc[t][col] == 1]
        points_events = np.array(
            list(zip(gdf_events_t.centroid.x, gdf_events_t.centroid.y))
        )
        if len(points_events) > 0:
            # Build the KDTree of the points
            btree = cKDTree(data=points_events)  # pylint: disable=not-callable
            # Find distance to closest k points, discard idx
            dist, _ = btree.query(points_canvas, k=k)
            # If more than one neighbor get the mean distance
            if k > 1:
                dist = np.mean(dist, axis=1)
            gdf_results.loc[t, "distance"] = dist
        else:
            gdf_results.loc[t, "distance"] = fill_value

    s = gdf_results["distance"]
    return s

# Function to get the moving minimum.
def moving_min(s: pd.Series, t: int):
    "Moving minimum"
    
    # Group by pg_id
    y = s.groupby(level=1)
    y = y.rolling(t, min_periods=0) # min_period = n option simply means that you require at least n valid observations to compute your rolling stats
    #  Get minimum value. 
    y = y.min()
    # Groupby and rolling do stuff to indices, return to original form
    y = y.reset_index(level=0, drop=True).sort_index().values
    return y


def divide_by_pop_cm(df, varname, multiplier):
    
    var_out = df[f'{varname}']/df['wdi_sp_pop_totl']
    var_out = var_out.replace(np.inf, np.nan)
    var_out = var_out.fillna(0) # due to pgd_pop_gpw_sum == 0 
    
    var_out = var_out*multiplier
    
    return var_out