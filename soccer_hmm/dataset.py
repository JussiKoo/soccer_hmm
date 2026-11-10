from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import numpy as np
import pandas as pd
import os
from scipy.stats import zscore
from scipy.signal import butter, filtfilt
from datetime import datetime

from src.soccer_hmm.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def remove_idle(df,
                time_col="timestamp",
                speed_col="speed",
                speed_threshold=10/3.6,     # m/s
                max_idle_minutes=5):   # maximum allowed idle time in minutes
    """
    Detects and removes idle periods in data.

    Parameters:
        df (pd.DataFrame): Input dataframe with time and speed columns.
        time_col (str): Name of the time column (datetime).
        speed_col (str): Name of the speed column.
        speed_threshold (float): Speed below which the player is considered idle.
        max_idle_minutes (float): Maximum duration in minutes to consider as idle.

    Returns:
        pd.DataFrame: DataFrame with idle periods removed.
    """

    df = df.copy()

    # Ensure datetime type
    #df[time_col] = pd.to_datetime(df[time_col])

    #smoothed speed
    #speed_smooth = (
    #    df[speed_col]
    #    .rolling(window=5, center=True, min_periods=1)
    #    .median()
    #)

    speed_smooth = (
        df["speed"]
        .rolling("10s", min_periods=1)
        .median()
    )

    df["idle"] = speed_smooth < speed_threshold

    # Group contiguous idle periods
    df["idle_group"] = (df["idle"].diff() != 0).cumsum()

    idle_periods = (
        df[df["idle"]]
        .groupby("idle_group")
        .agg(
            start=("idle", lambda x: x.index.min()),
            end=("idle", lambda x: x.index.max()),
            samples=("idle", "count"),
        )
    )

    # Idle period duration in minutes
    idle_periods["duration_min"] = (
        idle_periods["end"] - idle_periods["start"]
    ).dt.total_seconds() / 60

    # Select idle periods longer than allowed
    idle_periods = idle_periods[idle_periods["duration_min"] > max_idle_minutes]
    print("Idle periods:", idle_periods.shape[0])

    if idle_periods.empty:
        print("No idle periods detected.")
        return df.drop(columns=["idle", "idle_group"])

    # Remove idle periods
    for _, row in idle_periods.iterrows():
        df.loc[
            (df.index >= row["start"]) & (df.index <= row["end"]),
            ["x", "y", "playerload", "speed", "turn_angle"],
        ] = np.nan
        print(
            f"Removed idle period from {row['start']} to {row['end']} "
            f"lasting {row['duration_min']:.2f} minutes."
        )

    return df.drop(columns=["idle", "idle_group"]).dropna(subset=["x", "y", "playerload", "speed", "turn_angle"], how="all")


#===============================================================================================================================

def identify_first_and_second_half(df, time_col="timestamp"):
    """
    Identifies first and second halves in match data based on the largest time gap.

    Parameters:
        df (pd.DataFrame): Input dataframe with datetime column.

    Returns:
        pd.Series: Series indicating half number (1 or 2).
    """

    df = df.copy()

    # Compute time gaps in seconds
    time_gap = df[time_col].diff().dt.total_seconds()

    # Find index of the maximum time gap
    split_idx = time_gap.idxmax()

    # Initialize all as first half
    half = pd.Series(0, index=df.index)

    # Mark second half (strictly after the gap)
    half.loc[split_idx:] = 1

    print("Detected 2 halves (based on largest time gap).")

    return half

#===============================================================================================================================

def remove_short_segments(
    df,
    time_col="timestamp",
    gap_threshold_seconds=60*5,
    min_segment_minutes=20
):
    """
    Splits data into segments based on time gaps and removes short segments.

    Parameters:
        df (pd.DataFrame): Input dataframe sorted by time.
        time_col (str): Name of datetime column.
        gap_threshold_seconds (float): Time gap (in seconds) that starts a new segment.
        min_segment_minutes (float): Minimum allowed segment duration.

    Returns:
        pd.DataFrame: Filtered dataframe with short segments removed.
    """

    df = df.copy()

    # Ensure sorted
    df = df.sort_values(time_col)

    # Time difference between consecutive samples
    dt = df[time_col].diff().dt.total_seconds()

    # New segment starts after a big gap
    df["segment_id"] = (dt > gap_threshold_seconds).cumsum()

    # Segment start/end times
    segments = (
        df.groupby("segment_id")
        .agg(
            start=(time_col, "min"),
            end=(time_col, "max")
        )
    )

    segments["duration_min"] = (
        segments["end"] - segments["start"]
    ).dt.total_seconds() / 60

    # Keep only long enough segments
    valid_segments = segments[
        segments["duration_min"] >= min_segment_minutes
    ].index

    print(f"Removed {len(segments) - len(valid_segments)} short segments.")

    return df[df["segment_id"].isin(valid_segments)]


#===============================================================================================================================

#Function to cut start and end until certain speed is reached
def trim_start_end(df, speed_col="speed", speed_threshold=4):
    """
    Sets start and end of the data to NaN until smoothed speed exceeds a threshold.
    """

    df = df.copy()

    # Smooth speed to avoid spike-based trimming
    #speed_smooth = (
    #    df[speed_col]
    #    .rolling(window=5, center=True, min_periods=1)
    #    .median()
    #)

    speed_smooth = (
        df["speed"]
        .rolling("10s", min_periods=1)
        .median()
    )

    # Identify movement
    moving = speed_smooth > speed_threshold

    if not moving.any():
        print("No movement detected above threshold.")
        return df.iloc[0:0]

    # First movement index
    start_index = moving.idxmax()

    # Last movement index
    end_index = moving[::-1].idxmax()

    print(f"Setting start to NaN until index {start_index}")
    print(f"Setting end to NaN from index {end_index}")

    # Set BEFORE start to NaN
    df.loc[df.index < start_index, 
           ["x", "y", "playerload", "speed", "turn_angle"]] = np.nan

    # Set AFTER end to NaN
    df.loc[df.index > end_index,
           ["x", "y", "playerload", "speed", "turn_angle"]] = np.nan

    return df.dropna(subset=["x", "y", "playerload", "speed", "turn_angle"], how="all")


#=================

def segment_long_tracks(
    df,
    time_col="timestamp",
    track_col="track_id",
    segment_col="segment_id",
    max_segment_seconds=300,   # 5 minutes
    gap_threshold_seconds=5
):
    """
    Split a single player's track into shorter segments for HMM efficiency.
    
    Each new segment starts if:
      2) original segment changes
      3) a large time gap occurs
      4) the current continuous block exceeds max_segment_seconds
    """

    df = df.sort_values([segment_col, time_col]).copy()

    # Time differences
    dt = df[time_col].diff().dt.total_seconds()

    # Step 1: natural segment boundaries
    new_segment = (
        (df[segment_col] != df[segment_col].shift()) |
        (dt > gap_threshold_seconds)
    )

    # Step 2: label continuous blocks
    block_counter = new_segment.cumsum()

    # Step 3: measure time since block start
    time_since_start = (
        df[time_col] - df.groupby(block_counter)[time_col].transform("first")
    ).dt.total_seconds()

    # Step 4: split long blocks into multiple sub-segments
    # For each row, compute how many full chunks of max_segment_seconds have passed
    forced_subsegment = (time_since_start // max_segment_seconds).astype(int)

    # Step 5: combine natural segments + forced sub-segments
    df["segment_id_new"] = (new_segment.cumsum() * 1000 + forced_subsegment).astype(int)

    return df

#===============================================================================================================

def preprocess(df, entry_name, track_id):

    df = df.copy()

    col_renames = dict({
        "Excel Timestamp": "timestamp",
        "Latitude": "lat",
        "Longitude": "lon",
        "Speed": "speed",
        "Accel X": "acc_x",
        "Accel Y": "acc_y",
        "Accel Z": "acc_z"
    })

    df.columns = df.columns.str.strip()
    #df.drop(columns=["Heart Rate"], axis=1, inplace=True)
    df.rename(columns=col_renames, inplace=True)

    #parse time, sort index
    df["timestamp"] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df["timestamp"], unit="D")
    df = df.sort_values("timestamp")
    df = df.set_index("timestamp")
    df.index = df.index.round("100ms")

    #reindex to exact 10 Hz
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=f"{int(100)}ms"
    )

    df = df.reindex(full_index)
    #df.reset_index(names="timestamp", inplace=True)

    #convert lat/lon to local meters
    R = 6371000  # Earth radius (m)

    lat_rad = np.deg2rad(df["lat"])
    lon_rad = np.deg2rad(df["lon"])

    lat0 = lat_rad.iloc[0]
    lon0 = lon_rad.iloc[0]
    
    #local equirectangular projection
    df["x"] = R * (lon_rad - lon0) * np.cos(lat0)
    df["y"] = R * (lat_rad - lat0)

    # interpolate only short gaps (limit=1 means max 1 consecutive NaN)
    df[["acc_x", "acc_y", "acc_z", "speed", "x", "y"]] = (
        df[["acc_x", "acc_y", "acc_z", "speed", "x", "y"]]
          .interpolate(method="linear", limit_direction="both", limit=1)
    )

    # Apply high-pass filter for every continuous segment of valid acceleration data

    acc_cols = ["acc_x", "acc_y", "acc_z"]
    lowcut = 0.1  # Hz

    # Create mask of rows where all acceleration channels are present
    valid_mask = df[acc_cols].notna().all(axis=1)

    valid_groups = (valid_mask & ~valid_mask.shift(fill_value=False)).cumsum()
    df["segment_id"] = valid_groups.where(valid_mask)

    # Prepare output columns
    for col in acc_cols:
        df[f"{col}_filt"] = np.nan

    # Filtering parameters
    fs = 10
    lowcut = 0.1
    nyq = fs / 2
    low = lowcut / nyq
    b, a = butter(4, low, btype="highpass")

    padlen = 3 * max(len(a), len(b))

    for seg in df[valid_mask].groupby("segment_id"):
        idx = seg[1].index
        segment_length = len(idx)

        if segment_length >= padlen:
            for col in acc_cols:
                try:
                    filtered = filtfilt(b, a, df.loc[idx, col].values)
                    df.loc[idx, f"{col}_filt"] = filtered
                except Exception:
                    # in case filtfilt fails for numerical reasons
                    continue

    # Compute gravity-removed Accel'Rate
    df["acc_mag"] = np.sqrt(
        df["acc_x_filt"]**2 +
        df["acc_y_filt"]**2 +
        df["acc_z_filt"]**2
    )

    df["ar_10hz"] = np.abs(np.diff(df["acc_mag"], prepend=df["acc_mag"].iloc[0]))


    #calculate instantaneous player load
    df["pl_10hz"] = np.sqrt(
        np.diff(df["acc_x_filt"], prepend=df["acc_x_filt"].iloc[0])**2 +
        np.diff(df["acc_y_filt"], prepend=df["acc_y_filt"].iloc[0])**2 +
        np.diff(df["acc_z_filt"], prepend=df["acc_z_filt"].iloc[0])**2
    )/10

    df["bin_2s"] = df.index.floor("2s")

    min_valid = 16

    ar_05hz = (
        df.groupby("bin_2s")
            .agg(ar_raw=("ar_10hz", "sum"),
                 n_valid=("ar_10hz", "count"))
    )

    ar_05hz["coverage"] = ar_05hz["n_valid"] / 20
    ar_05hz["ar_scaled"] = ar_05hz["ar_raw"] / ar_05hz["coverage"]
    ar_05hz.loc[ar_05hz["n_valid"] < min_valid, "ar_scaled"] = np.nan

    pl_05hz = (
        df.groupby("bin_2s")
            .agg(pl_raw=("pl_10hz", "sum"),
                 n_valid=("pl_10hz", "count"))
    )

    pl_05hz["coverage"] = pl_05hz["n_valid"] / 10
    pl_05hz["pl_scaled"] = pl_05hz["pl_raw"] * (10 / pl_05hz["n_valid"])
    pl_05hz.loc[pl_05hz["n_valid"] < min_valid, "pl_scaled"] = np.nan

    #Speed to 1 Hz
    speed_05hz = (
        df.groupby("bin_2s")
            .agg(speed=("speed", "mean"),
                 n_valid=("speed", "count"))
    )

    speed_05hz.loc[speed_05hz["n_valid"] < min_valid, "speed"] = np.nan

    loc_05hz = (
        df.groupby("bin_2s")
            .agg(x=("x", "mean"),
                 y=("y", "mean"),
                 n_valid=("x", "count"))
    )

    loc_05hz.loc[loc_05hz["n_valid"] < min_valid, ["x", "y"]] = np.nan

    df_05hz = pd.DataFrame({
                          "x": loc_05hz["x"],
                          "y": loc_05hz["y"],
                          "speed": speed_05hz["speed"],
                          "playerload": pl_05hz["pl_scaled"],
                          "accelrate": ar_05hz["ar_scaled"]})
    
    #df_05hz.reset_index(names="timestamp", inplace=True)
    
    df_05hz["track_id"] = track_id

    entry_name_split = entry_name.split("_")

    df_05hz["game"] = entry_name_split[1]
    df_05hz["position"] = entry_name_split[2]
    df_05hz["player"] = entry_name_split[3]
    df_05hz["track_id"] = track_id

    #calculate turn angle
    dx = df_05hz["x"].diff()
    dy = df_05hz["y"].diff()

    heading = np.arctan2(dy, dx)
    dtheta = heading.diff()

    df_05hz["turn_angle"] = (dtheta + np.pi) % (2 * np.pi) - np.pi

    #All have to be valid for a single time point
    df_05hz.loc[df_05hz.isna().any(axis=1), ["x", "y", "speed", "playerload", "accelrate", "turn_angle"]] = np.nan

    #remove idle periods in start and end
    df_05hz = trim_start_end(df_05hz)
    
    #remove other long idle periods
    df_05hz = remove_idle(df_05hz)

    # mark consistent 0.5 Hz segments
    expected_dt = pd.Timedelta(seconds=2)
    tol = pd.Timedelta(milliseconds=100)  # tolerance

    dt = df_05hz.index.to_series().diff()

    # True whenever a new segment should begin
    new_segment = (
        dt.isna() |
        ((dt - expected_dt).abs() > tol)
    )

    # Assign segment IDs: 0, 1, 2, ...
    df_05hz["segment_id"] = new_segment.cumsum() - 1

    min_samples = 5  # 10 seconds at 0.5 Hz

    segment_sizes = df_05hz.groupby("segment_id").size()
    valid_segments = segment_sizes[segment_sizes >= min_samples].index

    df_05hz = df_05hz[df_05hz["segment_id"].isin(valid_segments)]

    #df_05hz["is_valid"] = df_05hz["x"].notna()

    ##mark continuous segments
    #df_05hz["segment_id"] = (
    #    df_05hz["is_valid"]
    #    .ne(df_05hz["is_valid"].shift())
    #    .cumsum()
    #)

    ##calculate segment lengths
    #df_05hz["segment_length"] = (
    #    df_05hz.groupby("segment_id")["segment_id"]
    #    .transform("size")
    #)

    #minimum segment length
    # 10 -> 20 seconds
    #min_len = 10

    #too short segments are deleted
    #df_05hz.loc[df_05hz["segment_length"] < min_len, ["x", "y", "speed", "playerload", "accelrate", "turn_angle"]] = np.nan
    #df_05hz = df_05hz.dropna()

    #if speed is less than 0.5 m/s turn angle is set to zero
    #df_05hz.loc[df_05hz["speed"] < 0.1, "turn_angle"] = 0

    #speed and load have to be strictly positive for sampling
    df_05hz.loc[df_05hz["speed"]<=0, "speed"] = 0.01
    df_05hz.loc[df_05hz["playerload"]<=0, "playerload"] = 0.01
    df_05hz.loc[df_05hz["accelrate"]<=0, "accelrate"] = 0.01

    df_05hz["playerload_cumulative"] = df_05hz["playerload"].cumsum()
    df_05hz["playerload_cumulative"] = df_05hz["playerload_cumulative"]/df_05hz["playerload_cumulative"].max()
    df_05hz["playerload_cumulative_t-1"] = df_05hz["playerload_cumulative"].shift(1, fill_value=0)

    df_05hz["accelrate_cumulative"] = df_05hz["accelrate"].cumsum()
    df_05hz["accelrate_cumulative"] = df_05hz["accelrate_cumulative"]/df_05hz["accelrate_cumulative"].max()
    df_05hz["accelrate_cumulative_lagged"] = df_05hz["accelrate_cumulative"].shift(1, fill_value=0)

    df_05hz["accelrate_cumulative_30s"] = (
        df_05hz
        .groupby("segment_id")["accelrate"]
        .rolling(15, closed="left", min_periods=0)
        .sum()
        .reset_index(level=0, drop=True)
    )

    #df_05hz["accelrate_cumulative_10s"] = df_05hz["accelrate"].rolling(window=5).sum().shift(1, fill_value=0)

    df_05hz.reset_index(names="timestamp", inplace=True)
    df_05hz["timestamp"] = pd.to_datetime(df_05hz["timestamp"])

    #df_05hz = segment_long_tracks(df_05hz)

    #detect 1st and 2nd half
    #df_05hz["half"] = identify_first_and_second_half(df_05hz)

    return df_05hz

#===============================================================================================================

def make_data_dict(directory, preprocess_data=True):
    """
    Loads all raw data files from a directory into a dictionary.

    Args:
        directory (str): Path to the directory containing raw data files.
    Returns:
        dict: A dictionary where keys are filenames and values are DataFrames with raw data.
    """
    dict = {}

    track_id = 0

    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.lower().endswith(".csv"):
            #Game_1_hyökkääjä_JM_Catapult-Export-1770541727119
            logger.info(f"Processing file {entry.name}...")
            entry_name = os.path.splitext(entry.name)[0]
            #drop file extension from name
            df = pd.read_csv(entry.path, sep=",")
            if preprocess_data:
                df = preprocess(df, entry_name, track_id)
                #df = preprocess2(df)
            dict[track_id] = df
            track_id = track_id + 1
            logger.info(f"File {entry} processed succesfully")
    return dict

#===============================================================================================================

def process_raw_data(input_path, output_path):
    logger.info("Processing dataset...")

    data_dict = make_data_dict(input_path)
    df = pd.concat([data_dict[key] for key in data_dict.keys()], ignore_index=True)

    df["player_id"] = pd.factorize(df["player"])[0]
    df["position_id"] = pd.factorize(df["position"])[0]

    # Important: For testing with smaller dataset
    #df = df[df["player_id"] == 18]

    mean_val = df["accelrate_cumulative_30s"].mean()
    std_val = df["accelrate_cumulative_30s"].std()

    df["accelrate_cumulative_30s_z"] = (
        df["accelrate_cumulative_30s"] - mean_val
    ) / std_val

    scaling_params = {
        "accelrate_cumulative_30s": {
            "mean": mean_val,
            "std": std_val
        }
    }

    df["position"] = df["position"].replace({"hyökkääjä": "forward",
                                             "keskikenttä": "midfielder",
                                             "laitapakki": "full-back",
                                             "toppari": "center-back",
                                             "linkki": "winger",
                                             "Linkki": "winger"})
    df.reset_index(inplace=True)
    df.to_csv(output_path, index=False)
    return df, scaling_params


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / "soccerdata_processed.csv",
    # ----------------------------------------------
):
    process_raw_data(input_path, output_path)

    # -----------------------------------------


if __name__ == "__main__":
    app()
