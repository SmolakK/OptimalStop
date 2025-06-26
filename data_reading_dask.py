import os
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client
import dask




def run():
    def process_user_data(group):
        """
        This function processes a single user's data to check if they have at least 85% data coverage
        in hourly intervals over a 30-day (720-hour) period. It returns the filtered dataframe
        if criteria are met, else returns an empty dataframe.
        """
        hours = 30 * 24
        # Sort by datetime and set it as index for resampling
        group = group.sort_values('datetime')
        group = group.set_index('datetime')

        # Resample to 1H and compute coverage
        rsmpl = group['lat'].resample('1H').count()
        rsmpl[rsmpl > 0] = 1
        roll = rsmpl.rolling(hours).mean()

        # Check coverage criteria
        if roll.max() >= 0.85:
            end = roll.idxmax()
            start = end - pd.Timedelta(f'{hours} hours')
            filtered = group[(group.index >= start) & (group.index <= end)]
            filtered = filtered.reset_index()  # restore datetime as a column
            return filtered
        else:
            # Return empty DataFrame with same columns
            return group.iloc[0:0].reset_index()

    def filter_dataset(ddf):
        # Convert datetime
        ddf['datetime'] = dd.to_datetime(ddf['datetime'])

        # We apply our processing function to each user_id group
        # Dask requires a meta object that matches the schema of the returned DataFrame.
        meta = ddf._meta
        # Apply the process_user_data function to each group
        result = ddf.groupby('user_id').apply(process_user_data)

        return result

    # -------------------------------
    # NZ Data Filtering
    # -------------------------------
    fpath = r'Z:\NZ_DATA\NZ_users'
    filename = 'NZ_active.csv'
    fpath_full = os.path.join(fpath, filename)

    # Read CSV in parallel with Dask
    ddf_nz = dd.read_csv(fpath_full, names=['user_id', 'lon', 'lat', 'datetime'], skiprows=1)

    # Filter dataset
    df_filtered_nz = filter_dataset(ddf_nz)

    # Write output to Parquet
    df_filtered_nz.to_parquet('NZ_filtered.parquet', write_index=False)

    # -------------------------------
    # RIO Data Filtering
    # -------------------------------
    fpath = r'Z:\RJ_DATA\RIO\RIO_active'
    filename = r'RIO_second_filter.csv'
    fpath_full = os.path.join(fpath, filename)

    ddf_rio = dd.read_csv(fpath_full, names=['user_id', 'lon', 'lat', 'datetime'], skiprows=1)

    # Filter dataset
    df_filtered_rio = filter_dataset(ddf_rio)

    # Write output to Parquet
    df_filtered_rio.to_parquet('RIO_filtered.parquet', write_index=False)


if __name__ == "__main__":
    client = Client(memory_limit='20GB')
    dask.config.set(shuffle='tasks')
    run()
