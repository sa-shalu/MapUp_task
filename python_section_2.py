from datetime import time

import pandas as pd


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    ids = sorted(df['id'].unique())
    distance_matrix = pd.DataFrame(0, index=ids, columns=ids, dtype=float)
    for index, row in df.iterrows():
        id_from = row['id']
        id_to = row['id_2']
        distance = row['distance']
        distance_matrix.loc[id_from, id_to] = distance
        distance_matrix.loc[id_to, id_from] = distance
    for k in ids:
        for i in ids:
            for j in ids:
                if distance_matrix.loc[i, j] > distance_matrix.loc[i, k] + distance_matrix.loc[k, j]:
                    distance_matrix.loc[i, j] = distance_matrix.loc[i,j]
    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    rows = []
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:
                distance = df.loc[id_start, id_end]
                rows.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})
    unrolled_df = pd.DataFrame(rows, columns=['id_start', 'id_end', 'distance'])
    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1
    filtered_df = df[
        (df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold) & (df['id_start'] != reference_id)]
    result_ids = filtered_df['id_start'].unique()
    result_df = pd.DataFrame(sorted(result_ids), columns=['id_start'])

    return result_df


def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    return df

def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    discount_factors = {
        'weekdays': {
            (time(0, 0), time(10, 0)): 0.8,
            (time(10, 0), time(18, 0)): 1.2,
            (time(18, 0), time(23, 59, 59)): 0.8
        },
        'weekends': {
            (time(0, 0), time(23, 59, 59)): 0.7
        }
    }

    def get_discount_factor(day: str, current_time: time) -> float:
        if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            for time_range, factor in discount_factors['weekdays'].items():
                if time_range[0] <= current_time <= time_range[1]:
                    return factor
        elif day in ['Saturday', 'Sunday']:
            for time_range, factor in discount_factors['weekends'].items():
                if time_range[0] <= current_time <= time_range[1]:
                    return factor
        return 1
    results = []
    for idx, row in df.iterrows():
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            for time_range, _ in discount_factors['weekdays'].items() if day not in ['Saturday', 'Sunday'] else \
            discount_factors['weekends'].items():
                start_time = time_range[0]
                end_time = time_range[1]
                discount_factor = get_discount_factor(day, start_time)

                result_row = row.copy()
                result_row['start_day'] = day
                result_row['start_time'] = start_time
                result_row['end_day'] = day
                result_row['end_time'] = end_time

                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    result_row[vehicle] = row[vehicle] * discount_factor

                results.append(result_row)

        result_df = pd.DataFrame(results)

        return result_df
