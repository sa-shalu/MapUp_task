import datetime
import re
from itertools import permutations
from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    def reverse_sublist(start,end):
        while start < end:
            lst[start], lst[end] = lst[end],lst[start]
            start +=1
            end -=1
    length = len(lst)
    for i in range(0,length,n):
        end = min(i+1,length -1)
        reverse_sublist(i,end)
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    sorted_result = {k: result[k] for k in sorted(result)}
    return sorted_result

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.

    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """

    def flatten(nested, parent_key=''):
        items = []
        for k, v in nested.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(flatten({f"{k}[{i}]": item}, parent_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.

    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    unique_perms = set(permutations(nums))
    return [list(p) for p in unique_perms]


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.

    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = re.compile(r'\b(?:\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b')
    dates = date_pattern.findall(text)
    return dates


import pandas as pd
import polyline
from haversine import haversine, Unit
def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.

    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    coordinates = polyline.decode(polyline_str)
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]
    for i in range(1, len(df)):
        coord1 = (df.loc[i - 1, 'latitude'], df.loc[i - 1, 'longitude'])
        coord2 = (df.loc[i, 'latitude'], df.loc[i, 'longitude'])
        distance = haversine(coord1, coord2, unit=Unit.METERS)
        distances.append(distance)
    df['distance'] = distances
    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element
    by the sum of its original row and column index before rotation.

    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.

    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Step 1: Rotate the matrix by 90 degrees clockwise
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n-1-i] = matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    def covers_full_day(intervals):
        full_day_seconds = 24 * 3600
        total_covered_seconds = 0
        for start, end in intervals:
            start_seconds = (datetime.strptime(start, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds()
            end_seconds = (datetime.strptime(end, "%H:%M:%S") - datetime(1900, 1, 1)).total_seconds()
            total_covered_seconds += end_seconds - start_seconds
        return total_covered_seconds >= full_day_seconds
    results = []
    for (id, id_2), group in df.groupby(['id', 'id_2']):
        days_covered = set(group['startDay']) | set(group['endDay'])
        if len(days_covered) == 7:
            intervals = zip(group['startTime'], group['endTime'])
            if covers_full_day(intervals):
                results.append(((id, id_2), False))
            else:
                results.append(((id, id_2), True))

    return pd.Series()
