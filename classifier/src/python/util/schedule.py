"""
    schedule.py
    Author: Milan Marocchi
    
    Purpose: Contains functions for working with the schedule files
"""

import os
import json

def validate_schedule(schedule: dict) -> dict:
    # Check for the relevant keys
    try: 
        test = schedule["test_set"]
        valid = schedule["valid_set"]

        test["data"]
        valid["data"]
        test["split"]
        valid["split"]
        test["segment"]
        valid["segment"]

        datasets = schedule["datasets"]
        for dataset in datasets.values():
            dataset["path"]
            dataset["split"]
            dataset["segment"]
            dataset["gen_data"]

        schedule = schedule["schedule"]
        for val in schedule:
            if val["key"] not in datasets.keys():
                raise ValueError

    except ValueError as e:
        raise ValueError("Invalid format for the schedule: " + str(e))

    return schedule


def get_schedule(schedule_str: str) -> dict:
    schedule_path = os.path.abspath(schedule_str)

    with open(schedule_path, "r") as json_file:
        schedule = json.load(json_file)

    validate_schedule(schedule)

    return schedule


def get_data_paths(schedule: dict) -> list[str]:
    data_paths: list[str] = list()

    data_paths.append(schedule["test_set"]["data"])
    data_paths.append(schedule["valid_set"]["data"])

    for dataset in schedule["datasets"]:
        data_paths.append(schedule["datasets"][dataset]["path"])

    return data_paths


def get_split_paths(schedule: dict) -> list[str]:
    split_paths: list[str] = list()

    split_paths.append(schedule["test_set"]["split"])
    split_paths.append(schedule["valid_set"]["split"])

    for dataset in schedule["datasets"]:
        split_paths.append(schedule["datasets"][dataset]["split"])

    return split_paths


def get_segment_paths(schedule: dict) -> list[str]:
    segment_paths: list[str] = list()

    segment_paths.append(schedule["test_set"]["segment"])
    segment_paths.append(schedule["valid_set"]["segment"])

    for dataset in schedule["datasets"]:
        segment_paths.append(schedule["datasets"][dataset]["segment"])

    return segment_paths