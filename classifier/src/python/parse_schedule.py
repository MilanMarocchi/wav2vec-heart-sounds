#!/usr/bin/env pipenv-shebang
"""
    parse_schedule.py
    Author: Milan Marocchie

    Purpose: Parse the schedule files and get information from them
"""
# Note this script is used for train.sh to make sure everything is set up correctly.

from util.schedule import (
    get_schedule,
    get_data_paths,
    get_segment_paths,
    get_split_paths
)

import click


@click.command()
@click.option('--schedule_path', '-S', required=True, help="The path to the schedule file")
def display_data_paths(schedule_path):
    schedule = get_schedule(schedule_path)
    data_paths = get_data_paths(schedule)

    for path in data_paths:
        print(path, end=" ")


@click.command()
@click.option('--schedule_path', '-S', required=True, help="The path to the schedule file")
def display_split_paths(schedule_path):
    schedule = get_schedule(schedule_path)
    data_paths = get_split_paths(schedule)

    for path in data_paths:
        print(path, end=" ")


@click.command()
@click.option('--schedule_path', '-S', required=True, help="The path to the schedule file")
def display_segment_paths(schedule_path):
    schedule = get_schedule(schedule_path)
    data_paths = get_segment_paths(schedule)

    for path in data_paths:
        print(path, end=" ")


@click.group()
def cli():
    pass

cli.add_command(display_data_paths, "display_data_paths")
cli.add_command(display_split_paths, "display_split_paths")
cli.add_command(display_segment_paths, "display_segment_paths")

if __name__ == "__main__":
    cli()