import gzip
import os
import pickle
import re
import statistics
import time
from collections import defaultdict, Counter
from datetime import datetime
import json
from typing import Dict, List, DefaultDict, Tuple, Set

import zstandard
import logging
from matplotlib import pyplot as plt

from tqdm import tqdm

"""
This cell contains some helpful methods to dump notebook variables
to a file so you don't have to rerun expensive computations every
time.
"""

from typing import Union
from data_processing_util import load_var_gz, read_lines_zst, dump_var_gz, does_var_exists_gz, read_and_decode

def process_files(file_paths: List, author_to_lines: Dict[str, List[Tuple[int, str]]] = None):
    """
    :param file_paths:
    :param author_to_lines: key: str representing author name
                            value: tuple with the epoch integer timestamp of the msg (utc) and str msg
    :return:
    """

    if author_to_lines is None:
        author_to_lines: DefaultDict[str, List[Tuple[int, str]]] = defaultdict(list)

    for file_path in file_paths:
        file_lines = 0
        file_size = os.stat(file_path).st_size
        file_bytes_processed = 0
        created = None
        bad_lines = 0

        # try:
        for line, file_bytes_processed in read_lines_zst(file_path):
            try:
                obj = json.loads(line)
                created = datetime.utcfromtimestamp(int(obj['created_utc']))

                if obj["author"] == "[deleted]":
                    continue

                # if either this is jan 2011 or if this author was already active in
                # jan 2011, only then do we add to the author_to_lines dict
                if obj["author"] in author_to_lines or 'RC_2011-01' in file_path:
                    author_to_lines[obj["author"]] += [(obj['created_utc'], obj["body"])]

            except (KeyError, json.JSONDecodeError) as err:
                bad_lines += 1

            file_lines += 1
            if file_lines % 100000 == 0:
                log.info(
                    f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : "
                    f"{file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")
        print(f"Finished processing {file_path}.")

    return author_to_lines

def remove_authors_small_sample(author_to_lines: Dict[str, List[Tuple[int, str]]]):
    """
        Filters to only include people with more than 543 posts to trim out people
        with very few posts. 543 was chosen because our data sample spans over roughly
        543 days. i.e. these people have 1/post per day avg.
    """
    to_remove_authors = []
    for author in tqdm(author_to_lines):
        if len(author_to_lines[author]) < 546:
            to_remove_authors.append(author)
    for to_remove_author in tqdm(to_remove_authors):
        author_to_lines.__delitem__(to_remove_author)

    print(f"Removed {len(to_remove_authors)} authors. Now we have {len(author_to_lines)} authors.")
    return author_to_lines


sample_month_years = {("01", 2011), ("02", 2011), ("03", 2011), ("04", 2011), ("05", 2011), ("06", 2011), ("07", 2011), ("08", 2011),
                      ("09", 2011), ("10", 2011), ("11", 2011), ("12", 2011), ("01", 2012), ("02", 2012), ("03", 2012), ("04", 2012),
                      ("05", 2012), ("06", 2012)}
def remove_irregular_authors_monthly(author_to_lines: Dict[str, List[Tuple[int, str]]]):
    """
        Filters the authors_to_lines dict to only authors who posted at least 18 times
        a month for every month
    """

    sample_month_years_corrected: List[Tuple[int, int]] = \
        [(int(sample_month_year[0]), int(sample_month_year[1])) for sample_month_year in sample_month_years]

    deleted_marked_authors = set()

    for author in tqdm(author_to_lines):

        posts_data: List[Tuple[int, str]] = author_to_lines[author]
        counter: Counter[Tuple[int, int]] = Counter()

        for date, post in posts_data:
            utc_datetime = datetime.utcfromtimestamp(int(date))
            month_and_year = (utc_datetime.month, utc_datetime.year)
            assert month_and_year in sample_month_years_corrected
            counter[month_and_year] += 1

        for month_and_year in sample_month_years_corrected:
            if counter[month_and_year] < 20:
                deleted_marked_authors.add(author)

    for deleted_marked_author in deleted_marked_authors:
        author_to_lines.__delitem__(deleted_marked_author)

    print(f"Removed {len(deleted_marked_authors)} authors. Now we have {len(author_to_lines)} authors.")
    return author_to_lines

def remove_irregular_authors_weekly(author_to_lines: Dict[str, List[Tuple[int, str]]]):
    """
        Filters the authors_to_lines dict to only authors who posted at least twice a week
        for every single week
    """

    start_week = 1293840000
    week_len = 60 * 60 * 24 * 7
    total_weeks = 78
    deleted_marked_authors = set()

    for author in tqdm(author_to_lines):

        date_and_posts = author_to_lines[author]
        week_posts = [0 for _ in range(total_weeks)]
        current_week = 0

        # exploits the fact the iteration is in order of date
        # for fast processing
        for date, post in date_and_posts:
            date = int(date)

            start_time = start_week + (week_len * current_week)
            end_time = start_time + (week_len * (current_week + 1))

            if date >= start_time and date <= end_time:
                week_posts[current_week] += 1
            elif week_posts[current_week] >= 3:  # if they made at least 2 posts this week
                current_week += 1
                start_time = start_week + (week_len * current_week)
                end_time = start_time + (week_len * (current_week + 1))
                if date >= start_time and date <= end_time:
                    week_posts[current_week] += 1
                else:
                    deleted_marked_authors.add(author)
                    break
            else:
                deleted_marked_authors.add(author)
                break

    for deleted_marked_author in deleted_marked_authors:
        author_to_lines.__delitem__(deleted_marked_author)

    print(f"Removed {len(deleted_marked_authors)} authors. Now we have {len(author_to_lines)} authors.")
    return author_to_lines

def remove_irregular_authors_daily(author_to_lines: Dict[str, List[Tuple[int, str]]]):
    """
        Remove authors who didn't post on at least 80% of the possible days
    """

    max_days = 546  # the max num of days someone could have
    threshold = round(max_days * 0.8)  # must post on 80% of days
    deleted_marked_authors = set()
    author_to_days_posted = dict()

    for author in tqdm(author_to_lines):

        date_and_posts = author_to_lines[author]
        month_day_yr_set: Set[Tuple[int, int, int]] = set()

        # exploits the fact the iteration is in order of date
        # for fast processing
        for date, post in date_and_posts:
            date = datetime.utcfromtimestamp(int(date))
            month_day_yr_set.add((date.month, date.day, date.year))

        author_to_days_posted[author] = len(month_day_yr_set)

    for author in author_to_days_posted:
        if author_to_days_posted[author] < threshold:
            deleted_marked_authors.add(author)

    for deleted_marked_author in deleted_marked_authors:
        author_to_lines.__delitem__(deleted_marked_author)

    print(f"Removed {len(deleted_marked_authors)} authors. Now we have {len(author_to_lines)} authors.")
    return author_to_lines


def analyze_data(author_to_lines: Dict[str, List[Tuple[int, str]]]):
    print(F"There are {len(author_to_lines)} users in the data set.")

    author_posts = []
    author_words = []
    for author in tqdm(author_to_lines):
        total_words = 0
        author_post_cnt = 0
        for date, post in author_to_lines[author]:
            author_post_cnt += 1
            total_words += len(re.split("[ \n]", post))
        if total_words < 10000:
            print([x[1] for x in author_to_lines[author]])
        author_words += [total_words]
        author_posts += [author_post_cnt]

    avg_words = statistics.mean(author_words)
    std_words = statistics.stdev(author_words)
    avg_posts = statistics.mean(author_posts)
    std_posts = statistics.stdev(author_posts)
    print(F"The average author wrote {avg_words} words "
          F"(std={std_words}, min={min(author_words)},max={max(author_words)}) over 1.5 years.")
    print(F"They did this over an average of {avg_posts} posts"
          F" (std={std_posts}, min={min(author_posts)},max={max(author_posts)}).")

    author_words = list(sorted(author_words))
    author_posts = list(sorted(author_posts))

    plt.hist(author_words, bins=100)
    plt.axvline(x=avg_words, color='r', linestyle='--', label='Mean')
    plt.axvline(x=avg_words + std_words, color='g', linestyle='--', label='Mean + 1 Std Dev')
    plt.axvline(x=avg_words - std_words, color='g', linestyle='--', label='Mean - 1 Std Dev')
    plt.legend()  # Show legend with labels
    plt.show()

    plt.hist(author_posts, bins=100)
    plt.axvline(x=avg_posts, color='r', linestyle='--', label='Mean')
    plt.axvline(x=avg_posts + std_posts, color='g', linestyle='--', label='Mean + 1 Std Dev')
    plt.axvline(x=avg_posts - std_posts, color='g', linestyle='--', label='Mean - 1 Std Dev')
    plt.legend()  # Show legend with labels
    plt.show()


def main():
    log.info("Starting...")

    base_path = "/projectnb/cs505ws/projects/NextType/raw_reddit_data"
    target_files = [f"{base_path}/RC_{sample_month_year[1]}-{sample_month_year[0]}.zst" for sample_month_year in
                    sample_month_years]
    author_to_lines = process_files(target_files)

    author_to_lines = remove_authors_small_sample(author_to_lines)
    author_to_lines = remove_irregular_authors_monthly(author_to_lines)
    author_to_lines = remove_irregular_authors_weekly(author_to_lines)
    author_to_lines = remove_irregular_authors_daily(author_to_lines)

    dump_var_gz("author_to_lines", author_to_lines)
    author_to_lines = load_var_gz("author_to_lines")
    analyze_data(author_to_lines)



log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
main()
