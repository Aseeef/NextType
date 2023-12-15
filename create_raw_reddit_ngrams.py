import concurrent.futures
import json
import logging
import multiprocessing
import os
import re
from collections import Counter
from concurrent.futures import Future, as_completed
from datetime import datetime
from typing import List

import nltk
from nltk import ngrams
from nltk.probability import FreqDist
from tqdm import tqdm

from data_processing_util import read_lines_zst, dump_var_gz, load_var_gz

"""
This cell contains some helpful methods to dump notebook variables
to a file so you don't have to rerun expensive computations every
time.
"""

import html
# Precompile regular expressions
new_line_regex = re.compile("\n")
html_chars_regex = re.compile(r"&[a-zA-Z]+;")
headers_regex = re.compile(r'^#{1,6}\s')
hyperlinks_regex = re.compile(r'\[([^]]+)]\(([^)]+)\)')
inline_code_regex = re.compile(r'`([^`]+)`')
block_code_regex = re.compile(r'```[^`]*```')
lists_regex = re.compile(r'^\s*([*\-+]\s|(\d+\.)\s)')
double_spaces_regex = re.compile(" {2,}")
urls_regex = re.compile(
    r"https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_+.~#?&/=]*)")
subreddit_regex = re.compile(r"(\W)(r/[a-z0-9A-Z_]{2,10})(\W)|(\W)(/[a-z0-9A-Z_]{2,10})(\W)")

def normalize_text(post_text: str):
    # Apply regular expressions
    post_text = new_line_regex.sub(" ", post_text)
    post_text = html.unescape(post_text)
    post_text = post_text.replace("*", "")  # italics/bold
    post_text = headers_regex.sub('', post_text)
    post_text = hyperlinks_regex.sub(r'\1', post_text)
    post_text = inline_code_regex.sub(r'\1', post_text)
    post_text = block_code_regex.sub('', post_text)
    post_text = lists_regex.sub('', post_text)
    post_text = double_spaces_regex.sub(" ", post_text)
    post_text = urls_regex.sub("{URL}", post_text)
    post_text = subreddit_regex.sub(r"\1{SUB_REDDIT}\3\4{SUB_REDDIT}\6", post_text)
    post_text = post_text.strip()
    return post_text

def add_data_to_freq_dict(current_freq_dist, new_data, n):
    # Tokenize and flatten the corpus
    tokens = [word for sent in nltk.sent_tokenize(new_data) for word in nltk.word_tokenize(sent)]
    # Generate n-grams
    ngrams_list = list(ngrams(tokens, n))
    new_freq_dist = FreqDist(ngrams_list)
    for new_elem in new_freq_dist:
        current_freq_dist[new_elem] += new_freq_dist[new_elem]
    return current_freq_dist

def create_ngram_distribution(current_freq_dist):
    total_ngrams = 0
    for k in current_freq_dist:
        total_ngrams += current_freq_dist[k]
    ngram_probabilities = {ngram: freq / total_ngrams for ngram, freq in current_freq_dist.items()}

    return ngram_probabilities


def preprocess_files(file_paths: List):
    """
    :param file_paths:
    :param author_to_lines: key: str representing author name
                            value: tuple with the epoch integer timestamp of the msg (utc) and str msg
    :return:
    """

    for file_path in file_paths:
        file_lines = 0
        file_size = os.stat(file_path).st_size
        file_bytes_processed = 0
        created = None
        bad_lines = 0
        post_lines = []
        print(f'Starting pre-processing {file_path}')

        # try:
        for line, file_bytes_processed in read_lines_zst(file_path):
            try:
                obj = json.loads(line)
                created = datetime.utcfromtimestamp(int(obj['created_utc']))
                post_data = obj["body"]
                post_data = normalize_text(post_data)
                post_data = post_data.lower()
                post_lines += [post_data]

            except (KeyError, json.JSONDecodeError) as err:
                bad_lines += 1

            file_lines += 1
            if file_lines % 10000 == 0:
                log.info(
                    f"[Pre-processing] "
                    f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : "
                    f"{file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")

        print(f"Finished processing {file_path}.")
        os.remove(file_path)
        dump_name = file_path.replace('/projectnb/cs505ws/projects/NextType/raw_reddit_data/', '')
        dump_name = dump_name.replace('.zst', '')
        dump_var_gz(dump_name, post_lines)


def process_file(dump_name, current_freq_dist):
    iter_num = 0
    post_lines = load_var_gz(dump_name)
    for post_data in tqdm(post_lines, desc=dump_name):
        current_freq_dist = add_data_to_freq_dict(current_freq_dist, post_data, n=3)
        iter_num += 1
        if iter_num % 100000 == 0:
            print(f'{dump_name} - Current n-gram size:', len(current_freq_dist))
    return current_freq_dist

def process_files(file_paths: List):
    """
    :param file_paths:
    :param author_to_lines: key: str representing author name
                            value: tuple with the epoch integer timestamp of the msg (utc) and str msg
    :return:
    """
    current_freq_dist = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        futures: List[Future] = []
        # process 6 at a time
        for sublist_files in [file_paths[0:6], file_paths[6:12], file_paths[12:18]]:
            for file_path in sublist_files:
                dump_name = file_path.replace('/projectnb/cs505ws/projects/NextType/raw_reddit_data/', '')
                dump_name = dump_name.replace('.zst', '')
                print(f'Submitting task for post-processing dump {dump_name}')
                future = executor.submit(process_file, dump_name, Counter())
                futures.append(future)

            for future in as_completed(futures):
                compiled_counter = future.result()
                for val in compiled_counter:
                    current_freq_dist[val] += compiled_counter[val]
                print('A future was just completed! Combined n-gram size:', len(current_freq_dist))

    print('All done!')
    return current_freq_dist

def delete_rare_ngrams(current_freq_dist: Counter):
    filtered_counter = Counter({key: value for key, value in current_freq_dist.items() if value >= 3})
    return filtered_counter

def main():
    log.info("Starting...")

    sample_month_years = [("01", 2011), ("02", 2011), ("03", 2011), ("04", 2011), ("05", 2011), ("06", 2011),
                          ("07", 2011), ("08", 2011),
                          ("09", 2011), ("10", 2011), ("11", 2011), ("12", 2011), ("01", 2012), ("02", 2012),
                          ("03", 2012), ("04", 2012),
                          ("05", 2012), ("06", 2012)]
    base_path = "/projectnb/cs505ws/projects/NextType/raw_reddit_data"
    target_files = [f"{base_path}/RC_{sample_month_year[1]}-{sample_month_year[0]}.zst" for sample_month_year in
                   sample_month_years]

    #print('Preprocessing...')
    #preprocess_files(target_files)
    print('Post processing...')
    current_freq_dist = process_files(target_files)
    #reddit_ngram_prob_dict = create_ngram_distribution(current_freq_dist)
    current_freq_dist = delete_rare_ngrams(current_freq_dist)
    dump_var_gz("reddit_ngram_prob_dict", current_freq_dist)

log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
main()