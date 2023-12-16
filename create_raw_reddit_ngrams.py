import base64
import concurrent.futures
import gzip
import json
import logging
import multiprocessing
import os
import pickle
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

processed_data_dir = "/projectnb/cs505ws/projects/NextType/data/"
reddit_data_dir = "/projectnb/cs505ws/projects/NextType/raw_reddit_data/"

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


def add_data_to_freq_dict(current_freq_dist, tokenized_reddit_post, n):
    # Generate n-grams
    ngrams_list = list(ngrams(tokenized_reddit_post, n))
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


def preprocess_file(file_path):

    def tokenize_post(pre_tokenized_post_data):
        return [word for sent in nltk.sent_tokenize(pre_tokenized_post_data) for word in nltk.word_tokenize(sent)]

    try:
        file_lines = 0
        file_size = os.stat(file_path).st_size
        file_bytes_processed = 0
        created = None
        bad_lines = 0

        dump_name = file_path.replace(reddit_data_dir, '')
        dump_name = dump_name.replace('.zst', '')
        print(f'Starting pre-processing of {dump_name}')

        # try:
        with gzip.open(processed_data_dir + dump_name + ".gz", 'wt', encoding='utf-8', compresslevel=3) as f:
            for line, file_bytes_processed in read_lines_zst(file_path):
                try:
                    obj = json.loads(line)
                    created = datetime.utcfromtimestamp(int(obj['created_utc']))
                    post_data = obj["body"]
                    post_data = normalize_text(post_data)
                    post_data = post_data.lower()
                    tokenized_post = tokenize_post(post_data)
                    serialized_bytes = pickle.dumps(tokenized_post)
                    serialized_string = base64.b64encode(serialized_bytes).decode('utf-8')
                    f.write(serialized_string + '\n')

                except Exception as err:
                    bad_lines += 1
                    print(err)

                file_lines += 1
                if file_lines % 50000 == 0:
                    log.info(
                        f"[Pre-processing] [{dump_name}] "
                        f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {file_lines:,} : {bad_lines:,} : "
                        f"{file_bytes_processed:,} : {(file_bytes_processed / file_size) * 100:.0f}%")

        print(f"Finished processing {file_path}.")
        os.remove(file_path)
    except Exception as e:
        print(e)

    return file_path

def preprocess_files(file_paths: List):
    """
    Preprocesses the raw reddit data into normalized tokens that can easily be converted into ngrams
    """

    print('Spawning workers...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=18) as executors:
        futures = []
        for file_path in file_paths:
            print('Submit task for file path', file_path)
            future = executors.submit(preprocess_file, file_path)
            futures.append(future)
        concurrent.futures.wait(futures)
        for result in concurrent.futures.as_completed(futures):
            try:
                print(f"{result.result()} completed preprocessing!")
            except Exception as e:
                print(e)



def delete_rare_ngrams(rarity, current_freq_dist):
        print(f'Filtering ngrams - current size={len(current_freq_dist)}')
        # filter to only include n-grams that appeared at least 2 times
        filtered_counter = Counter({key: value for key, value in current_freq_dist.items() if value >= rarity})
        print(f'Filtering ngrams - new size={len(filtered_counter)}')
        return filtered_counter
        

def process_files(file_paths: List):
    """
    :param file_paths:
    :param author_to_lines: key: str representing author name
                            value: tuple with the epoch integer timestamp of the msg (utc) and str msg
    :return:
    """

    current_freq_dist = Counter()

    for file_path in file_paths:
        dump_name = file_path.replace(reddit_data_dir, '')
        dump_name = dump_name.replace('.zst', '')
        print(f'Starting dump {dump_name}')

        line_count = 0
        # Open the file in read mode
        with gzip.open(processed_data_dir + dump_name + ".gz", 'rt', encoding='utf-8', compresslevel=3) as file:
            # Iterate through the file line by line
            for line in file:
                line_count += 1

        with gzip.open(processed_data_dir + dump_name + ".gz", 'rt', encoding='utf-8', compresslevel=3) as f:
            iter_num = 0
            for i in tqdm(range(line_count), desc=dump_name):
                picked_data = f.readline()
                picked_bytes = base64.b64decode(picked_data)
                post_data = pickle.loads(picked_bytes)
                current_freq_dist = add_data_to_freq_dict(current_freq_dist, post_data, n=3)
                iter_num += 1
                if iter_num % 100000 == 0:
                    print(f'{dump_name} - Current n-gram size:', len(current_freq_dist))

        # at the end of each file, we free up some memory by deleting rare n-grams
        current_freq_dist = delete_rare_ngrams(2, current_freq_dist)

    print('All done!')
    return current_freq_dist


def main():
    log.info("Starting...")
    sample_month_years = [
        ("01", 2011), ("02", 2011), ("03", 2011), ("04", 2011), ("05", 2011), ("06", 2011),
        ("07", 2011), ("08", 2011),
        ("09", 2011), ("10", 2011), ("11", 2011), ("12", 2011), ("01", 2012), ("02", 2012),
        ("03", 2012), ("04", 2012),
        ("05", 2012), ("06", 2012)
    ]
    target_files = [f"{reddit_data_dir}RC_{sample_month_year[1]}-{sample_month_year[0]}.zst" for sample_month_year in
                    sample_month_years]

    print('Preprocessing...')
    preprocess_files(target_files)
    print('Post processing...')
    current_freq_dist = process_files(target_files)
    current_freq_dist = load_var_gz("reddit_ngram_prob_dict")
    current_freq_dist = delete_rare_ngrams(11, current_freq_dist)
    print("Final n-gram size is:", len(current_freq_dist))
    dump_var_gz("reddit_ngram_prob_dict_2", current_freq_dist)


log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
main()
