import base64
import concurrent.futures
import gzip
import json
import logging
import multiprocessing
import os
import pickle
import re
from collections import Counter, defaultdict
from concurrent.futures import Future, as_completed
from datetime import datetime
from typing import List, Dict, Tuple

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


def process_files(file_paths: List, n_size: int, del_amnt: int):
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
                current_freq_dist = add_data_to_freq_dict(current_freq_dist, post_data, n=n_size)
                iter_num += 1
                if iter_num % 100000 == 0:
                    print(f'{dump_name} - Current n-gram size:', len(current_freq_dist))

        # at the end of each file, we free up some memory by deleting rare n-grams
        current_freq_dist = delete_rare_ngrams(del_amnt, current_freq_dist)

    print('All done!')
    return current_freq_dist


def pmf_from_3_grams(reddit_3_grams: Counter) -> Dict[Tuple[str, str], Dict[str, float]]:
    from_2_gram_to_word_counter: Dict[Tuple[str, str], Counter[str]] = defaultdict(lambda: Counter())
    # first convert loaded data into a dict [2-tuple] -> possible words and counter
    for reddit_3_gram in tqdm(reddit_3_grams):
        lookup = tuple(reddit_3_gram[:2])
        result_word = reddit_3_gram[2]

        possible_next_words_counter = from_2_gram_to_word_counter[lookup]
        possible_next_words_counter[result_word] += 1

    # delete all result words less common than top 5
    for a_2_gram in tqdm(from_2_gram_to_word_counter):
        possible_next_words_counter = from_2_gram_to_word_counter[a_2_gram]
        most_common_tuples = possible_next_words_counter.most_common(5)
        most_common_elems_counter = Counter()
        for element in most_common_tuples:
            most_common_elems_counter[element[0]] = element[1]

    from_2_gram_to_word_probability: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: defaultdict(float))

    # then convert the counter into a probability distribution
    for a_2_gram in tqdm(from_2_gram_to_word_counter):
        possible_next_words_counter = from_2_gram_to_word_probability[a_2_gram]
        total_count = sum(possible_next_words_counter.values())
        probability_distribution = {key: count / total_count for key, count in possible_next_words_counter.items()}
        from_2_gram_to_word_probability[a_2_gram] = probability_distribution

    return from_2_gram_to_word_probability


def pmf_from_2_grams(reddit_2_grams: Counter) -> Dict[Tuple[str, str], Dict[str, float]]:
    from_1_gram_to_word_counter: Dict[Tuple[str, str], Counter[str]] = defaultdict(lambda: Counter())
    # first convert loaded data into a dict [1-tuple] -> possible words and counter
    for reddit_2_gram in tqdm(reddit_2_grams):
        lookup = tuple(reddit_2_gram[:1])
        result_word = reddit_2_gram[1]

        possible_next_words_counter = from_1_gram_to_word_counter[lookup]
        possible_next_words_counter[result_word] += 1

    # delete all result words less common than top 5
    for a_1_gram in tqdm(from_1_gram_to_word_counter):
        possible_next_words_counter = from_1_gram_to_word_counter[a_1_gram]
        most_common_tuples = possible_next_words_counter.most_common(5)
        most_common_elems_counter = Counter()
        for element in most_common_tuples:
            most_common_elems_counter[element[0]] = element[1]

    from_1_gram_to_word_probability: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
        lambda: defaultdict(float))

    # then convert the counter into a probability distribution
    for a_1_gram in tqdm(from_1_gram_to_word_counter):
        possible_next_words_counter = from_1_gram_to_word_probability[a_1_gram]
        total_count = sum(possible_next_words_counter.values())
        probability_distribution = {key: count / total_count for key, count in possible_next_words_counter.items()}
        from_1_gram_to_word_probability[a_1_gram] = probability_distribution

    return from_1_gram_to_word_probability


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

    # first pre-processing the data to normalize raw reddit data
    # and then tokenize it - this part can be done in parallel
    print('Preprocessing...')
    # preprocess_files(target_files)  # no need to return anything, this data is dumped to a file since its too large to
    # fit into memory

    # then process 2-grams
    print('Processing 2-grams')
    reddit_2_grams = process_files(target_files, 2, 5)
    reddit_2_grams = delete_rare_ngrams(25, reddit_2_grams)
    print("Final 2-gram size is:", len(reddit_2_grams))
    dump_var_gz("reddit_2_gram", reddit_2_grams)

    # next processes 3-grams
    print('Processing 3-grams')
    reddit_3_grams = process_files(target_files, 3, 3)  # 3,3 orginal
    reddit_3_grams = delete_rare_ngrams(16, reddit_3_grams)  # 11 orginal
    print("Final 3-gram size is:", len(reddit_3_grams))
    dump_var_gz("reddit_3_gram", reddit_3_grams)

    # final post-processing to convert everything into a pmf
    print('Post-Processing 2-grams')
    reddit_2_grams_pmf = pmf_from_2_grams(reddit_2_grams)
    dump_var_gz("reddit_2_gram_pmf", dict(reddit_2_grams_pmf))  # need to convert into a regular dict to dump

    print('Post-Processing 3-grams')
    reddit_3_grams_pmf = pmf_from_3_grams(reddit_3_grams)
    dump_var_gz("reddit_3_grams_pmf", dict(reddit_3_grams_pmf))  # need to convert into a regular dict to dump


log = logging.getLogger("bot")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())
main()
