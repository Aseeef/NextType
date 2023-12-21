import pickle
import zstandard
import gzip
import os
from typing import Dict, List, DefaultDict, Tuple, Union

data_dir = '/projectnb/cs505ws/projects/NextType'

def does_var_exists_gz(var_name: str) -> bool:
    return os.path.isfile(F'{data_dir}/data/{var_name}.pkl.gz')


def dump_var_gz(var_name: str, obj) -> None:
    os.makedirs("./data", exist_ok=True)
    with gzip.open(F'{data_dir}/data/{var_name}.pkl.gz', 'wb', compresslevel=1) as file:
        pickle.dump(obj, file)


def load_var_gz(var_name: str) -> Union[None, object]:
    if not does_var_exists_gz(var_name):
        print(f"Error: The file {data_dir}/data/{var_name}.pkl.gz does not exist!")
        return None

    file_path = F'{data_dir}/data/{var_name}.pkl.gz'  # Updated file extension
    with gzip.open(file_path, 'rb', compresslevel=1) as file:
        return pickle.load(file)


def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        print(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2 ** 27, (2 ** 29) * 2)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, file_handle.tell()

            buffer = lines[-1]

        reader.close()