import gzip
import os
import shutil
from urllib.request import urlretrieve

import wget

if not os.path.exists("language_model"):
    os.mkdir("language_model")
os.chdir("language_model")

lm_gzip_path = "3-gram.arpa.gz"
if not os.path.exists(lm_gzip_path):
    print("Downloading pruned 3-gram model.")
    lm_url = "https://www.openslr.org/resources/11/3-gram.arpa.gz"
    lm_gzip_path = wget.download(lm_url)
    print("\n Downloaded the 3-gram language model.")
else:
    print("\n Pruned .arpa.gz already exists.")

uppercase_lm_path = "3-gram.arpa"
if not os.path.exists(uppercase_lm_path):
    with gzip.open(lm_gzip_path, "rb") as f_zipped:
        with open(uppercase_lm_path, "wb") as f_unzipped:
            shutil.copyfileobj(f_zipped, f_unzipped)
    print("\n Unzipped the 3-gram language model.")
else:
    print("\n Unzipped .arpa already exists.")

lm_path = "lowercase_3-gram.arpa"
if not os.path.exists(lm_path):
    with open(uppercase_lm_path, "r") as f_upper:
        with open(lm_path, "w") as f_lower:
            for line in f_upper:
                f_lower.write(line.lower())
    print("\n Converted language model file to lowercase.")
else:
    print("\n Converted language model already exists.")

vocab_path = "librispeech-vocab.txt"
if not os.path.exists(vocab_path):
    print("Downloading vocab.")
    vocab_url = "http://www.openslr.org/resources/11/librispeech-vocab.txt"
    vocab_path = wget.download(vocab_url)
    print("\n Librispeech vocab.")
else:
    print("\n Librispeech vocab already exists.")

if os.path.exists(lm_gzip_path):
    os.remove(lm_gzip_path)

if os.path.exists(uppercase_lm_path):
    os.remove(uppercase_lm_path)
