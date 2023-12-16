import os
import zipfile

import gdown


def setup_file(url, filename):
    gdown.download(url, filename, fuzzy=True)

    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()

    os.remove(filename)


setup_file(
    "https://drive.google.com/file/d/188xgMRA7frpUEqXUTSr_JV3N1grAAZ6p/view?usp=sharing",
    "small.zip",
)
setup_file(
    "https://drive.google.com/file/d/1JUzIy6q6t6aARgM-p_B6UNIwuGjYc9zw/view?usp=sharing",
    "big.zip",
)
# setup_file(
#     "https://drive.google.com/file/d/1eOL503j3SrunU5ZpoChGbbLJqR07sRou/view?usp=sharing",
#     "test_audio.zip",
# )
