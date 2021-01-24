#!/usr/bin/env python
import codecs
import fnmatch
import os
import subprocess
import sys
import tarfile
import unicodedata

import pandas
import progressbar
from sox import Transformer
from tensorflow.python.platform import gfile

from deepspeech_training.util.downloader import maybe_download

SAMPLE_RATE = 16000


def _download_and_preprocess_data(data_dir, language="english"):
    # Conditionally download data to data_dir
    print(
        "Downloading MLS data set into {} if not already present...".format(
            data_dir
        )
    )
    with progressbar.ProgressBar(max_value=2, widget=progressbar.AdaptiveETA) as bar:
        MLS_URL = (
            "https://dl.fbaipublicfiles.com/mls/mls_{}.tar.gz".format(language)
        )

        def filename_of(x):
            return os.path.split(x)[1]

        bar.update(0)
        mls_all = maybe_download(
            filename_of(MLS_URL), data_dir, MLS_URL
        )
        bar.update(1)

    # Conditionally extract LibriSpeech data
    # We extract each archive into data_dir, but test for existence in
    # data_dir/LibriSpeech because the archives share that root.
    print("Extracting mls data if not already extracted...")
    with progressbar.ProgressBar(max_value=1, widget=progressbar.AdaptiveETA) as bar:
        MLS_DIR = "MLS"
        work_dir = os.path.join(data_dir, MLS_DIR)

        _maybe_extract(
            data_dir, MLS_DIR, mls_all
        )
        bar.update(0)

    # Convert FLAC data to wav, from:
    #  data_dir/LibriSpeech/split/1/2/1-2-3.flac
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-3.wav
    #
    # And split LibriSpeech transcriptions, from:
    #  data_dir/LibriSpeech/split/1/2/1-2.trans.txt
    # to:
    #  data_dir/LibriSpeech/split-wav/1-2-0.txt
    #  data_dir/LibriSpeech/split-wav/1-2-1.txt
    #  data_dir/LibriSpeech/split-wav/1-2-2.txt
    #  ...
    print("Converting FLAC to WAV and splitting transcriptions...")
    with progressbar.ProgressBar(max_value=3, widget=progressbar.AdaptiveETA) as bar:
        train = _convert_audio_and_split_sentences(
            data_dir, os.path.join("mls_{}".format(language), "train"), os.path.join("mls_{}".format(language), "train-wav")
        )
        bar.update(0)
        test = _convert_audio_and_split_sentences(
            data_dir, os.path.join("mls_{}".format(language), "test"), os.path.join("mls_{}".format(language), "test-wav")
        )
        bar.update(1)
        dev = _convert_audio_and_split_sentences(
            data_dir, os.path.join("mls_{}".format(language), "dev"), os.path.join("mls_{}".format(language), "dev-wav")
        )
        bar.update(2)

    # Write sets to disk as CSV files
    train.to_csv(
        os.path.join(data_dir, "mls-{}-train.csv".format(language)), index=False
    )
    test.to_csv(
        os.path.join(data_dir, "mls-{}-test.csv".format(language)), index=False
    )
    dev.to_csv(
        os.path.join(data_dir, "mls-{}-dev.csv".format(language)), index=False
    )


def _maybe_extract(data_dir, extracted_data, archive):
    # If data_dir/extracted_data does not exist, extract archive in data_dir
    if not gfile.Exists(os.path.join(data_dir, extracted_data)):
        tar = tarfile.open(archive)
        tar.extractall(data_dir)
        tar.close()


def _convert_audio_and_split_sentences(extracted_dir, data_set, dest_dir):
    source_dir = os.path.join(extracted_dir, data_set)
    target_dir = os.path.join(extracted_dir, dest_dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Loop over transcription files and split each one
    #
    # The format for each file 1-2.trans.txt is:
    #  1-2-0 transcription of 1-2-0.flac
    #  1-2-1 transcription of 1-2-1.flac
    #  ...
    #
    # Each file is then split into several files:
    #  1-2-0.txt (contains transcription of 1-2-0.flac)
    #  1-2-1.txt (contains transcription of 1-2-1.flac)
    #  ...
    #
    # We also convert the corresponding FLACs to WAV in the same pass
    files = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, "transcripts.txt"):
            trans_filename = os.path.join(root, filename)
            with codecs.open(trans_filename, "r", "utf-8") as fin:
                for line in fin:
                    # Parse each segment line
                    first_space = line.find("\t")
                    seqid, transcript = line[:first_space], line[first_space + 1 :]

                    # We need to do the encode-decode dance here because encode
                    # returns a bytes() object on Python 3, and text_to_char_array
                    # expects a string.
                    transcript = (
                        unicodedata.normalize("NFKD", transcript)
                        .encode("utf-8", "ignore")
                        .decode("utf-8", "ignore")
                    )

                    transcript = transcript.lower().strip()

                    # Convert corresponding FLAC to a WAV
                    split_seq = seqid.split('_')
                    flac_file = os.path.join(root, 'audio', split_seq[0], split_seq[1], seqid + ".flac")
                    wav_file = os.path.join(target_dir, seqid + ".wav")
                    if not os.path.exists(wav_file):
                        tfm = Transformer()
                        tfm.set_output_format(rate=SAMPLE_RATE)
                        tfm.build(flac_file, wav_file)
                    wav_filesize = os.path.getsize(wav_file)

                    files.append((os.path.abspath(wav_file), wav_filesize, transcript))

    return pandas.DataFrame(
        data=files, columns=["wav_filename", "wav_filesize", "transcript"]
    )


if __name__ == "__main__":
    _download_and_preprocess_data(sys.argv[1], sys.argv[2])
