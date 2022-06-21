"""CARB: Dataset."""

from __future__ import absolute_import, division, print_function

import csv
import logging

import datasets

import math
from collections import defaultdict

_DESCRIPTION = """\
CARB dataset
"""
MAX_ARGUMENTS = 4

_URL = ""
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "val.json",
    "test": _URL + "test.json",
}

TRIPLET_TOKEN_FORMAT = '<triplet>'
ARGUMENT_TOKEN_FORMAT = '<arg{}>'


class CARBConfig(datasets.BuilderConfig):
    """BuilderConfig for CARB."""
    def __init__(self, **kwargs):
        """BuilderConfig for NYT.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CARBConfig, self).__init__(**kwargs)


class CARB(datasets.GeneratorBasedBuilder):
    """NYT: Version 1.0."""

    BUILDER_CONFIGS = [
        CARBConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "context": datasets.Value("string"),
                "triplets": datasets.Value("string"),
            }),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        if self.config.data_files:
            downloaded_files = {
                "train":
                self.config.data_files["train"],  # self.config.data_dir + "en_train.jsonl",
                "dev": self.config.data_files["dev"],  #self.config.data_dir + "en_val.jsonl",
                "test": self.config.data_files["test"],  #self.config.data_dir + "en_test.jsonl",
            }
        else:
            downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logging.info("generating examples from = %s", filepath)

        with open(filepath) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            prev_text, triplets = '', ''
            for id_, row in enumerate(csv_reader):
                id_str = str(id_)
                text, rel = row[0], row[1]
                args = row[2:]
                n_arguments = min(len(args), MAX_ARGUMENTS)
                args = args[:n_arguments]

                triplets = TRIPLET_TOKEN_FORMAT + f' {rel}'
                for arg_num, arg in enumerate(args):
                    triplets += f' {ARGUMENT_TOKEN_FORMAT.format(arg_num)} {arg}'

                if text != prev_text:
                    yield id_str, {
                        "title": id_str,
                        "context": text,
                        "id": id_str,
                        "triplets": triplets,
                    }
                    prev_text = text
                    triplets = ''

            if triplets != '':
                yield id_str, {
                    "title": id_str,
                    "context": text,
                    "id": id_str,
                    "triplets": triplets,
                }
