"""
A :class:`~allennlp_data.dataset_readers.dataset_reader.DatasetReader`
reads a file and converts it to a collection of
:class:`~allennlp_data.instance.Instance` s.
The various subclasses know how to read specific filetypes
and produce datasets in the formats required by specific models.
"""


from allennlp_data.dataset_readers.dataset_reader import (
    DatasetReader,
    WorkerInfo,
)
from allennlp_data.dataset_readers.babi import BabiReader
from allennlp_data.dataset_readers.conll2003 import Conll2003DatasetReader
from allennlp_data.dataset_readers.interleaving_dataset_reader import InterleavingDatasetReader
from allennlp_data.dataset_readers.multitask import MultiTaskDatasetReader
from allennlp_data.dataset_readers.sequence_tagging import SequenceTaggingDatasetReader
from allennlp_data.dataset_readers.sharded_dataset_reader import ShardedDatasetReader
from allennlp_data.dataset_readers.text_classification_json import TextClassificationJsonReader
