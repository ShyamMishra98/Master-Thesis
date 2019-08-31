import os
import pathlib
import pickle
from dataclasses import InitVar, dataclass, field
from typing import Dict, List, Union, Optional
import codecs

import numpy as np

from blingfire import text_to_words

def word_tokenize(string):
    """Tokenize space delimited string with blingfire."""
    return text_to_words(string).split(' ')


@dataclass
class Vocabulary:
    """
    Ancillary word2id, id2word, and embeddings container class.

    :attr len: int, number of tokens in Dictionary
    :attr word2id: dict, keys: tokens, values: id
    :attr id2word: dict, k-v pair inverse of word2id
    :attr embeddings: list[list[np.ndarray]], tokens embedded
                      -- N: number of words
                      -- d: word vector dimensionality
    """

    word2id: dict = field(default_factory=dict)
    id2word: dict = field(default_factory=dict)
    embeddings: np.ndarray = field(init=False)
    dim: int = field(init=False)

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, index):
        return self.id2word[index]

    def add(self, word):
        """
        Add a new word to word2id & id2word.

        :param word:
            (a) str, word to be added, e.g. "hello"
            (b) list[str], words to be added, e.g. ["first", "second"]
        """
        if isinstance(word, list):
            for token in word:
                self.add(token)
        else:
            assert isinstance(word, str), "Passed argument not a string"
            if word not in self.word2id:
                len_ = len(self)
                self.word2id[word] = len_
                self.id2word[len_] = word

    @classmethod
    def from_embeddings(cls,
                        path: str = None,
                        pass_header: bool = True,
                        top_n_words: int = 200_000,
                        normalize: bool = False,
                        dtype=np.float32):
        """
        Instantiate Dictionary from pretrained word embeddings.

        :param path: str, path to pretrained word embeddings
        :param top_n_words: int, restrict Dictionary top_n_words frequent words
        :return: Dictionary, populated word2id and id2word from document tokens
        """
        assert os.path.exists(path), f'{path} not found!'
        cls_ = cls()
        with codecs.open(path, 'r', encoding='utf8', errors='ignore') as f:
            embeddings = []
            if pass_header:
                next(f)
            for idx, line in enumerate(f):
                if len(cls_) == top_n_words:
                    break
                token, vector = line.rstrip().split(' ', maxsplit=1)
                if token not in cls_.word2id:
                    embeddings.append(np.fromstring(vector.strip(), sep=' '))
                    cls_.add(token)
        cls_.embeddings = np.asarray(np.stack(embeddings), dtype=dtype)
        cls_.dim = cls_.embeddings.shape[-1]
        if normalize:
            norm = np.linalg.norm(cls_.embeddings, ord=2,
                                  axis=-1, keepdims=True)
            cls_.embeddings /= norm
        assert len(cls_.embeddings) == len(cls_.word2id), 'Reading error!'
        return cls_

    @classmethod
    def from_dictionary(cls,
                        dict_: Dict[str, int],
                        embeddings: Optional[np.ndarray] = None):
        """Instantiate Vocabulary instance from word2id dictionary."""
        assert isinstance(dict_, dict), 'Please pass a dictionary!'
        cls_ = cls()
        cls_.word2id = dict_
        cls_.id2word = {v: k for k, v in dict_.items()}
        if embeddings is not None:
            assert len(cls_) == len(embeddings), 'Shapes do not align!'
            assert isinstance(embeddings, np.ndarray), 'Not an np.ndarray'
            cls_.embeddings = embeddings
            cls_.dim = cls_.embeddings.shape[-1]
        return cls_

    @classmethod
    def from_pretrained(cls,
                        word2id_path: str,
                        embeddings_path: str):
        cls_ = cls()
        with open(word2id_path, 'rb') as file:
            word2id = pickle.load(file)
            cls_.word2id = word2id

        cls_.id2word = {v: k for k, v in cls_.word2id.items()}
        cls_.embeddings = np.load(embeddings_path)
        cls_.dim = cls_.embeddings.shape[-1]
        return cls_

    def to_vec(self, filepath: str,
                     write_header: bool = True):

        with open(filepath, 'w') as file:
            if write_header:
                N, dim = self.embeddings.shape
                file.write(f'{N} {dim}\n')
            for token, embedding in zip(self.word2id, self.embeddings):
                emb_str = ' '.join(embedding.astype('U').tolist())
                out = token + ' ' + emb_str + '\n'
                file.write(out)
