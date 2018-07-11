"""
License (MIT)

Copyright (c) 2018 by Vincent Matthys

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import numpy as np
import re
import pickle
import unidecode
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import NotFittedError

import tensorflow as tf


class Preprocessor(object):
    """
    Preprocessor class.
    - fit: fit the preprocessor with text data and labels categories
    - build_sequence: vectorise sentences
    - label_transform: one-hot encode labels (CategoricalEncoder sklearn 0.20)
    """
    def __init__(self, max_features=None, max_sentence_size=None):
        self.max_features = max_features
        self.max_sentence_size = max_sentence_size
        self.freezed = False

    def clean_str(self, s):
        """
        Remove some noise from the strings
        """
        # lossy ASCII transliterations of Unicode text
        # https://github.com/avian2/unidecode
        s = re.sub(r"['’ʼ]", " ", unidecode.unidecode(s.lower()))
        return s

    def clean_seq(self, seq):
        if type(seq) == str:
            return [self.clean_str(seq)]
        elif type(seq) == list:
            return [self.clean_str(sentence) for sentence in seq]
        else:
            raise TypeError("Input is not a str or list object")

    def fit_tokenizer(self, X):
        self.tokenizer.fit_on_texts(X)
        self.inv_vocab = {v: k for k, v in self.tokenizer.word_index.items()}

    def fit_label_encoder(self, y):
        # y must be a list or ndarray with ndim <= 1
        return self.le.fit(y)
        # y_num = self.label_transform(y)
        # self.ohe.fit(y_num.reshape(-1, 1))

    def label_transform(self, y):
        if not self.freezed:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        y_num = self.le.transform(y).reshape(-1, 1)
        if len(self.classes_) > 2:
            if not hasattr(self.ohe, "feature_indices_"):
                return self.ohe.fit_transform(y_num).toarray()
            else:
                return self.ohe.transform(y_num).toarray()
        else:
            return y_num
        # if type(y) == np.ndarray and y.ndim <= 2:
        #     if y.ndmin == 1:
        #         return self.le.transform(y)
        #     # If onehotencoder (ndim = 2)
        #     else:
        #         return self.ohe.transform(y)
        # elif type(y) == list:
        #     return self.le.transform(y)
        # else:
        #     raise TypeError("Can transform only list or ndarray (ndim <= 2)")

    def label_inverse_transform(self, y):
        if not self.freezed:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        return self.le.inverse_transform(y)

    def fit(self, X, y):
        # Call tf tokenizer
        self.tokenizer =\
            tf.keras.preprocessing.text.Tokenizer(
                num_words=self.max_features)

        # Call sklearn labelEncoder and OneHotEncoder
        self.le = LabelEncoder()
        self.ohe = OneHotEncoder()
        # Fit tokenizer
        self.fit_tokenizer(self.clean_seq(X))
        print("Number of unique tokens: {}"
              .format(len(self.inv_vocab)))
        # Fit LabelEncoder
        self.fit_label_encoder(y)
        self.classes_ = self.le.classes_
        print("Number of classes: {}".format(len(self.classes_)))
        # Get inverse mapping
        self.inverse_labels_map = {k: v
                                   for k, v in enumerate(self.classes_)}
        self.freeze()

    def build_sequence(self, X):
        if not self.freezed:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        X_seq = tf.keras.preprocessing.sequence.pad_sequences(
            self.tokenizer.texts_to_sequences(self.clean_seq(X)),
            maxlen=self.max_sentence_size,
            padding="post")

        # Recover the max_sentence_size
        if self.max_sentence_size is None:
            self.max_sentence_size = X_seq.shape[1]

        return X_seq

    def freeze(self):
        self.freezed = True
