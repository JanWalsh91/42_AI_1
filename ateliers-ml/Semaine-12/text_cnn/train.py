"""
License (MIT)

Copyright (c) 2018 by Vincent Matthys

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import pandas as pd
import argparse
from datetime import datetime

from utils.preprocessor import Preprocessor
from CNN_sentence_classifier import intent_classifier
import config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Training intent model"
        )
    parser.add_argument('-d',
                        '--train',
                        type=str,
                        help='Train dataset (filename, csv)', required=True)
    parser.add_argument('-v',
                        '--test',
                        type=str,
                        help='Test dataset (filename, csv)', required=True)
    args = parser.parse_args()

    # Load data
    df_train = pd.read_csv(args.train, usecols=[1, 2])
    X_train_raw = df_train["question"].tolist()
    y_train_raw = df_train["intention"].tolist()
    df_test = pd.read_csv(args.test, usecols=[1, 2])
    X_test_raw = df_test["question"].tolist()
    y_test_raw = df_test["intention"].tolist()

    # Preproccessing
    preprocessor = Preprocessor()
    preprocessor.fit(X_train_raw, y_train_raw)
    X_train = preprocessor.build_sequence(X_train_raw)
    X_test = preprocessor.build_sequence(X_test_raw)
    y_train = preprocessor.label_transform(y_train_raw)
    y_test = preprocessor.label_transform(y_test_raw)

    # Intent classifier prediction
    params = {
        "batch_size": 64,
        "num_epochs": 1,
        "embedding_size": 32,
        "filter_sizes": [3, 4, 5],
        'num_filters': 258,
        "patience": 20,
        "dropout": 0.7,
        "sequence_length": preprocessor.max_sentence_size,
        "num_classes": (len(preprocessor.classes_)
                        if len(preprocessor.classes_) > 2 else 1),
        "vocab_size": len(preprocessor.inv_vocab),
        "threshold": 0.5
    }
    clf = intent_classifier(GPU=True)
    clf.configure(
            out_dir=config.intent_model_path +
            str(datetime.now().isoformat()).split(".")[0],
            params=params,
            inverse_labels_map=preprocessor.inverse_labels_map,
            preprocessor=preprocessor
            )
    clf.fit(X_train, y_train, n_epochs=params["num_epochs"], verbosity=1,
            X_valid=X_test, y_valid=y_test)

    # score = clf.score(X_test_raw, y_test_raw)
