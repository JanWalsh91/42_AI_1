"""
License (MIT)

Copyright (c) 2018 by Vincent Matthys

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# General utils
import os
import json
import pickle
from tqdm import tqdm
from datetime import datetime
import numpy as np

# Machine learning utils
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
import warnings
import sklearn.exceptions

# Import utility tools
from utils.plot_confusion import plot_confusion_matrix

"""
Inspired from https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/dnn%20classifier/dnn_classifier.py
""" # noqa

warnings.filterwarnings("ignore",
                        category=sklearn.exceptions.UndefinedMetricWarning)


def batch_iter(x_data, y_data, batch_size, random=True):
    n = x_data.shape[0]
    nbr_batch = n // batch_size + 1
    _index = np.random.permutation(n) if random else np.arange(n)
    if y_data is not None:
        for k in np.array_split(_index, nbr_batch):
            yield x_data[k], y_data[k]
    else:
        for k in np.array_split(_index, nbr_batch):
            yield x_data[k]


class intent_classifier(BaseEstimator, ClassifierMixin):
    """
    """

    def __init__(self, GPU=True):
        """
        """

        # Initialize session to None
        self._session = None
        # Set device environment
        device_count = {"GPU": 1} if GPU else {"GPU": 0}
        self.config_device = tf.ConfigProto(device_count=device_count)

        # Set writers to False
        self._writers = False

    def configure(self,
                  params,
                  out_dir,
                  inverse_labels_map,
                  preprocessor):

        self.inverse_labels_map = inverse_labels_map
        # TODO verify params keys
        # "batch_size": 64,
        # "num_epochs": 50,
        # "embedding_size": 128,
        # "filter_sizes": [3, 4, 5],
        # 'num_filters': 1024,
        # "patience": 3,
        # "dropout": 0.5,
        # "sequence_length": preprocessor.max_sentence_size,
        # "num_classes": len(preprocessor.classes_),
        # "vocab_size": len(preprocessor.inv_vocab),
        self.params = params

        # Store
        self.out_dir = out_dir
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        with open(os.path.join(self.out_dir, "preprocessor"), "wb") as f:
            pickle.dump(preprocessor, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.out_dir, "getstate"), "wb") as f:
            pickle.dump(self.__getstate__(), f, pickle.HIGHEST_PROTOCOL)

    def load(self, model_path):
        """Load model meta graph and tensor by names"""

        # Close session if necessary
        self.close_session()

        # Get model_path
        self.out_dir = model_path
        self.checkpoint_dir = os.path.join(self.out_dir, "checkpoints")
        preprocessor_file = os.path.join(self.out_dir, "preprocessor")
        getstate = os.path.join(self.out_dir, "getstate")

        # Retrieve preprocessor used during training
        with open(preprocessor_file, 'rb') as f:
            self.preprocessor = pickle.load(f)
        # Retrieve params used during training
        with open(getstate, "rb") as f:
            config = pickle.load(f)
        self.params = config['params']
        self.inverse_labels_map = config['inverse_labels_map']
        self.params = config['params']

        self._session = tf.Session(config=self.config_device)

        # Restore meta graph in _session
        with self._session.as_default():
            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
            saver = tf.train.import_meta_graph(
                        "{}.meta".format(checkpoint_file))
            saver.restore(self._session, checkpoint_file)
            # Get tensor by names
            with tf.get_default_graph().as_default() as graph:
                self.probabilities = graph.get_tensor_by_name(
                                "output/probabilities:0")
                self.input_x = graph.get_tensor_by_name("input_x:0")
                self.input_y = graph.get_tensor_by_name("input_y:0")
                self.acc_op = graph.get_tensor_by_name(
                                "metrics/acc_top_3/update_op:0")
                self.acc = graph.get_tensor_by_name(
                                "metrics/acc_top_3/value:0")
                self.dropout_keep_prob = graph.get_tensor_by_name(
                                        "dropout_keep_prob:0")
            print("Restored model checkpoint: {}\n".format(checkpoint_file))

    def _construct_graph(self,
                         sequence_length,
                         num_classes,
                         vocab_size,
                         embedding_size,
                         filter_sizes,
                         num_filters,
                         ):
        """
        """

        ########################## CODE HERE ###################################

        # Declare Placeholders
        # self.input_x =
        # self.input_y =
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        # On the fly embedding is provided, using a skip-gram model
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W"
                    )
            self.embedded_words = tf.nn.embedding_lookup(W, self.input_x)
            # Expand (single channel for conv2d)
            self.embedded_words_expanded =\
                tf.expand_dims(self.embedded_words, -1)

        # Pooling layers definition
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%d" % filter_size):
                # Convolution layer with num_filters features extracted
                # Filter shape definition. Look at tensorflow API for detailss
                # filter_shape =

                # Weights definition
                # Initialization with tf.truncated_normal
                # use stddef = 0.1
                # W =

                # 2D convolutional layer definition using tf API
                # Use stride of 1 in any dimension
                # conv =

                # Bias definition
                # Initialization with tf.constant at 0.1
                # b =

                # Apply non linearity (RELU) after appplying bias
                # to activation map
                # Use tensorflow API for both
                # h =

                # Max pooling over the outputs
                # Use tf.nn.max_poll
                # Be careful about ksize definition
                # pooled =

                # Append outputs
                pooled_outputs.append(pooled)

        # Total number of extracted features
        num_filters_total =

        # Concat all outpus and flatten before classifier
        # Use tf.concat for concatenation
        # Use tf.reshape to flatten - final dimension [-1, num_filters_total]
        # self.h_pool =
        # self.h_pool_flat =

        # Dropout definition provided
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,
                                        self.dropout_keep_prob)

        # Logistic regression definition
        with tf.name_scope("output"):
            # Weights definition
            # Initialization with tf.truncated_normal
            # use stddef = 0.1
            # W =

            # Bias definition
            # Initialization with tf.constant at 0.1
            # b =

            # Compute output score
            # Use tf.nn.xw_plus_b
            # self.scores =

            # Compute output probabilities
            # Hint: have to sum to 1
            # self.probabilities =

            # Get output class
            # Hint:
            # self.output =

            # Top 3 output
            # Definition profided
            _, self.output_top_3 =\
                tf.nn.top_k(self.scores, k=3, name="3_pred")
            self.correct_among_top_3 =\
                tf.nn.in_top_k(self.scores,
                               tf.argmax(self.input_y, 1),
                               k=3)

        # Batch loss definition
        # Using tf API
        # Hint: cross entropy
        with tf.name_scope("losses"):
            # self.losses =

        ################### END OF YOUR CODE ###################################
        # Global step definition
        self.global_step =\
            tf.Variable(0, name="global_step", trainable=False)

        # Training operation and optimizer definition
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads = optimizer.compute_gradients(self.losses)
        self.train_op =\
            optimizer.apply_gradients(grads, global_step=self.global_step)

        with tf.name_scope("avg_loss"):
            self.avg_loss = tf.reduce_mean(self.losses, name="avg_loss")

        with tf.name_scope("avg_accuracy"):
            # Total accuracy
            correct = tf.equal(self.output, tf.argmax(self.input_y, 1))
            self.avg_accuracy = tf.reduce_mean(
                                tf.cast(correct, "float"),
                                name="avg_accuracy")

        with tf.name_scope("avg_acc_top_3"):
            # Current acc in top_3 for training batch
            self.avg_acc_top_3 = tf.reduce_mean(
                                 tf.cast(self.correct_among_top_3, "float32"),
                                 name="avg_acc_top_3")

        with tf.name_scope("metrics"):
            self.acc, self.acc_op =\
                tf.metrics.accuracy(tf.argmax(self.input_y, 1),
                                    self.output,
                                    name="acc")
            # Accuracy among the 3 best intentions
            self.acc_top_3, self.acc_top_3_op =\
                tf.metrics.mean(tf.cast(self.correct_among_top_3, "float32"),
                                name="acc_top_3")
            self.acc_per_class, self.acc_per_class_op =\
                tf.metrics.mean_per_class_accuracy(tf.argmax(self.input_y, 1),
                                                   self.output,
                                                   num_classes,
                                                   name="class_acc")
            self.loss, self.loss_op = tf.metrics.mean(self.losses,
                                                      name="loss")
        # Initializer and saver
        self._init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()

        self._init_writers()
        self._init_summaries()

    def _init_train_writer(self):
        """
        """
        train_summary_dir = os.path.join(self.out_dir,
                                         "summaries",
                                         "train")
        self.train_summary_writer =\
            tf.summary.FileWriter(train_summary_dir, self._graph)

    def _init_test_writer(self):
        """
        """
        test_summary_dir = os.path.join(self.out_dir,
                                        "summaries",
                                        "validation")
        self.test_summary_writer = tf.summary.FileWriter(test_summary_dir,
                                                         self._graph)

        # Confusion map writer
        img_d_summary_dir = os.path.join(self.out_dir, "summaries", "img")
        self.img_d_summary_writer = tf.summary.FileWriter(img_d_summary_dir,
                                                          self._graph)

    def _init_writers(self):
        if self._writers is False:
            self._init_train_writer()
            self._init_test_writer()
            self._writers = True

    def _close_writers(self):
        """
        """
        if self._writers is True:
            # Close writers (equivalent to flush writers + close the file)
            self.train_summary_writer.close()
            self.test_summary_writer.close()
            self.img_d_summary_writer.close()

    def _flush_tf_writers(self):
        self.train_summary_writer.flush()
        self.test_summary_writer.flush()
        self.img_d_summary_writer.flush()

    def _init_summaries(self):
        """
        """
        # Scalar summaries
        self.avg_loss_summary = tf.summary.scalar("avg_loss",
                                                  self.avg_loss)
        self.avg_acc_summary = tf.summary.scalar("avg_accuracy",
                                                 self.avg_accuracy)
        self.avg_acc_top_3_summary = tf.summary.scalar("avg_acc_top_3",
                                                       self.avg_acc_top_3)

        self.train_summary_op = tf.summary.merge([self.avg_loss_summary,
                                                  self.avg_acc_summary,
                                                  self.avg_acc_top_3_summary])

        self.test_summary_op = tf.summary.merge(
                                [self.avg_loss_summary,
                                 self.avg_acc_summary,
                                 self.avg_acc_top_3_summary])

    def _reset_metrics_op(self):
        # name_scope of update_op : metrics in TextCNN model
        stream_vars = [i for i in tf.local_variables()
                       if i.name.split('/')[0] == 'metrics']
        # Reinitializer of local variables
        reset_op = [tf.variables_initializer(stream_vars, name='init')]
        self._session.run(reset_op)

    def _print_F1(self, F1, file=None):
        for label in range(len(self.inverse_labels_map)):
            print(" "*10 + "\033[92m{:40}: {:.4f}\033[0m"
                  .format(self.inverse_labels_map[label],
                          F1[label]),
                  file=file)

    def close_session(self):
        if self._session is not None:
            self._session.close()

    def _get_model_parameters(self):
        # Retrieves the value of all the variables in the network
        with self._graph.as_default():
            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        return {gvar.op.name: value
                for gvar, value in zip(gvars, self._session.run(gvars))}

    def _save_checkpoint(self, step):
        path = self._saver.save(self._session,
                                self.checkpoint_prefix,
                                global_step=step)
        print("Saved model checkpoint to {}\n".format(path))

    def _restore_model_parameters(self, model_params):
        """
        graph.get_operation_by_name(operation).inputs returns the input to the
        given operation; because these are all assignment operations, the
        second argument to inputs is the value assigned to the variable
        """
        # Restores the value of all variables using tf assign operations
        # First retrieve the list of all the graph variables
        gvar_names = list(model_params.keys())

        # Then retrieve all the assignment operations in the graph
        assign_ops = {gvar_name: self._graph.get_operation_by_name(
                    gvar_name + "/Assign") for gvar_name in gvar_names}

        # Fetch the initialization values of the assignment operations
        init_values = {gvar_name: assign_op.inputs[1]
                       for gvar_name, assign_op in assign_ops.items()}
        # Create a dictionary mapping initial values to values after training
        feed_dict = {init_values[gvar_name]: model_params[gvar_name]
                     for gvar_name in gvar_names}
        # Assign the trained value to all the variables in the graph
        self._session.run(assign_ops, feed_dict=feed_dict)
        print("\033[1;42mBest model has been restored after patience"
              " ended\033[0m")

    def _load_last_checkpoint(self):
        """Load model weights"""
        self.close_session()
        # self._session = tf.Session(config=self.config_device,
        #                            graph=self._graph)
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(self._sesssion, checkpoint_file)

        print("Restored model checkpoint: {}\n".format(checkpoint_file))

    def _train_step(self, x_batch, y_batch, writer=None):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: self.params['dropout']
        }

        targets = [self.train_op,
                   self.global_step,
                   self.train_summary_op,
                   self.avg_loss,
                   self.avg_accuracy,
                   self.avg_acc_top_3,
                   self.loss_op,
                   self.acc_op,
                   self.acc_top_3_op]
        # Per batch summary
        _, step, summaries, loss, acc, acc_top_3, loss_op, acc_op, acc_3_op =\
            self._session.run(targets, feed_dict)
        if writer is not None:
            writer.add_summary(summaries, step)
        return step, (loss, acc, acc_top_3), (loss_op, acc_op, acc_3_op)

    def _test_step_batch(self, x_batch, y_batch):
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0
        }

        targets = [self.global_step,
                   self.output,
                   self.loss_op,
                   self.acc_op,
                   self.acc_top_3_op]
        step, output, loss_op, acc_op, acc_top_3_op =\
            self._session.run(targets, feed_dict)
        return step, output, loss_op, acc_op, acc_top_3_op

    def _test_step(self, x_test, y_test,
                   writer=None,
                   img_writer=None,
                   save_cm=None,
                   random=True):
        nbr_batches = x_test.shape[0] // self.params['batch_size']
        # List of output for confusion matrix
        output = []

        for X_batch, y_batch in batch_iter(x_test,
                                           y_test,
                                           self.params['batch_size'],
                                           random=random):
            step, output_batch, loss_op, acc_op, acc_top_3_op =\
                self._test_step_batch(X_batch, y_batch)
            output.extend(output_batch)

        # Get value of loss and acc over entire batches
        loss, acc, acc_top_3 = self._session.run([self.loss,
                                                  self.acc,
                                                  self.acc_top_3])

        # Compute F1 score
        F1 = f1_score(y_test.argmax(1), output, average=None)
        # Build the corresponding test summaries
        if writer is not None:
            summaries = self._session.run(self.test_summary_op,
                                          {
                                            self.avg_loss: loss,
                                            self.avg_accuracy: acc,
                                            self.avg_acc_top_3: acc_top_3
                                           })
            writer.add_summary(summaries, step)
        # Build the confusion matrix summary
        if img_writer is not None:
            # Plot the confusion matrix
            output = np.array(output)
            img_d_summary = plot_confusion_matrix(
                            [self.inverse_labels_map[l]
                             for l in y_test.argmax(1)],
                            [self.inverse_labels_map[o] for o in output],
                            [self.inverse_labels_map[key]
                                for key in self.inverse_labels_map.keys()],
                            custom_figsize=(7, 7),
                            tensor_name='dev/cm',
                            normalize=False,
                            savefig=save_cm,
                            dpi=300)
            img_writer.add_summary(img_d_summary, step)
        return dict(zip(("loss", "acc", "acc_top_3"),
                        (float(loss), float(acc), float(acc_top_3)))), F1

    def fit(self,
            X,
            y,
            n_epochs,
            X_valid=None,
            y_valid=None,
            verbosity=1):
        """
        Early stopping if validation data is provided
        verbosity: Integer. 0, 1, or 2.
                   Verbosity mode.
                   0 = silent, 1 = progress bar, 2 = one line per batch
        """

        verbosity_level = [0, 1, 2]
        if verbosity not in verbosity_level:
            raise ValueError("Verbosity not in {}".format(verbosity_level))

        # self.close_session()
        self._graph = tf.Graph()

        # Build the computation graph with self as default graph
        with self._graph.as_default():
            self._construct_graph(
                sequence_length=self.params['sequence_length'],
                num_classes=self.params['num_classes'],
                vocab_size=self.params['vocab_size'] + 1,
                embedding_size=self.params['embedding_size'],
                filter_sizes=self.params['filter_sizes'],
                num_filters=self.params['num_filters']
                )

        # Early stopping parameters
        checks_without_progress = 0
        best_parameters = None
        best_valid_metrics = None
        best_F1 = None

        # Number of batches per epoch
        nbr_batch = X.shape[0] // self.params['batch_size']

        # Starting session
        self._session = tf.Session(config=self.config_device,
                                   graph=self._graph)

        with self._session.as_default() as sess:
            with self._session.graph.as_default():
                # Initialize all variables
                self._init.run()
                # Iterate over epochs
                for epoch in range(n_epochs):
                    # Reset update_op between epochs
                    self._reset_metrics_op()

                    if verbosity == 2:
                        print("\n---- EPOCH {} ----".format(epoch))
                    # Iterate over batches in X
                    with tqdm(batch_iter(X,
                                         y,
                                         self.params['batch_size']),
                              total=nbr_batch,
                              disable=False if verbosity == 1 else True,
                              bar_format="Epoch {desc}training "
                                         "{n_fmt}/{total_fmt}"
                                         "|{bar}|[{elapsed}{rate_fmt}]"
                                         "{postfix}") as pbar:
                        # Verbosity
                        if verbosity == 1:
                            pbar.set_description("{}".format(epoch + 1))

                        for x_batch, y_batch in pbar:
                            # Train_step
                            step, batch_metric, epoch_metric = (
                                self._train_step(x_batch,
                                                 y_batch,
                                                 writer=self
                                                 .train_summary_writer
                                                 ))

                            # Verbosity
                            if verbosity == 1:
                                pbar.set_postfix_str(
                                    "loss={:.4f}, acc={:.4f}, acc_top_3={:.4f}"
                                    .format(*epoch_metric)
                                    )
                            elif verbosity == 2:
                                t = datetime.now().isoformat("|").split(".")[0]
                                print("{}: Epoch {}, step {}/{}, loss"
                                      "{:.4f}, acc {:.4f}, acc_top_3 {:.4f}"
                                      .format(t, epoch, step,
                                              nbr_batch * epoch, *epoch_metric)
                                      )
                    # Validation step - end of epoch - if X_valid provided
                    if X_valid is not None and y_valid is not None:
                        # Reset update_op for evaluation
                        self._reset_metrics_op()
                        valid_metrics, F1 = self._test_step(
                                    X_valid,
                                    y_valid,
                                    writer=self.test_summary_writer,
                                    img_writer=self.img_d_summary_writer,
                                    save_cm=self.out_dir + "/cm.png"
                                    if epoch + 1 == n_epochs else None,
                                    random=False)
                        # Verbosity
                        if verbosity >= 1:
                            print(" "*10 + "\033[1;44m DEV SCORE EPOCH {} - "
                                  "loss={:.4f}, acc={:.4f}, acc_top_3={:.4f}"
                                  "\033[0m"
                                  .format(epoch + 1,
                                          valid_metrics["loss"],
                                          valid_metrics["acc"],
                                          valid_metrics["acc_top_3"]))

                        # Check to see if model is improving
                        if best_valid_metrics is None:
                            best_valid_metrics = valid_metrics
                        if valid_metrics["loss"] <= best_valid_metrics["loss"]:
                            best_valid_metrics = valid_metrics
                            best_F1 = F1
                            checks_without_progress = 0
                            # If improving, store the model_parameters
                            best_parameters = self._get_model_parameters()
                        else:
                            checks_without_progress += 1

                        if checks_without_progress > self.params['patience']:
                            print("Stopping Early! Validation loss has not"
                                  "decresed in {} epochs"
                                  .format(self.params['patience']))
                            break

                    # Verbosity F1 score
                    if verbosity == 2 and best_F1 is not None:
                        self._print_F1(best_F1)

                # In the case of early stopping, restore the best weight values
                if best_parameters:
                    self._restore_model_parameters(best_parameters)
                # Save metrics for the best_parameters
                self.vvv = best_valid_metrics
                with open(os.path.join(self.out_dir,
                                       "metrics.json"), "w") as fjson:
                    json.dump(best_valid_metrics, fjson, indent=4)
                    print("\nF1 score:", file=fjson)
                    self._print_F1(best_F1, file=fjson)

                # Verbosity F1 score
                if verbosity == 1 and best_F1 is not None:
                    self._print_F1(best_F1)

                # Save network
                self._save_checkpoint(step)
                self._close_writers()
                return self

    def _preprocess_entry(self, X, y=None):
        if y is None:
            return self.preprocessor.build_sequence(X)
        else:
            return (self.preprocessor.build_sequence(X),
                    self.preprocessor.label_transform(y))

    def predict_probabilities(self, X):
        # Check if model has been loaded
        if not self._session:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        # If X is a unique exemple, reshape it
        X = self._preprocess_entry(X)
        # Evaluate tensor for the given meta graph restored
        with self._session.as_default():
            return np.vstack([self.probabilities.eval(
                feed_dict={self.input_x: X,
                           self.dropout_keep_prob: 1.0})
                for X in batch_iter(
                x_data=X,
                y_data=None,
                batch_size=X.shape[0] // self.params['batch_size'] + 1,
                random=False)])

    def predict(self, X):
        # Check if model has been loaded
        if not self._session:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        return [self.inverse_labels_map[pred_label.argmax()]
                for pred_label in self.predict_probabilities(X)]

    def predict_top_3(self, X):
        # Check if model has been loaded
        if not self._session:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        # Negate probabilities to sort descending. Keep first 3 values
        return [[self.inverse_labels_map[label] for label in pred_label]
                for pred_label
                in np.argsort(-self.predict_probabilities(X), axis=1)[:, :3]]

    def score(self, X_valid, y_valid):
        """
        Returns the mean top 3 accuracy
        """
        # Check if model has been loaded
        if not self._session:
            raise NotFittedError("This {} instance is not fitted yet"
                                 .format(self.__class__.__name__))
        X_valid, y_valid = self._preprocess_entry(X_valid, y_valid)

        with self._session.as_default():
            with self._session.graph.as_default():
                self._reset_metrics_op()
                for X_batch, y_batch in batch_iter(X_valid,
                                                   y_valid,
                                                   self.params['batch_size'],
                                                   random=False):
                    self.acc_top_3_op.eval(feed_dict={
                        self.input_x: X_batch,
                        self.input_y: y_batch,
                        self.dropout_keep_prob: 1.0})

                return self.acc_top_3.eval()
