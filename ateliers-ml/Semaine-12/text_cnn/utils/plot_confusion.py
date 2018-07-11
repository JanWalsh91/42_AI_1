"""
License (MIT)

Copyright (c) 2018 by Vincent Matthys

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from textwrap import wrap
import re
import itertools
import tfplot
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix
# from sklearn.metrics import f1_score


def plot_confusion_matrix(correct_labels,
                          predict_labels,
                          labels,
                          title="Confusion matrix",
                          custom_figsize=(7, 7),
                          tensor_name='MyFigure/image',
                          normalize=False,
                          savefig=False,
                          dpi=640):
    """
    Parameters:
        correct_labels                  : These are your true\
                                          classification categories.
        predict_labels                  : These are you predicted\
                                          classification categories
        labels                          : This is a list of labels which\
                                          will be used to display the axix\
                                          labels
        title='Confusion matrix'        : Title for your matrix
        custom_figsize = (7,7)          : Size of matplotlib figure
        tensor_name = 'MyFigure/image'  : Name for the output summay tensor
        savefig = False                 : Path to save the figure

    Returns:
        summary: TensorFlow summary

    Other itema to note:
        - Depending on the number of category and the data,\
        you may have to modify the figzie, font sizes etc.
        - Currently, some of the ticks dont line up due to rotations.
    """
    cm = confusion_matrix(correct_labels, predict_labels, labels=labels)
    if normalize:
        cm = cm.astype('float')*10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)
    #fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(
                            figsize=custom_figsize,
                            dpi=dpi,
                            facecolor='w',
                            edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    # # Add table of metrics to confusion matrix
    # rows = ["F1 score"]
    # columns = ["Negative", "Positive"]
    # cell_text = []
    # cell_text.append(["{0:.3f}".format(score)\
    #     for score in f1_score(correct_labels, predict_labels, average = None)])
    #
    # print (cell_text[0])

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))',
                      r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes,
                           fontsize=4,
                           rotation=-90,
                           ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # Highlights diagonal
        color_text = "olive" if i == j else "black"
        withdash_text = True if i == j else False
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
                horizontalalignment="center",
                fontsize=5,
                verticalalignment='center',
                color=color_text)
                # withdash = withdash_text)
    # table = matplotlib.pyplot.table(cellText = cell_text,
    #                         rowLabels = rows,
    #                         colLabels = columns,
    #                         loc = "right"
    #                         )

    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    if savefig is not None:
        fig.savefig(savefig)
    return summary
