import tensorflow as tf
import pandas as pd
from data_utils import generate_sentences
from extract_states import *
from regular_rnn import RegularRNN
from utils import *
from config import Config

import numpy as np

config = Config()

NUM_EPOCHS = config.RNN.NUM_EPOCHS.int
state_size = config.RNN.state_size.int
path = config.Misc.output_path.str
init_state = np.zeros(state_size)


def train(X_train, y_train, X_val, y_val, sess, rnn):
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('board', sess.graph)
    correct = []
    correct_labels = []

    print("Start learning...")
    for epoch in range(NUM_EPOCHS):
        loss_train = 0
        accuracy_train = 0
        i = 0

        print("epoch: {}\t".format(epoch), end="")

        # Training
        for sentence, label in zip(X_train, y_train):
            acc, loss_tr, _, summary = sess.run([rnn.accuracy, rnn.loss, rnn.optimizer, merged],
                                                feed_dict={rnn.label_ph: label,
                                                           rnn.init_state_ph: [init_state],
                                                           rnn.input_ph: sentence})

            accuracy_train += acc
            loss_train += loss_tr
            train_writer.add_summary(summary, i)
            i += 1

        accuracy_train /= len(y_train)
        loss_train /= len(y_train)

        # Testing
        accuracy_test = 0
        loss_test = 0

        for sentence, label in zip(X_val, y_val):
            loss, acc = sess.run([rnn.loss, rnn.accuracy],
                                 feed_dict={rnn.label_ph: label, rnn.input_ph: sentence,
                                            rnn.init_state_ph: [init_state]})
            accuracy_test += acc
            loss_test += loss

            if epoch == NUM_EPOCHS - 1 and acc == 1:
                correct.append(sentence)
                correct_labels.append(label)

        accuracy_test /= len(y_val)
        loss_test /= len(y_val)

        print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(
            loss_train, loss_test, accuracy_train, accuracy_test
        ))

    train_writer.close()
    return correct, correct_labels


def extract_graphs():
    """
    after the net has trained enough, we calculate the states returned and print the graphs of them.
    :param X: the training data
    :param y: the labels
    :return: nothing
    """
    init_node = SearchNode(State(init_state, quantized=tuple(init_state)))
    X_val_distinct = list(set([tuple(x) for x in X_val]))
    analog_nodes = get_analog_nodes(X_val_distinct, init_node, rnn)
    analog_states = [node.state for node in analog_nodes if not node == init_node]
    pca_model = PCA(n_components=2)
    pca_model = pca_model.fit([node.state.vec for node in analog_nodes])

    if len(analog_nodes) < 300:
        print_graph(analog_nodes, path + 'orig.png')

    print('num of nodes in original graph:', len(analog_nodes))
    trimmed_states = retrieve_minimized_equivalent_graph(analog_nodes, 'original', init_node, pca_model, path=path)

    trimmed_graph = get_trimmed_graph(analog_nodes)
    states = [node.state.vec for node in analog_nodes]
    colors = [color(node, init_node) for node in analog_nodes]

    plot_states(states, colors, 'RNN\'s continuous state vectors', pca_model, path, True)

    print('num of nodes in the trimmed graph:', len(trimmed_graph))

    if config.ClusteringModel.use_model.boolean:
        predictions = list()
        for sentence in X_val_distinct:
            y_hat = sess.run(rnn.prediction, feed_dict={rnn.input_ph: sentence, rnn.init_state_ph: [init_state]})
            y_pred = 1 if y_hat > 0 else 0
            predictions.append(y_pred)

        quantized_nodes, init_node = get_quantized_graph(analog_states, init_node, rnn, X_val_distinct, predictions, pca_model)
        acc, errors = evaluate_graph(X_val_distinct, predictions, init_node)
        print('quantized graph is correct in {:.1f}% of test sentences'.format(acc * 100))
        print('the FSA was wrong in the following sentences:')
        print(errors)

        if len(quantized_nodes) < 300:
            print_graph(quantized_nodes, 'quantized_graph_reduced.png', init_node)

        retrieve_minimized_equivalent_graph(quantized_nodes, 'quantized', init_node, pca_model, path=path)


if __name__ == '__main__':
    alphabet_map, inv_alphabet_map = get_data_alphabet()
    print(alphabet_map)

    X_train, y_train, X_val, y_val, X_test, y_test = generate_sentences(alphabet_map)
    sess = tf.InteractiveSession()
    rnn = RegularRNN(sess, len(alphabet_map))
    sess.run(tf.global_variables_initializer())
    correct_X, correct_y = train(X_train, y_train, X_val, y_val, sess, rnn)
    extract_graphs()
