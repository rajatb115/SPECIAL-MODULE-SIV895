
from tqdm import tqdm
import pickle
import tensorflow as tf

from utils_test import *
import sys
from tensorflow.contrib import learn

config_path = sys.argv[1]

# Read the configuration file
config = read_config(config_path)

# For debugging the code
debug = config['config']['debug']

# Since we are not working with the real dataset given in the paper.
# We have to clean another dataset and load it for further use.

# Reading the URL dataset file
file2 = open(r'Data/urls.pk', 'rb')
urls1 = pickle.load(file2)
file2.close()

# using partial data
urls = urls1[:10000]

# Reading the labels associated to these datasets
file2 = open(r'Data/labels.pk', 'rb')
labels1 = pickle.load(file2)
file2.close()

# using partial labels
labels = labels1[:10000]


## Create the vocabularyprocessor object, setting the max lengh of the documents.
vocab_processor = learn.preprocessing.VocabularyProcessor(int(config['config']['words_max_len']), min_frequency=int(config['config']['min_freq']))

## Transform the documents using the vocabulary.
x = np.array(list(vocab_processor.fit_transform(urls)))

## Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

## Sort the vocabulary dictionary on the basis of values(id).
## Treat the id's as index into list and create a list of words in the ascending order of id's
## word with id i goes at index i of the list.
vocabulary_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))

if debug:
    print("Transforming the document using the vocabulary\n", x)
    print("Vocabulary Dictionary :\n", vocabulary_dict)

word_x = get_words_for_url(x, vocabulary_dict, int(config['config']['mode']), urls)

if debug:
    print("Word_x :\n", word_x)

# make it as a function
ngram_dict = pickle.load(open("Output/subwords_dict.p", "rb"))
word_dict = pickle.load(open("Output/words_dict.p", "rb"))

ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, int(config['config']['subwords_max_len']), ngram_dict, word_dict)
chars_dict = pickle.load(open("Output/chars_dict.p", "rb"))

chared_id_x = char_id_x(urls, chars_dict, int(config['config']['chars_max_len']))
print("Number of testing urls: {}".format(len(labels)))


def test_step(x, emb_mode):
    p = 1.0
    feed_dict = {
        input_x_char_seq: x[0],
        input_x_word: x[1],
        input_x_char: x[2],
        input_x_char_pad_idx: x[3],
        dropout_keep_prob: p
    }
    preds, s = sess.run([predictions, scores], feed_dict)
    return preds, s


checkpoint_file = tf.train.latest_checkpoint("Output/checkpoints")
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
        input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
        input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
        input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]


        batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), int(config['config']['batch_size']), 1,shuffle=False)
        all_predictions = []
        all_scores = []

        nb_batches = int(len(labels) / int(config['config']['batch_size']))
        if len(labels) % int(config['config']['batch_size']) != 0:
            nb_batches += 1
        print("Number of batches in total: {}".format(nb_batches))
        it = tqdm(range(nb_batches), desc="test_size {}".format(len(labels)), ncols=0)
        for idx in it:
            # for batch in batches:
            batch = next(batches)

            x_char, x_word, x_char_seq = zip(*batch)

            x_batch = []

            x_char_seq = pad_seq_in_word(x_char_seq, int(config['config']['chars_max_len']))
            x_batch.append(x_char_seq)

            x_word = pad_seq_in_word(x_word, int(config['config']['words_max_len']))
            x_batch.append(x_word)

            x_char, x_char_pad_idx = pad_seq(x_char, int(config['config']['words_max_len']), int(config['config']['subwords_max_len']), int(config['config']['emb_dim']))
            x_batch.extend([x_char, x_char_pad_idx])

            batch_predictions, batch_scores = test_step(x_batch, int(config['config']['emb_mode']))
            all_predictions = np.concatenate([all_predictions, batch_predictions])
            all_scores.extend(batch_scores)

            it.set_postfix()

    correct_preds = float(sum(all_predictions == labels))
    print("Accuracy: {}".format(correct_preds / float(len(labels))))
