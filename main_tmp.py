import pickle
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from bisect import bisect_left
from tflearn.data_utils import to_categorical
from CNN import *
import os
import datetime
from tqdm import tqdm

file2 = open(r'Data/urls.pk', 'rb')
urls = pickle.load(file2)
file2.close()

file2 = open(r'Data/labels.pk', 'rb')
labels = pickle.load(file2)
file2.close()


def get_word_vocab(urls, max_length_words, min_word_freq):
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length_words, min_frequency=min_word_freq)
    x = np.array(list(vocab_processor.fit_transform(urls)))
    vocab_dict = vocab_processor.vocabulary_._mapping
    reverse_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    return x, reverse_dict


# make a dictionary of the words and the number of words in a url
x, word_reverse_dict = get_word_vocab(urls, 200, 0)


def get_words(x, reverse_dict, delimit_mode, urls=None):
    processed_x = []
    if delimit_mode == 0:
        for url in x:
            words = []
            for word_id in url:
                if word_id != 0:
                    words.append(reverse_dict[word_id])
                else:
                    break
            processed_x.append(words)
    elif delimit_mode == 1:
        for i in range(x.shape[0]):
            word_url = x[i]
            raw_url = urls[i]
            words = []
            for w in range(len(word_url)):
                word_id = word_url[w]
                if word_id == 0:
                    words.extend(list(raw_url))
                    break
                else:
                    word = reverse_dict[word_id]
                    idx = raw_url.index(word)
                    special_chars = list(raw_url[0:idx])
                    words.extend(special_chars)
                    words.append(word)
                    raw_url = raw_url[idx + len(word):]
                    if w == len(word_url) - 1:
                        words.extend(list(raw_url))
            processed_x.append(words)
    return processed_x


# words with all the special characters in an url
word_x = get_words(x, word_reverse_dict, 1, urls)


# given a word and the length it will return the character list
def get_char_ngrams(ngram_len, word):
    word = "<" + word + ">"
    chars = list(word)
    begin_idx = 0
    ngrams = []
    while (begin_idx + ngram_len) <= len(chars):
        end_idx = begin_idx + ngram_len
        ngrams.append("".join(chars[begin_idx:end_idx]))
        begin_idx += 1
    return ngrams


def bisect_search(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError


def is_in(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return True
    else:
        return False


def ngram_id_x(word_x, max_len_subwords):
    char_ngram_len = 1
    all_ngrams = set()
    ngramed_x = []
    all_words = set()
    worded_x = []
    counter = 0
    for url in word_x:
        if counter % 100000 == 0:
            print("Processing #url {}".format(counter))
        counter += 1
        url_in_ngrams = []
        url_in_words = []
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if len(ngrams) > max_len_subwords:
                all_ngrams.update(ngrams[:max_len_subwords])
                url_in_ngrams.append(ngrams[:max_len_subwords])
                all_words.add("<UNKNOWN>")
                url_in_words.append("<UNKNOWN>")
            else:
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams)
                all_words.add(word)
                url_in_words.append(word)

        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words)

    # making the list and the dictionary
    all_ngrams = list(all_ngrams)
    ngrams_dict = dict()
    for i in range(len(all_ngrams)):
        ngrams_dict[all_ngrams[i]] = i + 1  # ngram id=0 is for padding ngrams
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict)))

    # making the list and the dictionary
    all_words = list(all_words)
    words_dict = dict()
    for i in range(len(all_words)):
        words_dict[all_words[i]] = i + 1  # word id=0 for padding word
    print("Size of word vocabulary: {}".format(len(words_dict)))

    ngramed_id_x = []
    for ngramed_url in ngramed_x:
        url_in_ngrams = []
        for ngramed_word in ngramed_url:
            ngram_ids = [ngrams_dict[x] for x in ngramed_word]
            url_in_ngrams.append(ngram_ids)
        ngramed_id_x.append(url_in_ngrams)

    worded_id_x = []
    for worded_url in worded_x:
        word_ids = [words_dict[x] for x in worded_url]
        worded_id_x.append(word_ids)

    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict


# ngramed_id_x = [[url_ngram_idx]]
# ngrams_dict = {ngram:idx}
# worded_id_x = [[url_word_idx]]
# words_dict = {word:idx}
ngramed_id_x, ngrams_dict, worded_id_x, words_dict = ngram_id_x(word_x, 20)


# cap the maximum number of words to 200
def char_id_x(urls, char_dict, max_len_chars):
    chared_id_x = []
    for url in urls:
        url = list(url)
        url_in_char_id = []
        l = min(len(url), max_len_chars)
        for i in range(l):
            c = url[i]
            try:
                c_id = char_dict[c]
            except KeyError:
                c_id = 0
            url_in_char_id.append(c_id)
        chared_id_x.append(url_in_char_id)
    return chared_id_x


chars_dict = ngrams_dict
# chared_id_x =
chared_id_x = char_id_x(urls, chars_dict, 200)

pos_x = []
neg_x = []

for i in range(len(labels)):
    label = labels[i]
    if label == 1:
        pos_x.append(i)
    else:
        neg_x.append(i)

print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
pos_x = np.array(pos_x)
neg_x = np.array(neg_x)


def prep_train_test(pos_x, neg_x, dev_pct):
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(pos_x)))
    pos_x_shuffled = pos_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(pos_x)))
    pos_train = pos_x_shuffled[:dev_idx]
    pos_test = pos_x_shuffled[dev_idx:]

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(neg_x)))
    neg_x_shuffled = neg_x[shuffle_indices]
    dev_idx = -1 * int(dev_pct * float(len(neg_x)))
    neg_train = neg_x_shuffled[:dev_idx]
    neg_test = neg_x_shuffled[dev_idx:]

    x_train = np.array(list(pos_train) + list(neg_train))
    y_train = len(pos_train) * [1] + len(neg_train) * [0]
    x_test = np.array(list(pos_test) + list(neg_test))
    y_test = len(pos_test) * [1] + len(neg_test) * [0]

    y_train = to_categorical(y_train, nb_classes=2)
    y_test = to_categorical(y_test, nb_classes=2)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices]

    print("Train Mal/Ben split: {}/{}".format(len(pos_train), len(neg_train)))
    print("Test Mal/Ben split: {}/{}".format(len(pos_test), len(neg_test)))
    print("Train/Test split: {}/{}".format(len(y_train), len(y_test)))
    print("Train/Test split: {}/{}".format(len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test


default_dev_pct = 0.001
x_train, y_train, x_test, y_test = prep_train_test(pos_x, neg_x, default_dev_pct)


def get_ngramed_id_x(x_idxs, ngramed_id_x):
    output_ngramed_id_x = []
    for idx in x_idxs:
        output_ngramed_id_x.append(ngramed_id_x[idx])
    return output_ngramed_id_x


x_train_char = get_ngramed_id_x(x_train, ngramed_id_x)
x_test_char = get_ngramed_id_x(x_test, ngramed_id_x)

x_train_word = get_ngramed_id_x(x_train, worded_id_x)
x_test_word = get_ngramed_id_x(x_test, worded_id_x)

x_train_char_seq = get_ngramed_id_x(x_train, chared_id_x)
x_test_char_seq = get_ngramed_id_x(x_test, chared_id_x)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num+1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


def make_batches(x_train_char_seq, x_train_word, x_train_char, y_train, batch_size, nb_epochs, shuffle=False):

    batch_data = list(zip(x_train_char, x_train_word, x_train_char_seq, y_train))
    batches = batch_iter(batch_data, batch_size, nb_epochs, shuffle)

    if nb_epochs > 1:
        nb_batches_per_epoch = int(len(batch_data)/batch_size)
        if len(batch_data)%batch_size != 0:
            nb_batches_per_epoch += 1
        nb_batches = int(nb_batches_per_epoch * nb_epochs)
        return batches, nb_batches_per_epoch, nb_batches
    else:
        return batches

def pad_seq_in_word(urls, max_d1=0, embedding_size=128):
    if max_d1 == 0:
        url_lens = [len(url) for url in urls]
        max_d1 = max(url_lens)
    pad_urls = np.zeros((len(urls), max_d1))
    #pad_idx = np.zeros((len(urls), max_d1, embedding_size))
    #pad_vec = [1 for i in range(embedding_size)]
    for d0 in range(len(urls)):
        url = urls[d0]
        for d1 in range(len(url)):
            if d1 < max_d1:
                pad_urls[d0,d1] = url[d1]
                #pad_idx[d0,d1] = pad_vec
    return pad_urls

def pad_seq(urls, max_d1=0, max_d2=0, embedding_size=128):
    if max_d1 == 0 and max_d2 == 0:
        for url in urls:
            if len(url) > max_d1:
                max_d1 = len(url)
            for word in url:
                if len(word) > max_d2:
                    max_d2 = len(word)
    pad_idx = np.zeros((len(urls), max_d1, max_d2, embedding_size))
    pad_urls = np.zeros((len(urls), max_d1, max_d2))
    pad_vec = [1 for i in range(embedding_size)]
    for d0 in range(len(urls)):
        url = urls[d0]
        for d1 in range(len(url)):
            if d1 < max_d1:
                word = url[d1]
                for d2 in range(len(word)):
                    if d2 < max_d2:
                        pad_urls[d0,d1,d2] = word[d2]
                        pad_idx[d0,d1,d2] = pad_vec
    return pad_urls, pad_idx

def prep_batches(batch):
    x_char, x_word, x_char_seq, y_batch = zip(*batch)
    x_batch = []

    x_char_seq = pad_seq_in_word(x_char_seq, 200)
    x_batch.append(x_char_seq)

    x_word = pad_seq_in_word(x_word, 200)
    x_batch.append(x_word)

    x_char, x_char_pad_idx = pad_seq(x_char, 200, 20, 32)
    x_batch.extend([x_char, x_char_pad_idx])
    return x_batch, y_batch

def train_dev_step(x, y, is_train=True):
    if is_train:
        p = 0.5
    else:
        p = 1.0

    feed_dict = {
        cnn.input_x_char_seq: x[0],
        cnn.input_x_word: x[1],
        cnn.input_x_char: x[2],
        cnn.input_x_char_pad_idx: x[3],
        cnn.input_y: y,
        cnn.dropout_keep_prob: p}
    if is_train:
        _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
    else:
        step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
    return step, loss, acc

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        cnn = CNN(
                char_ngram_vocab_size = len(ngrams_dict)+1,
                word_ngram_vocab_size = len(words_dict)+1,
                char_vocab_size = len(chars_dict)+1,
                embedding_size=32,
                word_seq_len=200,
                char_seq_len=200,
                l2_reg_lambda=0.0)

    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.001)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Save dictionary files
    ngrams_dict_dir = "Output/subwords_dict.p"
    pickle.dump(ngrams_dict, open(ngrams_dict_dir, "wb"))
    words_dict_dir = "Output/words_dict.p"
    pickle.dump(words_dict, open(words_dict_dir, "wb"))
    chars_dict_dir = "Output/chars_dict.p"
    pickle.dump(chars_dict, open(chars_dict_dir, "wb"))

    # Save training and validation logs
    train_log_dir = "Output/train_logs.csv"
    with open(train_log_dir, "w") as f:
        f.write("step,time,loss,acc\n")
    val_log_dir ="Output/val_logs.csv"
    with open(val_log_dir, "w") as f:
        f.write("step,time,loss,acc\n")

    # Save model checkpoints
    checkpoint_dir = "Output/checkpoints/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = checkpoint_dir + "model"
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    sess.run(tf.global_variables_initializer())

    train_batches, nb_batches_per_epoch, nb_batches = make_batches(x_train_char_seq, x_train_word, x_train_char,y_train, 128, 5, True)

    min_dev_loss = float('Inf')
    dev_loss = float('Inf')
    dev_acc = 0.0
    print("Number of baches in total: {}".format(nb_batches))
    print("Number of batches per epoch: {}".format(nb_batches_per_epoch))

    it = tqdm(range(nb_batches), desc="emb_mode {} delimit_mode {} train_size {}".format(5,1, x_train.shape[0]), ncols=0)

    for idx in it:
        batch = next(train_batches)
        x_batch, y_batch = prep_batches(batch)
        step, loss, acc = train_dev_step(x_batch, y_batch, is_train=True)
        if step % 50 == 0:
            with open(train_log_dir, "a") as f:
                f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), loss, acc))
            it.set_postfix(
                trn_loss='{:.3e}'.format(loss),
                trn_acc='{:.3e}'.format(acc),
                dev_loss='{:.3e}'.format(dev_loss),
                dev_acc='{:.3e}'.format(dev_acc),
                min_dev_loss='{:.3e}'.format(min_dev_loss))
        if step % 50 == 0 or idx == (nb_batches - 1):
            total_loss = 0
            nb_corrects = 0
            nb_instances = 0
            test_batches = make_batches(x_test_char_seq, x_test_word, x_test_char, y_test, 128, 1,False)
            for test_batch in test_batches:
                x_test_batch, y_test_batch = prep_batches(test_batch)
                step, batch_dev_loss, batch_dev_acc = train_dev_step(x_test_batch, y_test_batch, is_train=False)
                nb_instances += x_test_batch[0].shape[0]
                total_loss += batch_dev_loss * x_test_batch[0].shape[0]
                nb_corrects += batch_dev_acc * x_test_batch[0].shape[0]

            dev_loss = total_loss / nb_instances
            dev_acc = nb_corrects / nb_instances
            with open(val_log_dir, "a") as f:
                f.write("{:d},{:s},{:e},{:e}\n".format(step, datetime.datetime.now().isoformat(), dev_loss, dev_acc))
            if step % 500 == 0 or idx == (nb_batches - 1):
                if dev_loss < min_dev_loss:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)
                    min_dev_loss = dev_loss
