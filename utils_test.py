from tflearn.data_utils import to_categorical
import configparser
from bisect import bisect_left
import numpy as np

# For debugging the code
debug = True


def read_config(config_path):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    # Debug
    if debug:
        log_info = open("Data/log.txt", 'a')
        log_info.write("# Here are the configuration values  : \n")
        log_info.write("words_max_len : " + str(parser['config']['words_max_len']) + "\n")
        log_info.write("chars_max_len : " + str(parser['config']['chars_max_len']) + "\n")
        log_info.write("subwords_max_len : " + str(parser['config']['subwords_max_len']) + "\n")
        log_info.write("emb_dim : " + str(parser['config']['emb_dim']) + "\n")
        log_info.write("mode : " + str(parser['config']['mode']) + "\n")
        log_info.write("emb_mode : " + str(parser['config']['emb_mode']) + "\n")
        log_info.write("batch_size : " + str(parser['config']['batch_size']) + "\n")
        log_info.write("minimum_frequency : " + str(parser['config']['min_freq']) + "\n")
        log_info.write("Debug : " + str(parser['config']['debug']) + "\n")
        log_info.write("\n")
        log_info.close()

    return parser


def get_words_for_url(x, reverse_dict, urls):
    processed_x = []
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


def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):
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
            if (len(ngrams) > max_len_subwords) or \
                    (high_freq_words is not None and len(word) > 1 and not is_in(high_freq_words, word)):
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

    all_ngrams = list(all_ngrams)
    ngrams_dict = dict()
    for i in range(len(all_ngrams)):
        ngrams_dict[all_ngrams[i]] = i + 1  # ngram id=0 is for padding ngram
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict)))
    all_words = list(all_words)
    words_dict = dict()
    for i in range(len(all_words)):
        words_dict[all_words[i]] = i + 1  # word id=0 for padding word
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNKNOWN> word: {}".format(words_dict["<UNKNOWN>"]))

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


def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict=None):
    char_ngram_len = 1
    print("Index of <UNKNOWN> word: {}".format(word_dict["<UNKNOWN>"]))
    ngramed_id_x = []
    worded_id_x = []
    counter = 0
    if word_dict:
        word_vocab = sorted(list(word_dict.keys()))
    for url in word_x:
        if counter % 100000 == 0:
            print("Processing url #{}".format(counter))
        counter += 1
        url_in_ngrams = []
        url_in_words = []
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word)
            if len(ngrams) > max_len_subwords:
                word = "<UNKNOWN>"
            ngrams_id = []
            for ngram in ngrams:
                if ngram in ngram_dict:
                    ngrams_id.append(ngram_dict[ngram])
                else:
                    ngrams_id.append(0)
            url_in_ngrams.append(ngrams_id)
            if is_in(word_vocab, word):
                word_id = word_dict[word]
            else:
                word_id = word_dict["<UNKNOWN>"]
            url_in_words.append(word_id)
        ngramed_id_x.append(url_in_ngrams)
        worded_id_x.append(url_in_words)

    return ngramed_id_x, worded_id_x


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


def get_ngramed_id_x(x_idxs, ngramed_id_x):
    output_ngramed_id_x = []
    for idx in x_idxs:
        output_ngramed_id_x.append(ngramed_id_x[idx])
    return output_ngramed_id_x


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
                        pad_urls[d0, d1, d2] = word[d2]
                        pad_idx[d0, d1, d2] = pad_vec
    return pad_urls, pad_idx


def pad_seq_in_word(urls, max_d1=0, embedding_size=128):
    if max_d1 == 0:
        url_lens = [len(url) for url in urls]
        max_d1 = max(url_lens)
    pad_urls = np.zeros((len(urls), max_d1))
    # pad_idx = np.zeros((len(urls), max_d1, embedding_size))
    # pad_vec = [1 for i in range(embedding_size)]
    for d0 in range(len(urls)):
        url = urls[d0]
        for d1 in range(len(url)):
            if d1 < max_d1:
                pad_urls[d0, d1] = url[d1]
                # pad_idx[d0,d1] = pad_vec
    return pad_urls


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]
