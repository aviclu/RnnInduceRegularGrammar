import random
import re
from string import ascii_lowercase as al
from copy import copy
import rstr
import numpy as np
from config import Config

config = Config()

pos_category_map = \
    {
        'NN': 'N', 'NNS': 'N', 'NNP': 'N', 'NNPS': 'N', 'PRP$': 'N', 'PRP': 'N', 'WP': 'N', 'WP$': 'N',
        'MD': 'V', 'VB': 'V', 'VBC': 'V', 'VBD': 'V', 'VBF': 'V', 'VBG': 'V', 'VBN': 'V', 'VBP': 'V',
        'VBZ': 'V',
        'JJ': 'J', 'JJR': 'J', 'JJS': 'J', 'LS': 'J', 'RB': 'A', 'RBR': 'A', 'RBS': 'A', 'WRB': 'A',
        'DT': 'D', 'PDT': 'D', 'WDT': 'D',
        'SYM': 'S', 'POS': 'S', '-LRB-': 'S', '-RRB-': 'S', ',': 'S', '-': 'S', ':': 'S', ';': 'S', '.': 'S',
        '``': 'S',
        '"': 'S', '$': 'S', "''": 'S', '#': 'S',
        'CD': 'C', 'DAT': 'X', 'CC': 'B', 'EX': 'E', 'FW': 'F', 'IN': 'I', 'RP': 'R', 'TO': 'T',
        'UH': 'U'
    }
pos_category_to_num = {cat: i for i, cat in enumerate(sorted(set(pos_category_map.values())))}

def generate_sentences(alphabet_map):
    """Generate data and return it splitted to train, test and labels"""
    source = config.Data.grammatical_source.str
    size_train = config.Data.size_train.int
    size_val = config.Data.size_validation.int
    size_test = config.Data.size_test.int
    DATA_AMOUNT = size_train + size_val + size_test
    if source == 'regex':
        raw_x, raw_y = get_regex_sentences(DATA_AMOUNT, alphabet_map)
    elif source == 'ptb':
        raw_x, raw_y = get_penn_pos_data(DATA_AMOUNT, alphabet_map)
    elif source == 'phonology':
        raw_x, raw_y = get_phonology_data(DATA_AMOUNT, alphabet_map)
    elif source == 'phonology_plurals':
        raw_x, raw_y = get_phonology_plural_data(DATA_AMOUNT, alphabet_map)
    else:
        raise Exception('must provide source between: regex, ptb, phonology, phonology_plurals')

    zipped = list(zip(raw_x, raw_y))
    random.shuffle(zipped)
    raw_x, raw_y = zip(*zipped)

    X_train, y_train = raw_x[:size_train], raw_y[:size_train]
    X_val, y_val = raw_x[size_train:(size_train + size_val)], raw_y[size_train:(size_train + size_val)]
    X_test, y_test = raw_x[(size_train + size_val):], raw_y[(size_train + size_val):]

    X_train, y_train = zip(*(sorted(zip(X_train, y_train), key=lambda x: len(x[0]))))
    X_val, y_val = zip(*(sorted(zip(X_val, y_val), key=lambda x: len(x[0]))))
    X_test, y_test = zip(*(sorted(zip(X_test, y_test), key=lambda x: len(x[0]))))

    X_train, y_train = np.array([np.array(x) for x in X_train]), np.array([np.array(y) for y in y_train])
    X_val, y_val = np.array([np.array(x) for x in X_val]), np.array([np.array(y) for y in y_val])
    X_test, y_test = np.array([np.array(x) for x in X_test]), np.array([np.array(y) for y in y_test])
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_penn_pos_data(total_num_of_sents, alphabet_map):
    alphabet = config.PTB.alphabet.lst if config.PTB.filter_alphabet.boolean else list(set(alphabet_map.keys()))

    grammatical_sents = read_conll_pos_file("../Penn_Treebank/train.gold.conll")
    grammaticals = grammatical_sents[:total_num_of_sents//2] if config.PTB.use_orig_sent.boolean \
        else sample_concat_sentences(grammatical_sents, total_num_of_sents//2)
    # todo: Mor filter also for not orig sentences
    ungrammaticals = get_ungrammatical_sentences(alphabet, grammaticals, total_num_of_sents//2,
                                                 filter_out_grammatical_sentences, grammatical_sents)

    data = list(map(lambda sent: [alphabet_map[s] for s in list(sent)], grammaticals + ungrammaticals))
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels


def get_regex_sentences(num_sents, alphabet_map):
    alphabet = list(alphabet_map.keys())
    max_len, min_len = config.Regex.max_len.int, config.Regex.min_len.int
    regex = '^' + config.Regex.regex.str + '$'
    num_stars = regex.count('*')
    if num_stars == 0:
        raise Exception('must provide regex that contains at least one *')
    ranges = '{' + str(min_len // num_stars) + ',' + str(max_len // num_stars) + '}'
    regex = ranges.join(regex.split('*'))

    grammaticals = [list(rstr.xeger(regex)) for _ in range(num_sents // 2)]
    ungrammaticals = get_ungrammatical_sentences(alphabet, grammaticals, num_sents//2, filter_by_regex, regex)

    data = np.array([np.array([alphabet_map[word] for word in sent]) for sent in grammaticals + ungrammaticals])
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels


def get_ungrammatical_sentences(alphabet, grammaticals, num_of_sents, func, args):
    ungrammatical_by_trans = config.Data.ungrammatical_by_trans.boolean
    ungrammaticals = []
    lengths = [len(sent) for sent in grammaticals]
    left = num_of_sents
    while left > 0:
        if not ungrammatical_by_trans:
            curr_ungrammaticals = generate_random_strings(left, alphabet, lengths)
        else:
            sample = np.random.choice(grammaticals, left)
            curr_ungrammaticals = [random_trans(sentence, alphabet) for sentence in sample]
        ungrammaticals += list(filter(lambda sent: func(sent, args),
                                      curr_ungrammaticals))
        left = num_of_sents - len(ungrammaticals)
    return ungrammaticals


def generate_random_strings(num_of_sents, alphabet, lengths):
    random_lengths = np.random.choice(lengths, num_of_sents)
    return [rstr.rstr(alphabet, length) for length in random_lengths]


def filter_by_pos(sent):
    allowed_pos = config.PTB.alphabet.lst
    for pos in sent:
        if pos not in allowed_pos:
            return False
    return True

def list_flatten(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(item)
    return flat_list

def map_to_seq(word):
    alphabet_dict = {'s': 's', 'ss' : 's', 'z':'s', 'x':'s', 'ch':'s','sh':'s',
                   'a':'v', 'i':'v', 'u':'v', 'o':'v',
                   'e':'e'}
    new_word = ''
    i = 0
    while i < len(word):
        c = word[i]
        if i < len(word) - 1:
            if c + word[i+1] in alphabet_dict:
                new_word += alphabet_dict[c + word[i+1]]
                i += 1
            elif c in alphabet_dict:
                new_word += alphabet_dict[c]
            else:
                new_word += 'c'
        elif c in alphabet_dict:
            new_word += alphabet_dict[c]
        else:
            new_word += 'c'
        i += 1
    return new_word

def sample_concat_sentences(sents, num_of_sents):
    filtered_sents = list(filter(lambda sent: filter_by_pos(sent), sents))

    def sample(sentences):
        rand_idx = np.random.randint(0, len(sentences))
        return sentences[rand_idx]

    def concat(sentences):
        rand_idx1, rand_idx2 = np.random.randint(0, len(sentences), 2)
        conj = ['B']
        return sentences[rand_idx1] + conj + sentences[rand_idx2]

    output = []
    for i in range(num_of_sents):
        action = np.random.choice([sample, concat])
        output.append(action(filtered_sents))
    return output


def filter_by_regex(sent, regex):
    pattern = re.compile(regex)
    if pattern.match(sent):
        return False
    return True


def random_trans(sentence, alphabet):
    result = copy(sentence)
    if len(sentence) < 2:
        return sentence
    num_trans = np.random.randint(1, len(sentence))
    for i in range(num_trans):
        trans = np.random.randint(2)
        ind = np.random.randint(len(result))
        if trans == 0:  # deletion
            result.pop(ind)
        else:
            new_char = np.random.choice(alphabet)
            if trans == 1:  # addition
                result.insert(ind, new_char)
            else:  # replacement
                result[ind] = new_char
    return ''.join(result)


def filter_out_grammatical_sentences(ungrammatical_sent, grammatical_sents):
    if list(ungrammatical_sent) in grammatical_sents:
        return False
    return True


def get_pos_num(pos):
    return pos_category_to_num[pos_category_map[pos]]


def read_conll_pos_file(path):
    """
    Takes a path to a file and returns a list of tags
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                pos = tokens[3]
                curr.append(pos_category_map[pos])
    return list(filter(filter_by_verb, sents))

def read_conll_word_pos_file(path):
    """
        Takes a path to a file and returns a list of word/tag pairs
    """
    sents = []
    with open(path, "r") as f:
        curr = []
        for line in f:
            line = line.strip()
            if line == "":
                sents.append(curr)
                curr = []
            else:
                tokens = line.strip().split("\t")
                curr.append((tokens[1],tokens[3]))
    return sents


def filter_by_verb(sent):
    if 'V' not in sent:
        return False
    return True


def get_data_alphabet():
    source = config.Data.grammatical_source.str
    if source == 'regex':
        alphabet = config.Regex.alphabet.lst
        alphabet_map = {a: i for i, a in enumerate(alphabet)}
    elif source == 'ptb':
        alphabet_map = pos_category_to_num
        if config.PTB.filter_alphabet.boolean:  # returning the filtered alphabet
            alphabet_map = {pos: num for pos, num in pos_category_to_num.items()
                            if pos in config.PTB.alphabet.lst}
    elif source == 'phonology':
        alphabet_map = {'C': 0, 'V': 1}
    elif source == 'phonology_plurals':
        alphabet_map = {'s': 0, 'v': 1, 'e' : 2, 'c' : 3}

    return alphabet_map, {v: k for k, v in alphabet_map.items()}


def get_phonology_data(total_num_of_sents, alphabet_map):
    alphabet = list(alphabet_map.keys())

    vowels = 'aeiou'
    with open('alice_full_text.txt') as f:
        text = f.read().lower()
    words = text.split()
    words = filter(str.isalpha, words)
    tagged_words = [['V' if c in vowels else 'C' for c in word] for word in words]
    grammaticals = list(np.random.choice(tagged_words, total_num_of_sents // 2))
    ungrammaticals = get_ungrammatical_sentences(alphabet, grammaticals, total_num_of_sents // 2,
                                                 filter_out_grammatical_sentences, grammaticals)

    data = list(map(lambda sent: [alphabet_map[s] for s in list(sent)], grammaticals + ungrammaticals))
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels

def get_phonology_plural_data(total_num_of_sents, alphabet_map):
    alphabet = list(alphabet_map.keys())

    grammatical_sents = read_conll_word_pos_file("Penn_Treebank/train.gold.conll")
    allowed_pos = {'NNS'}
    grammatical_sents = list_flatten(grammatical_sents)
    grammatical_sents = filter(lambda x: True if x[1] in allowed_pos else False, grammatical_sents)
    grammaticals = list(np.random.choice([word[0] for word in grammatical_sents], total_num_of_sents // 2))
    grammaticals = list(map(lambda x: x.lower(),grammaticals))
    grammaticals = list(map(map_to_seq, grammaticals))

    ungrammaticals = get_ungrammatical_sentences(alphabet, grammaticals, total_num_of_sents // 2,
                                                 filter_out_grammatical_sentences, grammaticals)

    data = list(map(lambda sent: [alphabet_map[s] for s in list(sent)], grammaticals + ungrammaticals))
    labels = np.array([1] * len(grammaticals) + [0] * len(ungrammaticals))
    return data, labels



if __name__ == '__main__':
    get_phonology_plural_data(100,get_data_alphabet()[0])