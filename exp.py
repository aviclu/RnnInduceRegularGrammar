from data_utils import *

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


if __name__ == '__main__':
    total_num_of_sents = 100
    alpha = get_data_alphabet()[0]
    grammatical_sents = read_conll_word_pos_file("Penn_Treebank/train.gold.conll")
    allowed_pos = {'NNS'}
    grammatical_sents = list_flatten(grammatical_sents)
    grammatical_sents = filter(lambda x: True if x[1] in allowed_pos else False, grammatical_sents)
    grammaticals = list(np.random.choice([word[0] for word in grammatical_sents], total_num_of_sents // 2))
    grammaticals = map(lambda x: x.lower(),grammaticals)
    grammaticals = map(map_to_seq,[w for w in grammaticals])
    print(grammaticals[2])

