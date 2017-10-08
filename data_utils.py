#-*- coding: utf-8 -*-
#-*- coding: euc-kr -*-
import numpy as np
import os


# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

# special error message
class MyIOError(Exception):
    def __init__(self, filename):
        # custom error message
        message = """
ERROR: Unable to locate file {}.

FIX: Have you tried running python build_data.py first?
This will build vocab file from your train, test and dev sets and
trimm your word vectors.
""".format(filename)
        super(MyIOError, self).__init__(message)


class CoNLLDataset(object):
    """
    Class that iterates over CoNLL Dataset

    __iter__ method yields a tuple (words, tags)
        words: list of raw words
        tags: list of raw tags
    If processing_word and processing_tag are not None,
    optional preprocessing is appplied

    Example:
        ```python
        data = CoNLLDataset(filename)
        for sentence, tags in data:
            pass
        ```
    """
    def __init__(self, filename, processing_word=None, processing_tag=None, processing_pos=None,
                 max_iter=None):
        """
        Args:
            filename: path to the file
            processing_words: (optional) function that takes a word as input
            processing_tags: (optional) function that takes a tag as input
            max_iter: (optional) max number of sentences to yield
        """
        self.filename = filename
        self.processing_word = processing_word
        self.processing_tag = processing_tag
        self.processing_pos = processing_pos
        self.max_iter = max_iter
        self.length = None


    def __iter__(self):
        niter = 0
        with open(self.filename, encoding='utf-8') as f:
            words, tags = [], []
            poses, inputs = [], []
            index = []
            for line in f:
                line = line.strip()
                if (len(line) == 0 or line.startswith("-DOCSTART-")):
                    if len(words) != 0:
                        niter += 1
                        if self.max_iter is not None and niter > self.max_iter:
                            break
                        yield words, tags, poses, inputs, index
                        words, tags, poses, inputs = [], [], [], []
                        index = []
                else:
                    ls = line.split('\t')
                    word, tag = ls[0],ls[-1]

                    pos = word.split('/')[-1]

                    ts = tag.split(' ')
                    tag = ts[0]

                    idx =ts[1]

                    p = pos

                    inputs += [word]

                    if self.processing_word is not None:
                        word = self.processing_word(word)
                    if self.processing_tag is not None:
                        tag = self.processing_tag(tag)
                    if self.processing_pos is not None:
                        pos = self.processing_pos(pos)


                    words += [word]

                    tags += [tag]
                    poses +=[pos]

                    index +=[idx]



    def __len__(self):
        """
        Iterates once over the corpus to set and store length
        """
        if self.length is None:
            self.length = 0
            for _ in self:
                self.length += 1

        return self.length


def get_vocabs(datasets):
    """
    Args:
        datasets: a list of dataset objects
    Return:
        a set of all the words in the dataset
    """
    print("Building vocab...")
    vocab_words = set()
    vocab_tags = set()
    vocab_pos = set()
    for dataset in datasets:
        for words, tags, pos, _, _ in dataset:
            vocab_words.update(words)
            vocab_tags.update(tags)
            vocab_pos.update(pos)
            vocab_pos.update(pos)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words, vocab_tags, vocab_pos


def get_char_vocab(dataset):
    """
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    vocab_char = set()
    for words, _, _, _, _ in dataset:
        for word in words:
            vocab_char.update(word)
    return vocab_char


def get_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            line.encode('utf-8')
            word = line.strip().split(' ')[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def get_pos_glove_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    with open(filename, encoding='utf-8') as f:
        for line in f:
            word = line.strip().split(' ')[0]
            pos = word.split("/")[-1]
            vocab.add(pos)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def get_dic_vocab(filename):
    """
    Args:
        filename: path to the glove vectors
    """
    print("Building vocab...")
    vocab = set()
    with open(filename) as f:
        for line in f:
            line.encode('utf-8')
            word = line.strip().split("\t")[0]
            vocab.add(word)
    print("- done. {} tokens".format(len(vocab)))
    return vocab

def write_vocab(vocab, filename):
    """
    Writes a vocab to a file

    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w", encoding='utf-8') as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(len(vocab)))


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    try:
        d = dict()
        w_i_d = dict()
        with open(filename, encoding='utf-8') as f:
            for idx, word in enumerate(f):
                word = word.strip()
                d[word] = idx
                w_i_d[idx] = word

    except IOError:
        raise MyIOError(filename)
    return d, w_i_d

def export_trimmed_uni_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            if 48 < len(line) < 51:
                continue
            word = line[0]
            l = []
            for x in line[1:]:
                l.append(x)

            embedding = [float(x) for x in l]
            if word in vocab:
                word_idx = vocab.get(word)
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            l = []
            dic = False
            for i, x in enumerate(line[1:]):
                l.append(x)



            embedding = [float(x) for x in l]
            #print(np.array(l).shape)
            #print(np.array(line[1:]).shape)
            if word in vocab:
                word_idx = vocab.get(word)
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def export_trimmed_pos_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            l = []
            for x in line[1:]:
                if float(x) == 0:
                    x = -1
                l.append(x)

            embedding = [float(x) for x in l]
            #print(np.array(l).shape)
            #print(np.array(line[1:]).shape)
            if word in vocab:
                word_idx = vocab.get(word)

                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def export_trimmed_dic_vectors(vocab, glove_filename, trimmed_filename):
    """
    Saves glove vectors in numpy array

    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), 4])
    with open(glove_filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            word = line[0]
            l = []
            for x in line[1:]:
                l.append(x)

            embedding = [float(x) for x in l]

            if word in vocab:
                word_idx = vocab.get(word)
                embeddings[word_idx] = np.asarray(embedding)

    np.savez_compressed(trimmed_filename, embeddings=embeddings)



def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    try:
        with np.load(filename) as data:
            return data["embeddings"]

    except IOError:
        raise MyIOError(filename)


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, chars=False):
    """
    Args:
        vocab: dict[word] = idx
    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)
    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None and chars == True:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        words = word
       #print(words)
        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words.get(word)
            else:
                word = vocab_words.get(UNK)

        # 3. return tuple char ids, word id
        if vocab_chars is not None and chars == True:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok]*max(max_length - len(seq), 0)
        index = 0
        for i in seq:
            index +=1
        sequence_padded += [seq_]

        sequence_length += [min(len(seq), max_length)]


    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    if nlevels == 1:
        max_length = max(map(lambda x : len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                            pad_tok, max_length)

    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq)) for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x : len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok]*max_length_word,
                                            max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)

    #print("pad",np.array(sequence_padded).shape)
    #print("len", np.array(sequence_length).shape)
    return sequence_padded, sequence_length


def minibatches(data, minibatch_size):
    """
    Args:
        data: generator of (sentence, tags) tuples
        minibatch_size: (int)
    Returns:
        list of tuples
    """
    x_batch, y_batch, poses = [], [], []
    words = []
    index = []
    for x, y, pos, w, idx in data:
        if len(x_batch) == minibatch_size:

            yield x_batch, y_batch, poses, words, index
            x_batch, y_batch, poses = [], [], []
            words = []
            index = []

        if type(x[0]) == tuple:
            x = zip(*x)

        words += [w]
        x_batch += [x]
        y_batch += [y]
        poses += [pos]
        index += [idx]


    if len(x_batch) != 0:
        yield x_batch, y_batch, poses, words, index


def get_chunk_type(tok, idx_to_tag):
    """
    Args:
        tok: id of token, ex 4
        idx_to_tag: dictionary {4: "B-PER", ...}
    Returns:
        tuple: "B", "PER"
    """
    tag_name = idx_to_tag[tok]
    tag_class = tag_name.split('_')[0]
    tag_type = tag_name.split('_')[-1]
    return tag_class, tag_type


def get_chunks(seq, tags):
    """
    Args:
        seq: [4, 4, 0, 0, ...] sequence of labels
        tags: dict["O"] = 4
    Returns:
        list of (chunk_type, chunk_start, chunk_end)

    Example:
        seq = [4, 5, 0, 3]
        tags = {"B-PER": 4, "I-PER": 5, "B-LOC": 3}
        result = [("PER", 0, 2), ("LOC", 3, 4)]
    """
    default = tags.get(NONE)
    idx_to_tag = {idx: tag for tag, idx in tags.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None

        # End of a chunk + start of a chunk!
        elif tok != default:
            tok_chunk_class, tok_chunk_type = get_chunk_type(tok, idx_to_tag)
            if chunk_type is None:
                chunk_type, chunk_start = tok_chunk_type, i
            elif tok_chunk_type != chunk_type or tok_chunk_class == "B":
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok_chunk_type, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)

    return chunks

from konlpy.tag import Komoran
kkma = Komoran()

def convert(file):
    fw = open("data_format/test_convert.txt", 'w')
    list = []

    with open(file, 'r') as f:
        for line in f:
            if len(line) == 1:
                continue
            line = line.split("\t")
            first = "; "+line[1]
            second = "$"+line[1]
            fw.write(first)
            fw.write(second)

            inputs = line[1].split(" ")
            for i,input in enumerate(inputs):
                if input is None:
                    continue
                poses = kkma.pos(input)
                for pos in poses:
                    fw.write(str(i+1) + "\t" + pos[0] + "\t" + pos[1] + "\tO" + "\n")
            fw.write("\n")

class Setence(object):
    def __init__(self):
        self.words = []

    def addWord(self, w):
        self.words.append(w)

    def getWords(self):
        return self.words

    def setLine(self, line):
        self.line = line

    def getLine(self):
        return self.line

class NE(object):

    def __init__(self):
        self.list = []

    def setOne(self, one):
        self.list.append(one)

    def getOne(self):
        return self.list

class One(object):
    def set(self, ne, word, pos):
        self.ne = ne
        self.word = word
        self.pos = pos

    def getWord(self):
        return self.word

    def getPos(self):
        return self.pos

def tagging(file):
    with open(file, 'r') as f:
        sentence_set = []
        for line in f:

            input = line.split(" ")

            if input[0] == ';':
                s = Setence()
                s.setLine(line)
                for word in input[1:]:
                    s.addWord(word)
            else:
                continue
            sentence_set.append(s)


    predict_set = []
    count = 0
    with open("predict/prediction.txt", 'r') as f:
        start = True
        predict = None
        get = False
        for line in f:
            if start:
                count += 1
                predict = NE()
                start = False
            po = 1
            index = 0

            if len(line) == 1:
                predict_set.append(predict)
                start = True
                continue

            inputs = line.split("\t")
            if len(inputs) == 1:
                print(inputs)
            ne = inputs[1]

            if ne != 'O':
                one = One()
                if len(ne) == 1:
                    one.set(ne, inputs[0], int(inputs[2]))
                else:
                    ne = ne.split("_")[1]
                    one.set(ne,inputs[0], int(inputs[2]))
                predict.setOne(one)
                get = True

            else:
                one = One()
                one.set(ne, inputs[0], int(inputs[2]))
                predict.setOne(one)
                get = False


        predict_set.append(predict)



    print(count)


    print("{} : {}".format(len(sentence_set), len(predict_set)))
    fw = open("submit/result.txt", 'w')


    for sentences, Ne in zip(sentence_set, predict_set):



        fw.write(sentences.getLine())
        str = sentences.getLine()[2:]

        words = sentences.getWords()

        line = "$"
        if len(Ne.list) == 0:
            fw.write(line+sentences.getLine()[2:])
            fw.write("\n")
        else:
            start = True
            word = ""
            ner = ""
            idx = 0
            index = 0
            position = []
            ne_list = []
            for i, ne in enumerate(Ne.list):
                if ne.ne != "I" and ne.ne != 'O':

                    if start:
                        start = False

                    else:
                        ne_list.append((ner, tag))
                        for w_idx in range(idx, position[0]-1):
                            word += words[w_idx]+" "
                        replace = ""
                        for w_idx in position:
                            replace += words[w_idx-1]+" "

                        replace = replace.replace(ner, "<"+ner+":"+tag+">")
                        word += replace

                        idx = position[-1]
                        position = []

                    position.append(ne.pos)
                    ner = ne.word
                    tag = ne.ne


                elif ne.ne != 'O': #  I 태깅
                    if len(position) != 0:
                        if ne.pos != position[-1]:
                            position.append(ne.pos)
                            ner += " "+ne.word
                        else:
                            if ne.word == 'ㄴ':
                                if "지나" in ner:
                                    ner = ner.replace('지나', "지난")
                                elif "하" in ner:
                                    ner = ner.replace('하', "한")
                            elif ne.word == 'ㄹ':
                                if "오" in ner:
                                    ner = ner.replace("오", "올")
                            else:
                                ner += ne.word

            if len(position) != 0:
                ne_list.append((ner, tag))
                for w_idx in range(idx, position[0] - 1):
                    word += words[w_idx] + " "
                replace = ""
                for w_idx in position:
                    replace += words[w_idx - 1] + " "

                # print("이전 : {}  ner --> {}".format(replace, ner))
                replace = replace.replace(ner, "<" + ner +":"+tag+">")
                # print("이후 : {}".format(replace))
                word += replace

                idx = position[-1]
                for w in words[idx:]:
                    word += w+" "

            else:
                word = str + " "

            # $ + 태깅 결과
            line += word[:-1]
            #print(line)
            fw.write(line)
            for _w, _ne in ne_list:
                fw.write(_w + "\t" + _ne +"\n")
            fw.write("\n")