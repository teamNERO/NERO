#-*- coding: utf-8 -*-
#-*- coding: euc-kr -*-
from config import Config
from data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word, get_pos_glove_vocab,\
    export_trimmed_pos_vectors, export_trimmed_uni_vectors, get_dic_vocab, \
    export_trimmed_dic_vectors


def build_data(config):
    """
    Procedure to build data

    Args:
        config: defines attributes needed in the function
    Returns:
        creates vocab files from the datasets
        creates a npz embedding file from trimmed glove vectors
    """
    processing_word = get_processing_word(lowercase=True)

    # Generators
    dev   = CoNLLDataset(config.dev_filename, processing_word)
    #test  = CoNLLDataset(config.test_filename, processing_word)
    train = CoNLLDataset(config.train_filename, processing_word)

    # Build Word and Tag vocab
    vocab_words, vocab_tags, vocab_pos = get_vocabs([train, dev])
    vocab_glove = get_glove_vocab(config.glove_filename)
    vocab_glove_uni = get_glove_vocab(config.glove_uni_filename)

    vocab_feature = get_pos_glove_vocab(config.glove_filename)

   # vocab = vocab_words & vocab_glove
    vocab = vocab_glove | vocab_words
    vocab.add(UNK)
    vocab.add(NUM)

    vocab_pos = vocab_feature
    vocab_pos.add(UNK)
    vocab_pos.add(NUM)

    # Save vocab
    write_vocab(vocab, config.words_filename)
    write_vocab(vocab_glove_uni, config.uni_words_filename)
    write_vocab(vocab_tags, config.tags_filename)
    write_vocab(vocab_pos, config.pos_filename)

    # Trim GloVe Vectors
    vocab = load_vocab(config.words_filename)
    export_trimmed_glove_vectors(vocab, config.glove_filename, 
                                config.trimmed_filename, config.t_dim)

    vocab = load_vocab(config.uni_words_filename)

    export_trimmed_uni_vectors(vocab, config.NEdic_filename,
                                 config.trimmed_dic, config.dic_dim)

    export_trimmed_uni_vectors(vocab, config.glove_uni_filename,
                                 config.uni_trimmed_filename, config.dim)

    vocab_feature = load_vocab(config.pos_filename)
    export_trimmed_pos_vectors(vocab_feature, config.glove_feature,
                                 config.feature_trimmed_filename, config.pos_dim)

    # Build and save char vocab
    train = CoNLLDataset(config.train_filename)
    vocab_chars = get_char_vocab(train)
    write_vocab(vocab_chars, config.chars_filename)


if __name__ == "__main__":
    config = Config()
    build_data(config)
