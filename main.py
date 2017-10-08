from data_utils import get_trimmed_glove_vectors, load_vocab, \
    get_processing_word, CoNLLDataset, convert, tagging
from model import NERModel
from config import Config
from text2O import t2o
import sys

def main(config):
    # load vocabs
    vocab_words, idx2words = load_vocab(config.words_filename)
    vocab_tags, _  = load_vocab(config.tags_filename)
    vocab_chars, _ = load_vocab(config.chars_filename)
    vocab_pos, _ = load_vocab(config.pos_filename)


    # get processing functions
    processing_word = get_processing_word(vocab_words, vocab_chars,
                    lowercase=True, chars=config.chars)

    processing_tag  = get_processing_word(vocab_tags, 
                    lowercase=False)

    processing_pos = get_processing_word(vocab_pos,
                                         lowercase=False)




    # get pre trained embeddings
    embeddings = get_trimmed_glove_vectors(config.trimmed_filename)
    embeddings_uni = get_trimmed_glove_vectors(config.uni_trimmed_filename)
    pos_embeddings = get_trimmed_glove_vectors(config.feature_trimmed_filename)
    NE_dic = get_trimmed_glove_vectors(config.trimmed_dic)


    # create dataset
    dev   = CoNLLDataset(config.dev_filename, processing_word,
                        processing_tag, processing_pos, config.max_iter)

    train = CoNLLDataset(config.train_filename, processing_word,
                        processing_tag, processing_pos, config.max_iter)
    
    # build model
    model = NERModel(config, embeddings, embeddings_uni,
                     pos_embeddings, ntags=len(vocab_tags), nchars=len(vocab_chars), vocab_words=idx2words,
                    NE_dic=NE_dic)
    model.build()

    # train, evaluate and interact
    if state == "train":
        model.train(train, dev, vocab_tags)

    elif state == "evaluate":
        model.evaluate(dev, vocab_tags)

    else: #state == predict
        convert(file)
        t2o("data_format/test_convert.txt","data_format/test.txt")
        test = CoNLLDataset(config.test_filename, processing_word,
                            processing_tag, processing_pos, config.max_iter)

        model.evaluate(test, vocab_tags)

        tagging("data_format/test_convert.txt")


if __name__ == "__main__":

    if len(sys.argv)==1:
        print(sys.argv[0], 'INPUT_CORPUS')
        sys.exit(0)

    state = sys.argv[1]
    print(len(sys.argv))

    # python main predict 'filename'
    if state == "predict":
        if len(sys.argv) < 3:
            print("Input file name")
            sys.exit(0)
        file = sys.argv[2]

    # python main train
    elif state == "train":
        print("train start")

    # python main evaluate
    elif state == "evaluate":
        print("evaluate start")

    else:
        print("Input Correct CORPUS ('train' or 'evaluate')")
        sys.exit(0)

    # create instance of config
    config = Config()
    
    # load, train, evaluate and interact with model
    main(config)
    print("finish")