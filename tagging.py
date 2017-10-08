
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

class Tag(object):
    def tagging(self, file):
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
