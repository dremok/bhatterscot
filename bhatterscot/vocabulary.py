PAD_token = 0
SOS_token = 1
EOS_token = 2


class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.reset()

    def reset(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: 'PAD', SOS_token: 'SOS', EOS_token: 'EOS'}
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def replace_word(self, old_word, new_word):
        idx = self.word2index[old_word]
        self.index2word[idx] = new_word

    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print(f'Keeping {len(keep_words) / len(self.word2index):.2%} of the words.')

        self.reset()

        for word in keep_words:
            self.add_word(word)
