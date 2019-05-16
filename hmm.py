import numpy as np
import itertools
import time


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a,b)


class HMM(object):
    def __init__(self, annotated_sentences):
        self.tags = np.array('ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split())
        self.tags_index = dict()
        self.vocabulary_dict = dict()

        self.tags_count = {k:0 for k in self.tags}
        self.tags_product = np.array(list(itertools.product(self.tags, repeat=2)))
        self.tags_product_count = {k[0] + ',' +k[1]:0 for k in self.tags_product}

        self.N = self.tags.shape[0]
        self.annotated_sentences = np.array(annotated_sentences)
        self.D = self.annotated_sentences.shape[0]

        # list of unique words
        vocabulary = set()
        for sentence in self.annotated_sentences:
            for word,tag in sentence:
                vocabulary.add(word)

        self.vocabulary = np.array(list(vocabulary))

        self.initial_probabilities = np.zeros(shape=(self.N))
        self.total_initial_words = 0
        self.transition_matrix = np.zeros(shape=(self.N,self.N))
        self.emission_matrix = np.zeros(shape=(self.N,self.vocabulary.shape[0]))


        for idx, tag in enumerate(self.tags):
            self.tags_index[tag] = idx
        for idx, word in enumerate(self.vocabulary):
            self.vocabulary_dict[word] = idx

    def train(self):
        sentences = self.annotated_sentences

        initial_tags = np.array([sentence[0][1] for sentence in sentences])
        total_initial_tags_count = initial_tags.shape[0]

        for tag in initial_tags:
            self.initial_probabilities[self.tags_index[tag]] += 1

        self.initial_probabilities = self.initial_probabilities / total_initial_tags_count

        for sentence in sentences:
            for tag in [w[1] for w in sentence]:
                self.tags_count[tag] += 1
            pairs = pairwise(sentence)
            for pair in pairs:
                key = pair[0][1] +',' + pair[1][1]
                self.tags_product_count[key] += 1

        # filling the transition matrix
        for row in range(0,self.N):
            for col in range(0,self.N):
                r_tag = self.tags[row]
                c_tag = self.tags[col]
                total_count_pair = self.tags_product_count[r_tag + ',' + c_tag]
                total_count_tag =  self.tags_count[r_tag]
                self.transition_matrix[row][col] = total_count_pair / total_count_tag

        for sentence in sentences:
            for word,tag in sentence:
                row = self.tags_index[tag]
                col = self.vocabulary_dict[word]
                self.emission_matrix[row][col] += 1

        for i in range(self.N):
            self.emission_matrix[i,:] /= self.tags_count[self.tags[i]]

    def predict(self, sentence):
        y_length = len(sentence)
        viterbi_matrix = np.zeros(shape=(self.N, y_length))
        backpointer = np.zeros(shape=(self.N, y_length))

        ## initilization step
        for idx, tag in enumerate(self.tags):
            first_word_in_sentence = sentence[0]
            try:
                word_index = self.vocabulary_dict[first_word_in_sentence]
                b = self.emission_matrix[idx][word_index]
            except Exception:
                b = 1
            viterbi_matrix[idx][0] = self.initial_probabilities[idx] * b
            backpointer[idx][0] = 0
        ### recursive step
        for i in range(1,y_length):
            for j in range(self.N):
                try:
                    b = self.emission_matrix[j][self.vocabulary_dict[sentence[i]]]
                except Exception:
                    b = 1
                viterbi_matrix[j][i] = np.max(viterbi_matrix[:,i-1] * self.transition_matrix[:,j] * b)
                backpointer[j][i] = np.argmax(viterbi_matrix[:,i-1] * self.transition_matrix[:,j] * b)

        prediction = []
        k = int(np.argmax(viterbi_matrix[:,len(sentence) - 1]))
        for i in range(len(sentence)):
            prediction.append(self.tags[int(k)])
            k = backpointer[int(k)][len(sentence) - 1 - i]

        prediction.reverse()
        return prediction
