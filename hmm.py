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
        self.tags_product = np.array(list(itertools.product(self.tags, repeat=2)))
        self.N = self.tags.shape[0]
        self.annotated_sentences = np.array(annotated_sentences)
        self.D = self.annotated_sentences.shape[0]

        # list of unique words
        vocabulary = set()
        for sentence in self.annotated_sentences:
            for word,tag in sentence:
                vocabulary.add(word)

        self.vocabulary = np.array(list(vocabulary))
        self.tags_count = np.zeros(shape=(self.N))
        self.initial_probabilities = np.zeros(shape=(self.N))
        self.total_initial_words = 0
        self.transition_matrix = np.zeros(shape=(self.N+1,self.N))
        self.emission_matrix = np.zeros(shape=(self.N,self.vocabulary.shape[0]))
        # observations = tagged sentences - (string, string) - (words, tags)

        # calculating the initial probability distribution
       #for sentence in self.annotated_sentences:
       #    for idx,wt in enumerate(sentence):
       #        try:
       #            i = self.tags.index(wt[1])
       #        except:
       #            #print(wt[0], wt[1])
       #            continue
       #        if idx == 0:
       #            self.initial_probabilities[i] += 1
       #            self.total_initial_words += 1
       #        self.tags_count[i] += 1

       #self.initial_probabilities = self.initial_probabilities / self.total_initial_words

       # self.transition_matrix[0] = self.initial_probabilities
        # ira approves

       # tags_permutations = list(itertools.permutations(self.tags, 2))
       # for tag in self.tags:
       #     tags_permutations.append((tag,tag))
#
       # self.count_adjacent_tags = {k[0] + ',' + k[1]: 0 for k in tags_permutations}
       # for sentence in self.annotated_sentences:
       #     pairs = pairwise(sentence)
       #     for pair in pairs:
       #         key = pair[0][1] + ',' + pair[1][1]
       #         self.count_adjacent_tags[key] += 1
#
       # for row in range(1,n+1):
       #     for col in range(0,n):
       #         r_tag = self.tags[row-1]
       #         c_tag = self.tags[col]
#
       #         count_pair = self.count_adjacent_tags[r_tag + ',' + c_tag]
       #         try:
       #             i = self.tags.index(r_tag)
       #             count_tag = self.tags_count[i]
       #         except Exception:
       #             #print('ERROR : index not found! : ' + r_tag)
       #             continue
#
       #         self.transition_matrix[row][col] = count_pair / count_tag
       # # ira approves 2

        # B \ emission matrix \ B[i][j] = Count(word,tag)/Count(tag_total)
     #  for# sentence in self.annotated_sentences:
     #      for word,tag in sentence:
     #          try:
     #              row = self.tags.index(tag)
     #              col = self.vocabulary.index(word)
     #              self.emission_matrix[row][col] += 1 / self.tags_count[row]
     #          except Exception:
     #              pass
     #              #print('ERROR : tag not found! : ' + tag)
     #              #print('ERROR : word not found! : ' + word)

        # ira approves 3
    def tag_to_index(self, tag):
        s = np.where(self.tags == tag)[0]
        if s.shape[0] > 0:
            return s[0]
        raise Exception("Tag was not found : " + tag)

    def word_to_index(self, word):
        s = np.where(self.vocabulary == word)[0]
        if s.shape[0] > 0:
            return s[0]
        return -1
        #raise Exception("Word was not found : " + word)

    def tag_tuple_to_index(self, tag_tuple):
        s = np.where(self.tags_product == tag_tuple)[0]
        if s.shape[0] > 0:
            return s[0]
        raise Exception("Tag Tuple was not found : " + tag_tuple)


    def train(self):
        sentences = self.annotated_sentences
        tags_count = np.zeros(shape=self.N)
        # array of tags for first words
        initial_tags = np.array([sentence[0][1] for sentence in sentences])
        total_initial_tags_count = initial_tags.shape[0]

        for tag in initial_tags:
            self.initial_probabilities[self.tag_to_index(tag)] += 1

        self.initial_probabilities = self.initial_probabilities / total_initial_tags_count
        # set first row to initial probabilities
        self.transition_matrix[0] = self.initial_probabilities


        # set up memory for counting and set to zeros
        adjacent_tags_count = np.zeros(shape=self.tags_product.shape[0])

        # counting tag pairs and tag counts
        for sentence in sentences:
            for tag in [w[1] for w in sentence]:
                tags_count[self.tag_to_index(tag)] += 1
            pairs = pairwise(sentence)
            for pair in pairs:
                pair = (pair[0][1], pair[1][1])
                adjacent_tags_count[self.tag_tuple_to_index(pair)] += 1

        # filling the transition matrix
        for row in range(1,self.N+1):
            for col in range(0,self.N):
                r_tag = self.tags[row-1]
                c_tag = self.tags[col]
                total_count_pair = adjacent_tags_count[self.tag_tuple_to_index((r_tag ,c_tag))]
                total_count_tag =  tags_count[self.tag_to_index(r_tag)]
                self.transition_matrix[row][col] = total_count_pair / total_count_tag

        for sentence in sentences:
            for word,tag in sentence:
                row = self.tag_to_index(tag)
                col = self.word_to_index(word)
                self.emission_matrix[row][col] += 1 / tags_count[row]


    def predict(self, sentence):
        ## Initilize
        start = time.time()


        viterbi_matrix = np.zeros(shape=(len(self.tags), len(sentence)))
        backpointer = np.zeros(shape=(len(self.tags), len(sentence)))

        ##
        for idx,tag in enumerate(self.tags):
            first_word_in_sentence = sentence[0] ## MIGHT BE sentence[0] only if the sentence is unlabled./
            word_index = self.word_to_index(first_word_in_sentence)
            if word_index == -1: # If word is unknown then the probability of P(word|tag) = 0 without smoothing.
                b = 1
            else:
                b = self.emission_matrix[idx][word_index]
            viterbi_matrix[idx][0] = self.initial_probabilities[idx] * b


        for idx in range(1,len(sentence)):
            for i,tag in enumerate(self.tags):
                v_values = np.zeros(shape=(len(self.tags)))
                for t_index, ttag in enumerate(self.tags):
                    viterbi_value = viterbi_matrix[t_index][idx-1]
                    a = self.transition_matrix[t_index][i]
                    word_index = self.word_to_index(sentence[idx])
                    if word_index == -1:
                        b = 1
                    else:
                        b = self.emission_matrix[i][self.word_to_index(sentence[idx])]
                    v_values[t_index] = viterbi_value * a * b

                max_idx = np.argmax(v_values)
                viterbi_matrix[i][idx] = v_values[max_idx]
                backpointer[i][idx] = max_idx


        best_path_pointer = np.argmax([viterbi_matrix[row][len(sentence) -1] for row in range(len(self.tags))])
        best_path = viterbi_matrix[best_path_pointer][len(sentence) -1]
        prediction = []
        for i in range(len(sentence)):
            best_path_pointer = np.argmax([viterbi_matrix[row][i] for row in range(len(self.tags))])
            prediction.append(self.tags[best_path_pointer])

        end = time.time()
        print("Total time per sentence: " + str(end-start))
        return prediction


    def predict2(self):
        pass

#[('With', 'ADP'),
# ('the', 'DET'),
# ('death', 'NOUN'),
# ('of', 'ADP'),
# ('Garcia', 'PROPN'),
# (',', 'PUNCT'),
# ('the', 'DET'),
# ('pursuit', 'NOUN'),
# ('might', 'AUX'),
# ('cease', 'VERB'),
# (',', 'PUNCT'),
# ('since', 'SCONJ'),
# ('such', 'DET'),
# ('a', 'DET'),
# ('death', 'NOUN'),
# ('might', 'AUX'),
# ('frighten', 'VERB'),
# ('others', 'NOUN'),
# ('from', 'ADP'),
# ('the', 'DET'),
# ('task', 'NOUN'),
# ('.', 'PUNCT'),
# ('>>', 'PUNCT')]

