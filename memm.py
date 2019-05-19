from sklearn.linear_model import LogisticRegression

class MEMM(object):
    def __init__(self):
        pass

    @staticmethod
    def contains_number(word):
        return any(i.isdigit() for i in word)

    @staticmethod
    def contains_upper_case(word):
        return any(i.isupper() for i in word)

    def features(self, sentence, index):
        word = sentence[index]
        if index > 0:
            pre_word = sentence[index - 1]

        f1 = 1 if self.contains_number(word) else 0
        f2 = 1 if self.contains_upper_case(word) else 0
        f3 = 1 if '-' in word else 0
        f4 = 1 if word.isupper() else 0
        f5 = 1 if word.endswith('ing') else 0
        f6 = 1 if word.endswith('ed') else 0
        f7 = 1 if word.endswith('able') else 0
        f8 = 1 if word.endswith('s') else 0
        f9 = word[0]
        f10 = word[0:1]
        f11 = word[0:2]
        f12 = word[0:3]
        f13 = word[-1:]
        f14 = word[-2:]
        f15 = word[-3:]
        f16 = word[-4:]
