## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.linear_model import LogisticRegression
import re
from mmn12.hmm import HMM

## project supplied imports
from mmn12.submission_specs.SubmissionSpec12 import SubmissionSpec12

class Submission(SubmissionSpec12):
    ''' a contrived poorely performing solution for question one of this Maman '''
    
    def _estimate_emission_probabilites(self, annotated_sentences):
        pass
        
    
    def _estimate_transition_probabilites(self, annotated_sentences):
        pass
        
        
    def train(self, annotated_sentences):    
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
        hmm = HMM(annotated_sentences)
        hmm.train()
        self.hmm = hmm

        return self 

    def predict(self, sentence):
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        prediction = self.hmm.predict(sentence)
        assert (len(prediction) == len(sentence))
        return prediction
            