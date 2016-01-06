# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015
# Data structure  for QbE

from nltk.corpus import wordnet as wn
import pandas as pd

class Documents():
    def __init__ (self, datafile):#, query_doc_id= "q"):
        self.datafile = datafile
        self.data = pd.read_csv(self.datafile, sep=';', header=0, na_values=['null'], keep_default_na=False, encoding='utf-8')
        self.header = self.data.columns.values
        self.doc_ids = list(set(self.data['docID']))
        self.sent_ids = list(set(self.data['sentID']))
        """"if query_doc_id in self.doc_ids:
            self.query = query_doc_id
        else:
            # if no document id is `q`, the first doc_id will be the query
            self.query = self.doc_ids[0]"""
        self.pattern_ids = list(set(self.data['patID']))
        self.subjects = list(set(self.data['s']))
        self.all_verbs = list(set(self.data['v']))
        self.all_objects = list(set(self.data['o']))
        # self.nouns = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
        self.verbs = {x.name().split('.', 1)[0] for x in wn.all_synsets('v')}
    
    def get_doc_ids(self):
        return list(set(self.data['docID']))
    def get_sentence_ids(self):
        return list(set(self.data['sentID']))
    def get_pattern_ids(self):
        return list(set(self.data['patID']))
   
    def get_subjects_syn(self):
        answer = []
        for subj in self.get_subjects():
            if not pd.isnull(subj):
                answer.append(subj)
            for syn in self.get_sym(subj, 'noun'):
                answer.append(syn)
        return list(set(answer))
    
    def get_verbs_syn(self):
        answer = []
        for verb in self.get_verbs():
            if not pd.isnull(verb):
                answer.append(verb)
            for syn in self.get_sym(verb, 'verb'):
                answer.append(syn)
        return list(set(answer))
    
    def get_objects_syn(self):
        answer = []
        for obj in self.get_objects():
            if not pd.isnull(obj):
                answer.append(obj)
            for syn in self.get_sym(obj, 'noun'):
                answer.append(syn)
            for syn in self.get_sym(obj, 'adj'):
                answer.append(syn)
        return list(set(answer))

    def get_adverbs_syn(self):
        answer = []
        for adv in self.get_adverbs():
            if not pd.isnull(adv):
                answer.append(adv)
            for syn in self.get_sym(adv, 'adv'):
                answer.append(syn)
        return list(set(answer))
    
    def get_sym(self, word, pos): 
    # returns a list of POS synonyms
        synonyms =[]
        if pd.isnull(word):
            return [' ']
        else:
            if pos == 'noun':
                synsets = wn.synsets(word, wn.NOUN)
            elif pos == 'verb':
                synsets = wn.synsets(word, wn.VERB)
            elif pos == 'adj':
                synsets = wn.synsets(word, wn.ADJ)
            elif pos == 'adv':
                synsets = wn.synsets(word, wn.ADV)
            else:
                raise ValueError('Second argument allowed options: noun, verb, adj, adv')

            for synset in synsets:
                for lemma in synset.lemmas():
                    synonyms.append(lemma.name())

            
            synonyms = list(set(synonyms))
            return synonyms
    
    def get_subjects(self):
        return list(set(self.data['s']))
    def get_verbs(self):
        return list(set(self.data['v']))
    def get_objects(self):
        return list(set(self.data['o']))
    def get_adverbs(self):
        return list(set(self.data['a']))
    def get_header(self):
        return self.data.columns.values
    
    def clean_data(self):
        lst =[]
        for w in self.data['v']:
            lst.append(self.is_verb(w))
        self.data['is_verb_verb'] = lst
        self.data = self.data[self.data['is_verb_verb']]
        lst =[]
        for w in self.data['o']:
            lst.append(not self.is_verb(w))
        self.data['is_obj_verb'] = lst
        self.data = self.data[self.data['is_obj_verb']]
    #def is_noun(word):
    #    return (word in nouns)
    def is_verb(self, word):
        return (word in self.verbs)
        
class Document(Documents):
    def __init__(self, doc_id, doc):
        self.doc_id = doc_id
        self.data = doc 
        
class Sentence(Document):
    def __init__(self, sent_id, sentence):
        self.sent_id = sent_id
        self.data = sentence

class Pattern(Sentence):
    def __init__(self, pat_id, pattern):
        self.pat_id = pat_id
        self.data = pattern