# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015
# QbE Tf-IDF Vecotorization and Similarities

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as csim
import numpy as np
import operator
import subprocess
import os.path
from qbe.utils import vconcat, call_jar, sort_dic_desc, norm_dic, remove_keys_of_empty_value, pos_patterns_sim


class QbE():
    def __init__ (self, q, documents):
        self.documents = documents # dictionary of objects of type Doc
        self.q = q # doc_id which is of type string

    def qbe(self, s, v, o, a):
        vectorizer = TfidfVectorizer(min_df=1)
        res = {}
        # Compute POS pattern-based similarity of the query with each of all documents
        subj_sim = pos_patterns_sim(self.q, self.documents, vectorizer, pos='subj')
        verb_sim = pos_patterns_sim(self.q, self.documents, vectorizer, pos='verb')
        obj_sim = pos_patterns_sim(self.q, self.documents, vectorizer, pos='obj')
        adv_sim = pos_patterns_sim(self.q, self.documents, vectorizer, pos='adv')
        res[d] = np.average([subj_sim, verb_sim, obj_sim, adv_sim])
        res[d] = s * subj_sim + v * verb_sim + o * obj_sim + a * adv_sim
        return norm_dic(res)

    def tfidf(self, path="./bench/"): # This function is used for comparision with qbe
        vectorizer = TfidfVectorizer(stop_words='english')
        lst_files = []
        doc_dict = {}
        inv_doc_dict = {}
        i = 0
        for f in os.listdir(path):
            if f.endswith(".txt"):
                d = open(path+f)
                cont = d.read()
                cont = unicode(cont, errors='ignore')
                lst_files.append(cont)
                doc_dict[f[:-4]] = i #[:-4] is used for trimming '.txt' from the filename
                inv_doc_dict[i] = f[:-4]
                i+= 1

        tfidf_bow = vectorizer.fit_transform(lst_files)
        search =  csim(tfidf_bow[doc_dict[self.q]], tfidf_bow)

        search = list(search[0])

        i =0
        ans = {}
        for item in search:
            x = inv_doc_dict[i]
            ans[x] = item
            i+= 1
        tfidf_dic = ans
        tfidf = norm_dic(tfidf_dic)
        return tfidf
    
    def get_query_content(self, path="./bench/"):
        return unicode(open(path+self.q+'.txt').read(), errors='ignore')
