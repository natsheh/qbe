# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015
# QbE Tf-IDF Vecotorization and Similarities

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as csim
import numpy as np
import operator
import subprocess
import os.path
from qbe.utils import vconcat, call_jar, sort_dic_desc, norm_dic, remove_keys_of_empty_value


class QbE():
    def __init__ (self, q, documents, norm):
        self.documents = documents # dictionary of objects of type Doc
        self.q = q # doc_id which is of type string
        self.norm = norm

    def qbe(self, s, v, o, a):
        vectorizer = TfidfVectorizer(min_df=1)
        res = {}
        qst = remove_keys_of_empty_value(self.documents[self.q].get_subj_syn())
        len_sq = len(qst.keys())
        qvt = remove_keys_of_empty_value(self.documents[self.q].get_verb_syn())
        len_vq = len(qvt.keys())
        qot = remove_keys_of_empty_value(self.documents[self.q].get_obj_syn())
        len_oq = len(qot.keys())
        qat = remove_keys_of_empty_value(self.documents[self.q].get_adv_syn())
        len_aq = len(qat.keys())
        
        for d in self.documents:
            # subj
            dst = remove_keys_of_empty_value(self.documents[d].get_subj_syn())
            Xs = vconcat(qst, dst)
            if len_sq < 1 or len(dst.keys()) < 1:
                subj_sim = 0.0
            else:
                Xs_vec = vectorizer.fit_transform(Xs)
                subj_sim = np.average(csim(Xs_vec[0:len_sq],Xs_vec[len_sq:]))
            # verb
            dvt = remove_keys_of_empty_value(self.documents[d].get_verb_syn())
            Xv = vconcat(qvt, dvt)
            if len_vq < 1 or len(dvt.keys()) < 1:
                verb_sim = 0.0
            else:
                Xv_vec = vectorizer.fit_transform(Xv)
                verb_sim = np.average(csim(Xv_vec[0:len_vq],Xv_vec[len_vq:]))
            # obj
            dot = remove_keys_of_empty_value(self.documents[d].get_obj_syn())
            Xo = vconcat(qot, dot)
            if len_oq < 1 or len(dot.keys()) < 1:
                obj_sim = 0.0
            else:
                Xo_vec = vectorizer.fit_transform(Xo)
                obj_sim = np.average(csim(Xo_vec[0:len_oq],Xo_vec[len_oq:]))
            # adv
            dat = remove_keys_of_empty_value(self.documents[d].get_adv_syn())
            Xa = vconcat(qat, dat)
            if len_aq < 1 or len(dat.keys()) < 1:
                adv_sim = 0.0
            else:
                Xa_vec = vectorizer.fit_transform(Xa)
                adv_sim = np.average(csim(Xa_vec[0:len_aq],Xa_vec[len_aq:]))
            res[d] = s * subj_sim + v * verb_sim + o * obj_sim + a * adv_sim
        if self.norm:
            answer = norm_dic(res)
        else:
            answer = sort_dic_desc(res)
        return answer

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
