# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015
# QbE helper functions

from sklearn.metrics.pairwise import cosine_similarity as csim
from nltk.corpus import wordnet as wn
import numpy as np
import operator
import subprocess

def get_antonym(word, pos):
# return the antonym of a given word if exist
    antonym= ''
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
        if len(antonym) > 0:
            return antonym[0].name()
        else:
            word = wn.synset(synset.name())
            for lemma in word.lemmas():
                if len(antonym) > 0:
                    break
                else:
                    antonym = lemma.antonyms()
    return antonym

def norm_dic(dic):
    mx = np.max(dic.values())
    mn = np.min(dic.values())
    n_dic = {}
    for d in dic:
        n_dic[d] = (dic[d] - mn) / (mx - mn)
    return sort_dic_desc(n_dic)

def sort_dic_desc(n_dic):
    sorted_list = sorted(n_dic.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_list

def call_jar(filename):
    return subprocess.call(['java', '-jar', filename])

def vconcat(dic1, dic2): # Vercital dictionary concatenation 
    ans = dic1.values()
    for v in dic2.values():
        ans.append(v)
    return np.array(ans)

def qbe(q,documents, s, v, o, a, vectorizer):
    res = {}
    len_sq = len(q.get_subj_syn().keys())
    len_vq = len(q.get_verb_syn().keys())
    len_oq = len(q.get_obj_syn().keys())
    len_aq = len(q.get_adv_syn().keys())
    for d in documents:
        Xs = vconcat(q.get_subj_syn(), documents[d].get_subj_syn())
        Xs_vec = vectorizer.fit_transform(Xs)
        subj_sim = np.average(csim(Xs_vec[0:len_sq],Xs_vec[len_sq:]))
        Xv = vconcat(q.get_verb_syn(), documents[d].get_verb_syn())
        Xv_vec = vectorizer.fit_transform(Xv)
        verb_sim = np.average(csim(Xv_vec[0:len_vq],Xv_vec[len_vq:]))
        Xo = vconcat(q.get_obj_syn(), documents[d].get_obj_syn())
        Xo_vec = vectorizer.fit_transform(Xo)
        obj_sim = np.average(csim(Xo_vec[0:len_oq],Xo_vec[len_oq:]))
        Xa = vconcat(q.get_adv_syn(), documents[d].get_adv_syn())
        Xa_vec = vectorizer.fit_transform(Xa)
        adv_sim = np.average(csim(Xa_vec[0:len_aq],Xa_vec[len_aq:]))

        res[d] = np.average([subj_sim, verb_sim, obj_sim, adv_sim])
        res[d] = s * subj_sim + v * verb_sim + o * obj_sim + a * adv_sim
    return norm_dic(res)

def performance(results, n):
    y_true, y_pred, y_score = [], [], []
    len_relevant = 0
    for doc_id, score in results:
        y_score.append(score)
        y_true.append('EiS_' in doc_id)
        if 'EiS_' in doc_id:
            len_relevant += 1
    y_pred = [False] * len(results)
    y_pred[:len_relevant] = [True] * len_relevant
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    y_score = y_score[:n]
    # in case all the top n results are from the same class, add a dummy item from the other class
    # That to avoid error in sklearn.metrics.roc_auc_score incase of 1 class only
    if list(np.unique(y_pred)) == [True] and list(np.unique(y_true)) == [True]:
        y_pred.append(False)
        y_true.append(False)
        y_score.append(0.0)
    if list(np.unique(y_pred)) == [False] and list(np.unique(y_true)) == [False]:
        y_pred.append(True)
        y_true.append(True)
        y_score.append(0.0)
    return y_true, y_pred, y_score

def evalu(results, n):
    y_true, y_pred = [], []
    len_relevant = 0
    for doc_id, score in results:
        #y_true.append('EiS_' in doc_id)
        if 'EiS_' in doc_id:
            len_relevant += 1
            y_true.append('E')
        else:
            y_true.append('o')
    y_pred = ['o'] * len(results)
    y_pred[:len_relevant] = ['E'] * len_relevant
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    return y_true, y_pred

def remove_keys_of_empty_value(D):
    return dict((k, v) for k, v in D.iteritems() if v!=' ')

def pos_patterns_sim(query_id, documents, vectorizer, pos='subj'):
    if pos=='subj':
        qt = remove_keys_of_empty_value(documents[query_id].get_subj_syn())
    elif pos =='verb':
        qt = remove_keys_of_empty_value(documents[query_id].get_verb_syn())
    elif pos =='obj':
        qt = remove_keys_of_empty_value(documents[query_id].get_obj_syn())
    elif pos =='adv':
        qt = remove_keys_of_empty_value(documents[query_id].get_adv_syn())
    else:
        raise ValueError('The given pos value is not withen (subj, verb, obj, adv)')

    len_q = len(qt.keys())
    for d in documents:
        if pos=='subj':
            dt = remove_keys_of_empty_value(documents[d].get_subj_syn())
        elif pos =='verb':
            dt = remove_keys_of_empty_value(documents[d].get_verb_syn())
        elif pos =='obj':
            dt = remove_keys_of_empty_value(documents[d].get_obj_syn())
        elif pos =='adv':
            dt = remove_keys_of_empty_value(documents[d].get_adv_syn())
        else:
            raise ValueError('The given pos value is not withen (subj, verb, obj, adv)')
        X = vconcat(qt, dt)
        if len(np.unique(X)) == 1:
            if np.unique(X) == ' ':
                pos_sim = 0.0
        else:
            X_vec = vectorizer.fit_transform(X)
            pos_sim = np.average(csim(X_vec[0:len_q],X_vec[len_q:]))
    return pos_sim