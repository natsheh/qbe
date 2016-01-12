# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015
# Load Data into dictionaries for QbE
# NOTE: Needs Documents object
from qbe.structure import Document, Sentence, Pattern
import numpy as np

class Load_data():
    # Class that takes Documents objects and generates all patterns and their POS and POS's synonym lists
    def __init__ (self, docs):
        self.docs_dict = {}
        self.docs_data = {}
        self.sent_dict = {}
        self.sent_data = {}
        self.pat_data = {}
        self.pat_subj = {}
        self.pat_subj_syn = {}
        self.pat_verb = {}
        self.pat_verb_syn = {}
        self.pat_obj = {}
        self.pat_obj_syn = {}
        self.pat_adv = {}
        self.pat_adv_syn = {}
        for doc_id in docs.doc_ids:
            doc = Document(doc_id, docs.data.loc[docs.data['docID']== doc_id])
            self.docs_dict[doc_id] = doc.get_sentence_ids()
            self.docs_data[doc_id] = docs.data.loc[docs.data['docID']== doc_id]
            for sent_id in self.docs_dict[doc_id]:
                sent = Sentence(sent_id, self.docs_data[doc_id].loc[self.docs_data[doc_id]['sentID'] == sent_id])
                self.sent_dict[sent_id] = sent.get_pattern_ids()
                self.sent_data[sent_id] = self.docs_data[doc_id].loc[self.docs_data[doc_id]['sentID'] == sent_id]
                for pat_id in self.sent_dict[sent_id]:
                    pat = Pattern(pat_id, self.sent_data[sent_id].loc[self.sent_data[sent_id]['patID'] == pat_id])
                    self.pat_data[pat_id] = np.array(self.sent_data[sent_id].loc[self.sent_data[sent_id]['patID'] == pat_id, 's':'q'].to_records(index=False))
                    self.pat_subj[pat_id] = pat.get_subjects()
                    self.pat_subj_syn[pat_id] = pat.get_subjects_syn()
                    self.pat_verb[pat_id] = pat.get_verbs()
                    self.pat_verb_syn[pat_id] = pat.get_verbs_syn()
                    self.pat_obj[pat_id] = pat.get_objects()
                    self.pat_obj_syn[pat_id] = pat.get_objects_syn()
                    self.pat_adv[pat_id] = pat.get_adverbs()
                    self.pat_adv_syn[pat_id] = pat.get_adverbs_syn()
        self.docs_data = {}
        self.sent_data = {}

    def get_sent_pat(self, sent_id):
        if sent_id not in self.sent_dict.keys():
            raise ValueError('The given sentence ID does not exist')
        if len(self.sent_dict[sent_id]) > 0 :
            ans = self.pat_data [self.sent_dict[sent_id][0]]
            i=0
            for pat_id in self.sent_dict[sent_id]:
                if i == 0:
                    i+=1
                    continue
                else:
                    ans = np.hstack((ans, self.pat_data[pat_id]))
            return ans
        else:
            raise ValueError('The given sentence ID has no patterns')

    def get_doc_pat(self, doc_id):
        if doc_id not in self.docs_dict.keys():
            raise ValueError('The given document ID does not exist')
        first_sent_id = self.docs_dict[doc_id][0]
        if len(self.docs_dict[doc_id]) > 0 :
            ans = self.get_sent_pat(first_sent_id)
            i=0
            for sent_id in self.docs_dict[doc_id]:
                if i == 0:
                    i+=1
                    continue
                else:
                    ans = np.hstack((ans, self.get_sent_pat(sent_id)))
            return ans
        else:
            raise ValueError('The given sentence ID has no patterns')
    
    def get_doc_pat_ids(self, doc_id):
        ans = []
        for sent_id in self.docs_dict[doc_id]:
            ans.append(self.sent_dict[sent_id])
        return np.hstack(ans)

    def get_doc_subj_syn(self, doc_id):
        pat_subj = {}
        for pat_id in self.get_doc_pat_ids(doc_id):
            pat_subj[pat_id] = self.pat_subj_syn[pat_id]
        return pat_subj
    def get_doc_verb_syn(self, doc_id):
        pat_verb = {}
        for pat_id in self.get_doc_pat_ids(doc_id):
            pat_verb[pat_id] = self.pat_verb_syn[pat_id]
        return pat_verb
    def get_doc_obj_syn(self, doc_id):
        pat_obj = {}
        for pat_id in self.get_doc_pat_ids(doc_id):
            pat_obj[pat_id] = self.pat_obj_syn[pat_id]
        return pat_obj
    def get_doc_adv_syn(self, doc_id):
        pat_adv = {}
        for pat_id in self.get_doc_pat_ids(doc_id):
            pat_adv[pat_id] = self.pat_adv_syn[pat_id]
        return pat_adv