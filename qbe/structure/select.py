# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015
# To be used to select content from a given doc_id
# Note that data is a global object of the class Load_data. It's not passes nor attribute

class Doc(): 
    def __init__(self, doc_id, data):
        # Note that data is a global object of the class Load_data. It's not passes nor attribute
        self.data = data
        self.doc_id = doc_id
    def get_sent_ids(self):
        return self.data.docs_dict[self.doc_id]
    def get_pat_ids(self):
        return self.data.get_doc_pat_ids(self.doc_id)
    def get_pats(self):
        return self.data.get_doc_pat(self.doc_id)
    def get_subj_syn(self):
        return self.lst2str(self.data.get_doc_subj_syn(self.doc_id))
    def get_verb_syn(self):
        return self.lst2str(self.data.get_doc_verb_syn(self.doc_id))    
    def get_obj_syn(self):
        return self.lst2str(self.data.get_doc_obj_syn(self.doc_id))
    def get_adv_syn(self):
        return self.lst2str(self.data.get_doc_adv_syn(self.doc_id))
    def lst2str(self, dic):
        #Takes dict of list of strings and returns dict of concatinated string
        #We do so since the vectorizer expects a list of documents rather than a list of list of words
        ans = {}
        for key in dic:
            ans[key] = ' '.join(dic[key])
        return ans