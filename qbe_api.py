# -*- coding: utf-8 -*-

# Hussein AL-NATSHEH, CNRS, France, 2015, 2016
# QbE API
# Usage example; requests.post('http://localhost:5000/Query', data = {"query":"EiS_Notation","weight":"average"}).content

from flask import Flask, make_response, make_response, request, current_app
from flask.ext.httpauth import HTTPBasicAuth
from flask_restful import Resource, Api, reqparse

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import subprocess
import requests
import pickle

from qbe.vectorize import QbE
from qbe.structure import Documents, Load_data, Doc
from qbe.utils import performance

"""
to do:
- Work on a flask app using jinja2 or Twitter Bootstrap, the most popular CSS/Javascript layout framework with jQuery
- consider negation
"""

class Parse(Resource):
    def get(self):
        # Note that pos_parser.jar assume the documents text files located in "./bench/". This is hard coded in Java
        try:
        # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('pos_parser', type=str, help='pos_parser.jar input filename to use')
            args = parser.parse_args()
            if args['pos_parser'] is not None:
                _posParser = args['pos_parser']
            else:
                _posParser = 'pos_parser.jar'
            subprocess.call(['java', '-jar', _posParser])
            return 'data successfully parsed into triples.csv'

        except Exception as e:
            return {'error': str(e)}

class Read(Resource):
    def get(self):
        #Prepare query, documents and a vectorizer for the IR
        try:
        # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('file', type=str, help='triples.csv input filename to read')
            args = parser.parse_args()

            if args['file'] is not None:
                _fileName = args['file']
            else:
                _fileName = 'triples.csv'
            # _fileName = args['file']
            docs = Documents(_fileName)
            docs.clean_data()
            data = Load_data(docs)
            documents = {} #initialize a dictionary of all Doc objetcs
            for doc_id in docs.doc_ids:
                documents[doc_id] = Doc(doc_id, data)
            pickle.dump(documents, open('documents.p', 'wb'))

            return 'data successfully read into documents.p'

        except Exception as e:
            return {'error': str(e)}

class Eis(Resource):

    def get(self):
        #get the document doc_id's of the EiS_doc
        all_doc_id = _documents.keys()
        eis_doc_ids = []
        response = {}
        for doc_id in all_doc_id:
            if 'EiS_' in doc_id:
                eis_doc_ids.append(doc_id)
        response['doc'] = eis_doc_ids
        resp = make_response()
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['content-type'] = 'application/json'
        resp.data = dict(response)
        return response

class Query(Resource):
    def post(self):
        try:
        # Parse the arguments
            parser = reqparse.RequestParser()
            parser.add_argument('global', type=str, required=False, help='set to True to run for all document')
            parser.add_argument('query', type=str, required=True, help='the document ID of the query')
            parser.add_argument('method', type=str, required=False, help='tfidf, qbe, or custom_weight_qbe')
            parser.add_argument('path', type=str, required=False, help='the path to text documents for tfidf method')
            parser.add_argument('weight', type=str, required=False, help='weighting schema for qbe method: subj, verb, obj, adv or average')
            parser.add_argument('subj', type=float, required=False, help='custom weighting of subj for qbe method')
            parser.add_argument('verb', type=float, required=False, help='custom weighting of verb for qbe method')
            parser.add_argument('obj', type=float, required=False, help='custom weighting of obj for qbe method')
            parser.add_argument('adv', type=float, required=False, help='custom weighting of adv for qbe method')
            parser.add_argument('topn', type=int, required=False, help='Top n ranked results to use when computing the performance')
            parser.add_argument('norm', type=str, required=False, help='set to Ture to normalise the scores [0, 1]')
            args = parser.parse_args()

            if args['global'] is not None:
                _global = True
            else:
                _global = False
            _query = args['query']
            if args['method'] is not None:
                _method = args['method']
            else:
                _method = 'qbe'
            if args['path'] is not None:
                _path = args['path']
            else:
                _path = "./bench/"
            if args['weight'] is not None:
                _weight = args['weight']
            else:
                _weight = 'average'
            if args['subj'] is not None:
                _s = args['subj']
            if args['verb'] is not None:
                _v = args['verb']
            if args['obj'] is not None:
                _o = args['obj']
            if args['adv'] is not None:
                _a = args['adv']

            if args['topn'] is not None:
                _n = args['topn']
            else:
                _n = 50
            
            if args['norm'] is not None:
                _norm = True
            else:
                _norm = False

            def query(_query):
                exp = QbE(_query, _documents, _norm) # Creating Query_by_Example object for the experiment
                response = {}
                if _method == 'tfidf':
                    response['results'] = exp.tfidf(_path)
                elif _method == 'qbe':
                    if (_weight == 'subj'):
                        response['results'] = exp.qbe(1.0, 0.0, 0.0, 0.0)
                    elif (_weight == 'verb'):
                        response['results'] = exp.qbe(0.0, 1.0, 0.0, 0.0)
                    elif (_weight == 'obj'):
                        response['results'] = exp.qbe(0.0, 0.0, 1.0, 0.0)
                    elif (_weight == 'adv' or _weight == 'all'):
                        response['results'] = exp.qbe(0.0, 0.0, 0.0, 1.0)
                    elif (_weight == 'average' or _weight == 'all'):
                        response['results'] = exp.qbe(0.25, 0.25, 0.25, 0.25)
                    else:
                        raise ValueError('Wrong value for argument `weight`. \
                            Accepted values are: subj, verb, obj, adv or average')
                elif _method == 'custom_weight_qbe':
                        response['results'] = exp.qbe(_s, _v, _o, _a)
                else:
                    raise ValueError('Wrong value for argument `method`. \
                            Accepted values are: tfidf, qbe, custom_weight_qbe or compare_all')
                y_true, y_pred, y_score = performance(response['results'], _n)
                response['query_id'] = _query
                response['query_content'] = exp.get_query_content(_path)
                response['precision'] = precision_score(y_true, y_pred)
                response['recall'] = recall_score(y_true, y_pred)
                response['f1'] = f1_score(y_true, y_pred)
                response['auc'] = roc_auc_score(y_true, y_score)
                return response

            if _global:
                avg={}
                precision_sum = 0
                recall_sum = 0
                f1_sum = 0
                auc_sum = 0
                all_doc_id = _documents.keys()
                eis_doc_ids = []
                for doc_id in all_doc_id:
                    if 'EiS_' in doc_id:
                        eis_doc_ids.append(doc_id)
                glob_results = {}
                nb_docs = len(eis_doc_ids)
                for _query in eis_doc_ids:
                    res = query(_query)
                    del res['results']
                    precision_sum += res['precision']
                    recall_sum += res['recall']
                    f1_sum += res['f1']
                    auc_sum += res['auc']
                    glob_results[_query] = res
                avg['precision'] = precision_sum / nb_docs
                avg['recall'] = recall_sum / nb_docs
                avg['f1'] = f1_sum / nb_docs
                avg['auc'] = auc_sum / nb_docs
                glob_results['average'] = avg
                response = glob_results
            else:
                response = query(_query)
            
            response['nb_docs'] = len(_documents.keys())
            response['method'] = _method
            response['topn'] = _n
            if _method == 'qbe':
                response['weighting'] = _weight
            if _method == 'custom_weight_qbe':
                _weight = {}
                _weight['subject'] = _s
                _weight['verb'] = _v
                _weight['object'] = _o
                _weight['adverb'] = _a
                response['weighting'] = _weight
            return response

        except Exception as e:
            return {'error': str(e)}


app = Flask(__name__, static_url_path="")
auth = HTTPBasicAuth()

api = Api(app)
api.add_resource(Parse, '/Parse')
api.add_resource(Read,'/Read')
api.add_resource(Eis, '/Eis')
api.add_resource(Query, '/Query')

if __name__ == '__main__':
    _documents = pickle.load(open('documents.p'))
    app.run(debug=True, threaded=True)
