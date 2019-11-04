# coding=UTF-8

import json, os
import pickle
import random
import numpy as np 
import time, datetime 
import thulac # 
#import args 

def read_data(tr=True, d=False, te=False):
    # train_dataset: ./final_all_data/first_stage/
    # dev_dataset: ./final_all_data/exercise_contest/
    # test_dataset: ./final_all_data/final_test.json
    #path = os.getcwd() # 當前運行的絕對路徑，主文件夾下
    #data_path = os.path.dirname(path)+"/final_all_data/"

    if tr:
        #train_data_path = os.path.dirname(data_path)+"/first_stage/train.json"
        file_path = "/home/fangyi/ijcaiYang/final_all_data/first_stage/train.json"
    elif d: # 17131個
        #dev_data_path = os.path.dirname(data_path)+"/exercise_contest/data_valid.json"
        file_path = "/home/fangyi/ijcaiYang/final_all_data/exercise_contest/data_valid.json"
    elif te:
        #test_data_path = os.path.dirname(data_path)+"/final_test.json"
        file_path = "/home/fangyi/ijcaiYang/final_all_data/final_test.json"

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data_raw = f.readlines()
        for num, data_one in enumerate(data_raw):
            try:
                data.append(json.loads(data_one))
            except Exception as e:
                print('error: \n',e)
    # 返回list[dict{}]
    return data

def extract_labels(data, name='fact'): 
    # 'fact'\'relevant_articles'\'accusation'\'term_of_imprisonment'
    ''' extract key-value alone saved as dict'''
    extraction = {}
    if name == 'fact':
        extraction = list(map(lambda x: x['fact'], data))
    elif name in ['relevant_articles', 'accusation']:
        extraction = list(map(lambda x: x['meta'][name], data))
    elif name == 'term_of_imprisonment':
        extraction = []
        for i in data:
            if i['meta']['term_of_imprisonment']['death_penalty']:
                extraction.append([1000])
            elif i['meta']['term_of_imprisonment']['life_imprisonment']:
                extraction.append([2000])
            else:
                extraction.append([i['meta']['term_of_imprisonment']['imprisonment']])
    # 返回list[str'']
    return extraction #extraction.update({name: extraction})


def tokenize(fact_data, need_token=True,saved=False):
    thu1 = thulac.thulac(seg_only=True)
    #word_len = 1 #保留詞語長度
    data_cut = []
    if need_token:
        for one_fact in fact_data:
            one_data_cut = [thu1.cut(one_fact, text=True)]
            #data_cut.append(''.join(data_cut))
            data_cut.append(one_data_cut)
    else:
        data_cut = fact_data

    if saved:
        path = os.getcwd()
        save_path = os.path.dirname(path)+"/code_data/preprocess_tokenize_data.json"
        with open(save_path, 'w') as f:
            json.dump(data_cut, f)
    # data_cut 類型 -->
    return data_cut

def pretrained_wordvec():
    return None

def text2seq(tokenized_text, ):
    '''5000 each epoch to avoid overflow'''








