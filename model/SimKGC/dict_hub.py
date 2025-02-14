import os
import glob

from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph
from logger_config import logger
import json

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None
relations2id = None

def _init_relations2id():
    global relations2id
    if not relations2id:
        with open(os.path.join(os.path.dirname(args.train_path), 'relations2id.json'), 'r') as f:
            relations2id = json.load(f)
        logger.info('Load relations2id from {}'.format(os.path.join(os.path.dirname(args.train_path), 'relations2id.json')))
    

def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))


def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict


def get_link_graph():
    _init_link_graph()
    return link_graph

def get_relations2id():
    _init_relations2id()
    return relations2id


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))

def get_relations_cnt(args):
    global relations2id
    if relations2id is None:
        _init_relations2id()
    args.rel_num = len(relations2id)
    return args



def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
