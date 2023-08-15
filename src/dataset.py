from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.rnn import pad_sequence
#from EduNLP.Pretrain import BertTokenizer
from PaperLMProcessor import PaperLMProcessor
from configuration_paperlm import PaperLMConfig
from augment import Augment
import os
import json
import torch
from tqdm import tqdm
import pickle

def load_vector_pickle(data_type): 
    if data_type == "Pretrain":
        print("load pretrain data vector")
        folder = "../data/pretrain_data/vector/"
    elif data_type == "Diff":
        print("load diff data vector")
        folder = "../data/difficulty_data/vector/"
    elif data_type == "Similarity":
        print("load similarity data vector")
        folder = "../data/similarity_data/vector/"
    elif data_type == "Cluster":
        print("load cluster data vector")
        folder = "../data/similarity_data/vector/"
    
    with open(os.path.join(folder, "title_vector.pickle"), "rb") as pic:
        title_vector = pickle.load(pic)   
    with open(os.path.join(folder, "pq_vector.pickle"), "rb") as pic:
        pq_vector = pickle.load(pic)
    with open(os.path.join(folder, "q_vector.pickle"), "rb") as pic:
        q_vector = pickle.load(pic)
    print("load over")
    return title_vector, pq_vector, q_vector


def load_diff_json(paper_json, paper_ID, title_vector, pq_vector, q_vector):

    text_arr = []
    vector_arr = []
    eot_path_arr = []
    cq_vector = []
    
    cq_count = 0
    
    text = paper_json["text"]
    vector_arr.append(title_vector[paper_ID])
    path = paper_json["eot_path"]
    paper_diff = paper_json["paper_diff"]

    text_arr.append(text)
    eot_path_arr.append(path)

    for pq in list(paper_json["sub"].keys()):
        pq_json = paper_json["sub"][pq]
        pqid = pq[1:len(pq)-1].split(", ")[1]
        text = pq_json["text"]
        
        path = pq_json["eot_path"]
        text_arr.append(text)
        
        vector_arr.append(pq_vector[pqid])
        eot_path_arr.append(path)
        
        for q in list(pq_json["sub"].keys()):
            q_json = pq_json["sub"][q]
            text = q_json["text"]
            qid = q[1:len(q)-1].split(", ")[1]
            if "sub" not in q_json:
                vector_arr.append(q_vector[qid])
            else: 
                qv = q_vector[qid]
                cq_vector = []
                cq_count = 0
                for dim in range(qv.size(0)):
                    cq_vector.append(qv[dim, :].reshape(1,-1))
                vector_arr.append(cq_vector[cq_count])
                cq_count += 1
            path = q_json["eot_path"]

            text_arr.append(text)
            eot_path_arr.append(path)

            if "sub" in q_json and len(q_json["sub"]) > 0:
                for cq in list(q_json["sub"].keys()):
                    cq_json = q_json["sub"][cq]
                    text = cq_json["text"]
                    path = cq_json["eot_path"]
                    text_arr.append(text)
                    vector_arr.append(cq_vector[cq_count])
                    #print(cq_count)
                    cq_count += 1
                    eot_path_arr.append(path)
    data_item = {}
    data_item["text_arr"] = text_arr
    data_item["vector_arr"] = vector_arr
    data_item["eot_path_arr"] = eot_path_arr
    data_item["paper_diff"] = paper_diff

    return data_item

def load_similarity_json(paper_json, paper_ID, title_vector, pq_vector, q_vector):
    text_arr = []
    vector_arr = []
    eot_path_arr = []
    cq_vector = []
    cq_count = 0

    text = paper_json["text"]
    vector_arr.append(title_vector[paper_ID])
    text_arr.append(text)
    eot_path_arr.append(paper_json["eot_path"])

    for pq in list(paper_json["sub"].keys()):
        pq_json = paper_json["sub"][pq]
        text = pq_json["text"]
        text_arr.append(text)
        vector_arr.append(pq_vector[paper_ID])
        eot_path_arr.append(pq_json["eot_path"])
        for q in list(pq_json["sub"].keys()):
            q_json = pq_json["sub"][q]
            text = q_json["text"]
            qid = q[1:len(q)-1].split(", ")[1]
            if "sub" not in q_json:
                vector_arr.append(q_vector[qid])
            else: 
                qv = q_vector[qid]
                cq_vector = []
                cq_count = 0
                for dim in range(qv.size(0)):
                    cq_vector.append(qv[dim, :].reshape(1,-1))
                vector_arr.append(cq_vector[cq_count])
                cq_count += 1
            text_arr.append(text)
            eot_path_arr.append(q_json["eot_path"])

            if "sub" in q_json and len(q_json["sub"]) > 0:
                for cq in list(q_json["sub"].keys()):
                    cq_json = q_json["sub"][cq]
                    text = cq_json["text"]
                    text_arr.append(text)
                    try:
                        vector_arr.append(cq_vector[cq_count])
                    except:
                        exit()
                    #print(cq_count)
                    cq_count += 1
                    eot_path_arr.append(cq_json["eot_path"])
    data_item = {}
    data_item["text_arr"] = text_arr
    data_item["vector_arr"] = vector_arr
    data_item["eot_path_arr"] = eot_path_arr
    return data_item


def load_pretrain_json(paper_json, paper_ID, max_length, title_vector, pq_vector, q_vector, maskPQ):
    length = paper_json["length"]
    if length > max_length - 2:
        return None
    text_arr = []
    vector_arr = []
    eot_path_arr = []
    txt_type_arr = []
    kn_arr = []
    cq_vector = []
    
    cq_count = 0
    
    text = paper_json["text"]
    vector_arr.append(title_vector[paper_ID])
    path = paper_json["eot_path"]
    
    text_arr.append(text)
    eot_path_arr.append(path)
    txt_type_arr.append(13)
    kn_arr.append(None) # none question

    for pq in list(paper_json["sub"].keys()):
        pq_json = paper_json["sub"][pq]
        pqid = pq[1:len(pq)-1].split(", ")[1]
        text = pq_json["text"]
        
        path = pq_json["eot_path"]
        text_arr.append(text)
        if maskPQ:
            vector_arr.append(torch.zeros([1,768],dtype=torch.float))
        else:
            vector_arr.append(pq_vector[pqid])
        eot_path_arr.append(path)
        txt_type_arr.append(14)
        kn_arr.append(None) # none question
        
        for q in list(pq_json["sub"].keys()):
            q_json = pq_json["sub"][q]
            text = q_json["text"]
            qid = q[1:len(q)-1].split(", ")[1]
            kn_arr.append(q_json["kn_list"])
            if "sub" not in q_json:
                vector_arr.append(q_vector[qid])
            else: 
                qv = q_vector[qid]
                cq_vector = []
                cq_count = 0
                for dim in range(qv.size(0)):
                    cq_vector.append(qv[dim, :].reshape(1,-1))
                vector_arr.append(cq_vector[cq_count])
                cq_count += 1
            path = q_json["eot_path"]

            text_arr.append(text)
            eot_path_arr.append(path)
            txt_type_arr.append(q_json["type"])
            

            if "sub" in q_json and len(q_json["sub"]) > 0:
                for cq in list(q_json["sub"].keys()):
                    cq_json = q_json["sub"][cq]
                    text = cq_json["text"]
                    path = cq_json["eot_path"]
                    text_arr.append(text)
                    vector_arr.append(cq_vector[cq_count])
                    #print(cq_count)
                    cq_count += 1
                    eot_path_arr.append(path)
                    txt_type_arr.append(15)
                    kn_arr.append(None) # subquestion

    nodeRelation = generateNRPLabel(eot_path_arr, max_length)
    PQClsLabel, PQ_pos = generatePQClsLabel(txt_type_arr)
    eot_path_arr = eot_path_arr[1:1+length]
    data_item = {}
    data_item["vector_arr"] = vector_arr
    data_item["pq_mask_vector"] = [torch.zeros(1,768) if i in PQ_pos else vector_arr[i] for i in range(len(vector_arr))]
    data_item["eot_path_arr"] = eot_path_arr
    data_item["txt_type_arr"] = txt_type_arr
    data_item["kn_arr"] = kn_arr
    data_item["nodeRelation"] = nodeRelation
    data_item["PQClsLabel"] = PQClsLabel
    return data_item

def load_cluster_json(paper_json, paper_ID, max_length, title_vector, pq_vector, q_vector):
    length = paper_json["length"]
    if length > max_length - 2:
        return None
    cluster_label = paper_json["cluster_label"]
    vector_arr = []
    eot_path_arr = []
    cq_vector = []
    
    cq_count = 0
    
    vector_arr.append(title_vector[paper_ID])# 暂时不mask标题
    path = paper_json["eot_path"]
    
    eot_path_arr.append(path)

    for pq in list(paper_json["sub"].keys()):
        pq_json = paper_json["sub"][pq]
        pqid = pq[1:len(pq)-1].split(", ")[1]
        
        path = pq_json["eot_path"]

        vector_arr.append(pq_vector[pqid])
        eot_path_arr.append(path)
        
        for q in list(pq_json["sub"].keys()):
            q_json = pq_json["sub"][q]
            qid = q[1:len(q)-1].split(", ")[1]
            if "sub" not in q_json:
                vector_arr.append(q_vector[qid])
            else: 
                qv = q_vector[qid]
                cq_vector = []
                cq_count = 0
                for dim in range(qv.size(0)):
                    cq_vector.append(qv[dim, :].reshape(1,-1))
                vector_arr.append(cq_vector[cq_count])
                cq_count += 1
            path = q_json["eot_path"]

            eot_path_arr.append(path)
            
            if "sub" in q_json and len(q_json["sub"]) > 0:
                for cq in list(q_json["sub"].keys()):
                    cq_json = q_json["sub"][cq]
                    path = cq_json["eot_path"]
                    vector_arr.append(cq_vector[cq_count])
                    #print(cq_count)
                    cq_count += 1
                    eot_path_arr.append(path)

    data_item = {}
    data_item["vector_arr"] = vector_arr
    data_item["eot_path_arr"] = eot_path_arr
    data_item["cluster_label"] = cluster_label
    return data_item


def judgeRelation(node1, node2):
    if len(node1) == 0 or len(node2) == 0:
        return -100 # -100 ignore
    if node1 == node2:
        return 0
    min_len = min(len(node1), len(node2))
    i = 0
    while i < min_len and node1[i] == node2[i]:
        i += 1
    if i == len(node1):
        if len(node2) - len(node1) == 1:
            return 1
        else:
            return 4
    elif i == len(node2):
        if len(node1) - len(node2) == 1:
            return 2
        else:
            return 5
    elif i == len(node1)-1 == len(node2)-1:
        return 3
    else:
        return 6

def generateNRPLabel(eot_arr, max_length):
    # self 0, parent 1, child 2, sibling 3, ancestor 4, descendent 5, others 6 
    nodeRelation = []
    originLen = len(eot_arr)
    difference = max_length - 2 - originLen
    eot_arr.insert(0, [])
    eot_arr.append([])
    for i in range(difference):
        eot_arr.append([])
    for i in range(max_length):
        for j in range(max_length):
            nodeRelation.append(judgeRelation(eot_arr[i], eot_arr[j]))
    return nodeRelation

def generatePQClsLabel(txt_type_arr):
    typeMap = [2,0,1,1,2,2,0,0,2,2,2,2,2]
    PQCls_label = []
    PQ_pos = []
    for idx in range(len(txt_type_arr)):
        if txt_type_arr[idx] == 14 and idx < len(txt_type_arr)-1:
            PQCls_label.append(typeMap[txt_type_arr[idx+1]])
            PQ_pos.append(idx)
        else:
            PQCls_label.append(-100)
    return PQCls_label, PQ_pos

class MyDataset(Dataset):
    def __init__(self, config, processor, type, needAugment=True, augmentNum=1, maskPQ=True, split=None, end=348, n_cluster=4):
        self.config = config
        self.processor = processor
        self.type = type
        self.needAugment = needAugment
        self.maskPQ = maskPQ
        self.augmentNum = augmentNum
        max_length = config.max_length
        
        if type == "Diff":
            dir = "../data/difficulty_data/json"
            title_vector, pq_vector, q_vector = load_vector_pickle(type)
            papers = []
            # load json
            print("start loading " + split + " data")
            for filename in os.listdir(os.path.join(dir, split)):
                with open (os.path.join(dir, split, filename), "r") as f:
                    paper_json = json.load(f)
                json_path = os.path.join(paper_json, filename)
                paper = load_diff_json(json_path, title_vector, pq_vector, q_vector)
                papers.append(paper)

        elif type == "Similarity":
            dir = "../data/similarity_data/json"
            title_vector, pq_vector, q_vector = load_vector_pickle(type)
            papers = []
            num_group = 200
            for folder in tqdm(range(0, num_group)):
                json_folder_path = os.path.join(dir, str(folder))
                paper_group = []
                for filename in os.listdir(json_folder_path):
                    with open (os.path.join(json_folder_path, filename), "r") as f:
                        paper_json = json.load(f)
                    paper = load_similarity_json(paper_json, title_vector, pq_vector, q_vector)
                    paper_group.append(paper)
                papers.append(paper_group)

        elif type == "Pretrain":
            title_vector, pq_vector, q_vector = load_vector_pickle(type)
            dir = "../data/pretrain_data/json"
            papers = []
            # load json
            print("start loading data")
            for filename in os.listdir(dir):
                json_path = os.path.join(dir, filename)
                with open (json_path, "r") as f:
                    paper_json = json.load(f)
                paper_ID = os.path.splitext(filename)[0]
                paper = load_pretrain_json(paper_json, paper_ID, max_length, title_vector, pq_vector, q_vector, self.maskPQ)
                if paper is not None:
                    papers.append(paper)
            if self.needAugment:
                print("start augment")
                papers = self.apply_augment(papers=papers, aug=Augment(), q_vector_table=q_vector)

        elif type == "Cluster":
            title_vector, pq_vector, q_vector = load_vector_pickle(type)
            dir = "../data/cluster_data/json"
            if n_cluster == 4:
                dir = os.path.join(dir, "top_kn_data")
            elif n_cluster == 12:
                dir = os.path.join(dir, "mid_kn_data")
            papers = []
            # load json
            print("start loading data")
            for filename in tqdm(os.listdir(dir)):
                json_path = os.path.join(dir, filename)
                #print("loading "+filename)
                with open (json_path, "r") as f:
                    paper_json = json.load(f)
                paper_ID = os.path.splitext(filename)[0]
                paper = load_cluster_json(paper_json, paper_ID, max_length, title_vector, pq_vector, q_vector, json_path)
                if paper is not None:
                    papers.append(paper)
        self.data = papers

    def __getitem__(self, index):
        item = self.data[index]
        
        if self.type == "Diff":
            vector_arr = item["vector_arr"]
            eot_path_arr = item["eot_path_arr"]
            paper_diff = item["paper_diff"]
            encoding = self.processor(input_embeds=vector_arr, eot_path=eot_path_arr, diff_label=paper_diff) 
            return encoding
        elif self.type == "Similarity":
            group = {}
            group_inputs_embeds = []
            group_attention_mask = []
            group_path_tags_seq = []
            group_path_subs_seq = []
            for paper in item:
                vector_arr = paper["vector_arr"]
                eot_path_arr = paper["eot_path_arr"]
                encoding = self.processor(input_embeds=vector_arr, eot_path=eot_path_arr) 
                group_inputs_embeds.append(encoding["inputs_embeds"])
                group_attention_mask.append(encoding["attention_mask"])
                group_path_tags_seq.append(encoding["path_tags_seq"])
                group_path_subs_seq.append(encoding["path_subs_seq"])
            group["inputs_embeds"] = pad_sequence(group_inputs_embeds, batch_first=True)
            group["attention_mask"] = pad_sequence(group_attention_mask, batch_first=True)
            group["path_tags_seq"] = pad_sequence(group_path_tags_seq, batch_first=True)
            group["path_subs_seq"] = pad_sequence(group_path_subs_seq, batch_first=True)
            return group
        elif self.type == "Cluster":
            vector_arr = item["vector_arr"]
            eot_path_arr = item["eot_path_arr"]
            cluster_label = item["cluster_label"]
            encoding = self.processor(input_embeds=vector_arr, eot_path=eot_path_arr, cluster_label=cluster_label) 
            return encoding
        elif self.type == "Pretrain":
            vector_arr = item["vector_arr"]
            eot_path_arr = item["eot_path_arr"]
            nodeRelation = item["nodeRelation"]
            PQClsLabel = item["PQClsLabel"]
            json_path = item["json_path"]
            pq_mask_vector = item["pq_mask_vector"]
            encoding = self.processor(input_embeds=vector_arr, input_embeds_mask_pq=pq_mask_vector, eot_path=eot_path_arr, nodeRelation=nodeRelation, PQClsLabel=PQClsLabel, json_path=json_path) 
            if self.needAugment:
                augment_vector_arr = item["augment_vector_arr"]
                #print(len(augment_vector_arr))
                augment_encoding = self.processor(input_embeds=augment_vector_arr, eot_path=eot_path_arr)
                return {"original_encoding": encoding, "augment_encoding": augment_encoding}
            else:
                return encoding

    def __len__(self):
        return len(self.data)

    def apply_augment(self, papers, aug, q_vector_table):
        for idx in tqdm(range(len(papers))):
            papers[idx] = aug.replace_question(
                example=papers[idx], 
                qv_table=q_vector_table
                )
        return papers