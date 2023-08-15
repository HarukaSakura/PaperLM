import json
import os
import torch
import pickle
import re
from EduNLP.Pretrain import BertTokenizer
from EduNlp import get_pretrained_i2v
from transformers import AutoModel
from knowTreeNode import TreeNode

title_vector = {}
pq_vector = {}
q_vector = {}

print("download EduNLP bert_math")
save_dir = "../edunlp_pretrain"
pretrained_name = "bert_math_ptc"
get_pretrained_i2v(pretrained_name, model_dir=save_dir)

print("load bert_math")
pretrained_dir = f"{save_dir}/{pretrained_name}"
tokenizer = BertTokenizer.from_pretrained(pretrained_dir)
model = AutoModel.from_pretrained(pretrained_dir)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
model = model.to(device)
print("load over")

knowledgeTable = [None]*1000

def q2v(text):
    if isinstance(text, list): 
        for i in range(len(text)):
            text[i] = text[i].replace("\n", "")
    else:
        text = text.replace("\n", "")
    token_items = tokenizer(text).to(device)
    #print(token_items)
    attention_mask = token_items["attention_mask"]
    with torch.no_grad():
        output = model(**token_items)
        emb = output.last_hidden_state
        num_non_padding = attention_mask.sum(dim=1).unsqueeze(-1)
        non_padding_mask = attention_mask.unsqueeze(-1)
        emb = emb * non_padding_mask
        emb = emb.sum(dim=1) / num_non_padding.float()
        emb = emb.to(cpu)
    #print(outputs.shape)
    token_items = token_items.to(cpu)
    
    return emb

def handle_json(data, paper_ID):
    title = data["text"]
    title_vector[paper_ID] = q2v(title)
    for pq in data["sub"].keys():
        pq_json = data["sub"][pq]
        pqid = pq[1:len(pq)-1].split(", ")[1]
        pq_text = pq_json["text"]
        pq_vector[pqid] = q2v(pq_text)
        for q in pq_json["sub"].keys():
            q_json = pq_json["sub"][q]
            qid = q[1:-1].split(", ")[1]
            if qid not in q_vector.keys():
                q_text = q_json["text"]
                try:
                    st = re.search(r'^[0-9]*. ',q_text).span()[1]
                    q_text = q_text[st:]
                except:
                    pass
                if "sub" not in q_json:
                    q_vector[qid] = q2v(q_text)
                else:
                    subq_arr = [q_text]
                    for no in q_json["sub"].keys():
                        subq_json = q_json["sub"][no]
                        subq_arr.append(subq_json["text"])
                    q_vector[qid] = q2v(subq_arr)
            # build knowledge table
            try:
                kn_list = q_json["kn_list"]
                kn_name = q_json["kn_name"]
            except(KeyError):
                continue
            type = q_json["type"]
            
            for i in range(len(kn_list)):
                id = kn_list[i] 
                if knowledgeTable[id] == None: # 尚未添加该node
                    name = kn_name[i]
                    if i == 0:
                        knowNode = TreeNode(id=id, name=name, parentId=-1, layer=i)
                    else:
                        parentId = kn_list[i-1]
                        knowNode = TreeNode(id=id, name=name, parentId=parentId, layer=i)
                    if i != len(kn_list)-1:
                        knowNode.addChild(kn_list[i+1])
                    knowledgeTable[id] = knowNode
                else:
                    if i != len(kn_list)-1:
                        knowledgeTable[id].addChild(kn_list[i+1])
                if qid not in knowledgeTable[id].questionList[type]:
                    knowledgeTable[id].questionList[type].append(qid)



if __name__ == "__main__":
    
    folder_path = "../data/pretrain_data/json"

    for filename in os.listdir(folder_path):
        json_path = os.path.join(folder_path, filename)
        with open (json_path, "r") as f:
            paper_json = json.load(f)
        paper_ID = os.path.splitext(filename)[0]
        handle_json(paper_json, paper_ID)        
    
    with open("../data/pretrain_data/vector/title_vector.pickle", "wb") as pic:
        pickle.dump(title_vector, pic)
    with open("../data/pretrain_data/vector/pq_vector.pickle", "wb") as pic:
        pickle.dump(pq_vector, pic)
    with open("../data/pretrain_data/vector/q_vector.pickle", "wb") as pic:
        pickle.dump(q_vector, pic)
    
    