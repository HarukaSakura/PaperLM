import random
import pickle
class Augment():
    def __init__(self, knowledge_table_path="../data/knowledgeTable_new.pickle", p=0.3):
        self.p = p
        with open(knowledge_table_path, "rb") as pic:
            self.knowledgeTable = pickle.load(pic)

    def replace_question(self, example, qv_table):
        '''
        txt_type:
        0-12 quetion_type
        13 title
        14 packed_question
        15 sub_question
        '''
        ques_vector = example["vector_arr"]
        txt_type = example["txt_type_arr"]
        
        kn_arr = example["kn_arr"]
        augment_ques_vector = []
        assert len(ques_vector) == len(txt_type) == len(kn_arr)
        length = len(txt_type)
        idx = 0
        while(idx < length):
            if 0 <= txt_type[idx] < 13 and random.random() < self.p:
                if kn_arr[idx] == None or len(kn_arr[idx]) == 0 or type(kn_arr[idx][-1]) != int: 
                    augment_ques_vector.append(ques_vector[idx])
                    idx += 1
                    continue
                bottom_knid = kn_arr[idx][-1]
                replace_id_list = self.knowledgeTable[bottom_knid].questionList[txt_type[idx]]
                replace_id = replace_id_list[random.randint(0, len(replace_id_list)-1)]
                if idx == length-1 or txt_type[idx+1] != 15: 
                    while(replace_id not in qv_table.keys() or qv_table[replace_id].size(0) != 1): 
                        replace_id = replace_id_list[random.randint(0, len(replace_id_list)-1)]
                    augment_ques_vector.append(qv_table[replace_id])
                    idx += 1
                else: 
                    sub_cnt = idx + 1
                    while(sub_cnt < length and txt_type[sub_cnt] == 15):
                        sub_cnt += 1
                    sub_cnt = sub_cnt - idx
                    
                    while(replace_id not in qv_table.keys() or qv_table[replace_id].size(0) != sub_cnt): 
                        replace_id = replace_id_list[random.randint(0, len(replace_id_list)-1)]
                    qv = qv_table[replace_id]
                    for dim in range(qv.size(0)):
                        augment_ques_vector.append(qv[dim, :].reshape(1,-1))
                        idx += 1
            else:
                augment_ques_vector.append(ques_vector[idx])
                idx += 1
        example["augment_vector_arr"] = augment_ques_vector
        return example

        
if __name__ == "__main__":
    aug = Augment()

