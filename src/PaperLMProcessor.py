import torch
from configuration_paperlm import PaperLMConfig

class PaperLMProcessor():
    def __init__(self, config, datasetType="Pretrain", needPad=True):
        self.tags_dict = {"paper": 0, "packedques": 1, "ques": 2, "subques": 3}
        self.max_depth = config.max_depth # 10
        self.max_width = config.max_width # 100
        self.pad_width = config.max_width + 1 # 101
        self.unk_tag_id = len(self.tags_dict)
        self.pad_tag_id = self.unk_tag_id + 1
        self.pad_xpath_tags_seq = [self.pad_tag_id] * self.max_depth
        self.pad_xpath_subs_seq = [self.pad_width] * self.max_depth
        self.cls_input_embeds_seq = torch.rand(1,config.hidden_size)
        self.sep_input_embeds_seq = self.cls_input_embeds_seq
        self.pad_input_embeds_seq = torch.zeros([1,config.hidden_size])
        self.max_length = config.max_length # 200
        self.needPad = needPad
        self.datasetType = datasetType
        self.NRP_num_labels = config.NRP_num_labels
        self.pad_label_id = -100
        
    def format_eot(self, path_units):# paper/1/2/1
        eotpath = "/"
        tag = ["paper", "packedques", "ques", "subques"]
        for i in range(len(path_units)):
            if i == 0:
                eotpath += tag[i]
            else:
                eotpath += "/" + tag[i] + "[" + str(path_units[i]) + "]"
        return eotpath
    
    def get_path_seq(self, xpath):
        
        xpath_tags_list = []
        xpath_subs_list = []

        xpath_units = xpath.split("/")
        
        for unit in xpath_units:
            if not unit.strip():
                continue
            name_subs = unit.strip().split("[")
            tag_name = name_subs[0]
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            xpath_tags_list.append(self.tags_dict.get(tag_name))
            xpath_subs_list.append(sub)

        xpath_tags_list = xpath_tags_list[: self.max_depth]
        xpath_subs_list = xpath_subs_list[: self.max_depth]
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))
        
        return xpath_tags_list, xpath_subs_list

    def get_html_path_seq(self, xpath):# paper/packedques[1]/ques[2]/subques[1]
        xpath_tags_list = []
        xpath_subs_list = []

        xpath_units = xpath.split("/")
        if len(xpath_units) == 1:
            html_path = "html/head/title"
        else:
            html_path = "html/body"
            for i in range(1,len(xpath_units)):
                html_path += "/"
                unit = xpath_units[i]
                name_subs = unit.strip().split("[")
                tag_name = name_subs[0]
                html_tag_name = self.html_tags_dict[tag_name]
                sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
                html_unit = html_tag_name + "["+ str(sub) +"]"
                html_path += html_unit
        print(html_path)
        xpath_units = html_path.split("/")
                
        for unit in xpath_units:
            if not unit.strip():
                continue
            name_subs = unit.strip().split("[")
            tag_name = name_subs[0]
            sub = 0 if len(name_subs) == 1 else int(name_subs[1][:-1])
            xpath_tags_list.append(self.tags_dict.get(tag_name))
            xpath_subs_list.append(sub)

        xpath_tags_list = xpath_tags_list[: self.max_depth]
        xpath_subs_list = xpath_subs_list[: self.max_depth]
        xpath_tags_list += [self.pad_tag_id] * (self.max_depth - len(xpath_tags_list))
        xpath_subs_list += [self.pad_width] * (self.max_depth - len(xpath_subs_list))
        
        return xpath_tags_list, xpath_subs_list

    def __call__(self, text=None, input_embeds=None, input_embeds_mask_pq=None, eot_path=None, nodeRelation=None, diff_label=None, PQClsLabel=None, cluster_label=None):
        # handle input_embeds
        input_embeds_example = []
        sequence_length = len(input_embeds) # 实际有意义的序列长度
        input_embeds_example.append(sum(input_embeds)/len(input_embeds)) # cls 取所有token均值
        input_embeds_example.extend(input_embeds)
        input_embeds_example.append(sum(input_embeds)/len(input_embeds))

        # handle eotpath
        path_tags_seq_example = []
        path_subs_seq_example = []
        
        path_tags_seq_example.append(self.pad_xpath_tags_seq)
        path_subs_seq_example.append(self.pad_xpath_subs_seq)

        for e in eot_path:
            path_tags_list, path_subs_list = self.get_path_seq(self.format_eot(e))
            path_tags_seq_example.extend([path_tags_list])
            path_subs_seq_example.extend([path_subs_list])

        path_tags_seq_example.append(self.pad_xpath_tags_seq)
        path_subs_seq_example.append(self.pad_xpath_subs_seq)

        attention_mask = [1] * (sequence_length + 2)

        if self.datasetType is "Pretrain":
            if PQClsLabel is not None:
                PQClsLabel_example = []
                PQClsLabel_example.append(self.pad_label_id)
                PQClsLabel_example.extend(PQClsLabel)
                PQClsLabel_example.append(self.pad_label_id)
            if input_embeds_mask_pq is not None:
                input_embeds_maskpq_example = []
                input_embeds_maskpq_example.append(sum(input_embeds_mask_pq)/len(input_embeds_mask_pq)) # cls 取所有token均值
                input_embeds_maskpq_example.extend(input_embeds_mask_pq)
                input_embeds_maskpq_example.append(sum(input_embeds_mask_pq)/len(input_embeds_mask_pq))
            
        if self.needPad:
            if sequence_length < self.max_length:
                difference = self.max_length - sequence_length - 2
                input_embeds_example = input_embeds_example + [self.pad_input_embeds_seq] * difference
                path_tags_seq_example = path_tags_seq_example + [self.pad_xpath_tags_seq] * difference
                path_subs_seq_example = path_subs_seq_example + [self.pad_xpath_subs_seq] * difference
                attention_mask = attention_mask + [0] * difference
                if self.datasetType is "Pretrain":
                    if PQClsLabel is not None:
                        PQClsLabel_example = PQClsLabel_example + [self.pad_label_id] * difference
                    if input_embeds_mask_pq is not None:
                        input_embeds_maskpq_example = input_embeds_maskpq_example + [self.pad_input_embeds_seq] * difference

        input_embeds_example = torch.cat(input_embeds_example, dim=0)
        if input_embeds_mask_pq is not None:
            input_embeds_maskpq_example = torch.cat(input_embeds_maskpq_example, dim=0)
        path_tags_seq_example = torch.tensor(path_tags_seq_example)
        path_subs_seq_example = torch.tensor(path_subs_seq_example)
        attention_mask = torch.tensor(attention_mask)

        encoding = {}
        encoding["inputs_embeds"] = input_embeds_example
        encoding["attention_mask"] = attention_mask
        encoding["path_tags_seq"] = path_tags_seq_example
        encoding["path_subs_seq"] = path_subs_seq_example

        if self.datasetType is "Diff" and diff_label is not None:
            encoding["diff_label"] = torch.tensor(diff_label)
        elif self.datasetType is "Pretrain":
            if nodeRelation is not None:
                nodeRelation_example = torch.tensor(nodeRelation)
                encoding["nodeRelation"] = nodeRelation_example
            if PQClsLabel is not None:
                PQClsLabel_example = torch.tensor(PQClsLabel_example)
                encoding["PQClsLabel"] = PQClsLabel_example
            if input_embeds_mask_pq is not None:
                encoding["inputs_embeds_maskpq"] = input_embeds_maskpq_example
        elif self.datasetType is "Cluster":
            encoding["cluster_label"] = torch.tensor(cluster_label)
        return encoding
