
import os
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import logging
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, classification_report, mean_absolute_error, mean_squared_error, ndcg_score, normalized_mutual_info_score, accuracy_score
from sklearn.cluster import KMeans
import numpy as np

def eval(model, test_loader, args, report=False, type='classification'):
    model.eval()
    pred = []
    label = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader, desc=f"[test] inference")):
            inputs_embeds = batch["inputs_embeds"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            path_tags_seq = batch["path_tags_seq"].to(args.device)
            path_subs_seq = batch["path_subs_seq"].to(args.device)
            diff_class_label = batch["diff_label"].to(args.device)
            logit = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, path_tags_seq=path_tags_seq, path_subs_seq=path_subs_seq)
            pred.extend(logit.cpu().detach().numpy())
            label.extend(diff_class_label.cpu().detach().numpy())
    model.train()
    if type == 'classification':
        return calculate_class_metrics(pred, label, report)
    elif type == 'regression':
        return calculate_regression_metrics(pred, label) 


def softmax(x):
    max = np.max(x,axis=-1,keepdims=True) #returns max of each row and keeps same dims
    e_x = np.exp(x - max) #subtracts each row with its max value
    sum = np.sum(e_x,axis=-1,keepdims=True) #returns sum of each row and keeps same dims
    f_x = e_x / sum 
    return f_x

def calculate_class_metrics(pred, truth, report=False):
    pred = np.array(pred)
    truth = np.array(truth)
    y_pred = np.argmax(pred, axis=-1)
    pred_softmax = softmax(pred)
    accuracy = np.sum(y_pred == truth) / len(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, y_pred, average='macro')
    try:
        auc = roc_auc_score(truth, pred_softmax, multi_class='ovo')
    except ValueError: 
        auc = 0
    if report:
        logging.info(classification_report(truth, y_pred))
    return accuracy, precision, recall, f1, auc


def calculate_regression_metrics(pred, truth):
    # (#samples)
    pred = np.array(pred).squeeze()
    truth = np.array(truth).squeeze()
    mae = mean_absolute_error(truth, pred)
    rmse = np.sqrt(mean_squared_error(truth, pred))
    p_corr = pearsonr(truth, pred)[0]
    sp_corr = spearmanr(truth, pred)[0]
    doa = calculate_doa(truth, pred)
    return mae, rmse, p_corr, sp_corr, doa

def calculate_doa(truth, pred):
    doa_score = 0.0
    all = 0.0
    for i in range(truth.shape[0]):
        for j in range(truth.shape[0]):
            if truth[i] > truth[j]:
                all += 1
                if pred[i] > pred[j]:
                    doa_score += 1
    return doa_score / all

def CosineSimilarity(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
    cs = (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
    return format(float(cs), ".4f")

def finetune_diff(model, train_dataset, test_dataset, out_model_path, args):
        
    train_loader = DataLoader(dataset=train_dataset, drop_last=True, batch_size=args.batch_size, num_workers=8, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, drop_last=True, batch_size=args.batch_size, num_workers=8, shuffle=True)
    loss_fn = nn.MSELoss()
    #optimizer = AdamW(model.parameters(), lr=args.lr)
    #t_total = len(train_loader) * args.epochs
    #scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.6)
    model.train()
    global_step = 0
    for epoch in range(args.epoch_start, args.epoch_start + args.epochs):
        tr_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(epoch_iterator):
            inputs_embeds = batch["inputs_embeds"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            path_tags_seq = batch["path_tags_seq"].to(args.device)
            path_subs_seq = batch["path_subs_seq"].to(args.device)
            diff_label = batch["diff_label"].to(args.device) 
            logit = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, path_tags_seq=path_tags_seq, path_subs_seq=path_subs_seq)
            loss = loss_fn(logit.squeeze(), diff_label)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.print_step == 0:
                logging.info('epoch:{0}  iter:{1}/{2}  loss:{3}'.format(epoch, step, len(train_loader), tr_loss / (step+1)))
        # validation
        mae, rmse, p_corr, sp_corr, doa = eval(model, test_loader, args, type='regression')
        logging.info(f'[test-diff] MAE: {mae:.4f}\tRMSE: {rmse:.4f}\tPearson: {p_corr:.4f}\tSpearman: {sp_corr:.4f}\tDOA: {doa:.4f}')
        # save
        if (epoch+1) % 10 == 0:
            ckpt_path = os.path.join(out_model_path, "{}-{}".format('checkpoint_epoch_', epoch))
            save_model(ckpt_path, model, optimizer)

def test_similarity(model, test_dataset, args):
    with torch.no_grad():
        ndcg_5 = 0
        ndcg_10 = 0
        idx = 0
        for group in test_dataset:
            inputs_embeds = group["inputs_embeds"].to(args.device)
            attention_mask = group["attention_mask"].to(args.device)
            path_tags_seq = group["path_tags_seq"].to(args.device)
            path_subs_seq = group["path_subs_seq"].to(args.device)
            logit = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, path_tags_seq=path_tags_seq, path_subs_seq=path_subs_seq)
            rel_pred = [[]]
            rel_true = [[10,9,8,7,6,5,4,3,2,1]]
            tensor_a = logit[0]
            for i in range(1, logit.size(0)):
                rel_pred[0].append(CosineSimilarity(tensor_a.cpu(), logit[i].cpu()))
            print("idx:{} rel_pred:{}".format(idx, rel_pred))
            n5 = ndcg_score(rel_true, rel_pred, k=5)
            n10 = ndcg_score(rel_true, rel_pred, k=10)
            #logging.info(f'[test-similarity] idx: {idx}\t NDCG@5: {n5:.4f}\tNDCG@10: {n10:.4f}')
            ndcg_5 += n5
            ndcg_10 += n10
            idx += 1
        ndcg_5 /= len(test_dataset)
        ndcg_10 /= len(test_dataset)
    logging.info(f'[test-similarity] average result: NDCG@5: {ndcg_5:.4f}\tNDCG@10: {ndcg_10:.4f}')

def paper_cluster(model, test_dataset, args):
    dataloader = DataLoader(dataset=test_dataset, drop_last=False, batch_size=args.batch_size, num_workers=8, shuffle=False)
    with torch.no_grad():
        logit_list = []
        label_list = []
        for step, batch in enumerate(dataloader):
            inputs_embeds = batch["inputs_embeds"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            path_tags_seq = batch["path_tags_seq"].to(args.device)
            path_subs_seq = batch["path_subs_seq"].to(args.device)
            cluster_label = batch["cluster_label"].to(args.device) 
            logit = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, path_tags_seq=path_tags_seq, path_subs_seq=path_subs_seq) # (bz,hidden)         
            logit_list.append(logit)
            label_list.append(cluster_label)
        outputs = torch.cat(logit_list, dim=0).cpu()
        label = torch.cat(label_list, dim=0).cpu()
        kmeans = KMeans(n_clusters=args.n_cluster)
        kmeans.fit(outputs)
        y_pred = kmeans.predict(outputs)
        nmi = normalized_mutual_info_score(label, y_pred)
        
    logging.info(f'[test-cluster] n_clusters: {args.n_cluster}  NMI: {nmi:.4f}')
    
            
def save_model(ckpt_path, model, optimizer):
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
    logging.info(f"[test-con] Saving model checkpoint to %s", ckpt_path)

    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    logging.info(f"[test-con] Saving optimizer and scheduler states to %s", ckpt_path)
