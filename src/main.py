import logging
import os
from utils import setuplogger, parse_args, set_seed
from utils import set_seed
from PaperLMProcessor import PaperLMProcessor
from configuration_paperlm import PaperLMConfig
from modeling_paperlm import MultiTaskModel, Predictor
from dataset import MyDataset
from train import train
from finetune import finetune_diff, test_similarity, paper_cluster, visual_cluster, debug

if __name__ == "__main__":
    args = parse_args()
    log_path = args.log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    setuplogger(args, log_path)
    set_seed(args)

    if args.mode == "pretrain":
        out_model_path = os.path.join(args.out_path, "pretrain")
        if not os.path.exists(out_model_path):
            os.makedirs(out_model_path)
        config = PaperLMConfig()
        processor = PaperLMProcessor(config=config, datasetType="Pretrain")
        logging.info("[train] loading dataset")
        train_dataset = MyDataset(config=config, processor=processor, type="Pretrain", needAugment=True, maskPQ=True, end=348)
        logging.info("[train] loading model")
        model = MultiTaskModel(config, device=args.device)
        model.to(args.device)
        logging.info("[train] train from zero")
        logging.info("[train] begin training")
        train(model, train_dataset, out_model_path, args)
    elif args.mode == "finetune":
        out_model_path = os.path.join(args.out_path, "finetune")
        if not os.path.exists(out_model_path):
            os.makedirs(out_model_path)
        config = PaperLMConfig()
        if args.downstream_task == "diff":
            logging.info("[test-diff] task: difficulty prediction")
            processor = PaperLMProcessor(config=config, datasetType="Diff")
            logging.info("[test] loading dataset")
            train_dataset = MyDataset(config=config, processor=processor, type="Diff", split="train", end=50)
            test_dataset = MyDataset(config=config, processor=processor, type="Diff", split="test", end=50)
            logging.info("[test] loading model")
            model = Predictor(config=config, device=args.device, cls_head=True, n_classes=1, output_strategy="avg")
            model.encoder.from_pretrained(args.pretrain_path)
            model.to(args.device)
            finetune_diff(model, train_dataset, test_dataset, out_model_path, args)
        elif args.downstream_task == "similarity":
            logging.info("[test-similarity] task: find similar paper")
            processor = PaperLMProcessor(config=config, datasetType="Similarity")
            logging.info("[test] loading dataset")
            test_dataset = MyDataset(config=config, processor=processor, type="Similarity")
            logging.info("[test] loading model")
            model = Predictor(config=config, device=args.device, cls_head=False, output_strategy="avg")
            model.encoder.from_pretrained(args.pretrain_path)
            model.to(args.device)
            test_similarity(model, test_dataset, args)
        elif args.downstream_task == "cluster":
            logging.info("[test-cluster] task: paper cluster")
            processor = PaperLMProcessor(config=config, datasetType="Cluster")
            logging.info("[test] loading dataset")
            test_dataset = MyDataset(config=config, processor=processor, type="Cluster", n_cluster=args.n_cluster)
            logging.info("[test] loading model")
            model = Predictor(config=config, device=args.device, cls_head=False, output_strategy="avg")
            model.encoder.from_pretrained(args.pretrain_path)
            model.to(args.device)
            paper_cluster(model, test_dataset, args)

