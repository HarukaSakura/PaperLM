import torch
import os
import logging
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data.dataloader import DataLoader


def train(model, dataset, out_model_path, args, checkpoint_path=None):

    if not args.train_from_zero:
        #model.encoder_q.from_pretrained(checkpoint_path)
        model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model.pt"), map_location=torch.device("cuda:3")), strict=True)

    train_loader = DataLoader(dataset=dataset, drop_last=True, batch_size=args.batch_size, num_workers=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        tr_total_loss = 0.0
        tr_NRP_loss = 0.0
        tr_PQCls_loss = 0.0
        tr_Moco_loss = 0.0
        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(epoch_iterator):
            original_encoding = batch["original_encoding"]
            augment_encoding = batch["augment_encoding"]
            result_dict = model(original_encoding=original_encoding, augment_encoding=augment_encoding)
            NRP_loss = result_dict["NRP_loss"]
            PQCls_loss = result_dict["PQCls_loss"]
            Moco_loss = result_dict["Moco_loss"]
            loss = NRP_loss/NRP_loss.detach() + PQCls_loss/PQCls_loss.detach() + 2*Moco_loss/Moco_loss.detach()
            tr_total_loss += loss.item()
            tr_NRP_loss += NRP_loss.item()
            tr_PQCls_loss += PQCls_loss.item()
            tr_Moco_loss += Moco_loss.item()
            # debug Nan
            assert torch.isnan(loss).sum() == 0, logging.info('loss Nan:{0}'.format(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.print_step == 0:
                logging.info('epoch:{0}  iter:{1}/{2}  loss:{3}  NRP_loss:{4}  PQCls_loss:{5}  Moco_loss:{6}'.format(epoch, step, len(train_loader), tr_total_loss / (step+1), tr_NRP_loss / (step+1), tr_PQCls_loss / (step+1), tr_Moco_loss / (step+1)))

        ckpt_path = os.path.join(out_model_path, "{}-{}".format('checkpoint_epoch_', epoch))
        save_model(ckpt_path, model, optimizer)

    ckpt_path = os.path.join(out_model_path, "{}".format('checkpoint'))
    save_model(ckpt_path, model, optimizer)


def save_model(ckpt_path, model, optimizer):
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
    model.encoder_q.encoder.save_pretrained(ckpt_path)
    logging.info("Saving model checkpoint to %s", ckpt_path)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    logging.info("Saving optimizer and scheduler states to %s", ckpt_path)


if __name__ =="__main__":

    logging.info("test")