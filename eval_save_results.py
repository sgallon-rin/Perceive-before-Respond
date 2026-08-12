"""
Evaluation script for ALBEF sticker response selection with results saving.

Usage examples:
  # Basic evaluation (no saving)
  python eval_save_results.py --config ./configs/toru.yaml --checkpoint output/exp01/checkpoint_best.pth
  
  # Save evaluation results
  python eval_save_results.py --config ./configs/toru.yaml --checkpoint output/exp01/checkpoint_best.pth --output_dir ./eval_results
  
  # Custom doc_num
  python eval_save_results.py --config ./configs/toru.yaml --checkpoint output/exp01/checkpoint_best.pth --output_dir ./eval_results --doc_num 10

Output filename format: {checkpoint_basename}_{test_file_path}_{timestamp}.json
"""

import argparse
import os
import json
import shutil
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from einops import rearrange, repeat

import ruamel.yaml as yaml

from models.model_srs import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from evaluation import evaluation
from dataset import create_dataset, create_sampler, create_loader


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def generate_output_filename(args, timestamp: str) -> str:
    """
    Generate output filename with format: 
    {checkpoint_basename}_{test_file_path}_{timestamp}.json
    """
    checkpoint_name = os.path.basename(args.checkpoint.rstrip('/')).replace('.pth', '')
    
    test_file_path = args.test_file.lstrip('/').replace('/', '_')
    
    filename = f"{checkpoint_name}_{test_file_path}_{timestamp}.json"
    
    return filename


@torch.no_grad()
def eval_with_scores(model, data_loader, tokenizer, device, doc_num=10):
    """
    Evaluate model and return ITA scores, ITM logits (pos/neg), and labels.
    """
    model.eval()

    print('Computing features for evaluation...')
    start_time = time.time()

    pred_ita = []
    pred_itm_pos = []  # positive class logits[:,1]
    pred_itm_neg = []  # negative class logits[:,0]
    gt_label = []
    sample_ids = []

    for image, text, idx in tqdm(data_loader):
        image = image.to(device)
        label = F.one_hot(idx, num_classes=doc_num)
        bs = image.shape[0]
        image = rearrange(image, 'b k c h w -> (b k) c h w')
        
        #### ITA ####
        image_feat = model.visual_encoder(image)  # [bs*10, 65, 768]
        image_atts = torch.ones(image_feat.size()[:-1], dtype=torch.long).to(image.device)
        image_embed = model.vision_proj(image_feat[:, 0, :])  # [bs*10, 256]
        image_embed = F.normalize(image_embed, dim=-1)
        image_embed = rearrange(image_embed, '(b k) c-> b k c', k=doc_num)  # [bs, 10, 256]

        text_input = tokenizer(text, padding='longest', truncation=True,
                               max_length=512, return_tensors="pt").to(device)
        text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask,
                                         mode='text')
        text_feat = text_output.last_hidden_state  # [bs, 512, 768]
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))  # [bs, 256]
        sim_t2i = text_embed.unsqueeze(1) @ image_embed.permute(0, 2, 1)
        sim_t2i = sim_t2i.squeeze(1)  # [bs, 10] text->image alignment score
        
        #### ITM ####
        repeat_text_embeds = repeat(text_feat, 'b w c->(b k) w c', k=doc_num)
        repeat_attention_mask = repeat(text_input.attention_mask, 'b w->(b k) w', k=doc_num)
        output = model.text_encoder(encoder_embeds=repeat_text_embeds,
                                    attention_mask=repeat_attention_mask,
                                    encoder_hidden_states=image_feat,
                                    encoder_attention_mask=image_atts,
                                    return_dict=True,
                                    mode='fusion')
        
        # Get full logits [bs*10, 2]
        logits = model.itm_head(output.last_hidden_state[:, 0, :])  # [bs*10, 2]
        logits = rearrange(logits, '(b k) c->b k c', k=doc_num)  # [bs, 10, 2]
        
        itm_pos_score = logits[:, :, 1]  # positive class [bs, 10]
        itm_neg_score = logits[:, :, 0]  # negative class [bs, 10]
        
        #### Save ####
        ita_scores = sim_t2i.view(-1).cpu().numpy()
        pos_scores = itm_pos_score.view(-1).cpu().numpy()
        neg_scores = itm_neg_score.view(-1).cpu().numpy()
        labels = label.view(-1).cpu().numpy()
        
        pred_ita.extend([safe_float(x) for x in ita_scores])
        pred_itm_pos.extend([safe_float(x) for x in pos_scores])
        pred_itm_neg.extend([safe_float(x) for x in neg_scores])
        gt_label.extend([int(x) for x in labels])
        
        # Track sample IDs
        for i in range(bs):
            sample_ids.append(i)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Evaluation time {total_time_str}')
    print(f"Total samples: {len(sample_ids)}, Total candidates: {len(pred_ita)}")

    return pred_ita, pred_itm_pos, pred_itm_neg, gt_label, sample_ids


def build_predictions_data(n_samples, ita_scores, itm_pos_scores, itm_neg_scores, 
                          labels, doc_num):
    """
    Build predictions data with ITA and ITM scores.
    """
    predictions = []
    
    ita_2d = np.array(ita_scores).reshape(-1, doc_num)
    pos_2d = np.array(itm_pos_scores).reshape(-1, doc_num)
    neg_2d = np.array(itm_neg_scores).reshape(-1, doc_num)
    labels_2d = np.array(labels).reshape(-1, doc_num)
    
    for i in range(n_samples):
        predictions.append({
            "sample_id": int(i),
            "ita_scores": ita_2d[i].tolist(),
            "itm_pos_scores": pos_2d[i].tolist(),
            "itm_neg_scores": neg_2d[i].tolist(),
            "labels": labels_2d[i].tolist()
        })
    
    return predictions


def save_evaluation_results(output_path, metadata, predictions):
    """
    Save evaluation results to JSON file.
    """
    result = {
        "metadata": metadata,
        "predictions": predictions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating sticker response selection dataset")
    # Only create test dataset
    _, _, test_dataset = create_dataset('srs', config)
    samplers = [None]
    
    test_loader = create_loader([test_dataset], samplers,
                                batch_size=[config.get('batch_size_test', 4)],
                                num_workers=[4],
                                is_trains=[False],
                                collate_fns=[None])[0]

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=False)
        print(f'load checkpoint from {args.checkpoint}')
        print(msg)

    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    #### Evaluation ####
    doc_num = args.doc_num
    
    pred_ita, pred_itm_pos, pred_itm_neg, gt_label, sample_ids = eval_with_scores(
        model_without_ddp, test_loader, tokenizer, device, doc_num=doc_num
    )

    if args.distributed:
        dist.barrier()

    # Compute metrics using ITM positive scores
    test_result = evaluation(pred_itm_pos, gt_label, samples=doc_num)
    
    print("\n===== Final Metrics =====")
    print(f"MAP:   {test_result['MAP']:.6f}")
    print(f"MRR:   {test_result['MRR']:.6f}")
    print(f"P@1:   {test_result['p@1']:.6f}")
    print(f"R2@1:  {test_result['r2@1']:.6f}")
    print(f"R@1:   {test_result['r@1']:.6f}")
    print(f"R@2:   {test_result['r@2']:.6f}")
    print(f"R@5:   {test_result['r@5']:.6f}")

    if args.output_dir is not None:
        if utils.is_main_process():
            print("\nSaving evaluation results...")
            os.makedirs(args.output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            
            filename = generate_output_filename(args, timestamp)
            output_path = os.path.join(args.output_dir, filename)
            
            n_samples = len(sample_ids)
            predictions = build_predictions_data(
                n_samples, pred_ita, pred_itm_pos, pred_itm_neg, gt_label, doc_num
            )
            
            metadata = {
                "model": "ALBEF",
                "checkpoint": args.checkpoint,
                "config_file": args.config,
                "test_file": config.get('test_file', 'unknown'),
                "image_root": config.get('image_root', 'unknown'),
                "doc_num": doc_num,
                "batch_size": config.get('batch_size_test', 4),
                "timestamp": timestamp,
                "metrics": test_result
            }
            
            save_evaluation_results(output_path, metadata, predictions)
            print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/SRS.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-chinese')
    parser.add_argument('--output_dir', default=None, help='Directory to save results')
    parser.add_argument('--doc_num', type=int, default=10, help='Number of candidates per sample')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    
    # Allow overriding test_file from command line if needed
    if hasattr(args, 'test_file') and args.test_file:
        config['test_file'] = args.test_file

    main(args, config)
