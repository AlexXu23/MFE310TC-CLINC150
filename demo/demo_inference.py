"""Run two frozen seed-42 BERT pipelines on the two presentation examples.

Use the project's existing BERT_Rejection_Study_20261019/.venv Python.
No training, detector fitting or threshold tuning occurs. Outputs are written
only beside this script, never into the frozen experiment directories.
"""
from pathlib import Path
import csv
import gc
import hashlib
import json
import os
import sys
import time
import argparse

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'BERT_Typo_Augmentation_20261019/outputs'
sys.path.insert(0,str(ROOT/'BERT_Rejection_Study_20261019'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
import numpy as np
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from distance_model import SharedMahalanobisDetector
from threadpoolctl import threadpool_limits

def saved_rows(arm):
    with (SOURCE/arm/'seed42/typo_medium_predictions.csv').open(encoding='utf-8-sig',newline='') as f:
        return {int(r['row_id']):r for r in csv.DictReader(f)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interactive',action='store_true',help='Preload both models, then press Enter to infer each example.')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    cases=[{'row_id':180,'text':'can you help me find my pone, pease','true_label':'find_phone'},
           {'row_id':4883,'text':'what is a normal rseting hart rate','true_label':'oos'}]
    mapping=json.loads((ROOT/'CLINC150_Team_20261019/outputs/bert/seed42/label_mapping.json').read_text(encoding='utf-8'))['id_to_label']
    results={};pipelines={}
    print('FROZEN BERT INFERENCE DEMONSTRATION',flush=True)
    print('Seed 42; FP32; Mahalanobis; validation IS acceptance >=95%; device='+device,flush=True)
    for arm in ['clean_continue','typo_mix']:
        folder=SOURCE/arm/'seed42'
        print('\nLoading '+arm+' ...',flush=True)
        start=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(folder/'final_model',local_files_only=True,use_fast=True)
        model=BertForSequenceClassification.from_pretrained(folder/'final_model',local_files_only=True,attn_implementation='eager').to(device).float().eval()
        model.requires_grad_(False)
        detector=SharedMahalanobisDetector.load(folder/'distance_model.npz')
        threshold=json.loads((folder/'frozen_thresholds.json').read_text(encoding='utf-8'))['mahalanobis/is_acceptance_95']['threshold']
        pipelines[arm]=(model,tokenizer,detector,threshold,saved_rows(arm))
        results[arm]={'predictions':[],'load_seconds':time.perf_counter()-start}
    print('\nREADY: both models are loaded. Start the screen recording now.',flush=True)
    for case in cases:
        if args.interactive:input('\nPress Enter to run the next input ... ')
        print('\nInput: '+case['text'],flush=True)
        for arm,(model,tokenizer,detector,threshold,records) in pipelines.items():
            batch=tokenizer([case['text']],truncation=True,max_length=64,padding=True,pad_to_multiple_of=8,return_tensors='pt')
            with torch.inference_mode():
                output=model(**{k:v.to(device) for k,v in batch.items()},output_hidden_states=True,return_dict=True)
            features=output.hidden_states[-1][:,0,:].cpu().numpy()
            logits=output.logits.cpu().numpy()
            with threadpool_limits(limits=4):distance=float(detector.score_samples(features)[0])
            prediction=mapping[str(int(logits.argmax(1)[0]))]
            rejected=bool(distance>=threshold)
            saved=records[case['row_id']]
            assert saved['text']==case['text'] and prediction==saved['predicted_label']
            assert rejected==(saved['mahalanobis_is_acceptance_95_reject']=='True')
            row={**case,'prediction':prediction,'rejected':rejected,'distance':distance,
                 'threshold':threshold,'matches_frozen_prediction':True}
            results[arm]['predictions'].append(row)
            print(arm+': '+prediction+'    '+('REJECT' if rejected else 'ACCEPT'),flush=True)
    record={'kind':'actual_model_inference','seed':42,'device':device,'precision':'FP32','training_performed':False,
            'threshold_refit':False,'matches_saved_predictions':True,'cases_are_postselected':True,'results':results}
    (OUT/'demo_inference_results.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('\nPASS: fresh inference agrees with the frozen predictions for all four pipeline/example pairs.',flush=True)

if __name__=='__main__':main()
