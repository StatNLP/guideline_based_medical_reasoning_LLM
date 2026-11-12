import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Sampler,SequentialSampler,RandomSampler
from transformers.models.t5.modeling_t5 import T5Block

from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
 checkpoint_wrapper,
 CheckpointImpl)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap, size_based_auto_wrap_policy,
    wrap,
)

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime
import pickle
import transformers
tokenizer = None
import importlib
from datasets import load_dataset

import torch

from pathlib import Path
import random
import pandas as pd
pd.set_option('display.max_columns', 500)
import re

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
   apply_activation_checkpointing,
)


from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

print(f'current device: { torch.cuda.current_device()}')

print(f'available devices: {torch.cuda.device_count()}')




d_model, e_layers, d_layers, d_ff = 512, 2, 1, 2048
number_of_epochs = 100
V, D = 131, 2
fore_max_len = 2640
print('loading of joblib completed')



import argparse
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                            help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                            help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                            help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 

parser.add_argument('--embed_type', type=int, default=3, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=V*2, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=V, help='decoder input size')
parser.add_argument('--c_out', type=int, default=V, help='output size')
parser.add_argument('--d_model', type=int, default=d_model, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=e_layers, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=d_layers, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=d_ff, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                            help='whether to use distilling in encoder, using this argument means not using distilling',
                                                default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args(args=[])



import models.InformerAutoregressiveFullExtract as autoformer
importlib.reload(autoformer)
batch_size, lr, patience = 32, 0.0005, 6  # batch_size increased, patience 10 --> 6
d, N, he, dropout = 50, 2, 4, 0.2
forecasting_model = autoformer.Model(args).to('cuda', dtype=torch.bfloat16)
# Load pretrained weights here.
forecasting_model.load_state_dict(torch.load("pretrained_informer.model"))


var_to_ind= {'ALP': 1, 'ALT': 2, 'AST': 3, 'Albumin': 4, 'Albumin 25%': 5, 'Albumin 5%': 6,
              'Amiodarone': 7, 'Anion Gap': 8, 'Antibiotics': 9, 'BUN': 10, 'Base Excess': 11,
              'Basophils': 12, 'Bicarbonate': 13, 'Bilirubin (Direct)': 14, 'Bilirubin (Indirect)': 15,
              'Bilirubin (Total)': 16, 'CRR': 17, 'Calcium Free': 18, 'Calcium Gluconate': 19,
              'Calcium Total': 20, 'Cefazolin': 21, 'Chest Tube': 22, 'Chloride': 23, 'Colloid': 24,
              'Creatinine Blood': 25, 'Creatinine Urine': 26, 'D5W': 27, 'DBP': 28, 'Dextrose Other': 29,
              'Dobutamine': 30, 'Dopamine': 31, 'EBL': 32, 'Emesis': 33, 'Eoisinophils': 34,
              'Epinephrine': 35, 'Famotidine': 36, 'Fentanyl': 37, 'FiO2': 38, 'Fiber': 39,
              'Free Water': 40, 'Fresh Frozen Plasma': 41, 'Furosemide': 42, 'GCS_eye': 43,
              'GCS_motor': 44, 'GCS_verbal': 45, 'GT Flush': 46, 'Gastric': 47, 'Gastric Meds': 48,
              'Glucose (Blood)': 49, 'Glucose (Serum)': 50, 'Glucose (Whole Blood)': 51,
              'HR': 52, 'Half Normal Saline': 53, 'Hct': 54, 'Heparin': 55, 'Hgb': 56,
              'Hydralazine': 57, 'Hydromorphone': 58, 'INR': 59, 'Insulin Humalog': 60,
              'Insulin NPH': 61, 'Insulin Regular': 62, 'Insulin largine': 63,
              'Intubated': 64, 'Jackson-Pratt': 65, 'KCl': 66, 'KCl (Bolus)': 67,
              'LDH': 68, 'Lactate': 69, 'Lactated Ringers': 70, 'Levofloxacin': 71,
              'Lorazepam': 72, 'Lymphocytes': 73, 'Lymphocytes (Absolute)': 74,
              'MBP': 75, 'MCH': 76, 'MCHC': 77, 'MCV': 78, 'Magnesium': 79,
              'Magnesium Sulfate (Bolus)': 80,  'Magnesium Sulphate': 81,
              'Mechanically ventilated': 82, 'Metoprolol': 83, 'Midazolam': 84,
              'Milrinone': 85, 'Monocytes': 86, 'Morphine Sulfate': 87,
              'Neosynephrine': 88, 'Neutrophils': 89, 'Nitroglycerine': 90,
              'Nitroprusside': 91, 'Norepinephrine': 92, 'Normal Saline': 93,
              'O2 Saturation': 94, 'OR/PACU Crystalloid': 95, 'PCO2': 96,
              'PO intake': 97, 'PO2': 98, 'PT': 99, 'PTT': 100, 'Packed RBC': 101,
              'Pantoprazole': 102, 'Phosphate': 103, 'Piggyback': 104, 'Piperacillin': 105,
              'Platelet Count': 106, 'Potassium': 107, 'Pre-admission Intake': 108,
              'Pre-admission Output': 109, 'Propofol': 110, 'RBC': 111, 'RDW': 112,
              'RR': 113, 'Residual': 114, 'SBP': 115, 'SG Urine': 116, 'Sodium': 117,
              'Solution': 118, 'Sterile Water': 119, 'Stool': 120, 'TPN': 121,
              'Temperature': 122, 'Total CO2': 123, 'Ultrafiltrate': 124, 'Urine': 125,
              'Vancomycin': 126, 'Vasopressin': 127, 'WBC': 128, 'Weight': 129,
              'pH Blood': 130, 'pH Urine': 131}

limited_variables = {"GCS_eye", "GCS_motor", "GCS_verbal", "Bilirubin (Total)", "Platelet Count", "Urine", "Creatinine Blood", "FiO2", "PO2", "Weight", "Dopamine", "Dobutamine", "Epinephrine", "Norepinephrine", "SBP", "DBP"}
import math
import copy
from copy import deepcopy


class MimicDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, stage, target_note_list = ["synthetic_sepsis_label"], include_text_of_same_type=False, include_measurements_in_text=True, max_len=3500, encoder_model = forecasting_model):
        with open("mimic_iii_preprocessed_synthetic.pkl", "rb") as infile:
            self.df, oc, self.train_ind, self.val_ind, self.test_ind = pickle.load(infile)
        self.tokenizer = tokenizer
        self.encoder_model = encoder_model
        self.offset = -0
        self.split_target = True # Implement That
        self.use_susinf = False
        if stage=="train":
            self.df=self.df[self.df.ts_ind.isin(self.train_ind)]
            if self.use_susinf:
                self.get_positive_susinfs = self.df[self.df.hour < 24]
                self.susinfs = self.get_positive_susinfs[self.get_positive_susinfs.variable=="suspected-infection"]
                self.susinfs = self.susinfs[self.susinfs.value==1.0]
                self.new_indices = self.susinfs.ts_ind
                self.df = self.df[self.df.ts_ind.isin(self.new_indices)]
            self.df=self.df.reset_index(drop=True)
            self.target_texts = self.df[self.df.variable.isin(target_note_list)]
            self.target_texts = self.target_texts[self.target_texts.hour > abs(self.offset)]
        if stage=="val":
            self.df=self.df[self.df.ts_ind.isin(self.val_ind)]
            if self.use_susinf:
                self.get_positive_susinfs = self.df[self.df.hour < 24]
                self.susinfs = self.get_positive_susinfs[self.get_positive_susinfs.variable=="suspected-infection"]
                self.susinfs = self.susinfs[self.susinfs.value==1.0]
                self.new_indices = self.susinfs.ts_ind
                self.df = self.df[self.df.ts_ind.isin(self.new_indices)]
            self.df=self.df.reset_index(drop=True)
            self.target_texts = self.df[self.df.variable.isin(target_note_list)]
            self.target_texts = self.target_texts[self.target_texts.hour > abs(self.offset)]
        if stage=="test":
            self.df=self.df[self.df.ts_ind.isin(self.test_ind)]
            if self.use_susinf:
                self.get_positive_susinfs = self.df[self.df.hour < 24]
                self.susinfs = self.get_positive_susinfs[self.get_positive_susinfs.variable=="suspected-infection"]
                self.susinfs = self.susinfs[self.susinfs.value==1.0]
                self.new_indices = self.susinfs.ts_ind
                self.df = self.df[self.df.ts_ind.isin(self.new_indices)]
            self.df=self.df.reset_index(drop=True)
            self.target_texts = self.df[self.df.variable.isin(target_note_list)]
            self.target_texts = self.target_texts[self.target_texts.hour > abs(self.offset)]
        self.cache = dict()
        self.caching = True
        self.target_note_list=target_note_list
        self.include_text_of_same_type=include_text_of_same_type
        self.max_len = max_len
        self.include_measurements_in_text = include_measurements_in_text
        #print(self.target_texts)
        self.len_dict = {"train": 15000, "val":3000, "test": 3000}
        self.target_texts = self.target_texts.sample(n=self.len_dict[stage], random_state=1)#.reset_index(drop=True)
        self.mean_std_dict = dict()
    def cache_exists(self, inputdir):
        return os.path.exists(inputdir)
    def save_cache(self, outputdir):
        with open(outputdir, "wb") as outfile:
            pickle.dump(self.cache, outfile)
    def load_cache(self, inputdir):
        with open(inputdir, "rb") as infile:
            self.cache=pickle.load(infile)
    def precompute_cache(self):
        for i in tqdm.tqdm(range(len(self.index))):
            self[i]
    def __getitem__(self, idx):
        if idx in self.cache and self.caching:
            return self.cache[idx]
        datapoint = self.df.loc[self.target_texts.index[idx]]
        if self.offset < -10:
            datapoint_24hours_before = self.df[(self.df.hour < datapoint.hour-24+0.01) & (self.df.hour > datapoint.hour-24-0.01)]
            same_datapoint_24hours_before = datapoint_24hours_before[datapoint_24hours_before.variable == datapoint.variable]
            #print("WOOOP", same_datapoint_24hours_before)
        other_datapoints = self.df[self.df.ts_ind==datapoint.ts_ind]
        in_time_datapoints = other_datapoints[(other_datapoints.hour > datapoint.hour - 24 + self.offset) & (other_datapoints.hour < datapoint.hour + self.offset)]
        age_data = (other_datapoints[other_datapoints.variable=="Age"])
        if len(age_data) > 0:
            age = age_data.iloc[0].value
        else:
            age = 60
        gender_data = (other_datapoints[other_datapoints.variable=="Gender"])
        if len(gender_data)>0:
            gender = gender_data.iloc[0].value
        else:
            gender = 0
        start_string = "Patient is {} years old and is {}. Given all the information in this text, answer the question at the end.".format(age, "female" if gender else "male")
        textual_information_pre = "\nHere are the clinical texts: "
        measurement_information_pre = "\nHere are the measurements: "
        textual_information = []
        measurement_information = []
        current_time = datapoint.hour
        final_part = "\nNow answer the following question: \n"
        ts = torch.zeros((24, 262))
        for information in in_time_datapoints[::-1].itertuples():
            old_textual_information = deepcopy(textual_information)
            old_measurement_information = deepcopy(measurement_information)
            if information.TABLE == "TEXT" and information.variable in self.target_note_list and self.include_text_of_same_type:
                textual_information+=["Clinical Note: \n" + information.textvalue]
            else:
                #print("VARIABLE", information)
                if information.variable in limited_variables:
                    if self.include_measurements_in_text:
                        measurement_information+=[information.variable +" at time {}: ".format(round(information.hour-current_time+abs(self.offset), 2)) + str(round(information.value * information.std + information.mean, 3))]
                if information.variable in var_to_ind:
                    time = math.floor(information.hour-current_time+abs(self.offset))
                    index = var_to_ind[information.variable]-1
                    ts[time, index] = information.value
                    ts[time, index+131] = 1
                    self.mean_std_dict[information.variable] = (information.mean, information.std)
            current_text = start_string + ("\n".join(textual_information[::-1]) if self.include_text_of_same_type else "") + (", ".join(measurement_information[::-1]) if self.include_measurements_in_text else "") + final_part
            if len(self.tokenizer(current_text)["input_ids"]) > self.max_len:
                textual_information = old_textual_information
                measurement_information = old_measurement_information
                #break
        input_matrix = ts
        input_mark = torch.arange(0, input_matrix.size(1)).unsqueeze(0).repeat(input_matrix.size(0), 1).cuda()
        dec_inp = torch.zeros_like(input_matrix[:, :])
        output = self.encoder_model(torch.tensor(ts).unsqueeze(0).to("cuda", dtype=torch.bfloat16), input_mark, dec_inp, input_mark)
        string = ""
        for variable in limited_variables:
            if variable not in self.mean_std_dict: continue
            row = output[0, :, var_to_ind[variable]].cpu()
            mini, maxi, sumi = row.min().item(), row.max().item(), (row * self.mean_std_dict[variable][1] + self.mean_std_dict[variable][0]).sum().item()
            string+="minimum {} will be {},".format(variable, mini*self.mean_std_dict[variable][1]+self.mean_std_dict[variable][0])
            string+="maximum {} will be {},".format(variable, maxi*self.mean_std_dict[variable][1]+self.mean_std_dict[variable][0])
            string+="sum of {} will be {},".format(variable, sumi)

        current_text = start_string + (textual_information_pre + "\n".join(textual_information[::-1]) if self.include_text_of_same_type else "") + ( measurement_information_pre+", ".join(measurement_information[::-1]) if self.include_measurements_in_text else "") + "\nAnd a forecasting model forecasted the following values: " +string[:-1] + final_part
        target_text = datapoint.textvalue

        complete_text = current_text + target_text
        item = self.tokenizer(complete_text+self.tokenizer.eos_token, return_length=True)
        if self.split_target:
            target_start = target_text.split("?")[0]+"?"
            item2 = self.tokenizer(current_text+target_start)
        else:

            item2 = self.tokenizer(current_text)
        target_labels = copy.deepcopy(item.input_ids)
        for i in range(len(item2.input_ids)):
            target_labels[i]=-100
        item["labels"]=target_labels
        item["size_diff"]=len(item.input_ids) - len(item2.input_ids)
        item["ts"] = ts

        if self.offset < -10:
            item["previousnote"] = same_datapoint_24hours_before.iloc[0].textvalue
        if self.caching:
            self.cache[idx]=item
        return item

    def __len__(self):
        return len(self.target_texts)







def setup():
    # initialize the process group
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()


def setup_model(model_name):
    global tokenizer
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    load_4bit_use_double_quant=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.float16,
    #)
    #model = T5ForConditionalGeneration.from_pretrained(model_name)
    #tokenizer =  T5Tokenizer.from_pretrained(model_name)
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model_name, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")#, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]

    config = LoraConfig(

    r=16, lora_alpha=16, target_modules="all-linear", lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"

        )
    lora_model = get_peft_model(model, config)
    #lora_model = model
    return lora_model, tokenizer


def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

def format_metrics_to_gb(item):
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 1024**3
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    local_rank = 0#int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(2).cuda()


    if sampler:
        pass#sampler.set_epoch(epoch)
    if rank==0:
        inner_pbar = tqdm.tqdm(
            range(len(train_loader)), colour="blue", desc="r0 Training Epoch"
        )
    counter=0
    optimizer.zero_grad()
    for batch in train_loader:
        #print(batch)
        for key in batch.keys():
            #batch[key] = batch[key].to(local_rank)
            batch[key].cuda()
        #optimizer.zero_grad()
        #output = model(input_ids=batch["source_ids"],attention_mask=batch["source_mask"] )
        output = model(input_ids=batch["input_ids"].cuda(), labels=batch["labels"].cuda(), attention_mask=batch["attention_mask"].cuda())
        loss = output["loss"] / args.batch_multiplier
        loss.backward()
        counter+=1
        if counter%args.batch_multiplier == 0:

            optimizer.step()
            optimizer.zero_grad()
        fsdp_loss[0] += loss.cpu().item()
        fsdp_loss[1] += len(batch)
        if rank==0:
            inner_pbar.update(1)

    train_accuracy = fsdp_loss[0] / fsdp_loss[1]


    if rank == 0:
        inner_pbar.close()
        print(
                f"Train Epoch: \t{epoch}, Loss: \t{train_accuracy:.4f}"
            )
    return train_accuracy


def validation(model, rank, world_size, val_loader):
    model.eval()
    correct = 0
    local_rank = 0#int(os.environ['LOCAL_RANK'])
    fsdp_loss = torch.zeros(3).cuda()
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Validation Epoch"
        )
    if True:#with FSDP.summon_full_params(model):
        with torch.no_grad():
            for batch in val_loader:
                for key in batch.keys():
                    #batch[key] = batch[key].to(local_rank)
                    batch[key].cuda()
                output = model(input_ids=batch["input_ids"].cuda(),labels=batch["labels"].cuda(), attention_mask=batch["attention_mask"].cuda())
                fsdp_loss[0] += output["loss"].item()  # sum up batch loss
                fsdp_loss[1] += len(batch)
                #model.generate(**input,synced_gpus=True)
                if rank==0:
                    inner_pbar.update(1)
    val_loss = fsdp_loss[0] / fsdp_loss[1]
    if rank == 0:
        inner_pbar.close()
        print(f"Validation Loss: {val_loss:.4f}")
    return val_loss

def test(model, rank, world_size, val_loader, filename):
    model.eval()
    fsdp_loss = torch.zeros(3).cuda()
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Testing Epoch"
        )
    collection = list()
    if True:#with FSDP.summon_full_params(model):
        with torch.no_grad():
            #print("here 1")
            for batch in val_loader:
                #print("here 2")
                for key in batch.keys():
                    #batch[key] = batch[key].to(local_rank)
                    try:
                        batch[key].cuda()
                    except: continue
                #output = model(input_ids=batch["input_ids"])
                #model.generate(**input,synced_gpus=True)
                max_len = max([len(x) for x in batch["labels"]])
                #print("here 3", str(max_len))
                output = model.generate(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda(), max_new_tokens=max_len+100)
                for inputs, outputs, gold in zip(batch["input_ids"], output, batch["labels"]):
                    
                    print("generated_stuff", outputs[len(inputs):])
                    result = {"input": tokenizer.decode(inputs), "output": tokenizer.decode(outputs[len(inputs):]), "gold": tokenizer.decode(gold) }
                    print(result)
                    collection.append(result)
                if rank==0:
                    inner_pbar.update(1)
    if rank == 0:
        inner_pbar.close()
        #print(f"Validation Loss: {val_loss:.4f}")
    with open(".".join(filename.split(".")[:-1])+".generations", "wb") as outfile:
        pickle.dump(collection, outfile)
    return 0


def force_decoding(model, rank, world_size, val_loader, filename, tokenizer):
    model.eval()
    fsdp_loss = torch.zeros(3).cuda()
    if rank == 0:
        inner_pbar = tqdm.tqdm(
            range(len(val_loader)), colour="green", desc="Testing Epoch"
        )
    collection = list()
    if True:
        with torch.no_grad():
            for batch in val_loader:
                for key in batch.keys():
                    try:
                        batch[key].cuda()
                    except: continue
                max_len = max([len(x) for x in batch["labels"]])
                print(batch["labels"])
                tokens = tokenizer.convert_ids_to_tokens(batch["labels"][0])
                i=0
                continueing = False
                spans = []
                span = []
                while True:
                    print(i)
                    if i >= len(tokens): break
                    if re.match("\d+", tokens[i]) and continueing:
                        pass#continue
                    elif re.match("\.", tokens[i]) and continueing:
                        pass#continue
                    elif continueing:
                        continueing = False
                        span.append(i)
                        spans.append(span)
                        span = []
                    elif re.match("\d+", tokens[i]):
                        span.append(i)
                        continueing = True
                    i+=1
                if span:
                    spans.append(span)
                for idx in [31, 38, 46, 48, 50, 53, 54, -2]:
                    print(idx, len(spans))
                    print(spans)
                    print("blub", spans[idx][1])
                    gold_parts = torch.tensor(batch["labels"][0][:spans[idx][1]], dtype=torch.long).unsqueeze(0)
                    gold_mask = torch.ones((1,spans[idx][1]), dtype=torch.long) 
                    output = model.generate(input_ids=torch.cat([batch["input_ids"], gold_parts], dim=1).cuda(), attention_mask=torch.cat([batch["attention_mask"], gold_mask], dim=1).cuda(), max_new_tokens=(spans[idx+1][1]-spans[idx][1]+5))
                    for inputs, outputs, gold in zip(batch["input_ids"], output, batch["labels"]):
                        result = {"level":idx, "input": tokenizer.decode(torch.cat([inputs,torch.tensor(gold[:spans[idx][1]])], dim=0)), "output": tokenizer.decode(outputs[len(inputs)+spans[idx][1]:]), "gold": tokenizer.decode(gold[spans[idx][1]:spans[idx+1][1]+5]) }
                        print(result)
                        collection.append(result)
                if rank==0:
                    inner_pbar.update(1)
    if rank == 0:
        inner_pbar.close()
        #print(f"Validation Loss: {val_loss:.4f}")
    with open(".".join(filename.split(".")[:-1])+".forced", "wb") as outfile:
        pickle.dump(collection, outfile)
    return 0



def collate_fn_partial(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    
    max_len = max([a-b for x in data for a,b in zip(x["length"], [x["size_diff"]])])
    global tokenizer
    input_ids_batched = torch.ones((len(data), max_len), dtype=torch.long) * tokenizer(tokenizer.eos_token, add_special_tokens=False).input_ids[-1]
    labels_batched = []
    attention_masks_batched = torch.zeros((len(data), max_len), dtype=torch.long)
    for i, datapoint in enumerate(data):
        true_input = datapoint["input_ids"][:len(datapoint["input_ids"])-datapoint["size_diff"]]
        true_output = datapoint["input_ids"][len(true_input):]
        input_ids_batched[i, -len(true_input):] = torch.tensor(true_input)
        attention_masks_batched[i, -len(true_input):] = 1
        labels_batched.append(true_output)
    lengths = torch.stack([torch.tensor(x["length"])-torch.tensor(x["size_diff"]) for x in data])
    return {"input_ids":input_ids_batched, "attention_mask": attention_masks_batched, "labels": labels_batched, "lengths":lengths}

def collate_fn_full(data):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """
    
    max_len = max([y for x in data for y in x["length"]])
    global tokenizer
    input_ids_batched = torch.ones((len(data), max_len), dtype=torch.long) * tokenizer(tokenizer.eos_token, add_special_tokens=False).input_ids[-1]
    labels_batched = torch.ones((len(data), max_len), dtype=torch.long) * -100
    attention_masks_batched = torch.zeros((len(data), max_len), dtype=torch.long)
    for i, datapoint in enumerate(data):
        input_ids_batched[i, -len(datapoint["input_ids"]):] = torch.tensor(datapoint["input_ids"])
        attention_masks_batched[i, -len(datapoint["attention_mask"]):] = torch.tensor(datapoint["attention_mask"])
        labels_batched[i, -len(datapoint["labels"]):] = torch.tensor(datapoint["labels"])
    target_length = torch.stack([torch.tensor(x["size_diff"]) for x in data])
    lengths = torch.stack([torch.tensor(x["length"]) for x in data])

    return {"input_ids":input_ids_batched, "attention_mask": attention_masks_batched, "labels": labels_batched, "lengths":lengths }

def fsdp_main(args):

    #model, tokenizer = setup_model("meta-llama/Llama-3.2-3B")
    #model, tokenizer = setup_model("gradientai/Llama-3-8B-Instruct-Gradient-1048k")

    model, tokenizer = setup_model("meta-llama/Llama-3.1-8B-Instruct")
    local_rank = 0#int(os.environ['LOCAL_RANK'])
    rank = 0#int(os.environ['RANK'])
    world_size = 1#int(os.environ['WORLD_SIZE'])


    #wikihow(tokenizer, type_path, num_samples, input_length, output_length, print_text=False)
    train_dataset = MimicDataset(tokenizer, "train") 
    val_dataset = MimicDataset(tokenizer, "val") 
    test_dataset = MimicDataset(tokenizer, "test")
    """
    if not train_dataset.cache_exists("train_exp1.cache"):
        train_dataset.precompute_cache()
        train_dataset.save_cache("train_exp1.cache")
    else:
        train_dataset.load_cache("train_exp1.cache")
    if not val_dataset.cache_exists("val_exp1.cache"):
        val_dataset.precompute_cache()
        val_dataset.save_cache("val_exp1.cache")
    else:
        val_dataset.load_cache("val_exp1.cache")
    if not test_dataset.cache_exists("test_exp1.cache"):
        test_dataset.precompute_cache()
        test_dataset.save_cache("test_exp1.cache")
    else:
        test_dataset.load_cache("test_exp1.cache")
    """
    sampler1 = RandomSampler(train_dataset)
    sampler2 = SequentialSampler(val_dataset)



    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False}
    #train_kwargs.update(cuda_kwargs)
    #test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,collate_fn=collate_fn_full,**train_kwargs)
    val_loader_full = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn_full,**test_kwargs)
    val_loader_partial = torch.utils.data.DataLoader(val_dataset, collate_fn=collate_fn_partial,**test_kwargs)

    test_loader_full = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn_full,**test_kwargs)
    test_loader_partial = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn_partial,**test_kwargs)

    force_kwargs = {'batch_size': 1, 'sampler': sampler2}
    test_loader_force = torch.utils.data.DataLoader(test_dataset, collate_fn=collate_fn_partial,**force_kwargs)

    #t5_auto_wrap_policy = functools.partial(
    #    transformer_auto_wrap_policy,
    #    transformer_layer_cls={
    #        T5Block,
    #    },
    #)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=20000
        )

    my_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    #torch.cuda.set_device(local_rank)


    #init_start_event = torch.cuda.Event(enable_timing=True)
    #init_end_event = torch.cuda.Event(enable_timing=True)

    #init_start_event.record()

    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and dist.is_nccl_available()
    )
    print("bf16 ready", bf16_ready)
    mp_policy = None # defaults to fp32
    fpSixteen = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
         reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )
    model.cuda()
    #fpSixteen=None 
    # model is on CPU before input to FSDP
    """
    model = FSDP(model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=fpSixteen,
        #sharding_strategy=sharding_strategy,
        cpu_offload=CPUOffload(offload_params=True),
        device_id=torch.cuda.current_device(), use_orig_params=True)
    #print(model)

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    """
    check_fn = lambda submodule: isinstance(submodule, LlamaDecoderLayer)
    #apply_activation_checkpointing(
    #       model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    #)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "pretrain_textualisation-reduced-pipeline-"

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    if args.load_path:
        model.load_state_dict(torch.load(args.load_path))
    force_decoding(model, rank, world_size, test_loader_force, args.load_path, tokenizer)
    test(model, rank, world_size, test_loader_partial, args.load_path)
    sys.exit(0)

    curr_val_loss = validation(model, rank, world_size, val_loader_full)
    cpu_state = model.state_dict()

    epoch = "start" 
    if rank == 0:
        currEpoch = (
             "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
        )
        save_name = file_save_name + "-" + time_of_run + "-" + str(args.lr) + "-" + currEpoch

        torch.save(cpu_state, save_name)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader_full)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )
            print(f"completed save and stats zone...")

        if args.save_model and curr_val_loss < best_val_loss:

            # save
            if rank == 0:
                print(f"--> entering save model state")

            cpu_state = model.state_dict()
            #print(f"saving process: rank {rank}  done w state_dict")


            if rank == 0:
                print(f"--> saving model ...")
                currEpoch = (
                    "-" + str(epoch) + "-" + str(round(curr_val_loss.item(), 4)) + ".pt"
                )
                print(f"--> attempting to save model prefix {currEpoch}")
                #save_name = file_save_name + "-" + time_of_run + "-" + currEpoch
                save_name = file_save_name + "-" + time_of_run + "-" + str(args.lr) + "-" + currEpoch
                print(f"--> saving as model name {save_name}")

                torch.save(cpu_state, save_name)

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 FSDP Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--lr', type=float, default=.000002, metavar='LR',
                        help='learning rate (default: .002)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--save-model', action='store_false', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--batch_multiplier', type=int, default=1,
                        help='batch multiplier (default: 1)')
    parser.add_argument("--load_path", type=str, default="", help="which path to load (default="")")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)



