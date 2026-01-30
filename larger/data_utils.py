import pandas as pd
from llm_tools_new import (
    init_column_process, 
    judge_user_satisfied, 
    process_absolute_exec, 
    column_process_adjust, 
    Feature_Recognition, 
    Feature_adaption
)
import torch
import fm
import json
import numpy as np
from rdkit import Chem
from scipy import sparse
import os
from larger.preset_method import process_RNA, get_fingerprint

def process_rna_sequence(seq, model, batch_converter, device):
    batch_labels, batch_strs, batch_tokens = batch_converter([('hello', seq)])
    batch_tokens = batch_tokens.to(device)
    results = model(batch_tokens, repr_layers=[12])
    return results["representations"][12].tolist()[0]

def build_rna_feature(row, rna_col, region_col, model, batch_converter, device):
    rna_seq = row[rna_col]
    region_mask = row[region_col]
    output = process_rna_sequence(rna_seq, model, batch_converter, device)
    region_mask = [0] + list(region_mask) + [0]
    enriched_output = [emb + [region_mask[i]] for i, emb in enumerate(output)]
    return enriched_output

def split_smiles(smiles, kekuleSmiles=True):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
    except:
        pass
    splitted_smiles = []
    for j, k in enumerate(smiles):
        if len(smiles) == 1:
            return [smiles]
        if j == 0:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            else:
                splitted_smiles.append(k)
        elif j != 0 and j < len(smiles) - 1:
            if k.isupper() and smiles[j + 1].islower() and smiles[j + 1] != "c":
                splitted_smiles.append(k + smiles[j + 1])
            elif k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
        elif j == len(smiles) - 1:
            if k.islower() and smiles[j - 1].isupper() and k != "c":
                pass
            else:
                splitted_smiles.append(k)
    return splitted_smiles

def get_maxlen(all_smiles, kekuleSmiles=True):
    maxlen = 0
    for smi in all_smiles:
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        maxlen = max(maxlen, len(spt))
    return maxlen

def get_dict(all_smiles, kekuleSmiles=True):
    words = [' ']
    for smi in all_smiles:
        spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
        if spt is None:
            continue
        for w in spt:
            if w in words:
                continue
            else:
                words.append(w)
    return words

def one_hot_coding(smi, words, kekuleSmiles=True, max_len=1000):
    coord_j = []
    coord_k = []
    spt = split_smiles(smi, kekuleSmiles=kekuleSmiles)
    if spt is None:
        return None
    for j, w in enumerate(spt):
        if j >= max_len:
            break
        try:
            k = words.index(w)
        except:
            continue
        coord_j.append(j)
        coord_k.append(k)
    data = np.repeat(1, len(coord_j))
    output = sparse.csr_matrix((data, (coord_j, coord_k)), shape=(max_len, len(words)))
    return output

def ligand2onehot(ligand, words):
    return one_hot_coding(ligand, words, max_len=len(ligand)).toarray().tolist()

def str2list(text):
    return json.loads(text)

def hex_to_fixed_length_list(hex_string):
    return [int(char, 16) for char in hex_string]

def concatenate_lists(row):
    return row['coding_1'] + row['coding_2']

def check_and_process_data(input_path, output_path):
    try:
        print(f"Loading data from {input_path}...")
        data = pd.read_csv(input_path)
        print(f"Loaded {len(data)} rows")

        model, alphabet = fm.pretrained.rna_fm_t12()
        batch_converter = alphabet.get_batch_converter()
        device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
        model.eval()
        model.to(device)

        result = {}
        result['RNA'] = 'rna_sequence'
        result['region'] = 'region_mask'
        result['ligand'] = 'smile'
        result['label'] = 'label'
        result['explain'] = 'nothing'
        
        data[result['region']] = data[result['region']].apply(str2list)
        
        data['rna_feature'] = data.apply(
            lambda row: build_rna_feature(
                row,
                rna_col=result['RNA'],
                region_col=result['region'],
                model=model,
                batch_converter=batch_converter,
                device=device
            ),
            axis=1
        )
        data['rna_feature'] = data['rna_feature'].apply(np.array)

        words = get_dict(data[result['ligand']].tolist())
        data['ligand_feature'] = data[result['ligand']].apply(lambda x: ligand2onehot(x, words))
        data['ligand_feature'] = data['ligand_feature'].apply(np.array)

        print(f"Data processing completed!")
        
        data.to_csv(output_path, index=False)
        
        return data, ['rna_feature', 'ligand_feature'], result['label']

    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def column_process(column, data, new_column_name):
    plan = init_column_process(data[column], new_column_name)

    while True:
        description = plan['description']
        code = plan['code']
        print(f"\nProcessing plan for column '{column}':")
        print(description)
        
        asking = "\n[Is it reasonable? (yes/no)] → "
        response = input(asking).strip()
        
        if judge_user_satisfied(response, asking):
            data = process_absolute_exec(code, description, new_column_name, data)
            print(f"Column '{new_column_name}' created successfully!")
            break
        else:
            print("Adjusting the processing plan...")
            plan = column_process_adjust(data[column], code, description, new_column_name)
               
    return data
