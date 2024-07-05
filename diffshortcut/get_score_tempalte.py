# %%
import wandb 
api = wandb.Api()

# %%
import os 
import numpy as np 
import pickle as pkl
def all_reduce_metrics(runs, value =False, instance_num = None):
    prompt2scoredict_dict = {}
    for run in runs:
        if 'restult_artifact_name' not in run.summary:
            continue
        art_name=  run.summary['restult_artifact_name']
        artifact = api.artifact(f'/diffshortcut/'+art_name+":latest")
        if not os.path.exists(artifact.file()):
            artifact.download()
    
        with open(artifact.file(), 'rb') as f:
            file = pkl.load(f)
        prompt2scoredict_dict[run.config['instance_name']] = file['propmt2score']
    reducer = {}


    if instance_num is not None:
        prompt2scoredict_dict = {k: prompt2scoredict_dict[k] for k in list(prompt2scoredict_dict.keys())[:instance_num]}
        
    for instance in prompt2scoredict_dict:
        for prompt in prompt2scoredict_dict[instance]:
            for metric in prompt2scoredict_dict[instance][prompt]:
                if metric not in reducer:
                    reducer[metric] = []
                reducer[metric]+=prompt2scoredict_dict[instance][prompt][metric]
    print(f'all reduce over {len(prompt2scoredict_dict)} instances')
    if value:
        return {k: np.mean(v) for k,v in reducer.items()}, len(prompt2scoredict_dict)
    return reducer, len(prompt2scoredict_dict)

def reduce_over_one_variable(runs, control = 'note', metric_subset=None, instance_num = None):
    runs_note_unique_list = list(set([run.config[control] for run in runs]))
    res_over_control = {k:{} for k in runs_note_unique_list}
    runs_over_control = {k:[] for k in runs_note_unique_list}
    for run in runs:
        runs_over_control[run.config[control]].append(run)
    for k in runs_note_unique_list:
        print(k)
        res_over_control[k], instance_num_real = all_reduce_metrics(runs_over_control[k], value = True, instance_num = instance_num)
        res_over_control[k]['instance_num'] = instance_num_real
    if metric_subset is not None:
        metric_subset.append('instance_num')
        tmp_res = {
            k: {metric: res_over_control[k][metric] for metric in metric_subset} for k in res_over_control
        }
    else:
        tmp_res = res_over_control
    return tmp_res
    
    
def wrap_with_exp_name(exp_name, res_dict):
    return {exp_name+"-" + k: v for k,v in res_dict.items()}
def dict2df(dictt):
    import pandas as pd
    df = pd.DataFrame(dictt)
    return df

# %%
metrics = [
    'IMS_VGG-Face_cosine',
    'IMS_IP_embed',
    'LIQE_Quality',
    # 'LIQE_Scene_Human',
    'CLIP_IQAC',
    # 'IMS_CLIP_ViT-B/32',
    'SDS',
]

# %%
exp_names=['setting1']
dataset_name="VGGFace2-clean"
res_merge = {}
for exp_name in exp_names:
    runs = api.runs("/diffshortcut", {"State": "finished",'config.dataset_name':dataset_name, "config.exp_name": exp_name, 'config.status': 'score', })
    runs_all_note_unique = set([run.config['note'] for run in runs])
    res = reduce_over_one_variable(runs, control='note',  metric_subset=metrics,  instance_num=4)
    wrap_res = wrap_with_exp_name(exp_name, res)
    res_merge.update(wrap_res)
# !pip install pandas 

# %%


# %%
import pandas as pd 
df = dict2df(res_merge)
df = df.T
df['IMS'] = (df['IMS_IP_embed'] * 0.7 + df['IMS_VGG-Face_cosine'] * 0.3) 
# renormalize the LIQE_Quality
df['Q'] = ((((df['LIQE_Quality'] - 1) / 4) * 2 - 1 )/2+ df['CLIP_IQAC'] /2 )
df['overall_score'] = df['IMS']*0.5 + df['Q'] * 0.5


# clean up your df and save it