import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from scipy.stats import t
import torch
from utils import set_torch_device

""" Data loading and processing functions for the Leuven dataset."""

def make_training_data(num_objects: int = 350, num_outputs: int = 2541+2+3+350,
                       num_tasks: int = 36, device=None) -> list:
    """
    Make training data for the Leuven dataset.

    Parameters
    ----------
    num_objects : int, optional
        Number of objects to include in the training data. Default: 350.
    num_outputs : int, optional
        Number of outputs to include in the training data. Default: 2541+2+3+350.
    num_tasks : int, optional
        Number of tasks to include in the training data. Default: 36.
    device : torch.device, optional
        Device to use for the training data. Default: None.
    
    Returns
    -------
    list
        List of training data tensors. Shape [(train_in, train_context), train_out]
    """
    leuven_data = load_leuven()
    object_names = pd.read_csv('data/leuven_size.csv').name.values
    in_, out_, context_ = make_input(leuven_data, object_names, num_objects, num_outputs, num_tasks, device=device)
    return [in_,context_], out_


def load_feature_reps() -> list:
    """ Load feature representations for the Leuven dataset. """
    return pickle.load(open('data/object_features_in_context.pkl','rb'))


def get_raw_feature_values() -> np.array:
    """
    Get raw feature values for the Leuven dataset for each context.
    
    Returns
    -------
    np.array
        Array of raw feature values. Shape [num_contexts, num_objects, num_features]
    """
    leuven_data = load_leuven()
    object_names = pd.read_csv('data/leuven_size.csv').name.values
    in_, out_, context_ = make_input(leuven_data, object_names)
    in_, out_, context_ = in_.cpu().detach().numpy(), out_.cpu().detach().numpy(), context_.cpu().detach().numpy()
    dense_reps = np.zeros((context_.shape[1],in_.shape[1],out_.shape[1]))
    for i in range(len(in_)):
        in_idx, task_idx = in_[i].argmax(), context_[i].argmax()
        dense_reps[task_idx,in_idx] = out_[i]
    return dense_reps


def get_item_indices_by_category(category_list: str | list) -> np.array:
    """
    Get indices of objects in the Leuven dataset by category.

    Parameters
    ----------
    category_list : str or list
        Category or list of categories to get indices for.
    
    Returns
    -------
    np.array
        Array of indices of objects in the Leuven dataset.
    """
    if type(category_list) is str:
        category_list = [category_list]
    objects = pd.read_csv('data/leuven_size.csv')[['name','category','size_for_animal_instrument_dataset']]
    object_idxs = objects[objects.category.isin(category_list)].index.values
    return object_idxs


def get_item_indices_by_category_and_size(category_list: str | list, size: str | int) -> np.array:
    """
    Get indices of objects in the Leuven dataset by category and size.

    Parameters
    ----------
    category_list : str or list
        Category or list of categories to get indices for.
    size : str or int
        Size or size index to get indices for.
    
    Returns
    -------
    np.array
        Array of indices of objects in the Leuven dataset.
    """
    if type(category_list) is str:
        category_list = [category_list]
    if type(size) is str:
        size_idx = 0 if size=='small' else 1
    else:
        size_idx = size
    objects = pd.read_csv('data/leuven_size.csv')[['name','category','size_for_animal_instrument_dataset']]
    object_idxs = objects[objects.category.isin(category_list)&(objects.size_for_animal_instrument_dataset==size_idx)].index.values
    return object_idxs


""" Helper functions for loading and processing the Leuven dataset."""

def load_leuven() -> dict:
    """ Load the Leuven dataset. """
    patterns = {}
    leuven_size_labels = pd.read_csv('data/leuven_size.csv')
    leuven_size_labels = leuven_size_labels[leuven_size_labels.name!='viger']
    object_list = list(leuven_size_labels.name.values)
    category_list = sorted(list(set(leuven_size_labels.category.values)))
    category_list = ['reptile','tool','weapon','vehicle','clothing','fruit','mammal','fish',
     'instrument','vegetable','insect','bird','kitchen']
    category_map_= {'reptile':'animal',
                    'tool':'other',
                    'weapon':'other',
                    'vehicle':'other',
                    'clothing':'other',
                    'fruit':'other',
                    'mammal':'animal',
                    'fish':'animal',
                    'instrument':'instrument',
                    'vegetable':'other',
                    'insect':'other',
                    'bird':'animal',
                    'kitchen':'other'}
    with open('data/leuven_rum_nozeroes.ex') as f:
        idx = 0
        for line in f:
            obj,out_context = line.split()[1].split('C')
            if obj == 'viger':
                continue
            in_pat = [object_list.index(obj)]
            out_pat = [int(x) for x in line.split('t:')[1].strip().split()[:-1]]
            in_context = category_list.index(leuven_size_labels[leuven_size_labels.name==obj]['category'].values[0])
            context = int(out_context)-1

            if obj in patterns:
                patterns[obj]['context'].append([context])
                patterns[obj]['in_pat'].append(in_pat)
                patterns[obj]['out_pat'].append(out_pat)
            else:
                patterns[obj] = {'context':[[context]],'out_pat':[out_pat],'in_pat':[in_pat]}


    for i in range(len(leuven_size_labels)):
        try:
            obj = leuven_size_labels.at[i,'name']
        except:
            continue
        if obj=='viger':
            continue
        #size=int(leuven_size_labels.at[i,'size_7_rank'])+1
        size = bool(int(leuven_size_labels.at[i,'size_for_animal_instrument_dataset']))
        cat = ['animal','instrument','other'].index(category_map_[leuven_size_labels.at[i,'category']])
        size_out = [2542] if size else [2541]#[x+2541 for x in range(size)]
        cat_out = [2543+cat]
        name_out = [2546+object_list.index(obj)]
        context = [[33],[34],[35]]
        in_pat = [[object_list.index(obj)],[object_list.index(obj)],[object_list.index(obj)]]
        out_pat = [size_out,cat_out,name_out]
        if obj in patterns:
            patterns[obj]['context'].extend(context) #[size,cat,naming] output contexts
            patterns[obj]['out_pat'].extend(out_pat)
            patterns[obj]['in_pat'].extend(in_pat)
        else:
            patterns[obj] = {'context':context,'out_pat':out_pat,'in_pat':in_pat}
    return patterns











def load_object_data():
    object_data = pd.read_csv('data/leuven_size.csv')
    object_data['size'] = object_data.size_for_animal_instrument_dataset*4+1
    object_data['category'] = object_data.category.replace(['bird','fish','mammal','reptile'],'animal')
    return object_data



def load_behavioral_data():
    get_size_condition = lambda x: x[:4]
    get_cat_condition = lambda x: x[5:]
    behavioral_data = pd.read_csv('data/replication_data.csv')
    block_type_map = {'animal':'blocked','instrument':'blocked','interleaved':'interleaved','random1':'random'}
    blacklist = ['1457-2882-5959_replication', '4845-6126-5921_replication',
        '4989-6963-1797_replication', '5190-817-5024_replication',
        '5976-1010-9880_replication', '6401-5171-1258_replication',
        '7362-6472-5383_replication', '7947-4286-4432_replication',
        '9175-5989-1557_replication', '9305-2440-4556_replication',
                '2114-5445-1575_replication']
    random_list = ['7988-4602-5234_replication','8919-8336-2615_replication',
    '8601-8481-4307_replication','860-7124-4147_replication','7430-2301-8623_replication',
    '6367-6686-6886_replication','6048-3188-7751_replication','4364-9637-7395_replication',
    '9466-317-3968_replication','4058-8313-5939_replication','8752-2911-349_replication',
    '2988-2122-1287_replication','4401-2121-1972_replication','7710-7724-8792_replication',
    '5416-1430-5727_replication','7025-3763-2125_replication']
    behavioral_data = behavioral_data[~behavioral_data.participant.isin(blacklist)]
    behavioral_data['size_condition'] = behavioral_data.condition.apply(get_size_condition)
    behavioral_data['cat_condition'] = behavioral_data.condition.apply(get_cat_condition)
    behavioral_data['rand_condition'] = behavioral_data.condition_rand.apply(get_cat_condition)
    behavioral_data['correct'] = behavioral_data.correct.astype(float)
    behavioral_data['block_type_agg'] = [block_type_map[x] for x in behavioral_data.block_type]
    behavioral_data['participant_type'] = ['random' if x in random_list else 'non-random' for x in behavioral_data.participant]
    return behavioral_data

def calculate_mean_rts(behavioral_data,correct_only=True,random_intercept='participant'):
    if correct_only:
        mean_rts = behavioral_data[behavioral_data.correct==1]
    else:
        mean_rts = behavioral_data
    if random_intercept=='participant':
        participant_means = mean_rts[['participant_type','participant','rt']].groupby(['participant_type','participant'],as_index=False).transform('mean')
        mean_rts['rt'] -= participant_means.rt.values
    elif random_intercept=='block_type':
        participant_means = mean_rts[['participant_type','participant','rt','block_type']].groupby(['participant_type','participant','block_type'],as_index=False).transform('mean')
        mean_rts['rt'] -= participant_means.rt.values
    elif random_intercept is None:
        pass
    else:
        raise Exception(f'Unknown value {random_intercept} for parameter random_intercept. Try one of ["participant","block_type"].')
    if 'random' in set(mean_rts.participant_type):
        mean_rts = mean_rts[['participant_type','participant','size_condition','rand_condition','block_type_agg','rt']].groupby(['participant_type','participant','size_condition','rand_condition','block_type_agg'],as_index=False).mean()
    else:
        mean_rts = mean_rts[['participant_type','participant','size_condition','cat_condition','block_type_agg','rt']].groupby(['participant_type','participant','size_condition','cat_condition','block_type_agg'],as_index=False).mean()
    return mean_rts

def make_ohe(vals, n):
    ohe = np.zeros((n,))
    ohe[vals] = 1
    return ohe

def make_input(leuven_data, object_names, n_in=350, n_out=2541+2+3+350, n_tasks=36, device=None):
    if device is None:
        device = set_torch_device()
    in_patterns, out_patterns, task_patterns = [], [], []
    for name in object_names:
        obj_data = leuven_data[name]
        for i in range(len(obj_data['context'])):
            in_patterns.append(make_ohe(obj_data['in_pat'][i],n_in))
            out_patterns.append(make_ohe(obj_data['out_pat'][i],n_out))
            task_patterns.append(make_ohe(obj_data['context'][i],n_tasks))
    in_patterns = torch.tensor(np.stack(in_patterns),dtype=torch.float,device=device)
    out_patterns = torch.tensor(np.stack(out_patterns),dtype=torch.float,device=device)
    task_patterns = torch.tensor(np.stack(task_patterns),dtype=torch.float,device=device)
    return in_patterns, out_patterns, task_patterns

def make_input_comparison(leuven_data, object_names, n_in=350, n_out=2541+2+3+350, n_tasks=36, device=None):
    if device is None:
        device = set_torch_device()
    in_patterns, out_patterns, task_patterns = [], [], []
    in_patterns_comparison, out_patterns_comparison, task_patterns_comparison = [], [], []
    for name in object_names:
        obj_data = leuven_data[name]
        for i in range(len(obj_data['context'])):
            in_patterns.append(make_ohe(obj_data['in_pat'][i],n_in))
            out_patterns.append(make_ohe(obj_data['out_pat'][i],n_out))
            task_patterns.append(make_ohe(obj_data['context'][i],n_tasks))
            in_patterns_comparison.append(make_ohe([obj_data['in_pat'][i][0]+n_in//2],n_in))
            if i==len(obj_data['context'])-3:
                out_pat = [2541] if obj_data['out_pat'][i][0]==2542 else [2542]
            else:
                out_pat = obj_data['out_pat'][i]
            out_patterns_comparison.append(make_ohe(out_pat,n_out))
            task_patterns_comparison.append(make_ohe(obj_data['context'][i],n_tasks))
    in_patterns = torch.tensor(np.stack(in_patterns+in_patterns_comparison),dtype=torch.float,device=device)
    out_patterns = torch.tensor(np.stack(out_patterns+out_patterns_comparison),dtype=torch.float,device=device)
    task_patterns = torch.tensor(np.stack(task_patterns+task_patterns_comparison),dtype=torch.float,device=device)
    return in_patterns, out_patterns, task_patterns



def make_training_data_comparison(num_objects=700,num_outputs=2541+2+3+350,num_tasks=36,device=None):
    leuven_data = load_leuven()
    object_names = pd.read_csv('data/leuven_size.csv').name.values
    in_, out_, context_ = make_input_comparison(leuven_data, object_names, num_objects, num_outputs, num_tasks, device=device)
    return [in_,context_], out_



def make_full_distractor_training_data():
    obj_data = pd.read_csv('data/leuven_size.csv')
    names, sizes, categories = obj_data.name.values, obj_data.size_for_animal_instrument_dataset.values, obj_data.category.values
    experiment_stimulus_indices = get_item_indices_by_category(['instrument','mammal','bird','fish','reptile'])
    names, sizes, categories = names[experiment_stimulus_indices], sizes[experiment_stimulus_indices], categories[experiment_stimulus_indices]

    item_in = torch.zeros((5454,350),dtype=torch.float,device=set_torch_device())
    context_in = torch.zeros((5454,2),dtype=torch.float,device=set_torch_device())
    out_ = torch.zeros((5454,2541+2+3+350),dtype=torch.float,device=set_torch_device())

    contexts = np.array([1 if x=='instrument' else 0 for x in categories])
    sizes = np.array([2541 if size==0 else 2542 for size in sizes])
    target_idxs, distractor_idxs, context_idxs, size_idxs = [], [], [], []
    for i in range(len(experiment_stimulus_indices)):
        for j in range(len(experiment_stimulus_indices)):
            if contexts[i]==contexts[j]:
                continue
            target_idxs.append(experiment_stimulus_indices[i])
            distractor_idxs.append(experiment_stimulus_indices[j])
            context_idxs.append(contexts[i])
            size_idxs.append(sizes[i])
    item_in[list(range(5454)),target_idxs] = 1
    item_in[list(range(5454)),distractor_idxs] = 1
    context_in[list(range(5454)),context_idxs] = 1
    out_[list(range(5454)),size_idxs] = 1

    small_animal_idxs = (contexts==0)*(sizes==2541)
    large_animal_idxs = (contexts==0)*(sizes==2542)
    small_instrument_idxs = (contexts==1)*(sizes==2541)
    large_instrument_idxs = (contexts==1)*(sizes==2542)
    return [item_in,context_in],out_,experiment_stimulus_indices,small_animal_idxs,large_animal_idxs,small_instrument_idxs,large_instrument_idxs

def make_behavioral_experiment_training_data(distractor_strength=.975,target_strength=1):
    experiment_stimulus_indices = [118,104,30,48,116,
    105,115,29,57,59,
    248,252,261,263,257,
    262,253,266,267,260]
    random_stimuli = ['cello','elephant','goldfish','harp','iguana',
    'mouse','piano','recorder','shark','triangle']
    random_contexts = [1,0,1,0,0,0,1,1,0,1,1,0,0,1,1,0,1,0,1,0]

    obj_data = pd.read_csv('data/leuven_size.csv')
    names, sizes, categories = obj_data.name.values, obj_data.size_for_animal_instrument_dataset.values, obj_data.category.values
    names, sizes, categories = names[experiment_stimulus_indices], sizes[experiment_stimulus_indices], categories[experiment_stimulus_indices]
    item_in = torch.zeros((1140,350),dtype=torch.float,device=set_torch_device())
    context_in = torch.zeros((1140,5),dtype=torch.float,device=set_torch_device())
    out_ = torch.zeros((1140,2541+2+3+350),dtype=torch.float,device=set_torch_device())

    contexts = [0]*10+[1]*10
    random_contexts = [3 if c==0 else 4 for c in random_contexts]
    sizes = [2542]*5+[2541]*5+[2542]*5+[2541]*5
    target_idxs, distractor_idxs, context_idxs, size_idxs = [], [], [], []
    size_conditions, cat_conditions = [], []
    random_cat_conditions, random_context_idxs = [], []
    for i in range(20):
        for j in range(20):
            if i==j:
                continue
            target_idxs.append(experiment_stimulus_indices[i])
            distractor_idxs.append(experiment_stimulus_indices[j])
            context_idxs.append(contexts[i])
            size_idxs.append(sizes[i])
            size_conditions.append('s_ma' if sizes[i]==sizes[j] else 's_ms')
            cat_conditions.append('c_ma' if contexts[i]==contexts[j] else 'c_ms')
            random_cat_conditions.append('c_ma' if random_contexts[i]==random_contexts[j] else 'c_ms')
            random_context_idxs.append(random_contexts[i])
    item_in[list(range(380)),target_idxs] = target_strength
    item_in[list(range(380)),distractor_idxs] = distractor_strength
    context_in[list(range(380)),context_idxs] = 1
    out_[list(range(380)),size_idxs] = 1

    item_in[list(range(380,380*2)),target_idxs] = target_strength
    item_in[list(range(380,380*2)),distractor_idxs] = distractor_strength
    context_in[list(range(380,380*2)),2] = 1
    out_[list(range(380,380*2)),size_idxs] = 1

    item_in[list(range(380*2,380*3)),target_idxs] = target_strength
    item_in[list(range(380*2,380*3)),distractor_idxs] = distractor_strength
    context_in[list(range(380*2,380*3)),random_context_idxs] = 1
    out_[list(range(380*2,380*3)),size_idxs] = 1
    size_conditions *= 3
    cat_conditions *= 3
    random_cat_conditions *= 3
    #cat_conditions = cat_conditions+cat_conditions+random_cat_conditions
    blocks = ['blocked']*380+['interleaved']*380+['random']*380
    return [item_in,context_in],out_,size_conditions,cat_conditions,random_cat_conditions,blocks



def add_within_subject_error_bars(data,dv='rt',within=['cat_condition','size_condition'],subject='participant',remove_mean=True):
    #Morey-Cousineau correction
    n_participants = len(set(data[subject]))
    if len(within)==2:
        n_conditions = len(set(data[within[0]])) * len(set(data[within[1]]))
    else:
        raise Exception('Only 2-way data currently supported')
    grand_mean = data[dv].mean()
    participant_means = data.groupby([subject],as_index=False).transform('mean')
    data[f'{dv}_normalized'] = data[dv] - participant_means[dv] + grand_mean
    stds = data[[f'{dv}_normalized']+within].groupby(within,as_index=False).transform('std')
    t_statistic = t.ppf(1-.025,df=n_participants-1)
    morey_correction = np.sqrt(n_conditions/(n_conditions-1))
    data[f'{dv}_error'] = t_statistic * morey_correction * stds[f'{dv}_normalized'] / np.sqrt(n_participants)
    if remove_mean:
        data[f'{dv}_normalized'] -= grand_mean
    return data

def pretty_bar_chart(data, x, y, color, barmode='group', error_y=None, error_x=None, 
                     title=None, x_title=None, y_title=None, color_title=None,
                     range_x=None, range_y=None):
    if x_title is None:
        x_title = x
    if y_title is None:
        y_title = y
    if color_title is None:
        color_title = color
    fig = px.bar(data,x=x,color=color,range_x=range_x,range_y=range_y,
       y=y,barmode=barmode,error_y=error_y,error_x=error_x,
       title=title,labels={x:x_title,y:y_title,color:color_title})
    fig.update_layout(width=800,height=600,plot_bgcolor='white',title_x=0.5,
                      legend=dict(yanchor='top',y=.99,xanchor='left',x=.01,font=dict(size=14)))

    fig.update_xaxes(showline=True,linewidth=1.5,linecolor='black',tickfont=dict(size=14),
                     mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20)
                     )
    fig.update_yaxes(showline=True,linewidth=1.5,linecolor='black',
                     mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20),
                     zeroline=True,zerolinecolor='black',zerolinewidth=1)
    return fig


def format_fig(fig,height=600):
    fig.update_layout(width=800,height=height,plot_bgcolor='white',
                      legend=dict(orientation="v",yanchor="auto",
                                  y=1,
                                  xanchor="right",  # changed
                                  x=1,
                                  font=dict(size=16)
                                  )
    )
    fig.update_xaxes(showline=True,linewidth=1.5,linecolor='black',mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20))
    fig.update_yaxes(showline=True,linewidth=1.5,linecolor='black',mirror=True,ticks='outside',showgrid=False,titlefont=dict(size=20))
    return fig


def gen_distractor_patterns(distractor_strength,device=set_torch_device()):
    object_names = pd.read_csv('data/leuven_size.csv').name.values
    animal_names_experiment = ['elephant','giraffe','shark','alligator','rhinoceros',
                            'hamster','mouse','goldfish','frog','iguana']
    animal_idxs_experiment = [list(object_names).index(x) for x in animal_names_experiment]
    instrument_names_experiment = ['cello','drum_set','organ','piano','harp_',
                                'panpipe','flute','tambourine','triangle','harmonica']
    instrument_idxs_experiment = [list(object_names).index(x) for x in instrument_names_experiment]
    both_names_experiment = animal_names_experiment+instrument_names_experiment
    both_idxs_experiment = animal_idxs_experiment+instrument_idxs_experiment
    sizes=(['big']*5+['small']*5)*2
    categories = ['animal']*10+['instrument']*10
    def get_condition(target,distractor):
        target_idx = both_names_experiment.index(target)
        distractor_idx = both_names_experiment.index(distractor)
        cat_match = 'match' if target_idx//10==distractor_idx//10 else 'mismatch'
        size_match = 'match' if target_idx//5%2==distractor_idx//5%2 else 'mismatch'
        return cat_match, size_match, f'cat_{cat_match}_size_{size_match}'
    d_in, d_context, d_true = torch.zeros((20*20-20,350),device=device), torch.zeros((20*20-20,36),device=device), torch.zeros((20*20-20,2932),device=device)
    targets,distractors=[],[]
    counter = 0
    for o1 in range(20):
        for o2 in range(20):
            if o1==o2:
                continue
            d_in[counter,both_idxs_experiment[o1]] = 1
            d_in[counter,both_idxs_experiment[o2]] = distractor_strength
            d_true[counter,2541+['small','big'].index(sizes[o1])] = 1
            d_context[counter,-3] = 1
            counter += 1
            targets.append(both_names_experiment[o1])
            distractors.append(both_names_experiment[o2])
    conditions = np.vectorize(get_condition)(targets,distractors)
    return d_in,d_context,d_true,targets,distractors,conditions