import analysis
import data
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import utils


class BCEMetric:
    """
    Binary cross-entropy metric for tracking during training.
    """
    def __init__(self) -> None:
        self.name='bce'
        self.values = []
        self.fn = nn.BCELoss()


    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor, model=None) -> list[np.array]:
        self.values.append(self.fn(y,y_hat).cpu().detach().numpy())
    

class TPRMetric:
    """
    True positive rate metric for tracking during training.
    """
    def __init__(self) -> None:
        self.name='tpr'
        self.values = []


    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor, model=None) -> list[np.array]:
        p = y_hat[y>0.5]
        self.values.append((p>0.5).cpu().detach().numpy().mean())


class SizeOrthogonalityMetric:
    def __init__(self):
        self.name='ortho'
        self.values = {'x':[],'y':[],'color':[]}
        self.small_animal_idxs = data.get_item_indices_by_category_and_size(['mammal','bird','fish','reptile'],'small')
        self.large_animal_idxs = data.get_item_indices_by_category_and_size(['mammal','bird','fish','reptile'],'large')
        self.small_instrument_idxs = data.get_item_indices_by_category_and_size(['instrument'],'small')
        self.large_instrument_idxs = data.get_item_indices_by_category_and_size(['instrument'],'large')

    def __call__(self,y_hat,y,model):
        learned_context_dependent_reps = model.get_context_dependent_reps()[1]
        small_animal_reps = learned_context_dependent_reps[self.small_animal_idxs]
        large_animal_reps = learned_context_dependent_reps[self.large_animal_idxs]
        small_instrument_reps = learned_context_dependent_reps[self.small_instrument_idxs]
        large_instrument_reps = learned_context_dependent_reps[self.large_instrument_idxs]
        animal_within = []
        for i in range(len(small_animal_reps)):
            for j in range(len(large_animal_reps)):
                animal_within.append(large_animal_reps[j]-small_animal_reps[i])
        instrument_within = []
        for i in range(len(small_instrument_reps)):
            for j in range(len(large_instrument_reps)):
                instrument_within.append(large_instrument_reps[j]-small_instrument_reps[i])
        within_angles, across_angles = analysis.cosine_splithalf(small_animal_reps, large_animal_reps,
                                                                    small_instrument_reps, large_instrument_reps,1000)
        self.values['y'].append(within_angles.mean())
        self.values['y'].append(across_angles.mean())
        self.values['y'].append(across_angles.mean()-within_angles.mean())
        self.values['y'].append(((within_angles-across_angles)>0).mean())
        self.values['x'].extend([len(self.values['x'])//4]*4)
        self.values['color'].extend(['within','across','delta','pval'])


class ISCModel(nn.Module):
    """
    Creates a Controlled Semantic Cognition (CSC) Model.

    Parameters
    ----------
    num_objects (int): Number of objects/inputs for the model. Default: 350
    num_hub_hidden_units (int): Number of hidden units in the hub layer. Default: 64
    num_context_dependent_hidden_units (int): Number of hidden units in the context-dependent layer. Default: 128
    num_task_context_units (int): Number of hidden units in the task context layer. Default: 16
    num_output (int): Number of output units. Default: 2541+2+3+350
    num_tasks (int): Number of tasks. Default: 36
    lr (float): Learning rate for the model. Default: 0.05
    device (str): Device to use for training. Default: None
    biases (bool): If ``True``, uses biases in the model. Default: True

    Attributes
    ----------
    item_input_to_hub_weights (torch.nn.Linear): Weights from the item input to the hub.
    context_input_to_task_context_rep_weights (torch.nn.Linear): Weights from the context input to the task context layer.
    task_context_rep_to_context_dependent_rep_weights (torch.nn.Linear): Weights from the task context layer to the context dependent layer.
    hub_to_context_dependent_rep_weights (torch.nn.Linear): Weights from the hub to the context dependent layer.
    context_dependent_rep_to_output_weights (torch.nn.Linear): Weights from the context dependent layer to the output layer.
    hub_to_output_weights (torch.nn.Linear): Weights from the hub to the output layer.
    loss_fn (torch.nn.BCEWithLogitsLoss): Loss function for the model.
    optimizer (torch.optim.Adam): Optimizer for the model.
    metrics (list): List of metrics to track during training.
    num_objects (int): Number of objects/inputs for the model.
    num_tasks (int): Number of tasks for the model.
    num_context_dependent_hidden_units (int): Number of hidden units in the context-dependent layer.
    device (str): Device to use for training.

    Methods
    -------
    freeze_weights()
        Freezes the weights of the model.
    load_old_model_weights(state_dict,use_old_size_starting_point=True)
        Loads weights from a previous model.
    get_context_independent_rep(x)
        Gets the context-independent representation of the model for a given input.
    get_task_context_rep(x)
        Gets the task context representation of the model for a given input.
    get_context_dependent_rep(x)
        Gets the context-dependent representation of the model for a given input.
    forward(x,take_sigmoid=True)
        Forward pass of the model.
    train(x,y,epochs=1,batch_size=64)
        Trains the model.
    plot_metrics()
        Plots the metrics of the model.
    get_task_context_reps()
        Gets the task context representations of the model for all input combinations.
    get_context_independent_reps()
        Gets the context-independent representations of the model for all input combinations.
    get_context_dependent_reps()
        Gets the context-dependent representations of the model for all input combinations.
    """
    def __init__(self, num_objects: int = 350, num_hub_hidden_units: int = 64,
                 num_context_dependent_hidden_units: int = 128,
                 num_task_context_units: int = 16,
                 num_output: int = 2541+2+3+350,
                 num_tasks: int = 36, lr: float = .05,
                 device=None, biases: bool = True) -> None:
        super().__init__()
        if device is None:
            device = utils.set_torch_device()

        self.item_input_to_hub_weights = nn.Linear(num_objects,num_hub_hidden_units,device=device,bias=biases)
        self.context_input_to_task_context_rep_weights = nn.Linear(num_tasks,num_task_context_units,device=device,bias=biases)
        self.task_context_rep_to_context_dependent_rep_weights = nn.Linear(num_task_context_units,num_context_dependent_hidden_units,device=device,bias=biases)
        self.hub_to_context_dependent_rep_weights = nn.Linear(num_hub_hidden_units,num_context_dependent_hidden_units,device=device,bias=biases)
        self.context_dependent_rep_to_output_weights = nn.Linear(num_context_dependent_hidden_units,num_output,device=device,bias=biases)
        self.hub_to_output_weights = nn.Linear(num_hub_hidden_units,num_output,device=device)
        
        nn.init.uniform_(self.item_input_to_hub_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.context_input_to_task_context_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.task_context_rep_to_context_dependent_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.hub_to_context_dependent_rep_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.context_dependent_rep_to_output_weights.weight,a=-.01,b=.01)
        nn.init.uniform_(self.hub_to_output_weights.weight,a=-.01,b=.01)

        if biases:
            nn.init.uniform_(self.item_input_to_hub_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.context_input_to_task_context_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.task_context_rep_to_context_dependent_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.hub_to_context_dependent_rep_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.context_dependent_rep_to_output_weights.bias,a=-.01,b=.01)
            nn.init.uniform_(self.hub_to_output_weights.bias,a=-.01,b=.01)
        else:
            with torch.no_grad():
                self.hub_to_output_weights.bias.copy_(torch.ones(self.hub_to_output_weights.bias.shape,device=device)*-2)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.parameters(),lr=lr)
        self.metrics = [BCEMetric(), TPRMetric()]
        self.num_objects = num_objects
        self.num_tasks = num_tasks
        self.num_context_dependent_hidden_units = num_context_dependent_hidden_units
        self.device = device


    def freeze_weights(self) -> None:
        for param in self.parameters():
            param.requires_grad = False


    def load_old_model_weights(self, state_dict: dict, use_old_size_starting_point: bool=True) -> None:
        for name, param in state_dict.items():
            if name not in ['context_input_to_task_context_rep_weights.weight','context_input_to_task_context_rep_weights.bias']:
                self.state_dict()[name].copy_(param)
                param.requires_grad = False
        if use_old_size_starting_point:
            old_size_weights = state_dict['context_input_to_task_context_rep_weights.weight'][:,-3]
            old_size_bias = state_dict['context_input_to_task_context_rep_weights.bias'][-3]
            with torch.no_grad():
                for i in range(self.num_tasks):
                    self.context_input_to_task_context_rep_weights.weight[:,i] = old_size_weights
                    self.context_input_to_task_context_rep_weights.bias[i] = old_size_bias


    def get_context_independent_rep(self, x: torch.Tensor) -> torch.Tensor:
        hub_rep = torch.sigmoid(self.item_input_to_hub_weights(x[0]))
        return hub_rep


    def get_task_context_rep(self, x: torch.Tensor) -> torch.Tensor:
        task_context_rep = torch.sigmoid(self.context_input_to_task_context_rep_weights(x[1]))
        return task_context_rep


    def get_context_dependent_rep(self, x: torch.Tensor) -> torch.Tensor:
        hub_rep = self.get_context_independent_rep(x)
        task_context_rep = self.get_task_context_rep(x)
        item_in_context_rep = torch.sigmoid(self.task_context_rep_to_context_dependent_rep_weights(task_context_rep)+
                                            self.hub_to_context_dependent_rep_weights(hub_rep))
        return item_in_context_rep


    def forward(self, x: torch.Tensor, take_sigmoid: bool=True, noise: float=0) -> torch.Tensor:
        if noise:
            hub_rep = self.item_input_to_hub_weights(x[0])
            #hub_rep += torch.randn_like(hub_rep)*noise
            task_context_rep = self.context_input_to_task_context_rep_weights(x[1])
            #task_context_rep += torch.randn_like(task_context_rep)*noise
            hub_to_dep = self.hub_to_context_dependent_rep_weights(torch.sigmoid(hub_rep))
            tc_to_dep = self.task_context_rep_to_context_dependent_rep_weights(torch.sigmoid(task_context_rep))
            item_in_context_rep = torch.sigmoid(hub_to_dep+tc_to_dep)#+torch.randn_like(hub_to_dep)*noise)
            output = self.hub_to_output_weights(torch.sigmoid(hub_rep))+self.context_dependent_rep_to_output_weights(item_in_context_rep)
            #output += torch.randn_like(output)*noise
            if take_sigmoid:
                output = torch.sigmoid(output)+torch.randn_like(output)*noise
            return output

        hub_rep = torch.sigmoid(self.item_input_to_hub_weights(x[0]))
        task_context_rep = torch.sigmoid(self.context_input_to_task_context_rep_weights(x[1]))
        item_in_context_rep = torch.sigmoid(self.task_context_rep_to_context_dependent_rep_weights(task_context_rep)+
                                            self.hub_to_context_dependent_rep_weights(hub_rep))
        output = self.hub_to_output_weights(hub_rep)+\
                 self.context_dependent_rep_to_output_weights(item_in_context_rep)
        if take_sigmoid:
            output = torch.sigmoid(output)
        return output


    def train(self, x: torch.Tensor, y: torch.Tensor, epochs: int = 1, batch_size: int = 64) -> list:
        if epochs < 1:
            return [0]
        for metric in self.metrics:
            metric(self(x),y,self)
        for epoch in range(epochs):
            n_steps = 0
            batch_idxs = np.random.permutation(range(len(y)))
            for batch_start in range(0,len(y),batch_size):
                batch_idx = batch_idxs[batch_start:min(batch_start+batch_size,len(y))]
                self.optimizer.zero_grad()
                output = self([x[0][batch_idx],x[1][batch_idx]],take_sigmoid=False)
                loss = self.loss_fn(output,y[batch_idx])
                loss.backward()
                self.optimizer.step()
                n_steps += 1

            for metric in self.metrics:
                metric(self(x),y,self)
        return self.metrics
    

    def plot_metrics(self) -> None:
        for metric in self.metrics:
            print(metric.name)
            if type(metric.values) is list:
                fig = px.line(y=metric.values)
            else:
                fig = px.line(metric.values,x='x',y='y',color='color')
            fig.show()


    def get_task_context_reps(self) -> np.array:
        item_x = torch.zeros((self.num_tasks,self.num_objects),device=self.device)
        context_x = torch.eye(self.num_tasks,device=self.device)
        tc_reps = self.get_task_context_rep([item_x,context_x]).cpu().detach().numpy()
        return tc_reps
    

    def get_context_independent_reps(self) -> np.array:
        item_x = torch.eye(self.num_objects,device=self.device)
        context_x = torch.zeros((self.num_objects,self.num_tasks),device=self.device)
        ind_reps = self.get_context_independent_rep([item_x,context_x]).cpu().detach().numpy()
        return ind_reps
    

    def get_context_dependent_reps(self) -> np.array:
        item_x = torch.eye(self.num_objects,device=self.device)
        context_x = torch.zeros((self.num_objects,self.num_tasks),device=self.device)
        dep_reps = np.zeros((self.num_tasks, self.num_objects, self.num_context_dependent_hidden_units))
        for context in range(self.num_tasks):
            context_x = torch.zeros((self.num_objects,self.num_tasks),device=self.device)
            context_x[:,context] = 1
            dep_reps[context] = self.get_context_dependent_rep([item_x,context_x]).cpu().detach().numpy()
        return dep_reps


def load_isc_models(num_models: int, retrain_models: bool = False,
                    num_training_epochs: int = 30, model_path: str = 'models',
                    device = utils.set_torch_device()) -> list[ISCModel]:
    """
    Load ISC models from saved weights. Optionally retrains models if specified.

    Parameters
    ----------
    num_models (int): Number of models to load
    retrain_models (bool): If ``True``, retrains models instead of loading from disk. Default: False
    num_training_epochs (int): Number of training epochs for retrained models. Default: 30
    model_path (str): Path to use when loading/saving models to/from disk.
    device (str): Device to use for training.
    """

    train_x, train_y = data.make_training_data(device=device)
    isc_models = []
    if retrain_models:
        for model_num in range(num_models):
            isc_model = ISCModel(device=device)
            isc_model.train(train_x,train_y,epochs=num_training_epochs)
            torch.save(isc_model.state_dict(), f'{model_path}/csc_model_{model_num}.torch')
            isc_models.append(isc_model)
    else:
        for model_num in range(num_models):
            isc_model = ISCModel(device=device)
            try:
                isc_model.load_state_dict(torch.load(f'{model_path}/csc_model_{model_num}.torch'))
            except:
                print(f'Error loading model number {model_num}. Training model instead.')
                isc_model.train(train_x,train_y,epochs=num_training_epochs)
                torch.save(isc_model.state_dict(), f'{model_path}/csc_model_{model_num}.torch')
            isc_models.append(isc_model)
    return isc_models


def load_comparison_models(num_models: int, retrain_models: bool = False,
                           num_training_epochs: int = 30, model_path: str = 'models',
                           device = utils.set_torch_device()) -> list[ISCModel]:
    """
    Load comparison models (trained without size correlations) from saved weights. Optionally retrains models if specified.

    Parameters
    ----------
    num_models (int): Number of models to load
    retrain_models (bool): If ``True``, retrains models instead of loading from disk. Default: False
    num_training_epochs (int): Number of training epochs for retrained models. Default: 30
    model_path (str): Path to use when loading/saving models to/from disk.
    device (str): Device to use for training.
    """

    train_x_comparison, train_y_comparison = data.make_training_data_comparison(device=device)
    comparison_models = []
    if retrain_models:
        for model_num in range(num_models):
            comparison_model = ISCModel(device=device,num_objects=700,num_hub_hidden_units=64,
                                            num_context_dependent_hidden_units=128,biases=False)
            comparison_model.train(train_x_comparison,train_y_comparison,epochs=num_training_epochs)
            torch.save(comparison_model.state_dict(), f'{model_path}/comparison_model_{model_num}-128hidden_nobias.torch')
            comparison_models.append(comparison_model)
    else:
        for model_num in range(num_models):
            comparison_model = ISCModel(device=device,num_objects=700,num_hub_hidden_units=64,
                                            num_context_dependent_hidden_units=128,biases=False)
            try:
                comparison_model.load_state_dict(torch.load(f'{model_path}/comparison_model_{model_num}-128hidden_nobias.torch'))
            except:
                print(f'Error loading model number {model_num}. Training model instead.')
                comparison_model.train(train_x_comparison,train_y_comparison,epochs=num_training_epochs)
                torch.save(comparison_model.state_dict(), f'{model_path}/comparison_model_{model_num}-128hidden_nobias.torch')
            comparison_models.append(comparison_model)
    return comparison_models