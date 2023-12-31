from .meta_model import MetaModel
from core.model.abstract_model import AbstractModel
from core.utils import ModelType
from abc import abstractmethod
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

def update_params(loss, params, acc_gradients, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad
        if name == 'fc.weight':
            acc_gradients[0] = acc_gradients[0] + grad
        if name == 'fc.bias':
            acc_gradients[1] = acc_gradients[1] + grad
    return updated_params, acc_gradients

def inner_train_step(model, support_data, args):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    updated_params = OrderedDict(model.named_parameters())
    acc_gradients = [torch.zeros_like(updated_params['fc.weight']), torch.zeros_like(updated_params['fc.bias'])]
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)        
    
    for ii in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=args.gd_lr, first_order=True)
    return updated_params, acc_gradients

class UNICORN_MAML(MetaModel):
    def __init__(self, args, init_type="normal"):
        super().__init__(init_type, ModelType.META, args=args)
        self.args = args
        if args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12_maml import ResNetMAML
            self.encoder = ResNetMAML(dropblock_size=args.dropblock_size) 
        else:
            raise ValueError('')
        self.hdim = hdim
        self.encoder.fc = nn.Linear(self.hdim, args.way)        
        self.fcone = nn.Linear(self.hdim, 1)

    def set_forward(self, data_shot, data_query):
        self.encoder.fc.weight.data = self.fcone.weight.data.repeat(self.args.way, 1)
        self.encoder.fc.bias.data = self.fcone.bias.data.repeat(self.args.way)
        updated_params, acc_gradients = inner_train_step(self.encoder, data_shot, self.args)
        updated_params['fc.weight'] = self.fcone.weight.repeat(self.args.way, 1) - self.args.gd_lr * acc_gradients[0]
        updated_params['fc.bias'] = self.fcone.bias.repeat(self.args.way) - self.args.gd_lr * acc_gradients[1]
        
        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis

    def set_forward_loss(self, data_shot, data_query):
        logitis = self.set_forward(data_shot, data_query)
        label = torch.arange(self.args.way).repeat(self.args.query)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
        else:
            label = label.type(torch.LongTensor)
        loss = F.cross_entropy(logitis, label)
        return loss

    def set_forward_adaptation(self, data_shot, data_query):
        self.fcone.weight.data = torch.randn_like(self.fcone.weight.data) # use a random shared classifier initialization
        self.fcone.bias.data = torch.randn_like(self.fcone.bias.data)
        self.train()
        updated_params, acc_gradients = inner_train_step(self.encoder, data_shot, self.args)
        updated_params['fc.weight'] = self.fcone.weight.repeat(self.args.way, 1) - self.args.gd_lr * acc_gradients[0]
        updated_params['fc.bias'] = self.fcone.bias.repeat(self.args.way) - self.args.gd_lr * acc_gradients[1]
        
        self.eval()
        with torch.no_grad():        
            logitis_query = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis_query
