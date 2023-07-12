from torch import nn
from transformers import BertModel, BertTokenizer

from utils import ModelUtils
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE

from prototypes.PrototypeModel import PrototypeModel


class PrototypeMetaModel(nn.Module, PrototypeModel):

    def __init__(self):
        super(PrototypeMetaModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.hidden = nn.Linear(BERT_DIMS, HIDDEN_MODEL_SIZE)
        self.relu = nn.ReLU()
        self.tunableLayers = {str(l) for l in range(8, 12)}
        self.assignTrainableParams()

    def assignTrainableParams(self):
        # freeze the first 8 layers of BERT
        for name, param in self.bert.named_parameters():
            if not set.intersection(set(name.split('.')), self.tunableLayers):
                param.requires_grad = False

    def forward(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(ModelUtils.DEVICE)
        outputs = self.bert(**tokenized_inputs)
        encoding = outputs.last_hidden_state[:, 0, :]
        hidden_output = self.hidden(encoding)
        output = self.relu(hidden_output)
        del tokenized_inputs
        return output

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        for name, param in self.bert.named_parameters():
            if set.intersection(set(name.split('.')), self.tunableLayers):
                param.grad = param.grad * scalingFactor
        self.hidden.weight.grad = self.hidden.weight.grad * scalingFactor
        self.hidden.bias.grad = self.hidden.bias.grad * scalingFactor
