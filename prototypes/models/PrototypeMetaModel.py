from torch import nn
from transformers import BertModel, BertTokenizer

from utils import ModelUtils
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel


class PrototypeMetaModel(nn.Module, PrototypeModel):

    def __init__(self):
        super(PrototypeMetaModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-cased")
        # self.hidden = nn.Linear(BERT_DIMS, HIDDEN_MODEL_SIZE)
        # self.metaOutput = nn.Linear(HIDDEN_MODEL_SIZE, META_OUTPUT_SIZE)
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.1)
        self.tunableLayers = {str(l) for l in range(9, 12)}
        self.assignTrainableParams()

    def assignTrainableParams(self):
        # freeze the first n tunable layers of BERT
        for name, param in self.bert.named_parameters():
            if not set.intersection(set(name.split('.')), self.tunableLayers):
                param.requires_grad = False

    def forward(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(ModelUtils.DEVICE)
        outputs = self.bert(**tokenized_inputs)
        encoding = outputs.last_hidden_state[:, 0, :]
        # dropout = self.dropout(encoding)
        # hidden_output = self.hidden(inputs)
        # dropout = self.dropout(hidden_output)
        # output = self.metaOutput(dropout)
        # del tokenized_inputs
        return encoding

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        for name, param in self.bert.named_parameters():
            if set.intersection(set(name.split('.')), self.tunableLayers):
                param.grad = param.grad * scalingFactor
        # self.hidden.weight.grad = self.hidden.weight.grad * scalingFactor
        # self.hidden.bias.grad = self.hidden.bias.grad * scalingFactor
        # self.metaOutput.weight.grad = self.metaOutput.weight.grad * scalingFactor
        # self.metaOutput.bias.grad = self.metaOutput.bias.grad * scalingFactor
