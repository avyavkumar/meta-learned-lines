from torch import nn
from transformers import BertModel, BertTokenizer, AlbertModel, AutoTokenizer, DistilBertModel

from utils import ModelUtils
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel


class PrototypeMetaModel(nn.Module, PrototypeModel):

    def __init__(self):
        super(PrototypeMetaModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-cased")
        # self.tunableLayers = {str(l) for l in range(0, 12)}
        # self.embedder = nn.Linear(BERT_DIMS, HIDDEN_MODEL_SIZE)
        # self.dropout = nn.Dropout(0.1)
        # self.assignTrainableParams()

    def assignTrainableParams(self):
        # freeze the first n tunable layers of BERT
        for name, param in self.bert.named_parameters():
            if not set.intersection(set(name.split('.')), self.tunableLayers):
                param.requires_grad = False

    def forward(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(ModelUtils.DEVICE)
        outputs = self.bert(**tokenized_inputs)
        encoding = outputs.last_hidden_state[:, 0, :]
        # dropout_output = self.dropout(encoding)
        # embedding = self.embedder(dropout_output)
        del outputs, tokenized_inputs
        # return embedding
        return encoding

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        for name, param in self.bert.named_parameters():
            if "layer.9.output.dense.weight" in name:
                print(name, param.grad)
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                param.grad = param.grad * scalingFactor
        for name, param in self.bert.named_parameters():
            if "layer.9.output.dense.weight" in name:
                print(name, param.grad)
        # for name, param in self.embedder.named_parameters():
        #     if param.requires_grad:
        #         param.grad = param.grad * scalingFactor
