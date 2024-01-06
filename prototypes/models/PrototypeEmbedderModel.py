from torch import nn
from transformers import BertModel, BertTokenizer, AlbertModel, AutoTokenizer, DistilBertModel

from utils import ModelUtils
from utils.Constants import BERT_DIMS, HIDDEN_MODEL_SIZE, META_OUTPUT_SIZE

from prototypes.PrototypeModel import PrototypeModel


class PrototypeEmbedderModel(nn.Module, PrototypeModel):

    def __init__(self):
        super(PrototypeEmbedderModel, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.bert = BertModel.from_pretrained("bert-base-cased")
        # self.tunableLayers = {str(l) for l in range(8, 12)}
        self.embedder = nn.Linear(BERT_DIMS, HIDDEN_MODEL_SIZE)
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
        embedding = self.embedder(encoding)
        del outputs, tokenized_inputs
        return embedding

    def getEncoder(self):
        return "BERT"

    def scaleGradients(self, scalingFactor):
        for name, param in self.bert.named_parameters():
            if param.requires_grad:
                param.grad = param.grad * scalingFactor
