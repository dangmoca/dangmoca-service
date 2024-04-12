from kobert_model.kobert_hf.kobert_tokenizer.kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

import numpy as np
import torch

from torch.utils.data import Dataset

import gluonnlp as nlp

from kobert_model.Classifier import BERTClassifier
from kobert_model.Dataset import BERTDataset

# device = torch.device("cuda:0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KoBERTPredictor:
    def __init__(self):
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        self.vocab = nlp.vocab.BERTVocab.from_sentencepiece(self.tokenizer.vocab_file, padding_token='[PAD]')
        self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower=False)
        self.bert_model = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
        self.model = BERTClassifier(self.bert_model, dr_rate=0.5).to(device)
        self.model.load_state_dict(
            torch.load("kobert_model/model_pt/kobert_model_state_dict.pt", map_location=torch.device('cpu')),
            strict=False)
        self.max_len = 128
        self.batch_size = 16
        self.warmup_ratio = 0.05
        self.num_epochs = 3
        self.max_grad_norm = 5
        self.log_interval = 200
        self.learning_rate = 5e-5

    def new_softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a - c)
        sum_exp_a = np.sum(exp_a)
        return (exp_a / sum_exp_a) * 100

    def predict(self, review_sentence):
        global emotion_probability
        data = [review_sentence, '0']
        dataset_another = [data]
        tok = self.tokenizer.tokenize
        another_test = BERTDataset(dataset_another, 0, 1, tok, self.vocab, self.max_len, True, False)
        test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=self.batch_size,
                                                      num_workers=5)  # torch 형식 변환

        self.model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)

            valid_length = valid_length

            out = self.model(token_ids, valid_length, segment_ids)

            for i in out:
                logits = i
                logits = logits.detach().cpu().numpy()
                emotion_probability = []
                logits = np.round(self.new_softmax(logits), 3).tolist()
                emotion_name = ["0,", "최악", "별로", "중립", "좋음", "완좋"]
                index = 0
                for logit in logits:
                    emotion_probability.append([emotion_name[index], np.round(logit, 3)])
                    index += 1

                if np.argmax(logits) == 1:
                    emotion = "최악"
                elif np.argmax(logits) == 2:
                    emotion = "별로"
                elif np.argmax(logits) == 4:
                    emotion = "좋음"
                elif np.argmax(logits) == 5:
                    emotion = "완좋"

                emotion_probability.append(emotion)
        return emotion_probability
