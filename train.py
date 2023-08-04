import os
import torch
import argparse
import logging
from utils import get_device, set_random_seeds
from load_data import get_dataloader, get_label_num, get_inv_label, MAX_LEN
from sklearn.metrics import f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='train.json', help='Filename of training data')
parser.add_argument('--dev_file', type=str, default='dev.json', help='Filename of dev data')
parser.add_argument('--test_file', type=str, default='test.json', help='Filename of testing data')
parser.add_argument('--epoch', type=int, help='Epoch', default=10)
parser.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
parser.add_argument('--model', type=str, help='Specify the BERT model in use', default='bert')
parser.add_argument('--mode', type=int, help='Specify the preprocessing mode', default=2)
parser.add_argument('--ner', action='store_true', help='Whether to include NER taggers', default=False)
parser_args = parser.parse_args()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_map = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'BioClinical': 'emilyalsentzer/Bio_ClinicalBERT',
    'MedBert': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    'bioBert': 'dmis-lab/biobert-base-cased-v1.2',
    'BioMegatron': 'EMBO/BioMegatron345mUncased',
    'deberta': 'microsoft/deberta-v3-large',
}


class Trainer:
    def __init__(self, model_name, train_set, dev_set, test_set, epoch, batch_size, lr, mode, ner, model_path):
        model_name = model_map[model_name]
        self.train_set = train_set
        self.dev_set = dev_set
        self.test_set = test_set
        self.epoch = epoch
        self.batch_size = batch_size
        self.mode = mode
        self.ner = ner
        self.model_path = f'{model_name}_{model_path}'
        self.inv_label_map = get_inv_label()

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=get_label_num())
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = get_device()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.train_dataloader, self.dev_dataloader = get_dataloader(
            tokenizer=self.tokenizer,
            train_filename=self.train_set,
            dev_filename=self.dev_set,
            device=self.device,
            batch_size=self.batch_size,
            mode=self.mode,
            ner=self.ner,
        )
        logging.info(
            f'Model {self.model_path}, lr {parser_args.lr}, batch size {parser_args.batch_size}, epoch {parser_args.epoch}, featurization mode {parser_args.mode}, NER tagger {parser_args.ner}')

    def train(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            logging.info(f'Pre-trained model loaded...')
        else:
            logging.info(f'{self.model_path} model training starts...')
            self.model.to(self.device)
            self.model.train()
            for i in range(self.epoch):
                epoch_loss = 0
                for j, (input_ids, attention_mask, labels) in enumerate(self.train_dataloader):
                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = self.loss_fn(outputs.logits, labels)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                logging.info(f'Epoch {i + 1} out of {self.epoch}: Loss {epoch_loss}')
                self.evaluate()
            torch.save(self.model.state_dict(), self.model_path)

    def evaluate(self):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for i, (input_ids, attention_mask, labels) in enumerate(self.dev_dataloader):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                y_pred.extend(predictions.cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        con_matrix = confusion_matrix(y_true, y_pred)
        logging.info(f'Dev accuracy: {micro_f1:.2f} | Dev F1:  {macro_f1:.2f}')
        logging.info(con_matrix)

    def inference(self, features):
        encoded_inputs = self.tokenizer(features,
                                        add_special_tokens=True,
                                        max_length=MAX_LEN,
                                        truncation=True,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(outputs, dim=-1)
        return self.inv_label_map[pred.item()]


class evidence_trainer(Trainer):
    def __init__(self, entail_trainer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entail_trainer = entail_trainer
        entail_trainer.train()
        self.train_dataloader, self.dev_dataloader = get_dataloader(
            tokenizer=self.tokenizer,
            train_filename=self.train_set,
            dev_filename=self.dev_set,
            device=self.device,
            batch_size=self.batch_size,
            mode=self.mode,
            ner=self.ner,
            evidence_model=self.entail_trainer,
        )

    def inference(self, features):
        encoded_inputs = self.tokenizer(features,
                                        add_special_tokens=True,
                                        max_length=MAX_LEN,
                                        truncation=True,
                                        padding='max_length',
                                        return_attention_mask=True,
                                        return_tensors='pt')
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask).logits
        pred = torch.argmax(outputs, dim=-1)
        return pred.tolist()


if __name__ == '__main__':
    set_random_seeds(42)
    entailTrainer = Trainer(
        model_name=parser_args.model,
        train_set=parser_args.train_file,
        dev_set=parser_args.dev_file,
        test_set=parser_args.test_file,
        epoch=parser_args.epoch,
        batch_size=parser_args.batch_size,
        lr=parser_args.lr,
        mode=parser_args.mode,
        ner=parser_args.ner,
        model_path='task1.pt',
    )
    evidenceTrainer = evidence_trainer(
        model_name=parser_args.model,
        train_set=parser_args.train_file,
        dev_set=parser_args.dev_file,
        test_set=parser_args.test_file,
        epoch=parser_args.epoch,
        batch_size=parser_args.batch_size,
        lr=parser_args.lr,
        mode=parser_args.mode,
        ner=parser_args.ner,
        entail_trainer=entailTrainer,
        model_path='task2.pt'
    )
    evidenceTrainer.train()
