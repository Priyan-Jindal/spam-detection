import pytorch_lightning as pl
import torch as th
from torch.utils import data as th_data
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

pipe = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-sms-spam-detection")

from sklearn.model_selection import train_test_split



class SpamDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")
        self.loss = th.nn.CrossEntropyLoss(reduction='none')
        self.seq_length = 102 # 95th percentile
        self.batch_size = 15
        self.learning_rate = 2e-4
        self.epochs = 5
        self.loss_outputs = []
        self.acc_outputs = []

    def prepare_data(self):
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")

        def _tokenize(data):
            data['input_ids'] = tokenizer.batch_encode_plus(data['messageBody'], max_length=self.batch_size, pad_to_max_length=True)['input_ids']
            return data

        def _prepare_dataset(split_name):
            data = pd.read_csv('https://raw.githubusercontent.com/PriyanJindal/spamdetection/main/inboxV4.csv')

            # Convert the 'spam' column to integer
            data["spam"] = data["spam"].astype(int)

            # Separate the features and labels
            X = data['messageBody']
            y = data['spam']

            # Use stratified sampling to split the data
            if split_name == 'train':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                df = pd.DataFrame({'messageBody': X_train, 'spam': y_train})
            elif split_name == 'test':
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
                df = pd.DataFrame({'messageBody': X_test, 'spam': y_test})
            else:
                raise ValueError("Invalid split_name. Use 'train' or 'test'.")


            # Convert the pandas DataFrame to a datasets.Dataset object
            dataset = datasets.Dataset.from_pandas(df)

            # Tokenize and preprocess the data using the _tokenize function
            dataset = dataset.map(_tokenize, batched=True)

            # Set the format of the dataset for Torch tensors
            dataset.set_format(type='torch', columns=['input_ids', 'spam'])

            return dataset

        self.train_ds, self.test_ds = map(_prepare_dataset, ('train', 'test'))


    def forward(self, input_ids):
        mask = (input_ids != 0).float()  # Mask padding tokens
        logits = self.model(input_ids, attention_mask=mask).logits  # Compute logits
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        labels = batch['spam']

        # Perform forward pass
        logits = self.forward(input_ids)

        # Calculate the loss
        labels_tensor = th.tensor(labels, dtype=th.long) #converting labels into tensor
        loss = self.loss(logits, labels_tensor).mean()
        accuracy = (logits.argmax(-1) == labels_tensor).float().mean()

        self.log('train_loss', loss)
        self.log('train_acc', accuracy)

        return {'loss': loss, 'train_acc': accuracy, 'log': {'train_loss': loss, 'train_acc': accuracy}}


    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['spam'])
        accuracy = (logits.argmax(-1) == batch['spam']).float()
        
        # Filter messages with logits less than 0.65
        filtered_indices = (logits.max(-1).values < 0.65)
        filtered_messages = [batch['messageBody'][i] for i in range(len(batch['messageBody'])) if filtered_indices[i]]
        
        self.loss_outputs.extend(loss)
        self.acc_outputs.extend(accuracy)
        
        return {'loss': loss, 'acc': accuracy, 'filtered_messages': filtered_messages}

    def on_validation_epoch_end(self):
        loss_average = th.stack(self.loss_outputs).mean()
        acc_average = th.stack(self.acc_outputs).mean()
        out = {'val_loss': loss_average, 'val_acc': acc_average}
        self.loss_outputs.clear()
        self.acc_outputs.clear()
        return {**out, 'log': out}

    def train_dataloader(self):
        train_sampler = th.utils.data.RandomSampler(self.train_ds)  # Randomly samples data from the training dataset
        train_loader = th.utils.data.DataLoader(self.train_ds, batch_size=self.batch_size, sampler=train_sampler)
        return train_loader

    def val_dataloader(self):
        val_sampler = th.utils.data.SequentialSampler(self.test_ds)  # Sequentially samples data from the validation dataset
        val_loader = th.utils.data.DataLoader(self.test_ds, batch_size=self.batch_size, sampler=val_sampler)
        return val_loader

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def predict_examples(self, examples):
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")

        for example in examples:
            input_text = example['messageBody']
            input_ids = tokenizer.encode(input_text, max_length=self.seq_length, truncation=True, padding='max_length', return_tensors='pt')
            logits = self.forward(input_ids).squeeze()

            predicted_label = th.argmax(logits).item()
            predicted_prob = th.softmax(logits, dim=-1)[predicted_label].item()

            if predicted_label == 0:
                predicted_class = 'ham'
            else:
                predicted_class = 'spam'

            print(f"Input Text: {input_text}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Predicted Probability: {predicted_prob:.4f}\n")