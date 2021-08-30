from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoTokenizer
from .utils import TextClassificationConfigBase
import os
import numpy as np
from sklearn.metrics import classification_report


TC_Config = TextClassificationConfigBase()


class TC_Trainer():
    def __init__(self,config_path):
        
        if os.path.exists(config_path):
            self.config = TC_Config.load_from_config_file(config_path)
            self.train_args = self.config["train_args"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_checkpoint"], 
                use_fast=True
                )
        else:
            print("can not read ", config_path)
    
    def _load_metric(self,config_name = "mnli"):
        metric = load_metric('glue', config_name)
        return metric

    def _load_dataset(self):
        return load_dataset("json", data_files=self.config["json_files"])

    def _process_data(self):
        def preprocess_function(examples):
            if len(self.config["sentence2_key"]) == 0:
                return self.tokenizer(
                    examples[self.config["sentence1_key"]],
                    truncation =True,
                    max_length = self.config["max_length"]
                    )
            return self.tokenizer(
                examples[self.config["sentence1_key"]],
                examples[self.config["sentence2_key"]], 
                truncation=True,
                max_length = self.config["max_length"]
                )
        dataset = self._load_dataset()

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        return encoded_dataset
    def train(self):
        encoded_dataset = self._process_data()

        model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_checkpoint"], 
            num_labels=self.config["num_labels"])
        args = TrainingArguments(
            **self.train_args
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            # if task != "stsb":
            #     predictions = np.argmax(predictions, axis=1)
            # else:
            #     predictions = predictions[:, 0]
            return classification_report(labels,predictions,output_dict = True)["weighted avg"]
        


        trainer = Trainer(
            model,
            args,
            train_dataset=encoded_dataset["train"],
            eval_dataset = encoded_dataset["valid"],
            # test_dataset = encoded_dataset["test"],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model()
        trainer.evaluate()
def gen_config_file(config_path):
    TC_Config.gen_config_template(config_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", help="text classification config path", type=str)
    args = parser.parse_args()
    tctrainer = TC_Trainer(args.config_path)


