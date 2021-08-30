from transformers import AutoTokenizer
import json
from tqdm import tqdm
from sklearn.metrics import classification_report
from .bert_text_classify_pipeline import TextClassificationPipeline
from .bert_model import BertForSequenceClassification
from .bert_model import BertCNNForSequenceClassification
import copy



class PredictBase():
    def __init__(self,model_checkpoint,device = 0):
        model,tokenizer = self.load_checkpoint(model_checkpoint)

        self.tc_pipline = TextClassificationPipeline(
            model = model,
            tokenizer = tokenizer,
            device = device,
            return_all_scores = True
            )
    def load_checkpoint(self,model_checkpoint):
        model = BertCNNForSequenceClassification.from_pretrained( 
            model_checkpoint
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_checkpoint, 
            use_fast=True
            )
        return model,tokenizer
    def pred_jsonline_file(self,
        data_path = "./json_data/task_1/v1/test.json",
        text_field = "text",
        write_to_json = True,
        output_json_path = "./prediction.json"
        ):
        res = []
        with open(data_path,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                data = json.loads(line)
                temp = copy.deepcopy(data)
                text = temp[text_field]
                tc_pred_res = self.tc_pipline(text)
                temp["prediction"] = tc_pred_res
                res.append(temp)
        if write_to_json:
            json.dump(res,open(output_json_path,"w+",encoding = "utf-8"),indent=4,ensure_ascii=False)
        return res
    def model_report(self,
        y_true, 
        y_pred,
        digits = 4,
        print_classification_report = True,
        return_classification_report_dict  =True
        ):
        if print_classification_report:
            print(classification_report(
                y_true, 
                y_pred,
                digits = digits)
            )
        if return_classification_report_dict:
            return classification_report(
                y_true, 
                y_pred,
                digits = digits,
                output_dict = True
                )
        else:
            return None