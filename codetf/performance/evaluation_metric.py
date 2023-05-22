import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import EvalPrediction

class EvaluationMetric:
    def __init__(self, metric, tokenizer):
        self.metric = metric
        self.tokenizer = tokenizer

    def compute_metrics(self, eval_pred: EvalPrediction):
        predictions = self.tokenizer.batch_decode(eval_pred.predictions, skip_special_tokens=True)
        references = self.tokenizer.batch_decode(eval_pred.label_ids, skip_special_tokens=True)

        if self.metric == "bleu":
            return {"bleu": sacrebleu.corpus_bleu(predictions, [references]).score}
        elif self.metric == "f1":
            return {"f1": self.compute_f1_score(predictions, references)}
        elif self.metric == "precision":
            return {"precision": self.compute_precision_score(predictions, references)}
        elif self.metric == "recall":
            return {"recall": self.compute_recall_score(predictions, references)}
        elif self.metric == "rouge":
            rouge_scores = self.compute_rouge(predictions, references)
            return {f"rouge_{key}": value for key, value in rouge_scores.items()}
        elif self.metric == "meteor":
            return {"meteor": self.compute_meteor(predictions, references)}
        else:
            raise ValueError("Invalid metric specified")

    def compute_f1_score(self, hypotheses, references):
        # Calculate F1 score for your use case, this is just a sample
        return f1_score(hypotheses, references, average='weighted')

    def compute_precision_score(self, hypotheses, references):
        # Calculate precision score for your use case, this is just a sample
        return precision_score(hypotheses, references, average='weighted')

    def compute_recall_score(self, hypotheses, references):
        # Calculate recall score for your use case, this is just a sample
        return recall_score(hypotheses, references, average='weighted')

    def compute_rouge(self, hypotheses, references):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hypotheses)]
        rouge1 = sum([score['rouge1'].fmeasure for score in scores]) / len(scores)
        rougeL = sum([score['rougeL'].fmeasure for score in scores]) / len(scores)
        return {"rouge1": rouge1, "rougeL": rougeL}

    def compute_meteor(self, hypotheses, references):
        return sum([meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)]) / len(hypotheses)
