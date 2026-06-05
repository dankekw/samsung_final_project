import evaluate
import numpy as np
import nltk

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from load_datasets import load_summarization_dataset


try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

MODEL_NAME = "google/pegasus-xsum"
metric = evaluate.load("rouge")

def tokenize_dataset(data, tokenizer):
    tokenized_data = tokenizer(
        data["Article"],
        max_length=512,
        padding="max_length",
        truncation=True,
    )

    labels = tokenizer(
        text_target=data["Summary"],
        max_length=64,
        padding="max_length",
        truncation=True,
    )

    labels_with_ignore = [
        [(label if label != tokenizer.pad_token_id else -100) for label in label_seq]
        for label_seq in labels['input_ids']
    ]

    tokenized_data['labels'] = labels_with_ignore
    return tokenized_data

def main():
    tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
    model = PegasusForConditionalGeneration.from_pretrained(MODEL_NAME)

    train_dataset, test_dataset = load_summarization_dataset("danidanou/Reuters_Financial_News", 10000, 5000)
    tokenized_train_dataset = train_dataset.map(
        lambda batch: tokenize_dataset(batch, tokenizer), 
        batched=True
    )
    tokenized_test_dataset = test_dataset.map(
        lambda batch: tokenize_dataset(batch, tokenizer), 
        batched=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
        )

    model = get_peft_model(model, lora_config)
    print(model.print_trainable_parameters())
    

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}

    data_collator = DataCollatorForSeq2Seq(tokenizer, model = model, return_tensors="pt")

    training_args = TrainingArguments(
        output_dir='./pegasus-thourght-lora',
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=10,
        save_steps=500,
        fp16=True,
        dataloader_num_workers=2
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = tokenized_train_dataset,
        data_collator=data_collator,
        ompute_metrics=compute_metrics
    )

    trainer.train()

    test_results = trainer.evaluate(eval_dataset=tokenized_test_dataset, metric_key_prefix="test")
    
    for key in ["test_rouge1", "test_rouge2", "test_rougeL", "test_rougeLsum"]:
        print(f"{key}: {test_results.get(key)}")

if __name__ == "__main__":
    main()