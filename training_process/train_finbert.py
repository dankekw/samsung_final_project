import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, classification_report
from load_datasets import load_sentiment_dataset


MODEL_NAME = "ProsusAI/finbert"


def tokenize(tokenizer, sample):
    return tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        max_length=200
    )


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    train_data, test_data = load_sentiment_dataset("nickmuchi/financial-classification")
    tokenized_train_dataset = train_data.map(
        lambda batch: tokenize(batch, tokenizer), 
        batched=True
    )

    tokenized_test_dataset = test_data.map(
        lambda batch: tokenize(batch, tokenizer), 
        batched=True
    )

    tokenized_train_dataset = tokenized_train_dataset.train_test_split(test_size=0.1)
    train_data = tokenized_train_dataset["train"]
    val_data = tokenized_train_dataset["test"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
        task_type=TaskType.SEQ_CLS
    )

    model_finbert = get_peft_model(model_finbert, lora_config)
    print(model_finbert.print_trainable_parameters())


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted")
        }


    training_args = TrainingArguments(
        output_dir="./finbert-lora",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        num_train_epochs=10,
        learning_rate=1e-4,
        logging_steps=25,
        metric_for_best_model="f1",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model_finbert,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    trainer.train()

    preds_output = trainer.predict(tokenized_test_dataset)

    logits = preds_output.predictions
    labels = preds_output.label_ids
    preds = np.argmax(logits, axis=1)

    print(classification_report(labels, preds))

if __name__ == "__main__":
    main()