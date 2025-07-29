
from datasets import load_from_disk
import torch
from transformers import ViTImageProcessor
import numpy as np
from transformers import ViTForImageClassification, TrainingArguments, EarlyStoppingCallback, Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import gc
import matplotlib.pyplot as plt
import evaluate
import os

os.makedirs("results", exist_ok=True)


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    rgb_images = [img.convert('RGB') for img in example_batch['image']]
    inputs = processor(rgb_images, return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs


def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1) # Get the predicted class
    labels = p.label_ids # Labels

    # Calculate accuracy, precision, recall, f1 (with average='weighted' for multi-class and to account for label imbalance)
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)
    precision = precision_metric.compute(predictions=preds, references=labels, average='weighted')
    recall = recall_metric.compute(predictions=preds, references=labels, average='weighted')
    f1 = f1_metric.compute(predictions=preds, references=labels, average='weighted')

    # Return dictionary of metrics
    return {
        "accuracy": accuracy["accuracy"],
        "precision": precision["precision"],
        "recall": recall["recall"],
        "f1": f1["f1"]
    }


def save_confusion_matrix(trainer, dataset, labels, exp_name):
    try:
        outputs = trainer.predict(dataset)
        y_true = outputs.label_ids
        y_pred = outputs.predictions.argmax(1)
        
        all_labels = list(range(len(labels)))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        
        plt.figure(figsize=(10, 8))
        disp.plot(xticks_rotation=45)
        plt.title(f"Confusion Matrix - {exp_name}")
        plt.tight_layout()
        
        plt.savefig(f"results/confusion_matrix_{exp_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved for {exp_name}")
        
    except Exception as e:
        print(f"Error creating confusion matrix for {exp_name}: {e}")



model_name_or_path = 'google/vit-base-patch16-224-in21k'

for dataset in ["dataset_1", "dataset_2", "dataset_3"]:
    for image_type in ["gri", "urz", "side_by_side"]:
        exp_name = f"{dataset}_{image_type}"
        print(f"\n{'='*60}")
        print(f"Training {exp_name.upper()}")
        print(f"{'='*60}")

        try:
            # Load processor
            processor = ViTImageProcessor.from_pretrained(model_name_or_path)

            # Load dataset
            print(f"Loading dataset: {dataset}_dictionary_{image_type}_images_224_x_224")
            dataset_dict = load_from_disk(f"{dataset}_dictionary_{image_type}_images_224_x_224")

            # Apply transforms
            prepared_ds = dataset_dict.with_transform(transform)

            # Get labels
            labels = dataset_dict['train'].features['label'].names
            print(f"Classes: {labels}")
            print(f"Training samples: {len(dataset_dict['train'])}")
            print(f"Validation samples: {len(dataset_dict['validation'])}")
            print(f"Test samples: {len(dataset_dict['test'])}")


            # Load model
            print("Loading model...")
            model = ViTForImageClassification.from_pretrained(
                model_name_or_path,
                num_labels=len(labels),
                id2label={str(i): c for i, c in enumerate(labels)},
                label2id={c: str(i) for i, c in enumerate(labels)}
            )

            # Training arguments
            training_args = TrainingArguments(
            output_dir=f"./vit_stellar_classification_{dataset}_{image_type}_images",
            per_device_train_batch_size=32,
            eval_strategy="steps",
            num_train_epochs=50,
            fp16=True,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=1000,
            learning_rate=1e-5,
            save_total_limit=10,
            metric_for_best_model="accuracy",
            remove_unused_columns=False,
            push_to_hub=False,
            report_to='tensorboard',
            load_best_model_at_end=True,
            )

            # Early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001
            )

            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=collate_fn,
                compute_metrics=compute_metrics,
                train_dataset=prepared_ds["train"],
                eval_dataset=prepared_ds["validation"],
                tokenizer=processor,
                callbacks=[early_stopping_callback]
            )

            # Train model
            print("Starting training...")
            train_results = trainer.train()

            # Save model and metrics
            trainer.save_model()
            trainer.log_metrics("train", train_results.metrics)
            trainer.save_metrics("train", train_results.metrics)
            trainer.save_state()

            # Evaluate on test set
            print("Evaluating on test set...")
            test_results = trainer.predict(prepared_ds["test"])
            trainer.log_metrics("test", test_results.metrics)
            trainer.save_metrics("test", test_results.metrics)


            # outputs = trainer.predict(prepared_ds['test'])
            # y_true = outputs.label_ids
            # y_pred = outputs.predictions.argmax(1)
            # labels = prepared_ds['train'].features['label'].names
            # cm = confusion_matrix(y_true, y_pred)
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            # disp.plot(xticks_rotation=45)

            # Create confusion matrix
            save_confusion_matrix(trainer, prepared_ds['test'], labels, exp_name)
            
            print(f"✓ Completed training for {exp_name}")
            print(f"Final test accuracy: {test_results.metrics.get('test_accuracy', 'N/A'):.4f}")

        except Exception as e:
            print(f"Error during training {exp_name}: {e}")
            continue
        
        finally:
            # Clean up memory
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            if 'dataset_dict' in locals():
                del dataset_dict
            if 'prepared_ds' in locals():
                del prepared_ds
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n" + "="*60)
print("ALL TRAINING COMPLETED!")
print("="*60)