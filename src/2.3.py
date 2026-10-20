# Cell 1 — Install required packages
import sys
import subprocess
print('Python executable:', sys.executable)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'transformers', 'datasets', 'accelerate', 'bitsandbytes', 'scikit-learn', 'matplotlib', 'psutil', 'nbformat'])
print('Installed packages')

# Cell 2 — Mount Google Drive (Optional, for Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    print("Google Drive mounted.")
except ImportError:
    print("Not running in Google Colab. Ensure paths are correct for your local environment.")


# Cell 3 — CONFIG (EDIT these paths before running)
# ==============================================================================
# CRITICAL: This path MUST point to your raw fine-tuned .pth STATE DICT file.
# ==============================================================================
FINETUNED_CHECKPOINT_PATH = "/content/drive/MyDrive/gpt2_agnews_checkpoints1/final_backup_auto/model_state_dict_remapped_and_fixed.pth"

BASE_MODEL_NAME = "gpt2"
OUT_DIR = "/content/drive/MyDrive/AGNEWS_BNB_RESULTS_FROM_DICT"
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_LABELS = 4

import os
os.makedirs(OUT_DIR, exist_ok=True)
print('FINETUNED_CHECKPOINT_PATH =', FINETUNED_CHECKPOINT_PATH)
print('OUT_DIR =', OUT_DIR)


# Cell 4 — Imports & device
import time
import json
import torch
import numpy as np
import tempfile
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import psutil

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Torch:', torch.__version__, ' Device:', DEVICE)


# Cell 5 — Load AG News test set and tokenizer
print('Loading AG News and tokenizer...')
ds = load_dataset('ag_news')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Set tokenizer pad_token to eos_token.")

def preprocess_batch(examples):
    toks = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=MAX_LENGTH)
    toks['labels'] = examples['label']
    return toks

print('Tokenizing test set...')
test_dataset = ds['test'].map(preprocess_batch, batched=True, remove_columns=ds['test'].column_names)
test_dataset.set_format(type='torch')
print('Test size:', len(test_dataset))


# Cell 6 — Function to remap state_dict keys
def remap_and_prepare_state_dict(sd_dict):
    """
    Remaps keys from a custom training checkpoint to the standard Hugging Face
    AutoModelForSequenceClassification format and transposes necessary weights.
    - 'gpt.*' -> 'transformer.*'
    - 'classifier.*' -> 'score.*'
    - Transposes GPT-2 weights that require it.
    """
    remapped_sd = {}
    for k, v in sd_dict.items():
        new_k = k
        # Strip common prefixes from distributed training
        if new_k.startswith('module.'):
            new_k = new_k[len('module.'):]
        if new_k.startswith('model.'):
            new_k = new_k[len('model.'):]

        # Remap base model and classifier keys
        if new_k.startswith('gpt.'):
            new_k = 'transformer.' + new_k[len('gpt.'):]
        elif new_k.startswith('classifier.'):
            new_k = 'score.' + new_k[len('classifier.'):]

        # Transpose weights for specific GPT-2 layers (HF uses Conv1D)
        if v.ndim == 2 and any(x in new_k for x in ("attn.c_attn.weight",
                                                     "attn.c_proj.weight",
                                                     "mlp.c_fc.weight",
                                                     "mlp.c_proj.weight")):
            remapped_sd[new_k] = v.t().contiguous()
        else:
            remapped_sd[new_k] = v
    return remapped_sd

# Cell 7 — Evaluation & Artifact Saving Helpers (same as before)
def evaluate_model(model, dataset, batch_size=BATCH_SIZE, device=DEVICE):
    # FIX 1: Removed redundant .to(device) call, as device_map="auto" handles it.
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            # The input batch still needs to be moved to the primary device.
            b = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(input_ids=b['input_ids'], attention_mask=b['attention_mask'])
            logits = outputs.logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = b['labels'].cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    # FIX 2: Correctly compare all_labels with all_preds for metrics.
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    return {
        'accuracy': float(acc), 'classification_report': report,
        'confusion_matrix': cm.tolist(), 'num_samples': len(all_labels)
    }

def save_artifacts(metrics, mode, out_dir=OUT_DIR):
    json.dump(metrics, open(os.path.join(out_dir, f"metrics_{mode}.json"), "w"), indent=2)
    with open(os.path.join(out_dir, f"report_{mode}.txt"), "w") as f:
        f.write(metrics["classification_report"])
    cm = np.array(metrics["confusion_matrix"])
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    plt.figure(figsize=(6, 6)); plt.imshow(cm, cmap=plt.cm.Blues); plt.title(f"Confusion Matrix ({mode})")
    plt.colorbar(); ticks = np.arange(len(labels)); plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_{mode}.png")); plt.close()
    print(f"✅ Saved all {mode} artifacts to {out_dir}")

# ==============================================================================
# MAIN WORKFLOW: Load Dict -> Convert to HF Model -> Quantize and Evaluate
# ==============================================================================
print("\n--- Starting End-to-End Workflow ---")

# Step 1: Load the raw state dictionary from the .pth file
print(f"Loading state dict from: {FINETUNED_CHECKPOINT_PATH}")
if not os.path.exists(FINETUNED_CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {FINETUNED_CHECKPOINT_PATH}")
ckpt = torch.load(FINETUNED_CHECKPOINT_PATH, map_location='cpu')

# Handle checkpoints that might be nested in another dictionary
raw_sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt

# Step 2: Create a standard HF model structure in full precision
print("Creating full-precision Hugging Face model structure...")
model_fp32 = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=NUM_LABELS
)
model_fp32.config.pad_token_id = tokenizer.pad_token_id

# Step 3: Remap keys and load the state dict into the HF structure
print("Remapping state dict keys for Hugging Face compatibility...")
final_sd = remap_and_prepare_state_dict(raw_sd)
missing_keys, unexpected_keys = model_fp32.load_state_dict(final_sd, strict=False)
print(f"State dict loaded. Missing keys: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
if unexpected_keys or "classifier.bias" in missing_keys and "score.bias" not in final_sd: # often classifier bias is missing but it's ok
    print("NOTE: Some keys were missing/unexpected, which is normal if the checkpoint didn't save a classifier bias.")


# Step 4: Use a temporary directory to save the corrected model and then load quantized versions from it
with tempfile.TemporaryDirectory() as temp_model_dir:
    print(f"\nSaving corrected full-precision model to temporary directory: {temp_model_dir}")
    model_fp32.save_pretrained(temp_model_dir)

    # --- 8-bit Quantization Evaluation ---
    print("\n--- Starting 8-bit Evaluation ---")
    qcfg8 = BitsAndBytesConfig(load_in_8bit=True)
    model_8bit = AutoModelForSequenceClassification.from_pretrained(
        temp_model_dir, quantization_config=qcfg8, device_map="auto"
    )
    print("Loaded 8-bit quantized model from temporary directory.")
    t_start_8bit = time.time()
    metrics_8bit = evaluate_model(model_8bit, test_dataset)
    metrics_8bit["eval_seconds"] = time.time() - t_start_8bit
    print(f"8-bit evaluation finished in {metrics_8bit['eval_seconds']:.2f}s. Accuracy: {metrics_8bit['accuracy']:.4f}")
    save_artifacts(metrics_8bit, "8bit")

    # --- 4-bit Quantization Evaluation ---
    print("\n--- Starting 4-bit Evaluation ---")
    qcfg4 = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False
    )
    model_4bit = AutoModelForSequenceClassification.from_pretrained(
        temp_model_dir, quantization_config=qcfg4, device_map="auto"
    )
    print("Loaded 4-bit quantized model from temporary directory.")
    t_start_4bit = time.time()
    metrics_4bit = evaluate_model(model_4bit, test_dataset)
    metrics_4bit["eval_seconds"] = time.time() - t_start_4bit
    print(f"4-bit evaluation finished in {metrics_4bit['eval_seconds']:.2f}s. Accuracy: {metrics_4bit['accuracy']:.4f}")
    save_artifacts(metrics_4bit, "4bit")

print("\n--- All evaluations complete! ---")

