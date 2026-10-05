import os
import gc
import json
import logging
import torch
import torchaudio
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import cer
import warnings
import random
import sys
from datetime import datetime

# Import for LR Scheduler
from torch.optim import lr_scheduler

warnings.filterwarnings("ignore", module="transformers")

# === Configuration ===

# Output directory for this specific training run
OUTPUT_DIR = r"D:\whisper_training8" # Updated output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CHECKPOINT DIRECTORY
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "train.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout) # Explicitly use stdout
    ]
)
logger = logging.getLogger(__name__)

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# === MODEL LOADING ===
# Start from the best known checkpoint based on previous results.txt analysis
MODEL_NAME = r"D:\whisper_training5\checkpoints\checkpoint-11000" # Best CER: 0.1248

# === HYPERPARAMETERS ===

# 🔷 KEY IMPROVEMENT 1: Much Lower and Stable Learning Rate Strategy
# Using a very low, constant learning rate for stability initially.
# Can be reduced dynamically if needed.
INITIAL_LEARNING_RATE = 5e-7 # <<< CRITICAL CHANGE: Significantly lowered
WEIGHT_DECAY = 0.05  # Slightly increased weight decay for regularization

# Batch size and accumulation
BATCH_SIZE = 1
GRAD_ACCUM = 32     # Effective batch size = 1 * 32 = 32
MAX_TRAINING_STEPS = 30000 # Limit total steps to prevent overfitting if early stopping fails
SAVE_STEPS = 1000
EVAL_STEPS = 1000   # Evaluate more frequently

# Data loading
CHUNK_SIZE = 2000
EXPECTED_MEL_LENGTH = 3000

# KEY IMPROVEMENT 2: Enhanced Early Stopping and Situation-Saving Parameters
EARLY_STOPPING_PATIENCE = 5  # Number of evaluations without improvement before stopping
EARLY_STOPPING_MIN_DELTA = 0.005 # Minimum CER improvement considered significant

#  NEW: Situation-Saving Parameters
CRITICAL_CER_SPIKE_THRESHOLD = 1.5  # If CER > best_cer * this, it's considered a spike
CRITICAL_CER_SPIKE_PATIENCE = 3       # Tolerate this many spike epochs before action
CRITICAL_LR_REDUCTION_FACTOR = 0.5 # Factor to reduce LR by if CER spikes
RESTORE_BEST_MODEL_IF_CER_ABOVE = 2.0 # If CER exceeds this, immediately restore best model

# Memory and cleanup
CLEANUP_INTERVAL = 200
MAX_GPU_MEMORY_GB = 8.0

# Dataset paths - Please update these paths to your actual dataset locations
TRAIN_JSON = r"D:\Training_datasets\data_aishell\transcript\train.json"
EVAL_JSON = r"D:\Training_datasets\data_aishell\transcript\test.json"
TRAIN_WAV_DIR = r"D:\Training_datasets\data_aishell\wav\train"
EVAL_WAV_DIR = r"D:\Training_datasets\data_aishell\wav\test"

# Log paths
MISSING_LOG = os.path.join(OUTPUT_DIR, "missing_files.log")
SUCCESS_LOG = os.path.join(OUTPUT_DIR, "success_files.log")
CHUNK_STATE_FILE = os.path.join(OUTPUT_DIR, "chunk_state.json")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "results.csv")

# Path to store the best model checkpoint temporarily (in case we need to restore)
BEST_MODEL_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model_temp")

# Clear logs at the start of a fresh run
for log_file in [MISSING_LOG, SUCCESS_LOG]:
    if os.path.exists(log_file):
        open(log_file, 'w').close()

# Initialize results CSV
if not os.path.exists(RESULTS_CSV):
    pd.DataFrame(columns=["step", "loss", "cer", "timestamp", "learning_rate", "note"]).to_csv(RESULTS_CSV, index=False)
    logger.info(f"Created results.csv at {RESULTS_CSV}")

# === Functions ===

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, item)):
            try:
                step = int(item.split("-")[-1])
                checkpoints.append((step, os.path.join(checkpoint_dir, item)))
            except ValueError:
                continue
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]  # Return path of latest checkpoint
    return None

def load_dataset_from_json(json_path: str, audio_root_dir: str):
    """Load dataset from JSON and audio directory."""
    logger.info(f"Loading dataset from: {json_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    # Build file index: map filename (without path) to full path
    logger.info("Building file index...")
    file_index = {}
    for root, _, files in os.walk(audio_root_dir):
        for file in files:
            if file.lower().endswith(".wav"):
                file_index[file.lower()] = os.path.join(root, file)
    logger.info(f"Found {len(file_index)} .wav files")

    # Load transcript
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            raise ValueError("JSON must be list or dict")

    data_dict = {"filename": [], "chinese_text": []}
    for item in data:
        if not isinstance(item, dict) or "filename" not in item or "chinese_text" not in item:
            continue
        wav_filename = item["filename"].strip()
        if not wav_filename.lower().endswith(".wav"):
            wav_filename = wav_filename + ".wav"
        full_path = file_index.get(wav_filename.lower())
        if not full_path:
            logger.warning(f"Audio file not found for: {wav_filename}")
            continue
        text = item["chinese_text"].strip().replace(" ", "")
        if not text:
            continue
        # Add sample validation
        if len(text) > 500 or text.count(text[:10]) > 3:
            logger.warning(f"Skipping problematic sample: {wav_filename}")
            continue
        data_dict["filename"].append(full_path)
        data_dict["chinese_text"].append(text)

    dataset = Dataset.from_dict(data_dict)
    logger.info(f"Dataset created: {len(dataset)} samples")
    return dataset

def get_next_chunk_indices(dataset_len, chunk_size=CHUNK_SIZE):
    logger.debug(f"Getting next chunk indices. Dataset length: {dataset_len}, chunk_size: {chunk_size}")

    if not os.path.exists(CHUNK_STATE_FILE) or dataset_len <= 0:
        start_idx = 0
        logger.info("No chunk state file found or empty dataset. Starting from index 0")
    else:
        try:
            with open(CHUNK_STATE_FILE, "r") as f:
                state = json.load(f)
                start_idx = state.get("next_start", 0)
                epoch_count = state.get("epoch_count", 0)
            logger.info(f"Loaded chunk state. Next start index: {start_idx}, Epoch count: {epoch_count}")
        except Exception as e:
            logger.warning(f"Failed to load chunk state file: {e}. Starting from index 0")
            start_idx = 0
            epoch_count = 0

    # Handle case where we've completed full dataset cycle
    if start_idx >= dataset_len:
        logger.info(f"Completed dataset cycle (start_idx={start_idx} >= dataset_len={dataset_len}). Resetting to 0")
        start_idx = 0
        epoch_count = epoch_count + 1 if 'epoch_count' in locals() else 1
        save_next_chunk_index(start_idx, epoch_count)

    end_idx = min(start_idx + chunk_size, dataset_len)
    indices = list(range(start_idx, end_idx))
    next_start_idx = end_idx if end_idx < dataset_len else 0

    logger.info(f"Selected chunk indices: {start_idx} to {end_idx-1} (total: {len(indices)})")
    logger.info(f"Next chunk will start at index: {next_start_idx}")
    return indices, next_start_idx

def save_next_chunk_index(next_idx, epoch_count=0):
    """Save the next chunk index to track progress."""
    logger.info(f"Saving chunk state: next_start = {next_idx}, epoch_count = {epoch_count} to {CHUNK_STATE_FILE}")
    try:
        with open(CHUNK_STATE_FILE, "w") as f:
            json.dump({"next_start": next_idx, "epoch_count": epoch_count, "timestamp": datetime.now().isoformat()}, f)
        logger.info(f"✅ Progress saved: next training starts at index {next_idx}")
    except Exception as e:
        logger.error(f"❌ Failed to save chunk state: {e}")

def detect_repetition(text, threshold=0.3):
    """Detect if text has excessive repetition"""
    if len(text) < 20:
        return False
    words = text.split()
    if len(words) < 10:
        return False
    for i in range(len(words) - 3):
        substring = ' '.join(words[i:i+2])
        if text.count(substring) > 3:
            return True
    return False

def cleanup_memory(force=False):
    """Clean up memory and GPU cache"""
    if DEVICE == "cuda":
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.memory_allocated() / 1e9
            logger.debug(f"GPU Memory: {gpu_memory_gb:.2f} GB")
            if force or gpu_memory_gb > MAX_GPU_MEMORY_GB:
                logger.info("🧹 Cleaning GPU memory...")
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("✅ GPU memory cleanup completed")
    else:
        if force:
            logger.info("🧹 Cleaning CPU memory...")
            gc.collect()
            logger.info("✅ CPU memory cleanup completed")

def final_evaluation(output_dir, eval_json, eval_wav_dir):
    """Run final evaluation on the full test set."""
    logger.info("📊 Running final evaluation on test set...")
    try:
        processor = WhisperProcessor.from_pretrained(output_dir, language="chinese", task="transcribe")
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
            logger.info("Pad token set to EOS token.")

        model = WhisperForConditionalGeneration.from_pretrained(output_dir).to(DEVICE)
        model.eval()
        eval_dataset = load_dataset_from_json(eval_json, eval_wav_dir) # Load full test set

        if len(eval_dataset) == 0:
            logger.error("❌ Final evaluation dataset is empty!")
            return 1.0

        predictions = []
        references = []

        # Evaluate all samples in the final test set
        for i, item in enumerate(eval_dataset):
            try:
                if not os.path.exists(item["filename"]):
                    logger.warning(f"File not found: {item['filename']}")
                    continue

                waveform, sr = torchaudio.load(item["filename"])
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)

                inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding="longest")
                input_tensor = inputs.input_features.to(DEVICE)

                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_tensor,
                        language="chinese",
                        task="transcribe",
                        suppress_tokens=[-1],
                        begin_suppress_tokens=[220, 50257],
                        max_new_tokens=100,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=2,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.92,
                        top_k=40,
                        length_penalty=0.6,
                        early_stopping=True,
                        num_beams=3
                    )
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                if detect_repetition(transcription):
                    logger.warning(f"Repetitive transcription detected: {transcription[:50]}...")

                if len(transcription.strip()) > 0:
                    predictions.append(transcription)
                    references.append(item["chinese_text"])
                    if i % 100 == 0: # Log progress occasionally
                         logger.debug(f"Final eval sample {i}: '{transcription[:30]}...'")

            except Exception as e:
                logger.warning(f"Failed to process {item['filename']}: {e}")

        if predictions and references:
            final_cer = cer(references, predictions)
            logger.info(f"🎉 Final CER on FULL test set: {final_cer:.4f}")
            logger.info(f"Evaluated {len(predictions)} samples successfully")
        else:
            final_cer = 1.0
            logger.warning("No predictions generated for final evaluation")

        # Save to file
        with open(os.path.join(output_dir, "final_cer.txt"), "w") as f:
            f.write(f"Final CER: {final_cer:.4f}\n")
            f.write(f"Total evaluated files: {len(references)}\n")

        cleanup_memory(force=True)
        return final_cer
    except Exception as e:
        logger.error(f"Error in final evaluation: {e}")
        logger.exception(e)
        return 1.0

def auto_train():
    logger.info("🚀 Starting Stable Fine-tuning Script with Situation-Saving...")
    # Load datasets only once
    train_dataset_full = load_dataset_from_json(TRAIN_JSON, TRAIN_WAV_DIR)
    eval_dataset_full = load_dataset_from_json(EVAL_JSON, EVAL_WAV_DIR)

    total_files = len(train_dataset_full)
    total_eval = len(eval_dataset_full)
    logger.info(f"Total training files found: {total_files}")
    logger.info(f"Total evaluation files found: {total_eval}")

    if total_files == 0:
        logger.error("❌ No training files found! Check your dataset paths.")
        return False

    # Check for resuming from OUR OWN checkpoints (not from previous runs necessarily)
    resume_path = None
    step = 0
    epoch_count = 0
    regular_resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
    if regular_resume_path:
        logger.info(f"🔄 Resuming from OUR checkpoint: {regular_resume_path}")
        try:
            resume_path = regular_resume_path
            step = int(os.path.basename(resume_path).split("-")[-1])
            logger.info(f"Starting from step: {step}")
        except Exception as e:
            logger.warning(f"Failed to parse our checkpoint: {e}")

    # Load model and processor from the specified starting point (or resume)
    if resume_path:
        logger.info(f"Loading model from OUR checkpoint: {resume_path}")
        model = WhisperForConditionalGeneration.from_pretrained(resume_path)
        processor = WhisperProcessor.from_pretrained(resume_path, language="chinese", task="transcribe")
    else:
        logger.info(f"🆕 Starting FRESH training from model: {MODEL_NAME}")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="chinese", task="transcribe")

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        logger.info("Pad token was not set. Set to EOS token as a fallback.")

    # Configure model for Chinese transcription, # Maintain standard dropout
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
    model.config.use_cache = False
    model.config.dropout = 0.25
    model.config.attention_dropout = 0.25

    model = model.to(DEVICE)

    # 🔷 KEY IMPROVEMENT 3: Use a simple, constant, LOW learning rate optimizer
    # The learning rate can be reduced dynamically later.
    current_lr = INITIAL_LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, eps=1e-8, weight_decay=WEIGHT_DECAY)
    logger.info(f"Initialized optimizer with INITIAL CONSTANT LOW LR: {current_lr} and WEIGHT_DECAY: {WEIGHT_DECAY}")

    # No learning rate scheduler is used for maximum stability initially.
    # The learning rate will be managed manually if needed.

    loss_history = []
    # 🔷 KEY IMPROVEMENT 4: Enhanced tracking variables for situation-saving
    cer_history = []
    best_cer = float('inf')
    best_cer_step = 0
    epochs_without_improvement = 0 # Counter for standard early stopping

    # 🔷 NEW: Situation-Saving tracking variables
    last_good_cer = float('inf') # CER from the previous evaluation
    consecutive_spike_epochs = 0 # Counter for CER spikes
    # Ensure temp best model path exists
    os.makedirs(BEST_MODEL_CHECKPOINT_PATH, exist_ok=True)

    def collate_fn(batch):
        input_features = []
        labels = []
        for item in batch:
            path = item["filename"]
            text = item["chinese_text"]
            try:
                if not os.path.exists(path):
                    logger.warning(f"File not found: {path}")
                    continue
                waveform, sr = torchaudio.load(path)
                if waveform.numel() == 0:
                    logger.warning(f"Empty audio file: {path}")
                    continue
                if waveform.abs().max() < 0.01:
                    logger.warning(f"Nearly silent audio: {path}")
                    continue

                max_duration = 30
                max_samples = 16000 * max_duration
                if waveform.shape[1] > max_samples:
                    waveform = waveform[:, :max_samples]
                    logger.warning(f"Truncated long audio: {path}")

                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                max_len = 16000 * 30
                if waveform.shape[1] > max_len:
                    waveform = waveform[:, :max_len]
                with torch.no_grad():
                    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding="longest")
                feat = inputs.input_features[0]
                current_len = feat.shape[1]
                if current_len < EXPECTED_MEL_LENGTH:
                    pad_amount = EXPECTED_MEL_LENGTH - current_len
                    feat = F.pad(feat, (0, pad_amount), value=0)
                elif current_len > EXPECTED_MEL_LENGTH:
                    feat = feat[:, :EXPECTED_MEL_LENGTH]
                input_features.append(feat)

                encoded = processor.tokenizer(text.strip(), return_tensors="pt").input_ids[0]
                encoded[encoded == processor.tokenizer.pad_token_id] = -100
                labels.append(encoded)

                with open(SUCCESS_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{os.path.basename(path)}\n")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                with open(MISSING_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{os.path.basename(path)}: {e}\n")
                return None

        if not input_features:
            return None
        input_features = torch.stack(input_features)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {"input_features": input_features, "labels": labels}

    def evaluate_cer(model, eval_dataset, step=0):
        model.eval()
        predictions = []
        references = []
        logger.info(f"Starting CER evaluation on {len(eval_dataset)} samples")

        if len(eval_dataset) == 0:
            logger.error("❌ Evaluation dataset is empty!")
            model.train()
            return 1.0

        # Limit evaluation size for speed, but ensure it's representative
        eval_limit = min(100, len(eval_dataset)) # Evaluate first 100 or all if < 100
        eval_count = 0
        success_count = 0

        logger.info(f"Evaluating up to {eval_limit} samples...")

        for i, item in enumerate(eval_dataset):
            if eval_count >= eval_limit:
                break
            try:
                if not os.path.exists(item["filename"]):
                    logger.warning(f"File not found: {item['filename']}")
                    eval_count += 1
                    continue

                waveform, sr = torchaudio.load(item["filename"])
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(0, keepdim=True)

                inputs = processor(
                    waveform.squeeze(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding="longest"
                )
                input_tensor = inputs.input_features.to(DEVICE)

                with torch.no_grad():
                    predicted_ids = model.generate(
                        input_tensor,
                        language="chinese",
                        task="transcribe",
                        suppress_tokens=[-1],
                        begin_suppress_tokens=[220, 50257],
                        max_new_tokens=100,
                        repetition_penalty=1.3,
                        no_repeat_ngram_size=2,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.92,
                        top_k=40,
                        length_penalty=0.6,
                        early_stopping=True,
                        num_beams=3
                    )
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

                if len(transcription.strip()) > 0:
                    predictions.append(transcription)
                    references.append(item["chinese_text"])
                    success_count += 1
                eval_count += 1

            except Exception as e:
                logger.warning(f"❌ Failed to process {item.get('filename', 'unknown')}: {e}")
                eval_count += 1
                continue

        logger.info(f"Evaluation results: {success_count} successful, {eval_count} attempted")

        if not references or not predictions:
            logger.error("❌ No valid predictions or references collected for CER!")
            model.train()
            return 1.0

        try:
            current_cer = cer(references, predictions)
            logger.info(f"Sample pred: {predictions[0] if predictions else 'None'}")
            logger.info(f"Sample ref: {references[0] if references else 'None'}")
            logger.info(f"Computed CER: {current_cer:.4f}")
        except Exception as e:
            logger.error(f"Error computing CER: {e}")
            current_cer = 1.0

        model.train()
        cleanup_memory()
        return current_cer

    # Training loop
    model.train()
    optimizer.zero_grad()

    processed_files = 0
    training_complete = False
    epochs_completed = 0

    logger.info(f"Starting training loop...")
    logger.info(f"Initial Constant Learning Rate: {current_lr}")
    logger.info(f"Max Training Steps: {MAX_TRAINING_STEPS}")
    logger.info(f"Standard Early Stopping Patience: {EARLY_STOPPING_PATIENCE} evaluations")
    logger.info(f"Standard Early Stopping Min Delta: {EARLY_STOPPING_MIN_DELTA}")
    logger.info(f"Critical CER Spike Threshold Multiplier: {CRITICAL_CER_SPIKE_THRESHOLD}")
    logger.info(f"Critical CER Spike Patience: {CRITICAL_CER_SPIKE_PATIENCE} evaluations")
    logger.info(f"Critical LR Reduction Factor: {CRITICAL_LR_REDUCTION_FACTOR}")
    logger.info(f"Absolute CER Threshold for Restoration: {RESTORE_BEST_MODEL_IF_CER_ABOVE}")
    logger.info(f"Saving checkpoints to: {CHECKPOINT_DIR}")

    while not training_complete and step < MAX_TRAINING_STEPS:
        # Get current chunk
        train_indices, next_train_start = get_next_chunk_indices(total_files, CHUNK_SIZE)
        logger.info(f"Current chunk indices: {len(train_indices)}, next_train_start: {next_train_start}")

        if next_train_start == 0 and len(train_indices) > 0:
            epochs_completed += 1
            logger.info(f"Completed epoch {epochs_completed}")

        if len(train_indices) == 0:
            logger.warning("No training indices available. Breaking loop.")
            break

        # Use a consistent evaluation chunk
        eval_indices = list(range(min(100, len(eval_dataset_full)))) # Evaluate first 100 or all if < 100
        logger.info(f"Using fixed evaluation chunk with {len(eval_indices)} samples")

        train_dataset = train_dataset_full.select(train_indices)
        eval_dataset = eval_dataset_full.select(eval_indices)

        current_chunk_size = len(train_dataset)
        logger.info(f"Training on {current_chunk_size} files (indices {train_indices[0]} to {train_indices[-1]})")

        if current_chunk_size == 0:
            logger.warning("No training data in current chunk. Skipping iteration.")
            save_next_chunk_index(next_train_start, epochs_completed)
            continue

        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)

        for batch_idx, batch in enumerate(dataloader):
            if batch is None or batch.get("input_features") is None:
                continue

            batch = {k: v.to(DEVICE) for k, v in batch.items() if v is not None}

            try:
                outputs = model(input_features=batch["input_features"], labels=batch["labels"])
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                step += 1

                if step % GRAD_ACCUM == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    current_loss = loss.item() * GRAD_ACCUM
                    logger.info(f"Step {step}, Loss: {current_loss:.4f}")
                    loss_history.append(current_loss)

                # Periodic actions (Save, Eval, Cleanup)
                if step % min(SAVE_STEPS, EVAL_STEPS) == 0:
                    # Save checkpoint periodically
                    if step % SAVE_STEPS == 0:
                        ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint-{step}")
                        model.save_pretrained(ckpt_path)
                        processor.save_pretrained(ckpt_path)
                        logger.info(f"💾 Saved checkpoint: {ckpt_path}")

                    # Evaluate CER periodically
                    if step % EVAL_STEPS == 0:
                        try:
                            current_cer = evaluate_cer(model, eval_dataset, step)
                            logger.info(f"🔍 CER at step {step}: {current_cer:.4f}")

                            # 🔷 Track CER for monitoring and situation-saving
                            cer_history.append(current_cer)

                            # 🔷 --- START OF ENHANCED MONITORING & SITUATION-SAVING LOGIC ---

                            # 1. Track Best Model and Save Temporarily
                            if current_cer < best_cer:
                                best_cer = current_cer
                                best_cer_step = step
                                epochs_without_improvement = 0
                                consecutive_spike_epochs = 0 # Reset spike counter
                                logger.info(f"🎉 New best CER: {best_cer:.4f} at step {best_cer_step}")

                                # Save the current best model state temporarily
                                model.save_pretrained(BEST_MODEL_CHECKPOINT_PATH)
                                processor.save_pretrained(BEST_MODEL_CHECKPOINT_PATH)
                                logger.info(f"💾 Temporarily saved best model state at step {best_cer_step}")

                            # 2. Check for Standard Early Stopping Improvement
                            elif current_cer < last_good_cer - EARLY_STOPPING_MIN_DELTA:
                                 epochs_without_improvement = 0
                            else:
                                 epochs_without_improvement += 1

                            # 3. Check for Critical High CER (Spike)
                            is_current_cer_spike = current_cer > (best_cer * CRITICAL_CER_SPIKE_THRESHOLD)

                            if is_current_cer_spike:
                                consecutive_spike_epochs += 1
                                logger.warning(f"⚠️ CER spike detected ({current_cer:.4f} > {best_cer * CRITICAL_CER_SPIKE_THRESHOLD:.4f}). Count: {consecutive_spike_epochs}/{CRITICAL_CER_SPIKE_PATIENCE}")
                            else:
                                consecutive_spike_epochs = 0 # Reset if CER improves or stays reasonable

                            # 4. Action: Critical Spike - Reduce Learning Rate
                            if consecutive_spike_epochs >= CRITICAL_CER_SPIKE_PATIENCE:
                                logger.warning(f"🚨 CER has spiked for {CRITICAL_CER_SPIKE_PATIENCE} evaluations.")
                                old_lr = optimizer.param_groups[0]['lr']
                                new_lr = old_lr * CRITICAL_LR_REDUCTION_FACTOR
                                if new_lr > 1e-9: # Avoid making LR too small
                                    for param_group in optimizer.param_groups:
                                        param_group['lr'] = new_lr
                                    logger.warning(f"🔻 Reduced learning rate from {old_lr:.2e} to {new_lr:.2e}")

                                    # Log the LR reduction event
                                    log_entry = pd.DataFrame([{
                                        "step": step,
                                        "loss": round(current_loss, 4) if 'current_loss' in locals() else 0,
                                        "cer": round(current_cer, 4),
                                        "timestamp": pd.Timestamp.now(),
                                        "learning_rate": new_lr, # Log the NEW LR
                                        "note": "LR reduced due to CER spike" # Add a note
                                    }])
                                    log_entry.to_csv(RESULTS_CSV, mode='a', header=False, index=False, encoding='utf-8')
                                    logger.info(f"📈 Logged CER, NEW LR, and spike note at step {step} to {RESULTS_CSV}")

                                    # Reset spike counter after LR reduction
                                    consecutive_spike_epochs = 0
                                else:
                                    logger.error("🟥 Learning rate is already too low to reduce further due to CER spike.")

                            # 5. Action: Extremely High CER - Restore Best Model
                            if current_cer > RESTORE_BEST_MODEL_IF_CER_ABOVE:
                                logger.error(f"💥 CER ({current_cer:.4f}) exceeded absolute threshold ({RESTORE_BEST_MODEL_IF_CER_ABOVE}). IMMEDIATELY RESTORING BEST MODEL from step {best_cer_step}!")
                                # Load the best model state saved temporarily
                                if os.path.exists(BEST_MODEL_CHECKPOINT_PATH):
                                    try:
                                        model = WhisperForConditionalGeneration.from_pretrained(BEST_MODEL_CHECKPOINT_PATH).to(DEVICE)
                                        processor = WhisperProcessor.from_pretrained(BEST_MODEL_CHECKPOINT_PATH, language="chinese", task="transcribe")
                                        if processor.tokenizer.pad_token is None:
                                            processor.tokenizer.pad_token = processor.tokenizer.eos_token
                                        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")
                                        model.config.use_cache = False
                                        model.train() # Ensure model is back in training mode

                                        logger.info("✅ Successfully restored best model state.")

                                        # Reset counters after restoration to give training a chance
                                        consecutive_spike_epochs = 0
                                        epochs_without_improvement = 0
                                        # Note: last_good_cer will be updated below

                                        # Log the restoration event
                                        log_entry = pd.DataFrame([{
                                            "step": step,
                                            "loss": round(current_loss, 4) if 'current_loss' in locals() else 0,
                                            "cer": round(current_cer, 4),
                                            "timestamp": pd.Timestamp.now(),
                                            "learning_rate": optimizer.param_groups[0]['lr'], # Log current LR
                                            "note": f"Model RESTORED from best (step {best_cer_step}) due to CER > {RESTORE_BEST_MODEL_IF_CER_ABOVE}" # Add a note
                                        }])
                                        log_entry.to_csv(RESULTS_CSV, mode='a', header=False, index=False, encoding='utf-8')
                                        logger.info(f"📈 Logged CER, LR, and RESTORE note at step {step} to {RESULTS_CSV}")

                                    except Exception as e:
                                        logger.error(f"❌ Failed to restore best model: {e}")
                                        training_complete = True # If restoration fails, stop training
                                else:
                                     logger.error(f"❌ Cannot restore best model: checkpoint not found at {BEST_MODEL_CHECKPOINT_PATH}")
                                     training_complete = True # Stop if no backup

                            # 6. Update last_good_cer for next comparison
                            last_good_cer = current_cer

                            # 7. Log Normal Metrics (unless already logged due to special event)
                            # Check if a special log entry was already made
                            if not any("note" in entry for entry in log_entry.to_dict('records')) if 'log_entry' in locals() else True:
                                log_entry = pd.DataFrame([{
                                    "step": step,
                                    "loss": round(current_loss, 4) if 'current_loss' in locals() else 0,
                                    "cer": round(current_cer, 4),
                                    "timestamp": pd.Timestamp.now(),
                                    "learning_rate": optimizer.param_groups[0]['lr'] # Log CURRENT LR
                                }])
                                log_entry.to_csv(RESULTS_CSV, mode='a', header=False, index=False, encoding='utf-8')
                                logger.info(f"📈 Logged CER and LR at step {step} to {RESULTS_CSV}")

                            # 8. Check for Standard Early Stopping (based on no improvement)
                            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                                 logger.warning(f"⏹️ Standard early stopping triggered: No improvement for {EARLY_STOPPING_PATIENCE} evaluations.")
                                 training_complete = True # Signal to break out of training loop

                            # 🔷 --- END OF ENHANCED MONITORING & SITUATION-SAVING LOGIC ---

                        except Exception as e:
                            logger.error(f"Failed to evaluate or log CER: {e}")

                    # Periodic memory cleanup
                    if step % CLEANUP_INTERVAL == 0:
                        logger.info(f"🔧 Periodic memory cleanup at step {step}")
                        cleanup_memory()
                        logger.info("✅ Memory cleanup completed")

                # Safety check for catastrophic loss
                if 'current_loss' in locals() and current_loss > 10.0:
                     logger.error(f"🚨 Catastrophic loss detected (Loss: {current_loss}). Stopping training.")
                     training_complete = True
                     break

            except Exception as e:
                logger.error(f"Error in training step {step}, batch {batch_idx}: {e}")
                logger.exception(e)
                cleanup_memory(force=True)
                continue

        # Update progress and chunk state
        processed_files = next_train_start if next_train_start != 0 else total_files
        logger.info(f"Updating chunk state: next_start = {next_train_start}, epoch_count = {epochs_completed}")
        save_next_chunk_index(next_train_start, epochs_completed)

        # Clean up after each chunk
        logger.info("🔧 Chunk completion memory cleanup")
        cleanup_memory(force=True)
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Break if training is marked complete (e.g., by early stopping)
        if training_complete:
            break

    # Final save and evaluation
    logger.info("Saving final model...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to: {OUTPUT_DIR}")
    logger.info(f"Best CER was {best_cer:.4f} at step {best_cer_step}")

    try:
        final_cer = final_evaluation(OUTPUT_DIR, EVAL_JSON, EVAL_WAV_DIR)
        logger.info(f"Training completed successfully. Final CER: {final_cer:.4f}")
        logger.info(f"Best CER during training: {best_cer:.4f} at step {best_cer_step}")
    except Exception as e:
        logger.error(f"Error in final evaluation: {e}")

    logger.info("🧹 Final memory cleanup...")
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ Final cleanup completed successfully")
    except Exception as e:
        logger.warning(f"Warning during final cleanup: {e}")

    logger.info("🎉 Stable fine-tuning script with situation-saving completed. Exiting...")
    return True

def main():
    try:
        success = auto_train()
        if success:
            logger.info("✅ Training completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Training failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.exception(e)
        sys.exit(1)
    finally:
        logger.info("🔚 Script ending.")
        # Ensure handlers are closed properly
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        sys.exit(0)

if __name__ == "__main__":
    main()