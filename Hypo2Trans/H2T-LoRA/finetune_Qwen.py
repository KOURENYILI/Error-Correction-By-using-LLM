import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter


def train(
    # Model and data parameters
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    data_path: str = "data/train_wsj.json",
    output_dir: str = "./output",

    # Training hyperparameters
    batch_size: int = 32,            # Increased for better training
    micro_batch_size: int = 8,       # Increased for better utilization
    num_epochs: int = 10,            # Increased for better convergence
    learning_rate: float = 3e-4,     # Adjusted for stability
    cutoff_len: int = 512,          # Increased sequence length
    val_set_size: int = 100,        # Increased validation set

    # LoRA hyperparameters
    lora_r: int = 8,               # Increased rank
    lora_alpha: int = 32,          # Increased alpha
    lora_dropout: float = 0.05,    # Adjusted dropout
    lora_target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # LLM hyperparameters
    train_on_inputs: bool = True,
    add_eos_token: bool = True,
    group_by_length: bool = False,
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "H2T-LoRA",
    
    # New parameters
    use_wandb: bool = True,          # Enable wandb logging
    wandb_project: str = "h2t-lora", # Wandb project name
    wandb_run_name: str = None,      # Wandb run name
    gradient_checkpointing: bool = True,  # Enable gradient checkpointing
    max_grad_norm: float = 0.3,      # Gradient clipping
    weight_decay: float = 0.01,      # L2 regularization
):
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name or f"h2t-lora-{base_model.split('/')[-1]}",
            config=locals()
        )

    print(
        f"Training H2T-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"gradient_checkpointing: {gradient_checkpointing}\n"
        f"max_grad_norm: {max_grad_norm}\n"
        f"weight_decay: {weight_decay}\n"
    )

    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Model initialization with improved memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
        use_cache=not gradient_checkpointing,
    )

    # Enable gradient checkpointing
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left"  # Allow batched inference
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors="pt",
        )
        if (
            result["input_ids"][0, -1] != tokenizer.eos_token_id
            and result["input_ids"].shape[1] < cutoff_len
            and add_eos_token
        ):
            eos_token = torch.tensor([[tokenizer.eos_token_id]]).to(result["input_ids"].device)
            attention_mask = torch.tensor([[1]]).to(result["attention_mask"].device)
            
            result["input_ids"] = torch.cat([result["input_ids"], eos_token], dim=1)
            result["attention_mask"] = torch.cat([result["attention_mask"], attention_mask], dim=1)

        result["labels"] = result["input_ids"].clone()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            input=data_point["input1"],
            input2=data_point.get("input2"),
        )
        return tokenize(full_prompt)

    # Improved LoRA Configuration
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # Load and prepare data
    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    # Use full dataset instead of limiting to 200 samples
    train_data = data["train"]
    if val_set_size > 0:
        train_val = train_data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        val_data = None

    # Tokenize datasets
    train_data = train_data.map(
        generate_and_tokenize_prompt,
        batched=True,
        remove_columns=train_data.column_names,
        num_proc=4  # Parallel processing
    )
    if val_data:
        val_data = val_data.map(
            generate_and_tokenize_prompt,
            batched=True,
            remove_columns=val_data.column_names,
            num_proc=4
        )

    model.print_trainable_parameters()

    # Calculate training steps
    num_update_steps_per_epoch = (
        len(train_data) + micro_batch_size * gradient_accumulation_steps - 1
    ) // (micro_batch_size * gradient_accumulation_steps)
    max_steps = num_epochs * num_update_steps_per_epoch

    # Improved training arguments
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=0.03,  # Warm up over 3% of steps
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=20,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=group_by_length,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=max_steps,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
    )

    # Initialize trainer with improved configuration
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # Optimize for hardware
        ),
    )

    # Add callbacks
    early_stopping = transformers.EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.01
    )
    trainer.add_callback(early_stopping)

    # Compile model if possible
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Train and save
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the model
    model.save_pretrained(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save training arguments
    import json
    with open(os.path.join(output_dir, "training_args.json"), 'w') as f:
        json.dump(training_args.to_dict(), f, indent=2)

    if use_wandb:
        wandb.finish()

    print("\nTraining completed. Model saved to", output_dir)


if __name__ == "__main__":
    fire.Fire(train)