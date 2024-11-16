import torch
from torch.profiler import profile, record_function, ProfilerActivity
from datasets import load_dataset, Dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
import json
import pandas as pd
import os
from contextlib import contextmanager

class ProfiledSFTTrainer(SFTTrainer):
    """Extended SFTTrainer with profiling capabilities"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create phase pipe for BPF communication
        try:
            os.mkfifo("/tmp/llama_phase")
        except FileExistsError:
            pass
        self.phase_pipe = open("/tmp/llama_phase", "w")

    def _mark_phase(self, phase):
        """Helper to mark training phases"""
        try:
            self.phase_pipe.write(f"{phase}\n")
            self.phase_pipe.flush()
        except Exception as e:
            print(f"Warning: Could not mark phase: {e}")

    def training_step(self, *args, **kwargs):
        """Override training step to add profiling"""
        self._mark_phase("training")
        with record_function("training_step"):
            return super().training_step(*args, **kwargs)

    def save_model(self, *args, **kwargs):
        """Override save to track checkpoints"""
        self._mark_phase("checkpoint")
        with record_function("save_checkpoint"):
            result = super().save_model(*args, **kwargs)
        self._mark_phase("training")
        return result

    def _prepare_inputs(self, *args, **kwargs):
        """Override input preparation to track data loading"""
        self._mark_phase("dataload")
        with record_function("data_loading"):
            result = super()._prepare_inputs(*args, **kwargs)
        self._mark_phase("training")
        return result

    def close(self):
        """Cleanup phase pipe"""
        if hasattr(self, 'phase_pipe'):
            self.phase_pipe.close()
            try:
                os.remove("/tmp/llama_phase")
            except FileNotFoundError:
                pass

class LlamaTrainer:
    def __init__(
            self,
            model_id="meta-llama/Meta-Llama-3.1-8B",
            output_model="llama3.1-8B-Fine-tuned"
    ):
        self.model_id = model_id
        self.output_model = output_model

    @contextmanager
    def profile_section(self, name):
        """Context manager for profiling sections"""
        try:
            with record_function(name):
                yield
        except Exception as e:
            print(f"Profiling error in {name}: {e}")
            yield

    def prepare_data(self, data_path):
        with self.profile_section("data_preparation"):
            with open(data_path, 'r') as f:
                training_data = json.load(f)
            df = pd.DataFrame(training_data)
            df["text"] = df[["prompt", "response"]].apply(
                lambda x: f"<|im_start|>user\n{x['prompt']}<|im_end|>\n<|im_start|>assistant\n{x['response']}<|im_end|>\n",
                axis=1
            )
            return Dataset.from_pandas(df)

    def get_model_tokenizer(self):
        with self.profile_section("model_loading"):
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto"
            )
            model.config.use_cache = False
            model.config.pretraining_tp = 1
            return model, tokenizer

    def get_training_args(self):
        return TrainingArguments(
            output_dir=self.output_model,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=16,
            optim="paged_adamw_32bit",
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=10,
            num_train_epochs=3,
            max_steps=250,
            fp16=True,
            push_to_hub=False
        )

    def train(self, data_path):
        try:
            with profile(
                    activities=[
                        ProfilerActivity.CPU,
                        ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=3,
                        repeat=2
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        f'{self.output_model}/profile_logs'
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True
            ) as prof:
                with self.profile_section("full_training"):
                    data = self.prepare_data(data_path)
                    model, tokenizer = self.get_model_tokenizer()

                    peft_config = LoraConfig(
                        r=8,
                        lora_alpha=16,
                        lora_dropout=0.05,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )

                    trainer = ProfiledSFTTrainer(
                        model=model,
                        train_dataset=data,
                        peft_config=peft_config,
                        dataset_text_field="text",
                        args=self.get_training_args(),
                        tokenizer=tokenizer,
                        packing=False,
                        max_seq_length=1024
                    )

                    trainer.train()
                    trainer.close()

                    # Export profiling data
                    print("\nProfiling Summary:")
                    print(prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=10))
        except Exception as e:
            print(f"Training error: {e}")
            raise

if __name__ == "__main__":
    trainer = LlamaTrainer()
    trainer.train("data.txt")