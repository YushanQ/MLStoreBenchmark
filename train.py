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
import time

class LlamaTrainer:
    def __init__(
            self,
            model_id="meta-llama/Meta-Llama-3.1-8B",
            output_model="llama3.1-8B-Fine-tuned"
    ):
        self.model_id = model_id
        self.output_model = output_model
        # self.profiler = None
        # self.phase_pipe = None
        # self.current_phase = "unknown"

    # def mark_phase(self, phase):
    #     """Explicitly mark phase for I/O monitoring"""
    #     self.current_phase = phase
    #     if self.phase_pipe:
    #         try:
    #             self.phase_pipe.write(f"{phase}\n")
    #             self.phase_pipe.flush()
    #             print(f"\nEntering phase: {phase}")
    #         except Exception as e:
    #             print(f"Warning: Could not mark phase: {e}")

    # @contextmanager
    # def profile_section(self, name):
    #     """Enhanced context manager for profiling sections with phase marking"""
    #     previous_phase = self.current_phase
    #     # self.mark_phase(name)  # Mark start of section
    #
    #     try:
    #         with record_function(name):
    #             yield
    #     except Exception as e:
    #         print(f"Profiling error in {name}: {e}")
    #         raise
    #     finally:
    #         self.mark_phase(previous_phase)  # Restore previous phase

    def prepare_data(self, data_path):
        with self.profile_section("data_preparation"):
            print("\nPreparing training data...")
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
            print("\nLoading model and tokenizer...")
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

    def setup_training_args(self):
        with self.profile_section("training_setup"):
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

    def setup_profiler(self):
        """Setup PyTorch profiler with phase awareness"""
        try:
            os.makedirs(f'{self.output_model}/profile_logs', exist_ok=True)
            return profile(
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
            )
        except Exception as e:
            print(f"Warning: Could not setup profiler: {e}")
            return None

    def train(self, data_path):
        """Enhanced training with detailed phase tracking"""
        # Setup phase pipe for BPF
        pipe_path = "/tmp/llama_phase"
        try:
            os.mkfifo(pipe_path)
        except FileExistsError:
            pass

        print(f"\nOpening phase communication pipe at {pipe_path}")
        self.phase_pipe = open(pipe_path, "w")
        self.mark_phase("initialization")

        try:
            # Setup profiler
            self.profiler = self.setup_profiler()
            if self.profiler:
                self.profiler.start()

            print("\nStarting training process...")

            # Data preparation
            data = self.prepare_data(data_path)

            # Model initialization
            model, tokenizer = self.get_model_tokenizer()

            # Training setup
            with self.profile_section("training_setup"):
                print("\nConfiguring training parameters...")
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )

                trainer = SFTTrainer(
                    model=model,
                    train_dataset=data,
                    peft_config=peft_config,
                    dataset_text_field="text",
                    args=self.setup_training_args(),
                    tokenizer=tokenizer,
                    packing=False,
                    max_seq_length=1024
                )

            # Main training loop
            with self.profile_section("training"):
                print("\nStarting main training loop...")
                trainer.train()

            # Save final model
            with self.profile_section("model_saving"):
                print("\nSaving trained model...")
                trainer.save_model()

            # Print profiling summary if available
            if self.profiler:
                print("\nProfiling Summary:")
                print(self.profiler.key_averages().table(
                    sort_by="cpu_time_total", row_limit=10))

        except Exception as e:
            print(f"Training error: {e}")
            raise
        finally:
            print("\nCleaning up...")
            self.mark_phase("cleanup")

            # Cleanup profiler
            if hasattr(self, 'profiler') and self.profiler:
                try:
                    self.profiler.stop()
                except Exception:
                    pass

            # Cleanup phase pipe
            if self.phase_pipe:
                try:
                    self.phase_pipe.close()
                    os.remove(pipe_path)
                except Exception:
                    pass

if __name__ == "__main__":
    trainer = LlamaTrainer()
    trainer.train("data.txt")