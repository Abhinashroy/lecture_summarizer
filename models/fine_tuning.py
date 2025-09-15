"""
Fine-Tuning Pipeline for Academic Concept Extraction
Implements LoRA fine-tuning for specialized academic content understanding.
"""

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Dict, List, Any
import numpy as np
from pathlib import Path

import config

class FineTuningPipeline:
    """Pipeline for fine-tuning models with LoRA for academic concept extraction."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.training_data = None
        
    def prepare_training_data(self):
        """Prepare training data for academic concept extraction."""
        print("📚 Preparing training data...")
        
        # Create synthetic training data for demonstration
        # In practice, this would be real lecture transcripts with annotations
        training_examples = self._create_synthetic_training_data()
        
        # Save training data
        training_file = os.path.join("data", "training_data.json")
        os.makedirs(os.path.dirname(training_file), exist_ok=True)
        
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(training_examples, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training data prepared: {len(training_examples)} examples")
        return training_examples
    
    def _create_synthetic_training_data(self) -> List[Dict[str, str]]:
        """Create synthetic training data for academic concept extraction."""
        examples = []
        
        # Mathematics examples
        math_examples = [
            {
                "input": "Today we'll discuss polynomial equations. A polynomial is an expression consisting of variables and coefficients.",
                "output": "**Key Concepts:**\n- Polynomial equations\n- Expression with variables and coefficients\n\n**Definition:** Polynomial - Mathematical expression with variables and coefficients"
            },
            {
                "input": "The derivative of a function measures the rate of change. For f(x) = x², the derivative is 2x.",
                "output": "**Key Concepts:**\n- Derivative\n- Rate of change\n- Function differentiation\n\n**Formula:** d/dx(x²) = 2x\n\n**Definition:** Derivative - Measures rate of change of a function"
            },
            {
                "input": "Integration is the reverse process of differentiation. The integral of 2x is x² + C.",
                "output": "**Key Concepts:**\n- Integration\n- Reverse of differentiation\n- Constant of integration\n\n**Formula:** ∫2x dx = x² + C\n\n**Definition:** Integration - Reverse process of finding derivatives"
            }
        ]
        
        # Computer Science examples
        cs_examples = [
            {
                "input": "An algorithm is a step-by-step procedure for solving a problem. Time complexity measures efficiency.",
                "output": "**Key Concepts:**\n- Algorithm\n- Step-by-step procedure\n- Time complexity\n- Efficiency measurement\n\n**Definition:** Algorithm - Systematic procedure for problem-solving"
            },
            {
                "input": "Binary search is an efficient algorithm with O(log n) complexity. It works on sorted arrays.",
                "output": "**Key Concepts:**\n- Binary search\n- Logarithmic complexity\n- Sorted arrays\n- Algorithm efficiency\n\n**Complexity:** O(log n)\n\n**Requirement:** Array must be sorted"
            },
            {
                "input": "Object-oriented programming uses classes and objects. Inheritance allows code reuse.",
                "output": "**Key Concepts:**\n- Object-oriented programming (OOP)\n- Classes and objects\n- Inheritance\n- Code reuse\n\n**Definition:** Inheritance - Mechanism for code reuse in OOP"
            }
        ]
        
        # Physics examples
        physics_examples = [
            {
                "input": "Newton's second law states that force equals mass times acceleration, F = ma.",
                "output": "**Key Concepts:**\n- Newton's second law\n- Force, mass, acceleration relationship\n\n**Formula:** F = ma\n\n**Law:** Force equals mass times acceleration"
            },
            {
                "input": "Energy cannot be created or destroyed, only transformed. This is conservation of energy.",
                "output": "**Key Concepts:**\n- Conservation of energy\n- Energy transformation\n- Energy cannot be created or destroyed\n\n**Principle:** Energy conservation - fundamental law of physics"
            }
        ]
        
        # Biology examples
        biology_examples = [
            {
                "input": "Photosynthesis converts light energy into chemical energy. Chloroplasts contain chlorophyll.",
                "output": "**Key Concepts:**\n- Photosynthesis\n- Light to chemical energy conversion\n- Chloroplasts\n- Chlorophyll\n\n**Process:** Converting light energy to chemical energy in plants"
            },
            {
                "input": "DNA contains genetic information in four bases: adenine, thymine, guanine, and cytosine.",
                "output": "**Key Concepts:**\n- DNA structure\n- Genetic information storage\n- Four DNA bases\n\n**Components:** A (adenine), T (thymine), G (guanine), C (cytosine)"
            }
        ]
        
        # Combine all examples
        all_examples = math_examples + cs_examples + physics_examples + biology_examples
        
        # Add instruction format
        formatted_examples = []
        for example in all_examples:
            formatted_examples.append({
                "instruction": "Extract key concepts, definitions, and formulas from this lecture transcript:",
                "input": example["input"],
                "output": example["output"]
            })
        
        return formatted_examples
    
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print(f"🔧 Loading base model: {config.BASE_MODEL_NAME}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            print("✅ Model and tokenizer loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def setup_lora(self):
        """Setup LoRA configuration and apply to model."""
        print("🔧 Setting up LoRA configuration...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=config.LORA_CONFIG["r"],
            lora_alpha=config.LORA_CONFIG["lora_alpha"],
            target_modules=config.LORA_CONFIG["target_modules"],
            lora_dropout=config.LORA_CONFIG["lora_dropout"],
            bias=config.LORA_CONFIG["bias"],
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        print("✅ LoRA configuration applied")
    
    def prepare_dataset(self, training_examples: List[Dict[str, str]]):
        """Prepare dataset for training."""
        print("📊 Preparing dataset...")
        
        def format_prompt(example):
            """Format training example as prompt."""
            return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        
        # Format prompts
        formatted_texts = [format_prompt(ex) for ex in training_examples]
        
        # Tokenize function for batched processing
        def tokenize_function(examples):
            # Handle both single examples and batches
            texts = examples["text"] if isinstance(examples["text"], list) else [examples["text"]]
            
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors=None  # Return lists, not tensors
            )
            
            # Set labels to be the same as input_ids for language modeling
            tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
            return tokenized
        
        # Create dataset
        dataset = Dataset.from_dict({"text": formatted_texts})
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True,
            remove_columns=["text"]
        )
        
        print(f"✅ Dataset prepared: {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def fine_tune_model(self):
        """Execute the fine-tuning process."""
        print("🚀 Starting fine-tuning process...")
        
        # Prepare training data if not already done
        if not os.path.exists("data/training_data.json"):
            training_examples = self.prepare_training_data()
        else:
            with open("data/training_data.json", 'r', encoding='utf-8') as f:
                training_examples = json.load(f)
        
        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.setup_lora()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(training_examples)
        
        # Training arguments (optimized for CPU)
        training_args = TrainingArguments(
            output_dir=config.MODEL_SAVE_PATH,
            num_train_epochs=1,  # Reduced for faster training
            per_device_train_batch_size=1,  # Smaller batch size for CPU
            gradient_accumulation_steps=2,  # Reduced accumulation
            warmup_steps=5,  # Reduced warmup
            learning_rate=5e-5,  # Standard learning rate
            logging_steps=1,  # More frequent logging
            save_steps=50,  # More frequent saves
            eval_strategy="no",  # No evaluation during training
            save_strategy="steps",
            load_best_model_at_end=False,
            report_to=None,  # Disable wandb logging
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Disable pin_memory for CPU
            fp16=False,  # Disable mixed precision for CPU
        )
        
        # Data collator with padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling, not masked
            pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        print("🔥 Beginning training...")
        trainer.train()
        
        # Save the model
        os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
        trainer.save_model()
        self.tokenizer.save_pretrained(config.MODEL_SAVE_PATH)
        
        print(f"✅ Fine-tuning completed! Model saved to {config.MODEL_SAVE_PATH}")
        
    def load_fine_tuned_model(self):
        """Load a previously fine-tuned model."""
        if not os.path.exists(config.MODEL_SAVE_PATH):
            print("❌ No fine-tuned model found. Please run training first.")
            return False
        
        try:
            print("📥 Loading fine-tuned model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_PATH)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_NAME,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Load LoRA weights
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(base_model, config.MODEL_SAVE_PATH)
            
            print("✅ Fine-tuned model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            return False


class FineTunedConceptModel:
    """Wrapper class for fine-tuned concept extraction model."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        try:
            print("📥 Loading fine-tuned model...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.BASE_MODEL_NAME,
                torch_dtype=torch.float32,  # Use float32 for CPU
                device_map=None
            )
            
            # Load LoRA weights
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()  # Set to evaluation mode
            
            print("✅ Fine-tuned model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate_concept_extraction(self, text: str) -> str:
        """Generate concept extraction for given text."""
        if not self.model or not self.tokenizer:
            raise Exception("Model not loaded properly")
        
        try:
            # Format input prompt
            prompt = f"### Instruction:\nExtract key concepts, definitions, and formulas from this lecture transcript:\n\n### Input:\n{text}\n\n### Response:\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=400,
                padding=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part
            response_start = generated_text.find("### Response:\n") + len("### Response:\n")
            response = generated_text[response_start:].strip()
            
            return response
            
        except Exception as e:
            # Return simple fallback if generation fails
            return f"**Key Concepts:**\n- {text[:100]}..."
    
    def generate_concept_extraction(self, text: str) -> str:
        """Generate concept extraction for given text."""
        if self.peft_model is None or self.tokenizer is None:
            if not self.load_fine_tuned_model():
                raise RuntimeError("No model available for inference")
        
        # Format prompt
        prompt = f"### Instruction:\nExtract key concepts, definitions, and formulas from this lecture transcript:\n\n### Input:\n{text}\n\n### Response:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated response part
        response_start = full_response.find("### Response:\n") + len("### Response:\n")
        response = full_response[response_start:].strip()
        
        return response