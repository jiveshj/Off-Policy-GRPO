"""
Off-Policy GRPO System with Parameter Server and Worker Nodes
Uses Tina's GRPOTrainer for actual training
"""

# workers.py
import copy
import json
import logging
import os
import torch
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from tina.post_train_hf.grpo_trainer import GRPOTrainer
from tina.post_train_hf.grpo_config import GRPOConfig
from trl import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class WorkerResponse:
    """Response from a worker node after generation"""
    worker_id: int
    prompt: str
    completions: List[str]  # Multiple completions from worker
    rewards: List[float]  # Rewards for each completion
    # Store generation info for off-policy correction
    generation_metadata: Dict


class WorkerNode:
    """
    Worker node with its own LoRA adapter and local dataset partition.
    Uses Tina's GRPOTrainer for generation.
    """
    
    def __init__(
        self,
        worker_id: int,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        local_dataset: Dataset,
        model_config: ModelConfig,
        training_config: GRPOConfig,
        reward_funcs: List[Callable],
        device: str = "cuda"
    ):
        self.worker_id = worker_id
        self.tokenizer = tokenizer
        self.local_dataset = local_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.reward_funcs = reward_funcs
        self.device = device
        
        logger.info(f"Initializing Worker {worker_id}")
        
        # Clone the base model for this worker
        self.model = copy.deepcopy(base_model)
        
        # Apply LoRA to worker model
        if model_config.use_peft:
            peft_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules,
                inference_mode=False,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, peft_config)
        
        self.model.to(device)
        
        # Create a mini training config for this worker (for generation only)
        self.worker_training_config = copy.deepcopy(training_config)
        self.worker_training_config.output_dir = f"{training_config.output_dir}/worker_{worker_id}"
        
        # Initialize GRPOTrainer for this worker (used for generation)
        # Note: We won't call .train() on worker trainers, just use for generation
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=self.worker_training_config,
            train_dataset=local_dataset
        )
        
        logger.info(f"Worker {worker_id} initialized with LoRA adapter")
    
    def generate_responses(
        self,
        prompt_batch: List[str],
        num_generations_per_prompt: int = None
    ) -> List[Dict]:
        """
        Generate responses for a batch of prompts using worker's policy.
        Uses Tina's GRPOTrainer generation mechanism.
        
        Returns: List of dicts with {prompt, completions, rewards}
        """
        if num_generations_per_prompt is None:
            num_generations_per_prompt = self.training_config.num_generations
        
        self.model.eval()
        
        # Create a temporary dataset for these prompts
        temp_dataset = Dataset.from_dict({"prompt": prompt_batch})
        
        # Use trainer's generation mechanism
        # This is similar to how GRPOTrainer generates during training
        with torch.no_grad():
            generation_results = []
            
            for prompt in prompt_batch:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.training_config.max_prompt_length
                ).to(self.device)
                
                completions = []
                scores_list = []
                
                # Generate multiple completions
                for _ in range(num_generations_per_prompt):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.training_config.max_completion_length,
                        temperature=self.training_config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    # Extract completion
                    prompt_len = inputs.input_ids.shape[1]
                    completion_ids = outputs.sequences[0][prompt_len:]
                    completion = self.tokenizer.decode(
                        completion_ids,
                        skip_special_tokens=True
                    )
                    
                    completions.append(completion)
                    
                    # Store generation metadata (for off-policy correction)
                    scores_list.append({
                        'sequence': outputs.sequences[0],
                        'scores': outputs.scores if hasattr(outputs, 'scores') else None
                    })
                
                generation_results.append({
                    'prompt': prompt,
                    'completions': completions,
                    'generation_metadata': scores_list
                })
        
        return generation_results
    
    def update_from_parameter_server(self, state_dict: Dict):
        """Update worker's model weights from parameter server"""
        # Only update the trainable parameters (LoRA weights)
        self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Worker {self.worker_id} updated from parameter server")
    
    def get_state_dict(self) -> Dict:
        """Get worker's current model state"""
        return self.model.state_dict()


class ParameterServer:
    """
    Central parameter server that aggregates worker responses and 
    performs GRPO optimization using Tina's GRPOTrainer.
    """
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        global_dataset: Dataset,
        model_config: ModelConfig,
        training_config: GRPOConfig,
        reward_funcs: List[Callable],
        device: str = "cuda"
    ):
        self.tokenizer = tokenizer
        self.global_dataset = global_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.reward_funcs = reward_funcs
        self.device = device
        
        logger.info("Initializing Parameter Server")
        
        # Apply LoRA to parameter server model
        if model_config.use_peft:
            peft_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules,
                inference_mode=False,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(model, peft_config)
        else:
            self.model = model
        
        self.model.to(device)
        
        # Initialize aggregated dataset for off-policy training
        # This will be updated with worker responses
        self.aggregated_dataset = None
        
        # Initialize GRPOTrainer for parameter server
        # This is the actual trainer that will perform optimization
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=training_config,
            train_dataset=global_dataset  # Will be replaced with aggregated data
        )
        
        self.step_count = 0
        
        logger.info("Parameter Server initialized with GRPOTrainer")
    
    def aggregate_worker_responses(
        self,
        worker_generation_results: List[List[Dict]],
        compute_rewards: bool = True
    ) -> Dataset:
        """
        Aggregate responses from all workers into a dataset.
        
        Args:
            worker_generation_results: List of generation results from each worker
            compute_rewards: Whether to compute rewards here
            
        Returns:
            Dataset with (prompt, completion) pairs and rewards
        """
        aggregated_data = {
            "prompt": [],
            "completion": [],
            "worker_id": []
        }
        
        # If we need to track additional columns for rewards
        if compute_rewards and len(self.global_dataset) > 0:
            # Copy other columns from original dataset that might be needed for rewards
            sample_columns = self.global_dataset.column_names
            for col in sample_columns:
                if col not in ["prompt", "completion"]:
                    aggregated_data[col] = []
        
        # Flatten all worker responses
        for worker_id, generation_results in enumerate(worker_generation_results):
            for result in generation_results:
                prompt = result['prompt']
                completions = result['completions']
                
                for completion in completions:
                    aggregated_data["prompt"].append(prompt)
                    aggregated_data["completion"].append(completion)
                    aggregated_data["worker_id"].append(worker_id)
                    
                    # Add other columns if needed for reward computation
                    if compute_rewards:
                        # Find matching row in global dataset
                        matching_rows = [
                            i for i, p in enumerate(self.global_dataset["prompt"])
                            if p == prompt
                        ]
                        if matching_rows:
                            idx = matching_rows[0]
                            for col in sample_columns:
                                if col not in ["prompt", "completion"]:
                                    aggregated_data[col].append(
                                        self.global_dataset[idx][col]
                                    )
        
        dataset = Dataset.from_dict(aggregated_data)
        logger.info(f"Aggregated {len(dataset)} responses from {len(worker_generation_results)} workers")
        
        return dataset
    
    def update_with_off_policy_data(
        self,
        aggregated_dataset: Dataset
    ):
        """
        Update the trainer's dataset with aggregated off-policy data.
        This allows Tina's GRPOTrainer to train on worker-generated responses.
        """
        # Update the trainer's train_dataset
        self.trainer.train_dataset = aggregated_dataset
        
        # Recreate dataloader with new dataset
        self.trainer._train_dataloader = None  # Force recreation
        
        self.aggregated_dataset = aggregated_dataset
        logger.info(f"Parameter server dataset updated with {len(aggregated_dataset)} samples")
    
    def train_step(self, num_steps: int = 1):
        """
        Perform training steps using Tina's GRPOTrainer on aggregated data.
        
        Args:
            num_steps: Number of optimization steps to perform
        """
        if self.aggregated_dataset is None:
            logger.warning("No aggregated dataset available for training")
            return {}
        
        logger.info(f"Parameter server performing {num_steps} training steps")
        
        # Use Tina's trainer to perform actual GRPO optimization
        # We need to manually step through the training loop
        self.model.train()
        
        train_dataloader = self.trainer.get_train_dataloader()
        
        stats_accumulator = {
            "loss": 0.0,
            "rewards": []
        }
        
        for step in range(num_steps):
            try:
                batch = next(iter(train_dataloader))
                
                # Perform one training step using trainer's logic
                # This will do the GRPO optimization
                loss = self.trainer.training_step(self.model, batch)
                
                stats_accumulator["loss"] += loss.item() if torch.is_tensor(loss) else loss
                
                self.step_count += 1
                
            except StopIteration:
                logger.warning("Dataloader exhausted, recreating...")
                train_dataloader = self.trainer.get_train_dataloader()
        
        # Compute average stats
        stats = {
            "loss": stats_accumulator["loss"] / num_steps,
            "step": self.step_count
        }
        
        logger.info(f"Step {self.step_count}: Loss={stats['loss']:.4f}")
        
        return stats
    
    def get_state_dict(self) -> Dict:
        """Get current model state for broadcasting to workers"""
        return self.model.state_dict()
    
    def save_checkpoint(self, path: str):
        """Save parameter server checkpoint using trainer's save"""
        self.trainer.save_model(path)
        logger.info(f"Checkpoint saved to {path}")


class DistributedOffPolicyGRPO:
    """
    Main orchestrator for distributed off-policy GRPO training.
    Coordinates workers and parameter server, both using Tina's GRPOTrainer.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        global_dataset: Dataset,
        model_config: ModelConfig,
        training_config: GRPOConfig,
        reward_funcs: List[Callable],
        num_workers: int = 3,
        device: str = "cuda"
    ):
        self.model_name_or_path = model_name_or_path
        self.global_dataset = global_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.reward_funcs = reward_funcs
        self.num_workers = num_workers
        self.device = device
        
        # Ensure num_generations matches num_workers
        if training_config.num_generations != num_workers:
            logger.warning(
                f"Setting num_generations={num_workers} to match num_workers "
                f"(was {training_config.num_generations})"
            )
            training_config.num_generations = num_workers
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {model_name_or_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if training_config.bf16 else torch.float16,
            device_map=device,
            use_cache=not training_config.gradient_checkpointing
        )
        
        # Initialize parameter server
        logger.info("Initializing Parameter Server...")
        self.parameter_server = ParameterServer(
            model=copy.deepcopy(self.base_model),
            tokenizer=self.tokenizer,
            global_dataset=global_dataset,
            model_config=model_config,
            training_config=training_config,
            reward_funcs=reward_funcs,
            device=device
        )
        
        # Partition dataset for workers
        local_datasets = self._partition_dataset()
        
        # Initialize worker nodes
        logger.info(f"Initializing {num_workers} worker nodes...")
        self.workers = []
        for i in range(num_workers):
            worker = WorkerNode(
                worker_id=i,
                base_model=copy.deepcopy(self.base_model),
                tokenizer=self.tokenizer,
                local_dataset=local_datasets[i],
                model_config=model_config,
                training_config=training_config,
                reward_funcs=reward_funcs,
                device=device
            )
            self.workers.append(worker)
        
        logger.info("Distributed Off-Policy GRPO System initialized!")
    
    def _partition_dataset(self) -> List[Dataset]:
        """Partition global dataset for worker nodes"""
        dataset_size = len(self.global_dataset)
        partition_size = dataset_size // self.num_workers
        
        partitions = []
        for i in range(self.num_workers):
            start_idx = i * partition_size
            end_idx = (start_idx + partition_size 
                      if i < self.num_workers - 1 
                      else dataset_size)
            
            partition = self.global_dataset.select(range(start_idx, end_idx))
            partitions.append(partition)
            logger.info(f"Worker {i} dataset size: {len(partition)}")
        
        return partitions
    
    def sample_prompts(self, batch_size: int = 1) -> List[str]:
        """Sample prompts from global dataset (same for all workers)"""
        import random
        indices = random.sample(range(len(self.global_dataset)), batch_size)
        prompts = [self.global_dataset[i]["prompt"] for i in indices]
        return prompts
    
    def training_step(self, prompts_per_step: int = 1) -> Dict:
        """
        Execute one distributed training iteration:
        1. Sample prompts from global dataset
        2. All workers generate responses for same prompts
        3. Aggregate worker responses
        4. Parameter server trains on aggregated data using GRPO
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Step {self.parameter_server.step_count + 1}")
        
        # 1. Sample prompts (same for all workers)
        prompts = self.sample_prompts(batch_size=prompts_per_step)
        logger.info(f"Sampled {len(prompts)} prompts")
        for i, prompt in enumerate(prompts[:2]):  # Show first 2
            logger.info(f"Prompt {i+1}: {prompt[:100]}...")
        
        # 2. Collect responses from all workers
        all_worker_results = []
        for worker in self.workers:
            worker_results = worker.generate_responses(
                prompt_batch=prompts,
                num_generations_per_prompt=1  # Each worker generates 1 response per prompt
            )
            all_worker_results.append(worker_results)
            logger.info(f"Worker {worker.worker_id} generated {len(worker_results)} responses")
        
        # 3. Aggregate worker responses into dataset
        aggregated_dataset = self.parameter_server.aggregate_worker_responses(
            all_worker_results,
            compute_rewards=True
        )
        
        # 4. Update parameter server with off-policy data
        self.parameter_server.update_with_off_policy_data(aggregated_dataset)
        
        # 5. Parameter server trains on aggregated data
        stats = self.parameter_server.train_step(num_steps=1)
        
        logger.info(f"{'='*60}\n")
        
        return stats
    
    def train(
        self,
        num_iterations: int,
        prompts_per_iteration: int = 1,
        sync_frequency: int = 10,
        save_frequency: int = 100,
        output_dir: str = None
    ):
        """
        Train the distributed off-policy GRPO system.
        
        Args:
            num_iterations: Number of training iterations
            prompts_per_iteration: Number of prompts to sample per iteration
            sync_frequency: Sync workers every N iterations
            save_frequency: Save checkpoint every N iterations
            output_dir: Directory to save checkpoints
        """
        if output_dir is None:
            output_dir = self.training_config.output_dir
        
        logger.info(f"\nStarting Distributed Off-Policy GRPO Training")
        logger.info(f"Iterations: {num_iterations}")
        logger.info(f"Workers: {self.num_workers}")
        logger.info(f"Prompts per iteration: {prompts_per_iteration}")
        logger.info(f"Sync frequency: {sync_frequency}")
        logger.info(f"Output directory: {output_dir}")
        
        for iteration in range(num_iterations):
            # Execute training iteration
            stats = self.training_step(prompts_per_iteration)
            
            # Periodic worker synchronization
            if (iteration + 1) % sync_frequency == 0:
                logger.info(f"\n[Sync] Iteration {iteration + 1}: Broadcasting parameters to workers")
                state_dict = self.parameter_server.get_state_dict()
                
                for worker in self.workers:
                    worker.update_from_parameter_server(state_dict)
                
                logger.info(f"[Sync] All workers synchronized\n")
            
            # Save checkpoint
            if (iteration + 1) % save_frequency == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
                self.parameter_server.save_checkpoint(checkpoint_path)
        
        # Final save
        final_path = os.path.join(output_dir, "final_model")
        self.parameter_server.save_checkpoint(final_path)
        
        logger.info(f"\nTraining completed! Final model saved to {final_path}")
