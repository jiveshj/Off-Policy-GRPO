
"""
Off-Policy GRPO System with Parameter Server and Worker Nodes
Correct implementation with:
- Worker pre-training on local datasets
- π_base (π_old) frozen forever as reference
- Configurable importance weighting
- Uses Tina's GRPOTrainer
"""

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
import random
from tina.post_train_hf.grpo_trainer import GRPOTrainer
from tina.post_train_hf.grpo_config import GRPOConfig
from trl import ModelConfig
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ============================================================================
# Importance Weighting Strategies
# ============================================================================

class Weighting:
    """No importance correction - all weights = 1.0 (for ablation studies)"""
    def __init__(self,alpha):
        self.alpha = alpha
        self.weights = []
    def compute_weight(self, *args, **kwargs) -> float:
        return 1.0


# ============================================================================
# Worker Node
# ============================================================================

@dataclass
class WorkerResponse:
    """Response from a worker node after generation"""
    worker_id: int
    prompt: str
    completion: str
    reward: float
    prompt_input_ids: torch.Tensor = None
    completion_input_ids: torch.Tensor = None



class WorkerNode:
    """
    Worker node with its own LoRA adapter and local dataset partition.
    - Pre-trains on local dataset
    - Generates responses for global prompts
    - Computes rewards locally
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
        
        logger.info(f"[Worker {worker_id}] Initializing...")
        
        # Clone the base model for this worker
        self.model = copy.deepcopy(base_model)
        
        # Apply LoRA to worker model
        if model_config.use_peft:
            logger.info(f"\n For worker_{worker_id}, using PEFT with {model_config.lora_r} rank, {model_config.lora_alpha} alpha, {model_config.lora_dropout} dropout.")
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
        
        self.model.to(self.device)
        
        # Create worker-specific training config
        self.worker_training_config = copy.deepcopy(training_config)
        self.worker_training_config.output_dir = f"{training_config.output_dir}/worker_{worker_id}_finetuned"
        
        # Store whether this worker has been pre-trained
        self.is_finetuned = False
        
        logger.info(f"[Worker {worker_id}] Initialized with {len(local_dataset)} local samples")
    
    def finetune(self, num_epochs: int = 1):
        """
        Fine-tune worker on its local dataset using GRPO.
        This happens BEFORE the distributed off-policy training begins.
        
        Args:
            num_epochs: Number of epochs to train on local dataset
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"[Worker {self.worker_id}] Starting Pre-training")
        logger.info(f"Local dataset size: {len(self.local_dataset)}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"{'='*60}")
        
        # Create a temporary config for pre-training
        finetune_config = copy.deepcopy(self.worker_training_config)
        finetune_config.num_train_epochs = num_epochs
        
        # Initialize GRPOTrainer for pre-training (did not add callbacks as was done in the Tina Repo)
        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=self.reward_funcs,
            args=finetune_config,
            train_dataset=self.local_dataset
        )
        
        # Train on local dataset
        logger.info(f"[Worker {self.worker_id}] Training...")
        train_result = self.trainer.train()
        
        logger.info(f"[Worker {self.worker_id}] Fine-tuning completed!")
        logger.info(f"Final loss: {train_result.metrics.get('train_loss', 'N/A')}")
        
        # Mark as pre-trained
        self.is_finetuned = True
        
        #Clean up trainer
        del trainer
        torch.cuda.empty_cache()
        
        logger.info(f"[Worker {self.worker_id}] Pre-training finished\n")
    
    def generate_and_score(
        self,
        prompts: List[str]
    ) -> List[WorkerResponse]:
        """
        Generate responses for prompts and compute rewards locally.
        
        Args:
            prompts: List of prompts to generate completions for
            
        Returns:
            List of WorkerResponse objects with (prompt, completion, reward)
        """
        if not self.is_finetuned:
            logger.warning(
                f"[Worker {self.worker_id}] Generating without fine-training! "
                "Call finetune() first."
            )
        
        self.model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.training_config.max_prompt_length
                ).to(self.device)
                
                # Generate completion
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.training_config.max_completion_length,
                    temperature=self.training_config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Extract completion (I should not make a completion mask here because prompt is a single prompt)
                prompt_len = inputs.input_ids.shape[1]
                completion_ids = outputs[0][prompt_len:]
                completion = self.tokenizer.decode(
                    completion_ids,
                    skip_special_tokens=True
                )
                
                # Compute reward locally using reward functions
                # Need to create a sample dict for reward functions
                sample = {
                    "prompt": prompt,
                    "completion": completion
                }
                
                # Compute rewards (assuming reward_funcs return single values)
                total_reward = 0.0
                for reward_func in self.reward_funcs:
                    reward = reward_func(sample)
                    total_reward += reward
                
                # Average if multiple reward functions
                avg_reward = total_reward / len(self.reward_funcs) if self.reward_funcs else 0.0
                
                responses.append(WorkerResponse(
                    worker_id=self.worker_id,
                    prompt=prompt,
                    completion=completion,
                    reward=avg_reward,
                    prompt_input_ids=inputs.input_ids[0],
                    completion_input_ids=completion_ids
                ))
        
        logger.info(
            f"[Worker {self.worker_id}] Generated {len(responses)} responses, "
            f"avg reward: {sum(r.reward for r in responses) / len(responses):.4f}"
        )
        
        return responses
    
    def get_policy_state(self) -> Dict:
        """Get worker's current policy state for importance weighting"""
        return self.model.state_dict()


# ============================================================================
# Parameter Server
# ============================================================================

class ParameterServer:
    """
    Central parameter server that:
    - Maintains π_base (frozen reference model)
    - Optimizes π_server using aggregated worker responses
    - Computes importance weights for off-policy correction
    - Uses Tina's GRPOTrainer for optimization
    """
    
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        global_dataset: Dataset,
        model_config: ModelConfig,
        training_config: GRPOConfig,
        reward_funcs: List[Callable],
        importance_weight_strategy = None,
        device: str = "cuda"
    ):
        self.tokenizer = tokenizer
        self.global_dataset = global_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.reward_funcs = reward_funcs
        self.device = device
        
        logger.info("#"*60)
        logger.info("Initializing Parameter Server")
        logger.info("#"*60)
        
        # π_base: Frozen base model (π_old) - never changes!
        logger.info("Creating π_base (frozen reference model)...")
        self.pi_ref = copy.deepcopy(base_model)
        self.pi_ref.to(device)
        self.pi_ref.eval()
        
        # Freeze π_base completely
        for param in self.pi_ref.parameters():
            param.requires_grad = False
        
        logger.info("π_ref frozen - will be updated after each outer iteration")
        
        # π_server: Parameter server model being optimized
        logger.info("Creating π_server (optimizable model)...")
        self.pi_server = copy.deepcopy(base_model)
        
        # Apply LoRA to π_server
        if model_config.use_peft:
            logger.info(f"\n For parameter server, using PEFT with {model_config.lora_r} rank, {model_config.lora_alpha} alpha, {model_config.lora_dropout} dropout.")
            peft_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                target_modules=model_config.lora_target_modules,
                inference_mode=False,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.pi_server = get_peft_model(self.pi_server, peft_config)
        self.pi_server.to(device)
        
        # Store worker policies for importance weighting
        self.worker_policies: Dict[int, AutoModelForCausalLM] = {}
        
        # Importance weighting strategy
        if importance_weight_strategy is None:
            logger.info("Using StandardImportanceSampling by default")
            self.importance_weight_strategy = Weighting(alpha = 1.0)
        else:
            self.importance_weight_strategy = importance_weight_strategy
            logger.info(f"Using {importance_weight_strategy.__class__.__name__}")
        
        # Dataset for aggregated worker responses
        self.aggregated_dataset = None
        
        # Initialize GRPOTrainer for parameter server
        # Note: We'll update the dataset dynamically

        # #probably don't need GRPOTrainer. 
        # self.trainer = GRPOTrainer(
        #     model=self.pi_server,
        #     processing_class=tokenizer,
        #     reward_funcs=reward_funcs,
        #     args=training_config,
        #     train_dataset=global_dataset
        # )
        self._dataloader = None
        #have to thinkabout this optimizer
        self.optimizer = torch.optim.AdamW(
            self.pi_server.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay if hasattr(training_config, 'weight_decay') else 0.0
        )

        
        # Override the reference model in trainer to use π_base
        #print(self.trainer.ref_model)
        #self.trainer.ref_model = self.pi_base
        
        self.step_count = 0
        
        logger.info("Parameter Server initialized!")
        logger.info(f"π_ref parameters: {sum(p.numel() for p in self.pi_ref.parameters()):,}")
        logger.info(f"π_server trainable parameters: {sum(p.numel() for p in self.pi_server.parameters() if p.requires_grad):,}")
        logger.info("="*60 + "\n")
    
    def update_reference_model(self):
        """
        Algorithm 1, Line 3: π_ref ← π_θ
        Update the reference model to the current policy.
        This happens at the start of each outer iteration.
        """
        logger.info("[Parameter Server] Updating π_ref ← π_θ...")
        
        # Copy current policy parameters to reference model
        self.pi_ref.load_state_dict(
            self.pi_server.state_dict(),
            strict=False  # In case of LoRA, only update trainable params
        )
        
        # Ensure π_ref remains frozen
        self.pi_ref.eval()
        for param in self.pi_ref.parameters():
            param.requires_grad = False
        
        logger.info("[Parameter Server] Reference model updated and frozen")

    def register_worker_policy(self, worker_id: int, policy_state: Dict):
        """
        Register a worker's policy for importance weighting.
        
        Args:
            worker_id: Worker identifier
            policy_state: Worker's model state_dict
        """
        if self.importance_weight_strategy is None:
            logger.info("No parameter weighting is specified")
            return
        
        # Create a copy of the model to store worker policy
        worker_policy = copy.deepcopy(self.pi_base)
        worker_policy.load_state_dict(policy_state, strict=False)
        worker_policy.to(self.device)
        worker_policy.eval()
        
        # Freeze worker policy
        for param in worker_policy.parameters():
            param.requires_grad = False
        
        self.worker_policies[worker_id] = worker_policy
        logger.info(f"[Parameter Server] Registered Worker {worker_id} policy")
    
    def aggregate_worker_responses(
        self,
        worker_responses: List[WorkerResponse],
        compute_importance_weights: bool = True
    ) -> Dataset:
        """
        Aggregate responses from workers and compute importance weights.
        
        Args:
            worker_responses: List of WorkerResponse objects from all workers
            compute_importance_weights: Whether to compute importance weights
            
        Returns:
            Dataset with (prompt, completion, reward, importance_weight) columns
        """
        logger.info(f"\n[Parameter Server] Aggregating {len(worker_responses)} responses...")
        
        aggregated_data = {
            "prompt": [],
            "completion": [],
            "reward": [],
            "importance_weight": [],
            "worker_id": []
        }
        
        for response in worker_responses:
            aggregated_data["prompt"].append(response.prompt)
            aggregated_data["completion"].append(response.completion)
            aggregated_data["reward"].append(response.reward)
            aggregated_data["worker_id"].append(response.worker_id)
            
            # Compute importance weight if requested
            if compute_importance_weights: #and response.worker_id in self.worker_policies:
                # weight = self.importance_weight_strategy.compute_weight(
                #     prompt=response.prompt,
                #     completion=response.completion,
                #     worker_policy=self.worker_policies[response.worker_id],
                #     server_policy=self.pi_server,
                #     tokenizer=self.tokenizer,
                #     device=self.device
                # )
                # weighting = Weighting(alpha = 1.0)
                weight = self.importance_weight_strategy.compute_weight()
                aggregated_data["importance_weight"].append(weight)
            else:
                # Default weight = 1.0 (no correction)
                aggregated_data["importance_weight"].append(1.0)
        
        dataset = Dataset.from_dict(aggregated_data)
        
        avg_reward = sum(aggregated_data["reward"]) / len(aggregated_data["reward"])
        avg_weight = sum(aggregated_data["importance_weight"]) / len(aggregated_data["importance_weight"])
        
        logger.info(f"[Parameter Server] Aggregated dataset created:")
        logger.info(f"  - Samples: {len(dataset)}")
        logger.info(f"  - Avg reward: {avg_reward:.4f}")
        logger.info(f"  - Avg importance weight: {avg_weight:.4f}")
        self.aggregated_dataset = dataset
        return dataset
    
    def update_with_aggregated_data(self, aggregated_dataset: Dataset):
        """
        Update with aggregated worker responses.
        
        Args:
            aggregated_dataset: Dataset with worker responses and importance weights
        """
        self.aggregated_dataset = aggregated_dataset
        
        #not sure about these lines
        #self.trainer.train_dataset = self.aggregated_dataset
        self._dataloader = None  # Force recreation
        
        logger.info(f"[Parameter Server] Aggregated Response of {len(aggregated_dataset)} samples")
    
    # def compute_logprobs_with_aggregated_data(self,new_model):
    #     """
    #     Update trainer's dataset with aggregated worker responses.
        
    #     Args:
    #         aggregated_dataset: Dataset with worker responses and importance weights
    #     """
    #     #not sure of these lines
    #     self.trainer.train_dataset = self.aggregated_dataset
    #     # self.trainer._train_dataloader = None  # Force recreation
    #     log_probs = []
    #     for result in len(self.aggregated_dataset):
    #         prompt = self.aggregated_dataset["prompt"][result]
    #         completion = self.aggregated_dataset["completion"][result]
    #         tokenized_result =self.tokenizer.tokenize(prompt + completion)
    #         log_probs.append([new_model.compute_log_probs(tokenized_result[len(prompt):])])
        
    # def compute_logprobs_base_policy(self,base_model):
    #     dataset = self.aggregated_dataset
    #     log_probs = []
          
    #     for result in len(self.aggregated_dataset):
    #         prompt = self.aggregated_dataset["prompt"][result]
    #         completion = self.aggregated_dataset["completion"][result]
    #         tokenized_result =self.tokenizer.tokenize(prompt + completion)
    #         log_probs.append([base_model.compute_log_probs(tokenized_result[len(prompt):])])



        
    # logger.info(f"[Parameter Server] Dataset updated with {len(self.aggregated_dataset)} samples")

    #correct the function below
    def compute_log_probs(self, model, prompts, completions) -> torch.Tensor:
        """
        Compute log probabilities of completions under a given model.
        
        Args:
            model: The policy model (π_server or π_base or π_worker)
            prompts: List of prompts
            completions: List of completions
            
        Returns:
            Tensor of log probabilities for each completion
        """
        model.eval()
        
        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]
        
        # Tokenize full sequences
        inputs = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_config.max_prompt_length + self.training_config.max_completion_length
        ).to(self.device)
        
        # Tokenize prompts only to get prompt lengths  (can't we simply find the prompt lengths from the input arguments above?)
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_config.max_prompt_length
        ).to(self.device)

        prompt_lengths = prompt_inputs.input_ids.shape[1] #should this be prompt_inputs.attention_mask.sum(dim = 1)  ?

        with torch.no_grad():
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get completion tokens (everything after prompt)
            # Shape: [batch_size, seq_len - prompt_len]
            completion_ids = inputs.input_ids[:, prompt_lengths:]
            
            # Compute log probabilities
            # logits[:, prompt_lengths-1:-1] aligns with completion_ids
            #log_probs shape: [batch_size, completion_len, vocab_size] - This means that there are log probabilities for each token in the vocabulary at each position in the completion. 
            log_probs = torch.nn.functional.log_softmax(
                logits[:, prompt_lengths-1:-1, :], dim=-1
            )
            
            # Gather log probs for actual completion tokens
            # Shape: [batch_size, completion_len]
            token_log_probs = log_probs.gather(
                2, completion_ids.unsqueeze(-1) # unsqueeze to match dimensions with log_probs so completion_ids shape becomes [batch_size, completion_len, 1]
            ).squeeze(-1)
            
            # Sum over sequence length to get log P(completion | prompt)
            # Shape: [batch_size]   (summing log probabilities is multiplying probabilities)
            sequence_log_probs = token_log_probs.sum(dim=1)
        
        return sequence_log_probs   #in the training step, I will need to find the log probs ratio iteratively for each new word added
    
    # the prompts are the same for all the workers and the parameter server so do we need batch["prompts"]. Also, the batch size is 1 so do we need completion_mask (or is it to make it more gneral) It shoud be off policy data so the responses will not be generated by pi_old.
    def train_step(self, num_steps: int = 1, batch_size: int = 1) -> Dict:
        """
        Algorithm 1, Lines 10-11: Perform μ GRPO optimization steps.
        
        Implements the GRPO loss from Equation 3:
        J_GRPO = E[ (1/G) Σ (1/|o_i|) Σ min(ratio * A, clip(ratio) * A) ] - β * D_KL(π_θ || π_ref)
        
        Args:
            num_steps: Number of optimization steps (essentially minibatches)
            batch_size: How many question/ responses to be used per minibatch when updating the policy
            
        Returns:
            Training statistics
        """
        if self.aggregated_dataset is None:
            logger.warning("[Parameter Server] No aggregated data available!")
            return {}
        
        logger.info(f"[Parameter Server] Training for {num_steps} steps...")
        
        self.pi_server.train()

        batch_size = batch_size
        if self._dataloader is None:
            self._dataloader = DataLoader(
                self.aggregated_dataset, 
                batch_size=batch_size, 
                shuffle=True
            )
        
        dataloader_iter = iter(self._dataloader)


        stats = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "kl_divergence": 0.0,
            "avg_reward": 0.0,
            "avg_importance_weight": 0.0, 
            "avg_ratio": 0.0,
            "clip_fraction": 0.0
        }
        
        for step in range(num_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                logger.warning("Dataloader exhausted, recreating...")
                #not sure about these lines
                dataloader_iter = iter(self._dataloader)
                batch = next(dataloader_iter)
        
            prompts = batch['prompt']
            completions = batch['completion']
            rewards = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)
            importance_weights = torch.tensor(
                batch.get('importance_weight', [1.0] * len(prompts)),
                dtype=torch.float32
            ).to(self.device)
         
            # Tokenize prompts and completions
            full_texts = [p + c for p, c in zip(prompts, completions)]
            
            # Tokenize
            inputs = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_config.max_prompt_length + self.training_config.max_completion_length
            ).to(self.device)
            
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.training_config.max_prompt_length
            ).to(self.device)
            
            # Use attention mask to get actual prompt lengths (handles padding correctly)
            prompt_lengths = prompt_inputs.attention_mask.sum(dim = 1)  
            
            # Forward pass through π_server (current policy)
            #self.pi_server.train()
            server_outputs = self.pi_server(**inputs)
            server_logits = server_outputs.logits
            
            # Forward pass through π_base (reference policy)
            with torch.no_grad():
                ref_outputs = self.pi_ref(**inputs)
                ref_logits = ref_outputs.logits
            
        
            input_ids = inputs.input_ids
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            
            # Create mask for completion tokens (1 for completion, 0 for prompt)
            # This allows us to handle variable prompt lengths efficiently   - would we need this if all the prompts are the same for all the models
            completion_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)
            for i in range(batch_size):
                prompt_len = prompt_lengths[i].item()
                if prompt_len < seq_len:
                    completion_mask[i, prompt_len:] = True
            

            # Compute log probabilities over all positions
            server_log_probs_full = torch.nn.functional.log_softmax(
                server_logits, dim=-1
            )
            ref_log_probs_full = torch.nn.functional.log_softmax(
                ref_logits, dim=-1
            )
            
            # Gather log probs for actual tokens in the sequence
            #shape: [batch_size, seq_len,vocab_size]-> [batch_size, seq_len]
            server_token_log_probs = server_log_probs_full.gather(
                2, input_ids.unsqueeze(-1)
            ).squeeze(-1)
            ref_token_log_probs = ref_log_probs_full.gather(
                2, input_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Count completion tokens for proper averaging (1/|o_i| term in GRPO)
            completion_lengths = completion_mask.float().sum(dim=1)

            # Compute advantages (rewards - baseline)
            baseline = rewards.mean()
            advantages = rewards - baseline

            # Broadcast advantages to token level: all tokens in a sequence share same advantage
            # Shape: [batch_size] -> [batch_size, seq_len]
            advantages_per_token = advantages.unsqueeze(1).expand(-1, seq_len)

            # # Find the probability over completion_ids 
            # server_log_probs = (server_token_log_probs * completion_mask.float())
            # base_log_probs = (base_token_log_probs * completion_mask.float())
            

            # # New implementation of GRPO loss
            # per_token_reward.append()
            # kl_div = (server_log_probs.exp() * (server_log_probs - base_log_probs)).mean()
            # for batch in server_log_probs.shape[0]:
            # baseline = server_log_probs -  base_log_probs
            # min_term = min(baseline,clip(baseline,1-epsilon,1+epsilon))
            # per_token_reward = []
            # clipped_term = min_term*per_token_reward
            # policy_loss = - (importance_weights * clipped_term).mean()


            # Compute token-level policy ratios: π_θ(o_t) / π_θ_old(o_t)
            token_log_ratio = server_token_log_probs - ref_token_log_probs
            token_ratio = torch.exp(token_log_ratio)
            
            # PPO-style clipping
            epsilon = getattr(self.training_config, 'clip_range', 0.1)
            ratio_clipped = torch.clamp(token_ratio, 1.0 - epsilon, 1.0 + epsilon)
            
            # multiply with advantages
            surr1 = token_ratio * advantages_per_token
            surr2 = ratio_clipped * advantages_per_token
            
            # Take minimum and apply completion mask
            token_loss = -torch.min(surr1, surr2) * completion_mask.float()
            
            # Average over tokens (1/|o_i| term), then over batch (1/G term)
            sequence_loss = token_loss.sum(dim=1) / (completion_lengths + 1e-12)
            policy_loss = (importance_weights * sequence_loss).mean()
            
            # KL divergence regularization: KL(π_θ || π_ref) using the equation 4 from the GRPO paper
            kl_ratio = torch.exp(ref_token_log_probs.detach() - server_token_log_probs)# (π_ref/π_θ)
            log_kl_ratio =ref_token_log_probs.detach() - server_token_log_probs # log(π_ref/π_θ)
            
            # Apply completion mask to only compute KL over completion tokens
            token_kl = (kl_ratio - log_kl_ratio - 1) * completion_mask.float()
            #averaging 
            sequence_kl = token_kl.sum(dim=1) / (completion_lengths + 1e-12)
            kl_div = sequence_kl.mean()

            # Total loss 
            # L = -J_GRPO = policy_loss + β * D_KL
            beta = getattr(self.training_config, 'beta', 0.04)
            total_loss = policy_loss + beta * kl_div
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # # Gradient clipping
            max_grad_norm = getattr(self.training_config, 'max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(
                self.pi_server.parameters(),
                max_grad_norm
            )
            
            self.optimizer.step()
            
            # Accumulate stats
            stats["loss"] += total_loss.item()
            stats["policy_loss"] += policy_loss.item()
            stats["kl_divergence"] += kl_div.item()
            stats["avg_reward"] += rewards.mean().item()
            stats["avg_importance_weight"] += importance_weights.mean().item()
            
            # Track policy ratio statistics
            with torch.no_grad():
                masked_ratios = token_ratio[completion_mask]
                if len(masked_ratios) > 0:
                    stats["avg_ratio"] += masked_ratios.mean().item()
                    stats["clip_fraction"] += (
                        ((masked_ratios > 1 + epsilon).float().sum() +
                         (masked_ratios < 1 - epsilon).float().sum()) / len(masked_ratios)
                    ).item()
            
            self.step_count += 1
        
        # Compute averages
        for key in stats:
            stats[key] = stats[key] / num_steps
        
        stats["step"] = self.step_count
        
        logger.info(
            f"[Parameter Server] Step {self.step_count}: "
            f"Loss={stats['loss']:.4f}, "
            f"Policy Loss={stats['policy_loss']:.4f}, "
            f"KL={stats['kl_divergence']:.4f}, "
            f"Avg Reward={stats['avg_reward']:.4f}, "
            f"Avg Ratio={stats['avg_ratio']:.4f}, "
            f"Clip Frac={stats['clip_fraction']:.4f}"
        )
        
        return stats
        
    
    def save_checkpoint(self, path: str):
        """Save parameter server checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        if hasattr(self.pi_server, 'save_pretrained'):
            # For PEFT models (LoRA)
            self.pi_server.save_pretrained(path)
        else:
            # For regular models
            torch.save(self.pi_server.state_dict(), os.path.join(path, "model.pt"))
        
        # Also save optimizer state for resuming
        torch.save(self.optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        
        logger.info(f"[Parameter Server] Checkpoint saved to {path}")




# ============================================================================
# Distributed Off-Policy GRPO Orchestrator
# ============================================================================

class DistributedOffPolicyGRPO:
    """
    Main orchestrator for distributed off-policy GRPO training.
    
    Training Flow:
    1. Initialize π_θ (π_server)
    2. Workers fine-tuned on local datasets
    3. Outer loop (I iterations):
       a. Update π_ref ← π_θ (Line 3)
       b. Inner loop (M steps):
          - Sample prompts from global dataset (Line 5)
          - Update π_θ_old ← π_θ (Line 6, implicit in worker generation)
          - Workers generate responses with π_θ_old (Line 7)
          - Compute rewards (Line 8)
          - Aggregate responses and compute advantages (Line 9)
          - GRPO optimization loop (μ iterations, Lines 10-11)
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        global_dataset: Dataset,
        model_config: ModelConfig,
        training_config: GRPOConfig,
        reward_funcs: List[Callable],
        num_workers: int = 3,
        worker_finetune_epochs: int = 1,
        importance_weight_strategy: Weighting = None,
        device: str = "cuda"
    ):
        self.model_name_or_path = model_name_or_path
        self.global_dataset = global_dataset
        self.model_config = model_config
        self.training_config = training_config
        self.reward_funcs = reward_funcs
        self.num_workers = num_workers
        self.worker_finetune_epochs = worker_finetune_epochs
        self.device = device
        
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING DISTRIBUTED OFF-POLICY GRPO SYSTEM")
        logger.info("="*80)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model (π_base)
        logger.info(f"Loading base model: {model_name_or_path}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_config.attn_implementation,
            use_cache=False if training_config.gradient_checkpointing else True
        )
        
        logger.info(f"Base model loaded: {sum(p.numel() for p in self.base_model.parameters()):,} parameters")
        
        # Initialize parameter server
        self.parameter_server = ParameterServer(
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            global_dataset=global_dataset,
            model_config=model_config,
            training_config=training_config,
            reward_funcs=reward_funcs,
            importance_weight_strategy=importance_weight_strategy,
            device=device
        )
        
        # Partition dataset for workers
        logger.info(f"\nPartitioning dataset for {num_workers} workers...")
        local_datasets = self._partition_dataset()   #how to partition the dataset. 
        
        # Initialize worker nodes
        logger.info(f"\nInitializing {num_workers} worker nodes...")
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
        
        logger.info("\n" + "#"*80)
        logger.info("SYSTEM INITIALIZED SUCCESSFULLY")
        logger.info(f"Workers: {num_workers}")
        logger.info(f"Global dataset: {len(global_dataset)} samples")
        logger.info(f"Worker finetune epochs: {worker_finetune_epochs}")
        logger.info("#"*80 + "\n")
    
    def _partition_dataset(self) -> List[Dataset]:
        """Partition global dataset evenly for workers"""
        indices = list(range(len(self.global_dataset)))
        random.shuffle(indices)  # Shuffle for better distribution
        
        partition_size = len(self.global_dataset) // self.num_workers
        partitions = []
        
        for i in range(self.num_workers):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < self.num_workers - 1 else len(indices)
            partition_indices = indices[start_idx:end_idx]
            
            partition = self.global_dataset.select(partition_indices)
            partitions.append(partition)
            logger.info(f"  Worker {i}: {len(partition)} samples")
        
        return partitions
    
    def finetune_workers(self):
        """Fine-tune all workers on their local datasets"""
        logger.info("\n" + "#"*80)
        logger.info("PHASE 1: WORKER Fine-Tuning")
        logger.info("#"*80 + "\n")
        
        for worker in self.workers:
            worker.finetune(num_epochs=self.worker_finetune_epochs)
            
            # Register worker policy with parameter server for importance weighting
            #probably don't need this right now. 
            policy_state = worker.get_policy_state()
            # self.parameter_server.register_worker_policy(worker.worker_id,policy_state)
        
        logger.info("\n" + "#"*80)
        logger.info("WORKER Fine-Tuning COMPLETED")
        logger.info("#"*80 + "\n")
    
    def sample_prompts(self, batch_size: int = 1) -> List[str]:
        """Sample prompts from global dataset"""
        indices = random.sample(range(len(self.global_dataset)), batch_size)
        prompts = [self.global_dataset[i]["prompt"] for i in indices]
        return prompts
    
    #so for different num_outer_iterations, different questions will be sampled?
    def train(self, num_outer_iterations:int,num_inner_steps: int, num_grpo_iterations: int, num_workers: int, prompts_per_step: int = 1, save_frequency: int = 100, output_dir:str = None) -> Dict:
        """
        Train following Algorithm 1 from GRPO paper.
        
        Args:
            num_outer_iterations: I - number of outer iterations (π_ref updates)
            num_inner_steps: M - number of data collection steps per outer iteration
            num_grpo_iterations: μ - number of optimization steps per batch
            prompts_per_step: Number of prompts to sample per step
            save_frequency: Save checkpoint frequency
            output_dir: Directory to save checkpoints
       
        """
        if output_dir is None:
            output_dir = self.training_config.output_dir
        
        logger.info("\n" + "#"*80)
        logger.info("[Distributed Server] STARTING DISTRIBUTED OFF-POLICY GRPO TRAINING")
        logger.info(f"Outer iterations (I): {num_outer_iterations}")
        logger.info(f"Inner steps (M): {num_inner_steps}")
        logger.info(f"GRPO iterations (μ): {num_grpo_iterations}")
        logger.info(f"Prompts per step: {prompts_per_step}")
        logger.info("#"*80)
        
        # Phase 1: Fine-tune workers
        self.finetune_workers()

        # Phase 2: Distributed off-policy training following Algorithm 1
        logger.info("\nPHASE 2: DISTRIBUTED OFF-POLICY TRAINING")
        
        total_steps = 0
        
        # Algorithm 1, Line 2: OUTER LOOP - for iteration = 1, ..., I
        for outer_iter in range(num_outer_iterations):
            logger.info(f"\n{'#'*80}")
            logger.info(f"OUTER ITERATION {outer_iter + 1}/{num_outer_iterations}")
            logger.info(f"{'#'*80}")
            
            # Algorithm 1, Line 3: Update reference model π_ref ← π_θ
            logger.info("Updating π_ref ← π_θ...")
            self.parameter_server.update_reference_model()
            
            # Algorithm 1, Line 4: INNER LOOP - for step = 1, ..., M
            for inner_step in range(num_inner_steps):
                logger.info(f"\n--- Inner Step {inner_step + 1}/{num_inner_steps} ---")
                
                # Algorithm 1, Line 5: Sample a batch D_b from D
                prompts = self.sample_prompts(batch_size=prompts_per_step)
                logger.info(f" Sampled {len(prompts)} prompts")
                
                # # Algorithm 1, Line 6: Update old policy π_θ_old ← π_θ
                # # This is not relevant (maybe) because workers use their own current policy
                # logger.info("Algorithm 1, Line 6: π_θ_old ← π_θ (implicit in worker generation)")
                
                # Algorithm 1, Line 7: Sample G outputs from π_θ_old
                logger.info("Workers generating responses...")
                all_responses = []
                for worker in self.workers:
                    worker_responses = worker.generate_and_score(prompts)
                    all_responses.extend(worker_responses)
                
                # Algorithm 1, Line 8: Compute rewards (already done in generate_and_score)
                logger.info(f"Rewards computed ({len(all_responses)} samples)")
                
                # Algorithm 1, Line 9: Compute advantages (done inside train_step)
                # Aggregate responses
                aggregated_dataset = self.parameter_server.aggregate_worker_responses(
                    all_responses,
                    compute_importance_weights=True
                )
                
                self.parameter_server.update_with_aggregated_data(aggregated_dataset)
                
                # Algorithm 1, Lines 10-11: for μ GRPO iterations, update π_θ
                logger.info(f"Performing {num_grpo_iterations} GRPO optimization steps...")
                stats = self.parameter_server.train_step(num_steps=num_grpo_iterations,batch_size=num_workers)
                
                total_steps += 1
                
                # Save checkpoint periodically
                if total_steps % save_frequency == 0:
                    checkpoint_path = os.path.join(
                        output_dir,
                        f"checkpoint-outer{outer_iter+1}-step{total_steps}"
                    )
                    self.parameter_server.save_checkpoint(checkpoint_path)
                    logger.info(f"[Checkpoint] Saved to {checkpoint_path}")
            
            # End of outer iteration
            logger.info(f"\nCompleted outer iteration {outer_iter + 1}")
        
        # Final save
        final_path = os.path.join(output_dir, "final_model")
        self.parameter_server.save_checkpoint(final_path)
        
        logger.info("\n" + "#"*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Final model saved to: {final_path}")
        logger.info("#"*80 + "\n")


        

    # # 1. Sample prompts
    # prompts = self.sample_prompts(batch_size=prompts_per_step)
    # logger.info(f"\nSampled {len(prompts)} prompts from global dataset")
    # # for i, prompt in enumerate(prompts[:2]):
    # #     logger.info(f"  Prompt {i+1}: {prompt[:80]}...")
    
    # # 2. Collect responses from workers
    # logger.info("\nCollecting worker responses...")
    # all_responses = []
    # for worker in self.workers:
    #     worker_responses = worker.generate_and_score(prompts)
    #     all_responses.extend(worker_responses)
    
    # logger.info(f"Collected {len(all_responses)} total responses")
    
    # # 3. Aggregate responses and compute importance weights
    # aggregated_dataset = self.parameter_server.aggregate_worker_responses(
    #     all_responses,
    #     compute_importance_weights=True
    # )
    
    # # 4. Update parameter server with aggregated data (don't really need this line I think)
    # self.parameter_server.update_with_aggregated_data(aggregated_dataset)
    
    # # 5. Train parameter server
    # stats = self.parameter_server.train_step(num_steps=1)
    
    # logger.info(f"\n{'#'*80}\n")
    
    # return stats

    # def train(
    #     self,
    #     num_iterations: int,
    #     prompts_per_iteration: int = 1,
    #     save_frequency: int = 100,
    #     output_dir: str = None
    # ):
    #     """
    #     Train the distributed off-policy GRPO system.
        
    #     Args:
    #         num_iterations: Number of training iterations
    #         prompts_per_iteration: Number of prompts per iteration
    #         save_frequency: Save checkpoint every N iterations
    #         output_dir: Directory to save checkpoints
    #     """
    #     if output_dir is None:
    #         output_dir = self.training_config.output_dir
        
    #     logger.info("\n" + "#"*80)
    #     logger.info("STARTING DISTRIBUTED OFF-POLICY GRPO TRAINING")
    #     logger.info(f"Iterations: {num_iterations}")
    #     logger.info(f"Prompts per iteration: {prompts_per_iteration}")
    #     logger.info(f"Save frequency: {save_frequency}")
    #     logger.info(f"Output directory: {output_dir}")
    #     logger.info("\n" + "#"*80)

        
    #     # Phase 1: Pre-train workers
    #     logger.info("\nPHASE 1: WORKER Fine-tuning")
    #     self.finetune_workers()
        
    #     # Phase 2: Distributed off-policy training
    #     logger.info("\nPHASE 2: DISTRIBUTED OFF-POLICY TRAINING")
        
    #     for iteration in range(num_iterations):
    #         # Execute training iteration
    #         stats = self.training_step(prompts_per_iteration)
            
    #         # Save checkpoint periodically  (if we are saving here we don't need to save in the grpo.py file) and I am not sure if I am saving it correctly. 
    #         if (iteration + 1) % save_frequency == 0:
    #             checkpoint_path = os.path.join(
    #                 output_dir,
    #                 f"checkpoint-{iteration + 1}"
    #             )
    #             self.parameter_server.save_checkpoint(checkpoint_path)
    #             logger.info(f"\n[Checkpoint] Saved to {checkpoint_path}\n")
                
    #     # Final save
    #     final_path = os.path.join(output_dir, "final_model")
    #     self.parameter_server.save_checkpoint(final_path)
        
    #     logger.info("\n" + "#"*80)
    #     logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    #     logger.info(f"Final model saved to: {final_path}")
    #     logger.info("#"*80 + "\n")

#In the end we can do a final accuracy eval here on the entire dataset with the parameter server. 




# # from grpo import trainViaGRPO

# # class WorkerNodes:
# #     def __init__(self,reward_func,dataset):
# #         self.reward_func = reward_func
# #         self.dataset = dataset
# #     def GRPO_training(self):
# #         rewards=trainViaGRPO(reward_func)
# #         self.rewards = rewards
# #     def send_rewards(self):
# #         return self.rewards

# # class ParamterServer:
# #     def __init__(self,rewards,dataset):
# #         self.rewards = rewards
# #         self.dataset = dataset
# #     def optimize_policy(self):
 

# """
# Off-Policy GRPO System with Parameter Server and Worker Nodes
# Uses Tina's GRPOTrainer for actual training
# """

# # workers.py
# import copy
# import json
# import logging
# import os
# import torch
# from dataclasses import dataclass
# from typing import List, Dict, Optional, Callable
# from datasets import Dataset
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import get_peft_model, LoraConfig, TaskType

# from tina.post_train_hf.grpo_trainer import GRPOTrainer
# from tina.post_train_hf.grpo_config import GRPOConfig
# from trl import ModelConfig

# logger = logging.getLogger(__name__)


# @dataclass
# class WorkerResponse:
#     """Response from a worker node after generation"""
#     worker_id: int
#     prompt: str
#     completions: List[str]  # Multiple completions from worker
#     rewards: List[float]  # Rewards for each completion
#     # Store generation info for off-policy correction
#     generation_metadata: Dict


# class WorkerNode:
#     """
#     Worker node with its own LoRA adapter and local dataset partition.
#     Uses Tina's GRPOTrainer for generation.
#     """
    
#     def __init__(
#         self,
#         worker_id: int,
#         base_model: AutoModelForCausalLM,
#         tokenizer: AutoTokenizer,
#         local_dataset: Dataset,
#         model_config: ModelConfig,
#         training_config: GRPOConfig,
#         reward_funcs: List[Callable],
#         device: str = "cuda"
#     ):
#         self.worker_id = worker_id
#         self.tokenizer = tokenizer
#         self.local_dataset = local_dataset
#         self.model_config = model_config
#         self.training_config = training_config
#         self.reward_funcs = reward_funcs
#         self.device = device
        
#         logger.info(f"Initializing Worker {worker_id}")
        
#         # Clone the base model for this worker
#         self.model = copy.deepcopy(base_model)
        
#         # Apply LoRA to worker model
#         if model_config.use_peft:
#             peft_config = LoraConfig(
#                 r=model_config.lora_r,
#                 lora_alpha=model_config.lora_alpha,
#                 lora_dropout=model_config.lora_dropout,
#                 target_modules=model_config.lora_target_modules,
#                 inference_mode=False,
#                 bias="none",
#                 task_type=TaskType.CAUSAL_LM
#             )
#             self.model = get_peft_model(self.model, peft_config)
        
#         self.model.to(self.device)
        
#         # Create a mini training config for this worker (for generation only)
#         self.worker_training_config = copy.deepcopy(training_config)
#         self.worker_training_config.output_dir = f"{training_config.output_dir}/worker_{worker_id}"
        
#         # Initialize GRPOTrainer for this worker (used for generation)
#         # Note: We won't call .train() on worker trainers, just use for generation
#         self.trainer = GRPOTrainer(
#             model=self.model,
#             processing_class=tokenizer,
#             reward_funcs=reward_funcs,
#             args=self.worker_training_config,
#             train_dataset=local_dataset
#         )
        
#         logger.info(f"Worker {worker_id} initialized with LoRA adapter")
    
#     def generate_responses(
#         self,
#         prompt_batch: List[str],
#         num_generations_per_prompt: int = None
#     ) -> List[Dict]:
#         """
#         Generate responses for a batch of prompts using worker's policy.
#         Uses Tina's GRPOTrainer generation mechanism.
        
#         Returns: List of dicts with {prompt, completions, rewards}
#         """
#         if num_generations_per_prompt is None:
#             num_generations_per_prompt = self.training_config.num_generations
        
#         self.model.eval()
        
#         # Create a temporary dataset for these prompts
#         #temp_dataset = Dataset.from_dict({"prompt": prompt_batch})
        
#         # Use trainer's generation mechanism
#         # This is similar to how GRPOTrainer generates during training
#         with torch.no_grad():
#             generation_results = []
            
#             for prompt in prompt_batch:
#                 # Tokenize prompt
#                 inputs = self.tokenizer(
#                     prompt,
#                     return_tensors="pt",
#                     padding=True,
#                     truncation=True,
#                     max_length=self.training_config.max_prompt_length
#                 ).to(self.device)
                
#                 completions = []
#                 scores_list = []
                
#                 # Generate multiple completions
#                 for _ in range(num_generations_per_prompt):
#                     outputs = self.model.generate(
#                         **inputs,
#                         max_new_tokens=self.training_config.max_completion_length,
#                         temperature=self.training_config.temperature,
#                         do_sample=True,
#                         pad_token_id=self.tokenizer.pad_token_id,
#                         eos_token_id=self.tokenizer.eos_token_id,
#                         return_dict_in_generate=True,
#                         output_scores=True
#                     )
                    
#                     # Extract completion
#                     prompt_len = inputs.input_ids.shape[1]
#                     completion_ids = outputs.sequences[0][prompt_len:]
#                     completion = self.tokenizer.decode(
#                         completion_ids,
#                         skip_special_tokens=True
#                     )
                    
#                     completions.append(completion)
                    
#                     # Store generation metadata (for off-policy correction)
#                     scores_list.append({
#                         'sequence': outputs.sequences[0],
#                         'scores': outputs.scores if hasattr(outputs, 'scores') else None
#                     })
                
#                 generation_results.append({
#                     'prompt': prompt,
#                     'completions': completions,
#                     'generation_metadata': scores_list
#                 })
        
#         return generation_results
    
#     def update_from_parameter_server(self, state_dict: Dict):
#         """Update worker's model weights from parameter server"""
#         # Only update the trainable parameters (LoRA weights)
#         self.model.load_state_dict(state_dict, strict=False)
#         logger.info(f"Worker {self.worker_id} updated from parameter server")
    
#     def get_state_dict(self) -> Dict:
#         """Get worker's current model state"""
#         return self.model.state_dict()


# class ParameterServer:
#     """
#     Central parameter server that aggregates worker responses and 
#     performs GRPO optimization using Tina's GRPOTrainer.
#     """
    
#     def __init__(
#         self,
#         model: AutoModelForCausalLM,
#         tokenizer: AutoTokenizer,
#         global_dataset: Dataset,
#         model_config: ModelConfig,
#         training_config: GRPOConfig,
#         reward_funcs: List[Callable],
#         device: str = "cuda"
#     ):
#         self.tokenizer = tokenizer
#         self.global_dataset = global_dataset
#         self.model_config = model_config
#         self.training_config = training_config
#         self.reward_funcs = reward_funcs
#         self.device = device
        
#         logger.info("Initializing Parameter Server")
        
#         # Apply LoRA to parameter server model
#         if model_config.use_peft:
#             peft_config = LoraConfig(
#                 r=model_config.lora_r,
#                 lora_alpha=model_config.lora_alpha,
#                 lora_dropout=model_config.lora_dropout,
#                 target_modules=model_config.lora_target_modules,
#                 inference_mode=False,
#                 bias="none",
#                 task_type=TaskType.CAUSAL_LM
#             )
#             self.model = get_peft_model(model, peft_config)
#         else:
#             self.model = model
        
#         self.model.to(device)
        
#         # Initialize aggregated dataset for off-policy training
#         # This will be updated with worker responses
#         self.aggregated_dataset = None
        
#         # Initialize GRPOTrainer for parameter server
#         # This is the actual trainer that will perform optimization
#         self.trainer = GRPOTrainer(
#             model=self.model,
#             processing_class=tokenizer,
#             reward_funcs=reward_funcs,
#             args=training_config,
#             train_dataset=global_dataset  # Will be replaced with aggregated data
#         )
        
#         self.step_count = 0
        
#         logger.info("Parameter Server initialized with GRPOTrainer")
    
#     def aggregate_worker_responses(
#         self,
#         worker_generation_results: List[List[Dict]],
#         compute_rewards: bool = True
#     ) -> Dataset:
#         """
#         Aggregate responses from all workers into a dataset.
        
#         Args:
#             worker_generation_results: List of generation results from each worker
#             compute_rewards: Whether to compute rewards here
            
#         Returns:
#             Dataset with (prompt, completion) pairs and rewards
#         """
#         aggregated_data = {
#             "prompt": [],
#             "completion": [],
#             "worker_id": []
#         }
        
#         # If we need to track additional columns for rewards
#         if compute_rewards and len(self.global_dataset) > 0:
#             # Copy other columns from original dataset that might be needed for rewards
#             sample_columns = self.global_dataset.column_names
#             for col in sample_columns:
#                 if col not in ["prompt", "completion"]:
#                     aggregated_data[col] = []
        
#         # Flatten all worker responses
#         for worker_id, generation_results in enumerate(worker_generation_results):
#             for result in generation_results:
#                 prompt = result['prompt']
#                 completions = result['completions']
                
#                 for completion in completions:
#                     aggregated_data["prompt"].append(prompt)
#                     aggregated_data["completion"].append(completion)
#                     aggregated_data["worker_id"].append(worker_id)
                    
#                     # Add other columns if needed for reward computation
#                     if compute_rewards:
#                         # Find matching row in global dataset
#                         matching_rows = [
#                             i for i, p in enumerate(self.global_dataset["prompt"])
#                             if p == prompt
#                         ]
#                         if matching_rows:
#                             idx = matching_rows[0]
#                             for col in sample_columns:
#                                 if col not in ["prompt", "completion"]:
#                                     aggregated_data[col].append(
#                                         self.global_dataset[idx][col]
#                                     )
        
#         dataset = Dataset.from_dict(aggregated_data)
#         logger.info(f"Aggregated {len(dataset)} responses from {len(worker_generation_results)} workers")
        
#         return dataset
    
#     def update_with_off_policy_data(
#         self,
#         aggregated_dataset: Dataset
#     ):
#         """
#         Update the trainer's dataset with aggregated off-policy data.
#         This allows Tina's GRPOTrainer to train on worker-generated responses.
#         """
#         # Update the trainer's train_dataset
#         self.trainer.train_dataset = aggregated_dataset
        
#         # Recreate dataloader with new dataset
#         self.trainer._train_dataloader = None  # Force recreation
        
#         self.aggregated_dataset = aggregated_dataset
#         logger.info(f"Parameter server dataset updated with {len(aggregated_dataset)} samples")
    
#     def train_step(self, num_steps: int = 1):
#         """
#         Perform training steps using Tina's GRPOTrainer on aggregated data.
        
#         Args:
#             num_steps: Number of optimization steps to perform
#         """
#         if self.aggregated_dataset is None:
#             logger.warning("No aggregated dataset available for training")
#             return {}
        
#         logger.info(f"Parameter server performing {num_steps} training steps")
        
#         # Use Tina's trainer to perform actual GRPO optimization
#         # We need to manually step through the training loop
#         self.model.train()
        
#         train_dataloader = self.trainer.get_train_dataloader()
        
#         stats_accumulator = {
#             "loss": 0.0,
#             "rewards": []
#         }
        
#         for step in range(num_steps):
#             try:
#                 batch = next(iter(train_dataloader))
                
#                 # Perform one training step using trainer's logic
#                 # This will do the GRPO optimization
#                 loss = self.trainer.training_step(self.model, batch)
                
#                 stats_accumulator["loss"] += loss.item() if torch.is_tensor(loss) else loss
                
#                 self.step_count += 1
                
#             except StopIteration:
#                 logger.warning("Dataloader exhausted, recreating...")
#                 train_dataloader = self.trainer.get_train_dataloader()
        
#         # Compute average stats
#         stats = {
#             "loss": stats_accumulator["loss"] / num_steps,
#             "step": self.step_count
#         }
        
#         logger.info(f"Step {self.step_count}: Loss={stats['loss']:.4f}")
        
#         return stats
    
#     def get_state_dict(self) -> Dict:
#         """Get current model state for broadcasting to workers"""
#         return self.model.state_dict()
    
#     def save_checkpoint(self, path: str):
#         """Save parameter server checkpoint using trainer's save"""
#         self.trainer.save_model(path)
#         logger.info(f"Checkpoint saved to {path}")


# class DistributedOffPolicyGRPO:
#     """
#     Main orchestrator for distributed off-policy GRPO training.
#     Coordinates workers and parameter server, both using Tina's GRPOTrainer.
#     """
    
#     def __init__(
#         self,
#         model_name_or_path: str,
#         global_dataset: Dataset,
#         model_config: ModelConfig,
#         training_config: GRPOConfig,
#         reward_funcs: List[Callable],
#         num_workers: int = 3,
#         device: str = "cuda"
#     ):
#         self.model_name_or_path = model_name_or_path
#         self.global_dataset = global_dataset
#         self.model_config = model_config
#         self.training_config = training_config
#         self.reward_funcs = reward_funcs
#         self.num_workers = num_workers
#         self.device = device
        
#         # Ensure num_generations matches num_workers
#         if training_config.num_generations != num_workers:
#             logger.warning(
#                 f"Setting num_generations={num_workers} to match num_workers "
#                 f"(was {training_config.num_generations})"
#             )
#             training_config.num_generations = num_workers
        
#         # Load tokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         # Load base model
#         logger.info(f"Loading base model: {model_name_or_path}")
#         self.base_model = AutoModelForCausalLM.from_pretrained(
#             model_name_or_path,
#             torch_dtype=torch.bfloat16 if training_config.bf16 else torch.float16,
#             device_map=device,
#             use_cache=not training_config.gradient_checkpointing
#         )
        
#         # Initialize parameter server
#         logger.info("Initializing Parameter Server...")
#         self.parameter_server = ParameterServer(
#             model=copy.deepcopy(self.base_model),
#             tokenizer=self.tokenizer,
#             global_dataset=global_dataset,
#             model_config=model_config,
#             training_config=training_config,
#             reward_funcs=reward_funcs,
#             device=device
#         )
        
#         # Partition dataset for workers
#         local_datasets = self._partition_dataset()
        
#         # Initialize worker nodes
#         logger.info(f"Initializing {num_workers} worker nodes...")
#         self.workers = []
#         for i in range(num_workers):
#             worker = WorkerNode(
#                 worker_id=i,
#                 base_model=copy.deepcopy(self.base_model),
#                 tokenizer=self.tokenizer,
#                 local_dataset=local_datasets[i],
#                 model_config=model_config,
#                 training_config=training_config,
#                 reward_funcs=reward_funcs,
#                 device=device
#             )
#             self.workers.append(worker)
        
#         logger.info("Distributed Off-Policy GRPO System initialized!")
    
#     def _partition_dataset(self) -> List[Dataset]:
#         """Partition global dataset for worker nodes"""
#         dataset_size = len(self.global_dataset)
#         partition_size = dataset_size // self.num_workers
        
#         partitions = []
#         for i in range(self.num_workers):
#             start_idx = i * partition_size
#             end_idx = (start_idx + partition_size 
#                       if i < self.num_workers - 1 
#                       else dataset_size)
            
#             partition = self.global_dataset.select(range(start_idx, end_idx))
#             partitions.append(partition)
#             logger.info(f"Worker {i} dataset size: {len(partition)}")
        
#         return partitions
    
#     def sample_prompts(self, batch_size: int = 1) -> List[str]:
#         """Sample prompts from global dataset (same for all workers)"""
#         import random
#         indices = random.sample(range(len(self.global_dataset)), batch_size)
#         prompts = [self.global_dataset[i]["prompt"] for i in indices]
#         return prompts
    
#     def training_step(self, prompts_per_step: int = 1) -> Dict:
#         """
#         Execute one distributed training iteration:
#         1. Sample prompts from global dataset
#         2. All workers generate responses for same prompts
#         3. Aggregate worker responses
#         4. Parameter server trains on aggregated data using GRPO
#         """
#         logger.info(f"\n{'='*60}")
#         logger.info(f"Training Step {self.parameter_server.step_count + 1}")
        
#         # 1. Sample prompts (same for all workers)
#         prompts = self.sample_prompts(batch_size=prompts_per_step)
#         logger.info(f"Sampled {len(prompts)} prompts")
#         for i, prompt in enumerate(prompts[:2]):  # Show first 2
#             logger.info(f"Prompt {i+1}: {prompt[:100]}...")
        
#         # 2. Collect responses from all workers
#         all_worker_results = []
#         for worker in self.workers:
#             worker_results = worker.generate_responses(
#                 prompt_batch=prompts,
#                 num_generations_per_prompt=1  # Each worker generates 1 response per prompt
#             )
#             all_worker_results.append(worker_results)
#             logger.info(f"Worker {worker.worker_id} generated {len(worker_results)} responses")
        
#         # 3. Aggregate worker responses into dataset
#         aggregated_dataset = self.parameter_server.aggregate_worker_responses(
#             all_worker_results,
#             compute_rewards=True
#         )
        
#         # 4. Update parameter server with off-policy data
#         self.parameter_server.update_with_off_policy_data(aggregated_dataset)
        
#         # 5. Parameter server trains on aggregated data
#         stats = self.parameter_server.train_step(num_steps=1)
        
#         logger.info(f"{'='*60}\n")
        
#         return stats
    
#     def train(
#         self,
#         num_iterations: int,
#         prompts_per_iteration: int = 1,
#         sync_frequency: int = 10,
#         save_frequency: int = 100,
#         output_dir: str = None
#     ):
#         """
#         Train the distributed off-policy GRPO system.
        
#         Args:
#             num_iterations: Number of training iterations
#             prompts_per_iteration: Number of prompts to sample per iteration
#             sync_frequency: Sync workers every N iterations
#             save_frequency: Save checkpoint every N iterations
#             output_dir: Directory to save checkpoints
#         """
#         if output_dir is None:
#             output_dir = self.training_config.output_dir
        
#         logger.info(f"\nStarting Distributed Off-Policy GRPO Training")
#         logger.info(f"Iterations: {num_iterations}")
#         logger.info(f"Workers: {self.num_workers}")
#         logger.info(f"Prompts per iteration: {prompts_per_iteration}")
#         logger.info(f"Sync frequency: {sync_frequency}")
#         logger.info(f"Output directory: {output_dir}")
        
#         for iteration in range(num_iterations):
#             # Execute training iteration
#             stats = self.training_step(prompts_per_iteration)
            
#             # Periodic worker synchronization
#             if (iteration + 1) % sync_frequency == 0:
#                 logger.info(f"\n[Sync] Iteration {iteration + 1}: Broadcasting parameters to workers")
#                 state_dict = self.parameter_server.get_state_dict()
                
#                 for worker in self.workers:
#                     worker.update_from_parameter_server(state_dict)
                
#                 logger.info(f"[Sync] All workers synchronized\n")
            
#             # Save checkpoint
#             if (iteration + 1) % save_frequency == 0:
#                 checkpoint_path = os.path.join(output_dir, f"checkpoint-{iteration + 1}")
#                 self.parameter_server.save_checkpoint(checkpoint_path)
        
#         # Final save
#         final_path = os.path.join(output_dir, "final_model")
#         self.parameter_server.save_checkpoint(final_path)
        
#         logger.info(f"\nTraining completed! Final model saved to {final_path}")





# Compute advantages (rewards - baseline)
            # Using mean reward as baseline. 
            # baseline = rewards.mean()
            # advantages = rewards - baseline
            
            # # Normalize advantages. 
            # if advantages.std() > 1e-8:
            #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # # Compute policy ratio: π_server / π_base
            # token_ratio = exp(server_token_log_probs - base_token_log_probs) #[batch,seq_len] I believe this should be [batch,completion_length though I am not sure]
            # log_ratio = server_log_prob - base_log_prob
            
            # # GRPO policy loss with importance weighting
            # # Loss = -importance_weight * advantage * log(π_server / π_base)
            # policy_loss = -(importance_weights * advantages * log_ratio).mean()
            
            # # KL divergence regularization: KL(π_server || π_base)
            # kl_div = (server_log_prob.exp() * log_ratio).mean()
            
            # # Total loss
            # beta = self.training_config.beta if hasattr(self.training_config, 'beta') else 0.04
            # total_loss = policy_loss + beta * kl_div
            


