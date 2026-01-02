# import datasets
# from datasets import Dataset, load_dataset
# from datetime import datetime
# import json
# import logging
# import os
# from peft import get_peft_model, LoraConfig, TaskType
# import random
# import sys
# import torch
# import transformers
# from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
# from transformers.trainer_utils import get_last_checkpoint

# from trl import ModelConfig, TrlParser # GRPOTrainer, GRPOConfig
# from tina.post_train_hf.grpo_trainer import GRPOTrainer # use this new one for Dr.GRPO
# from tina.post_train_hf.grpo_config import GRPOConfig # use this new one for Dr.GRPO

# # from .config import ModelPTConfig
# from tina.post_train_hf.callback import FixedPromptEvaluationCallback, PushToHubRevisionCallback, GradientClippingLoggerCallback
# from tina.post_train_hf.preprocess import make_conv_for_grpo, make_conv_for_grpo_l1
# from tina.post_train_hf.rewards import accuracy_reward
# from tina.utils.chat_template import DEFAULT_CHAT_TEMPLATE, REASON_CHAT_TEMPLATE
# from tina.utils.constant import RL_POST_TRAIN_CONFIG_MAP
# from tina.utils.prompt import OPEN_R1_SYSTEM_PROMPT, OPEN_RS_SYSTEM_PROMPT

# from offPolicyGRPO.workers import DistributedOffPolicyGRPO
# from offPolicyGRPO.config import ModelPTconfig

# def main():
#     parser = TrlParser((ModelPTConfig, GRPOConfig, ModelConfig))
#     pt_args, training_args, model_args = parser.parse_args_and_config()
#     set_seed(training_args.seed)

#     #use LORA adapters
#     model_args.use_peft = True
#     model_args.lora_r = 32
#     model_args.lora_alpha = 128
#     model_args.lora_dropout = 0.05
#     model_args.lora_target_modules= ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
   

#     os.environ["WANDB_PROJECT"] = "offPolicy_GRPO_Training"

#     ################
#     # Set up logging
#     ################

#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)])
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()

#     # Log on each process a small summary
#     logger.warning(
#         f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
#         + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")
#     logger.info(f" In grpo.py file: Model parameters {model_args}")
#     logger.info(f" In grpo.py file: Post training parameters {pt_args}")
#     logger.info(f" In grpo.py file: Training parameters {training_args}")

#     #####################
#     # Set up output paths
#     #####################

#     current_time = datetime.now()
#     formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")

#     model_name_or_path = model_args.model_name_or_path
#     ckpt_dir = os.environ.get("CKPT_DIR", "./checkpoints") #ckpt_dir = os.environ["CKPT_DIR"]
#     ckpt_prefix = f"{ckpt_dir}/models/{model_name_or_path}"
#     if model_args.use_peft:
#         ckpt_postfix = f"offpolicy_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
#     else:
#         ckpt_postfix = f"offpolicyfull_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"

#     model_args.model_name_or_path = f"{ckpt_prefix}/base"
#     training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}"
#     # training_args.hub_model_id = f"{training_args.hub_model_id}_{ckpt_postfix}"
#     training_args.run_name = f"{model_name_or_path}_{ckpt_postfix}_{formatted_datetime}"

#     training_args.hub_model_id = f"{training_args.hub_model_id}/{model_name_or_path}"

#     #######################################################################
#     # Load and preprocess dataset (tokenization is handled by GRPO Trainer)
#     #######################################################################

#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
#     if "Llama" in model_args.model_name_or_path:
#         tokenizer.pad_token = "<|finetune_right_pad_id|>"
#     elif "Qwen" in model_args.model_name_or_path:
#         tokenizer.pad_token = "<|fim_pad|>"
#     tokenizer.chat_template = REASON_CHAT_TEMPLATE

#     model_post_train_dataset_name = RL_POST_TRAIN_CONFIG_MAP[pt_args.model_post_train_dataset_name]
#     if pt_args.model_post_train_dataset_config is not None:
#         train_dataset = load_dataset(model_post_train_dataset_name, split="train", name=pt_args.model_post_train_dataset_config)
#     else:
#         train_dataset = load_dataset(model_post_train_dataset_name, split="train")
#     # required by GRPOTrainer: (prompt, solution) columns
#     if 'solution' not in train_dataset.column_names and 'answer' in train_dataset.column_names:
#         train_dataset = train_dataset.rename_column('answer', 'solution')

#         # Wrap the 'solution' values in $...$
#         def wrap_in_math(example):
#             return {"solution": f"${example['solution']}$"}

#         # Apply the transformation to the entire dataset
#         train_dataset = train_dataset.map(wrap_in_math)
#     if 'problem' not in train_dataset.column_names and 'question' in train_dataset.column_names:
#         train_dataset = train_dataset.rename_column('question', 'problem')
#     if 'problem' not in train_dataset.column_names and 'prompt' in train_dataset.column_names:
#         train_dataset = train_dataset.rename_column('prompt', 'problem')
#     if "messages" in train_dataset.column_names:
#         train_dataset = train_dataset.remove_columns("messages")

#     # handle deepscaler separately
#     if "deepscaler" in pt_args.model_post_train_dataset_name:
#         train_dataset = train_dataset.rename_column('solution', 'solution_archive')
#         train_dataset = train_dataset.rename_column('answer', 'solution')

#         # Wrap the 'solution' values in $...$
#         def wrap_in_math(example):
#             return {"solution": f"${example['solution']}$"}

#         # Apply the transformation to the entire dataset
#         train_dataset = train_dataset.map(wrap_in_math)
#     elif "2thought" in pt_args.model_post_train_dataset_name:
#         train_dataset = train_dataset.rename_column('messages', 'problem')
#         train_dataset = train_dataset.rename_column('verification_info', 'solution')

#         def extract_problem(example):
#             problem = example['problem'][0]["content"]
#             return {"problem": problem}

#         def extract_solution(example):
#             solution = json.loads(example['solution'])
#             solution = solution["answer"]["value"]
#             return {"solution": f"${solution}$"}

#         # Apply the transformation to the entire dataset
#         train_dataset = train_dataset.map(extract_problem)
#         train_dataset = train_dataset.map(extract_solution)


#     SYSTEM_PROMPT = OPEN_RS_SYSTEM_PROMPT if "open-rs" in model_post_train_dataset_name else OPEN_R1_SYSTEM_PROMPT

#     if "l1" in pt_args.model_post_train_dataset_name:
#         # uniformly sample a target length between 100 and 4000
#         min_length = 100
#         max_length = 4000
#         train_dataset = train_dataset.map(
#             make_conv_for_grpo_l1,
#             fn_kwargs={"system_prompt": SYSTEM_PROMPT, "min_length": min_length, "max_length": max_length})
#     else:
#         train_dataset = train_dataset.map(
#             make_conv_for_grpo,
#             fn_kwargs={"system_prompt": SYSTEM_PROMPT})

#     ######################
#     # Initialize the model
#     ######################

#     # don't need to do these - already doing them in the parameter server and worker models. 


#     #############################
#     # Set up reward functions
#     #############################

#     RL_POST_TRAIN_REWARD_MAP = {
#         "accuracy": accuracy_reward
#     }
#     rl_reward_funcs = [RL_POST_TRAIN_REWARD_MAP[func] for func in pt_args.rl_post_train_reward_funcs]
#     training_args.reward_weights = pt_args.rl_post_train_reward_weights

#     #not sure of these lines
#     # if model_args.use_peft:
#     #     callbacks = [
#     #         FixedPromptEvaluationCallback(system_prompt=OPEN_R1_SYSTEM_PROMPT, eval_steps=training_args.save_steps),
#     #         # PushToHubRevisionCallback(dataset_name=pt_args.model_post_train_dataset_name, use_peft=model_args.use_peft)
#     #     ]
#     # else:
#     #     callbacks = [
#     #         GradientClippingLoggerCallback(),
#     #         FixedPromptEvaluationCallback(system_prompt=OPEN_R1_SYSTEM_PROMPT, eval_steps=training_args.save_steps),
#     #         # PushToHubRevisionCallback(dataset_name=pt_args.model_post_train_dataset_name, use_peft=model_args.use_peft)
#     #     ]

#     # trainer = GRPOTrainer(
#     #     model=model,
#     #     processing_class=tokenizer,
#     #     reward_funcs=rl_reward_funcs,
#     #     args=training_args,
#     #     train_dataset=train_dataset,
#     #     callbacks=callbacks)

#     #########################
#     # Training and Evaluation
#     #########################

#     # logger.info(f"\nStarting training for {training_args.num_train_epochs} epochs.")

#     logger.info("Initializing Distributed Off-Policy GRPO System...")
    
#     distributed_system = DistributedOffPolicyGRPO(
#         model_name_or_path=model_args.model_name_or_path,
#         global_dataset=train_dataset,
#         model_config=model_args,
#         training_config=training_args,
#         reward_funcs=rl_reward_funcs,
#         num_workers=pt_args.num_worker_nodes,
#         worker_finetune_epochs=1,  # Pre-train workers for 1 epoch
#         importance_weight_strategy=None,  # Use uniform weighting
#         device="cuda" if torch.cuda.is_available() else "cpu"
#     )
        
#     #######################################################################
#     # Calculate training iterations following Algorithm 1
#     #######################################################################
#     # Algorithm 1 has three levels of loops:
#     # I (outer iterations): How many times to update π_ref (reference model)
#     # M (inner steps): How many prompts/batches to collect per outer iteration
#     # μ (GRPO iterations): How many gradient updates per collected batch
    
#     # Understanding the loops:
#     # - OUTER (I): Updates π_ref ← π_θ 
#     # - INNER (M): Number of times we sample prompts and collect worker responses
#     # - GRPO (μ): How many gradient steps on each collected batch (usually 1-3)
    
#     # Calculate total optimization steps
#     # if hasattr(training_args, 'max_steps') and training_args.max_steps > 0:
#     #     total_optimization_steps = training_args.max_steps
#     # else:
#     # Each "step" in inner loop processes this many prompts
#     # prompts_per_inner_step = training_args.per_device_train_batch_size
#     prompts_per_step = 1  #1 question per inner step
#     samples_per_inner_step = prompts_per_step*pt_args.num_worker_nodes
#     inner_steps_per_epoch = len(train_dataset) // prompts_per_step
#     # Total inner steps across all epochs
#     total_inner_steps = int(inner_steps_per_epoch * training_args.num_train_epochs)
#     # μ (GRPO iterations): How many gradient steps per batch
#     # we default to 1 for simplicity
#     num_grpo_iterations = 1  # μ: gradient updates per batch
#     gradient_update_per_inner_step = samples_per_inner_step*num_grpo_iterations
#     total_gradient_updates = total_inner_steps*gradient_update_per_inner_step
    
#     # I (outer iterations): How often to update π_ref
#     # Paper doesn't specify, but updating too frequently defeats the purpose
#     num_outer_iterations = training_args.num_train_epochs  
    
#     # M (inner steps): Number of data collection rounds per outer iteration
#     # This is calculated to achieve the target total_optimization_steps
#     # total_optimization_steps = I × M × μ
#     num_inner_steps = inner_steps_per_epoch
    
#     # Recalculate actual total for logging
#     actual_total_steps = num_outer_iterations * num_inner_steps * num_grpo_iterations
    
#     logger.info(f"\n{'='*80}")
#     logger.info("TRAINING SCHEDULE (Algorithm 1)")
#     logger.info(f"{'='*80}")
#     logger.info(f"Dataset size: {len(train_dataset)} samples")
#     logger.info(f"Epochs: {training_args.num_train_epochs}")
#     logger.info(f"")
#     logger.info(f"Three-level loop structure:")
#     logger.info(f"  I (Outer iterations): {num_outer_iterations}")
#     logger.info(f"    └─ Updates π_ref ← π_θ at start of each iteration")
#     logger.info(f"  M (Inner steps): {num_inner_steps}")
#     logger.info(f"    └─ Data collection rounds (sample prompts → workers generate)")
#     logger.info(f"  μ (GRPO iterations): {num_grpo_iterations}")
#     logger.info(f"    └─ Gradient updates per collected batch")
#     logger.info(f"")
#     logger.info(f"Prompts per inner step: {training_args.per_device_train_batch_size}")
#     logger.info(f"Actual total optimization steps: {actual_total_steps}")
#     logger.info(f"π_ref update frequency: every {num_inner_steps * num_grpo_iterations} steps")
#     logger.info(f"Save frequency: {training_args.save_steps} steps")
#     logger.info(f"{'='*80}\n")
    
#     #######################################################################
#     # Train the distributed system
#     #######################################################################
    
#     logger.info("\n" + "="*80)
#     logger.info("STARTING DISTRIBUTED OFF-POLICY GRPO TRAINING")
#     logger.info("="*80 + "\n")
    
#     distributed_system.train(
#         num_outer_iterations=num_outer_iterations,  # I
#         num_inner_steps=num_inner_steps,            # M
#         num_grpo_iterations=num_grpo_iterations,    # μ
#         num_workers = pt_args.num_worker_nodes,
#         prompts_per_step=prompts_per_step,
#         save_frequency=training_args.save_steps,
#         output_dir=training_args.output_dir
#     )
    
#     logger.info("\n" + "="*80)
#     logger.info("TRAINING COMPLETED SUCCESSFULLY!")
#     logger.info(f"Final model saved to: {training_args.output_dir}/final_model")
#     logger.info("="*80 + "\n")
    
#     # Optional: Push to hub if configured
#     if training_args.push_to_hub:
#         logger.info("Pushing model to hub...")
#         try:
#             # The parameter server's model should be pushed
#             from huggingface_hub import HfApi
#             api = HfApi()
            
#             final_model_path = os.path.join(training_args.output_dir, "final_model")
#             api.upload_folder(
#                 folder_path=final_model_path,
#                 repo_id=training_args.hub_model_id,
#                 commit_message=f"Off-policy GRPO training completed: {actual_total_steps} steps"
#             )
#             logger.info(f"Model pushed to {training_args.hub_model_id}")
#         except Exception as e:
#             logger.error(f"Failed to push to hub: {e}")


  
#     # # Check for last checkpoint
#     # ckpt = None
#     # if training_args.resume_from_checkpoint is not None:
#     #     ckpt = training_args.resume_from_checkpoint
#     # elif os.path.isdir(training_args.output_dir):
#     #     ckpt = get_last_checkpoint(training_args.output_dir)
#     #     if ckpt:
#     #         logger.info(f"\nCheckpoint detected, resuming training at {ckpt=}.")
#     #     else:
#     #         logger.info("\nNo checkpoint detected, starting training from scratch.")

#     # train_result = trainer.train(resume_from_checkpoint=ckpt)
#     # train_metrics = train_result.metrics
#     # trainer.log_metrics("train", train_metrics)
#     # trainer.save_metrics("train", train_metrics)
#     # trainer.save_state()
#     # trainer.push_to_hub(commit_message=f"Add checkpoint {training_args.max_steps} post-trained on {pt_args.model_post_train_dataset_name}")

#     # del trainer
#     # torch.cuda.empty_cache()


# if __name__ == "__main__":
#     main()

import datasets
from datasets import Dataset, load_dataset
from datetime import datetime
import json
import logging
import os
import sys
import torch
import transformers
from transformers import set_seed, AutoTokenizer

from trl import ModelConfig, TrlParser
from tina.post_train_hf.grpo_config import GRPOConfig

from OffPolicyGRPO.config import ModelPTConfig
from OffPolicyGRPO.workers import DistributedOffPolicyGRPO
from tina.post_train_hf.preprocess import make_conv_for_grpo
from OffPolicyGRPO.rewards import accuracy_reward
from tina.utils.chat_template import REASON_CHAT_TEMPLATE
from tina.utils.constant import RL_POST_TRAIN_CONFIG_MAP
from tina.utils.prompt import OPEN_R1_SYSTEM_PROMPT, OPEN_RS_SYSTEM_PROMPT


def main():
    parser = TrlParser((ModelPTConfig, GRPOConfig, ModelConfig))
    pt_args, training_args, model_args = parser.parse_args_and_config()
    set_seed(training_args.seed)

    # Use LORA adapters
    model_args.use_peft = True
    model_args.lora_r = 32
    model_args.lora_alpha = 128
    model_args.lora_dropout = 0.05
    model_args.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                       "down_proj", "up_proj", "gate_proj"]

    os.environ["WANDB_PROJECT"] = "offPolicy_GRPO_Training"

    ################
    # Set up logging
    ################
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Post training parameters {pt_args}")
    logger.info(f"Training parameters {training_args}")

    #####################
    # Set up output paths
    #####################
    current_time = datetime.now()
    formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    model_name_or_path = model_args.model_name_or_path
    ckpt_dir = os.environ.get("CKPT_DIR", "./checkpoints")
    ckpt_prefix = f"{ckpt_dir}/models/{model_name_or_path}"
    
    if model_args.use_peft:
        ckpt_postfix = f"offpolicy_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
    else:
        ckpt_postfix = f"offpolicyfull_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"

    # model_args.model_name_or_path = f"{ckpt_prefix}/base"
    model_args.model_name_or_path = "/jet/home/jjain2/rl-reasoning/Off_Policy_GRPO/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
    training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}"
    training_args.run_name = f"{model_name_or_path}_{ckpt_postfix}_{formatted_datetime}"
    training_args.hub_model_id = f"{training_args.hub_model_id}/{model_name_or_path}"

    #######################################################################
    # Load and preprocess dataset
    #######################################################################
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if "Qwen" in model_args.model_name_or_path:
        tokenizer.pad_token = "<|fim_pad|>"
    tokenizer.chat_template = REASON_CHAT_TEMPLATE

    model_post_train_dataset_name = RL_POST_TRAIN_CONFIG_MAP[pt_args.model_post_train_dataset_name]
    if pt_args.model_post_train_dataset_config is not None:
        train_dataset = load_dataset(model_post_train_dataset_name, split="train", 
                                     name=pt_args.model_post_train_dataset_config)
    else:
        train_dataset = load_dataset(model_post_train_dataset_name, split="train")
    
    # Rename columns
    if 'solution' not in train_dataset.column_names and 'answer' in train_dataset.column_names:
        train_dataset = train_dataset.rename_column('answer', 'solution')
        def wrap_in_math(example):
            return {"solution": f"${example['solution']}$"}
        train_dataset = train_dataset.map(wrap_in_math)
    
    if 'problem' not in train_dataset.column_names and 'question' in train_dataset.column_names:
        train_dataset = train_dataset.rename_column('question', 'problem')
    if 'problem' not in train_dataset.column_names and 'prompt' in train_dataset.column_names:
        train_dataset = train_dataset.rename_column('prompt', 'problem')

    SYSTEM_PROMPT = OPEN_RS_SYSTEM_PROMPT if "open-rs" in model_post_train_dataset_name else OPEN_R1_SYSTEM_PROMPT
    train_dataset = train_dataset.map(
        make_conv_for_grpo,
        fn_kwargs={"system_prompt": SYSTEM_PROMPT})
    # CRITICAL FIX: The custom GRPOTrainer expects 'prompt', not 'messages'
    # Remove the conflicting 'messages' column (original from dataset)
    # Keep the 'prompt' column (created by make_conv_for_grpo with our system prompt)
    if 'messages' in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(['messages'])
        logger.info("Removed original 'messages' column to avoid conflict")

    # Clean up: Keep only what the custom GRPOTrainer needs: 'prompt' and 'solution'
    columns_to_keep = ['prompt']
    if 'solution' in train_dataset.column_names:
        columns_to_keep.append('solution')

    columns_to_remove = [col for col in train_dataset.column_names if col not in columns_to_keep]
    if columns_to_remove:
        logger.info(f"Removing columns: {columns_to_remove}")
        train_dataset = train_dataset.remove_columns(columns_to_remove)

    logger.info(f"Final dataset columns: {train_dataset.column_names}")
    logger.info(f"Sample prompt format: {train_dataset[0]['prompt']}")
    #############################
    # Set up reward functions
    #############################
    reward_func = accuracy_reward  # Binary reward: 0 or 1

    #######################################################################
    # Calculate training iterations
    #######################################################################
    prompts_per_step = 1  # 1 question per inner step
    inner_steps_per_epoch = len(train_dataset) // prompts_per_step
    total_inner_steps = int(inner_steps_per_epoch * training_args.num_train_epochs)
    num_outer_iterations = training_args.num_train_epochs
    num_inner_steps = inner_steps_per_epoch
    num_grpo_iterations = getattr(training_args, 'num_grpo_iterations', 1)  # Default to 1

    logger.info(f"\n{'='*80}")
    logger.info("TRAINING SCHEDULE")
    logger.info(f"{'='*80}")
    logger.info(f"Dataset size: {len(train_dataset)} samples")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Outer iterations (I): {num_outer_iterations}")
    logger.info(f"Inner steps (M): {num_inner_steps}")
    logger.info(f"Workers: {pt_args.num_worker_nodes}")
    logger.info(f"Save frequency: {training_args.save_steps} steps")
    logger.info(f"{'='*80}\n")
    
    #######################################################################
    # Initialize and train the distributed system
    #######################################################################
    logger.info("Initializing Distributed Off-Policy GRPO System...")
    
    distributed_system = DistributedOffPolicyGRPO(
        model_name_or_path=model_args.model_name_or_path,
        global_dataset=train_dataset,
        model_config=model_args,
        training_config=training_args,
        reward_funcs=[reward_func],
        num_workers=pt_args.num_worker_nodes,
        worker_finetune_epochs=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    logger.info("\n" + "="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    # distributed_system.train(
    #     num_outer_iterations=num_outer_iterations,
    #     num_inner_steps=num_inner_steps,
    #     num_grpo_iterations = num_grpo_iterations,
    #     num_workers = pt_args.num_worker_nodes,
    #     save_frequency=training_args.save_steps,
    #     output_dir=training_args.output_dir
    # )



    # ============================================================
    # SPEED TEST: Only test worker fine-tuning, skip main training
    # ============================================================
    logger.info("\n" + "="*80)
    logger.info("SPEED TEST MODE: Testing single worker training speed")
    logger.info("="*80 + "\n")

    # Only finetune workers (this is what we're testing)
    import time
    start_time = time.time()

    distributed_system.finetune_workers()

    end_time = time.time()
    total_time = end_time - start_time

    logger.info("\n" + "="*80)
    logger.info("SPEED TEST RESULTS")
    logger.info("="*80)
    logger.info(f"Total time for worker fine-tuning: {total_time/60:.2f} minutes")
    logger.info(f"Total time for worker fine-tuning: {total_time/3600:.2f} hours")
    logger.info("="*80 + "\n")

    # Skip the main distributed training for now
    logger.info("Skipping main distributed training for speed test")
    logger.info("Once speed is acceptable, re-enable the distributed_system.train() call")
        
        
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETED!")
    logger.info(f"Final model saved to: {training_args.output_dir}/final_model")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()




















# """
# Main training script for distributed off-policy GRPO
# Uses Tina's GRPOTrainer for both workers and parameter server
# Place this as grpo.py in your offPolicyGRPO folder
# """

# import logging
# import os
# import sys
# from datetime import datetime
# from datasets import load_dataset
# from transformers import set_seed
# import datasets
# import transformers

# from trl import ModelConfig, TrlParser
# from tina.post_train_hf.grpo_config import GRPOConfig
# from tina.post_train_hf.rewards import accuracy_reward
# from tina.utils.chat_template import REASON_CHAT_TEMPLATE
# from tina.utils.constant import RL_POST_TRAIN_CONFIG_MAP
# from tina.utils.prompt import OPEN_R1_SYSTEM_PROMPT, OPEN_RS_SYSTEM_PROMPT
# from tina.post_train_hf.preprocess import make_conv_for_grpo, make_conv_for_grpo_l1

# # Import your local modules
# from config import ModelPTConfig
# from workers import DistributedOffPolicyGRPO


# def setup_logging(training_args):
#     """Set up logging configuration"""
#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)]
#     )
    
#     log_level = training_args.get_process_log_level()
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.enable_default_handler()
#     transformers.utils.logging.enable_explicit_format()
    
#     return logger


# def prepare_dataset(pt_args, system_prompt):
#     """Load and prepare dataset following Tina's pattern"""
#     logger = logging.getLogger(__name__)
    
#     model_post_train_dataset_name = RL_POST_TRAIN_CONFIG_MAP[pt_args.model_post_train_dataset_name]
    
#     # Load dataset
#     if pt_args.model_post_train_dataset_config is not None:
#         train_dataset = load_dataset(
#             model_post_train_dataset_name,
#             split="train",
#             name=pt_args.model_post_train_dataset_config
#         )
#     else:
#         train_dataset = load_dataset(model_post_train_dataset_name, split="train")
    
#     # Rename columns to match GRPO format (prompt, solution)
#     if 'solution' not in train_dataset.column_names and 'answer' in train_dataset.column_names:
#         train_dataset = train_dataset.rename_column('answer', 'solution')
        
#         def wrap_in_math(example):
#             return {"solution": f"${example['solution']}$"}
        
#         train_dataset = train_dataset.map(wrap_in_math)
    
#     if 'problem' not in train_dataset.column_names and 'question' in train_dataset.column_names:
#         train_dataset = train_dataset.rename_column('question', 'problem')
    
#     if 'problem' not in train_dataset.column_names and 'prompt' in train_dataset.column_names:
#         train_dataset = train_dataset.rename_column('prompt', 'problem')
    
#     if "messages" in train_dataset.column_names:
#         train_dataset = train_dataset.remove_columns("messages")
    
#     # Handle special datasets
#     if "deepscaler" in pt_args.model_post_train_dataset_name:
#         train_dataset = train_dataset.rename_column('solution', 'solution_archive')
#         train_dataset = train_dataset.rename_column('answer', 'solution')
        
#         def wrap_in_math(example):
#             return {"solution": f"${example['solution']}$"}
        
#         train_dataset = train_dataset.map(wrap_in_math)
    
#     elif "2thought" in pt_args.model_post_train_dataset_name:
#         import json
#         train_dataset = train_dataset.rename_column('messages', 'problem')
#         train_dataset = train_dataset.rename_column('verification_info', 'solution')
        
#         def extract_problem(example):
#             problem = example['problem'][0]["content"]
#             return {"problem": problem}
        
#         def extract_solution(example):
#             solution = json.loads(example['solution'])
#             solution = solution["answer"]["value"]
#             return {"solution": f"${solution}$"}
        
#         train_dataset = train_dataset.map(extract_problem)
#         train_dataset = train_dataset.map(extract_solution)
    
#     # Apply conversation formatting
#     if "l1" in pt_args.model_post_train_dataset_name:
#         min_length = 100
#         max_length = 4000
#         train_dataset = train_dataset.map(
#             make_conv_for_grpo_l1,
#             fn_kwargs={"system_prompt": system_prompt, "min_length": min_length, "max_length": max_length}
#         )
#     else:
#         train_dataset = train_dataset.map(
#             make_conv_for_grpo,
#             fn_kwargs={"system_prompt": system_prompt}
#         )
    
#     logger.info(f"Dataset loaded: {len(train_dataset)} examples")
#     logger.info(f"Dataset columns: {train_dataset.column_names}")
    
#     return train_dataset


# def main():
#     """Main training function"""
    
#     # Parse arguments
#     parser = TrlParser((ModelPTConfig, GRPOConfig, ModelConfig))
#     pt_args, training_args, model_args = parser.parse_args_and_config()
    
#     # Setup logging
#     logger = setup_logging(training_args)
    
#     set_seed(training_args.seed)
    
#     # Configure LoRA (override ModelConfig defaults)
#     model_args.use_peft = True
#     model_args.lora_r = 32
#     model_args.lora_alpha = 128
#     model_args.lora_dropout = 0.05
#     model_args.lora_target_modules = [
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "down_proj", "up_proj", "gate_proj"
#     ]
    
#     # Set up WandB
#     os.environ["WANDB_PROJECT"] = "OffPolicy_GRPO_Training"
    
#     # Log on each process
#     logger.warning(
#         f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
#         f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
#         f"16-bits training: {training_args.fp16}"
#     )
#     logger.info(f"Model parameters: {model_args}")
#     logger.info(f"Post training parameters: {pt_args}")
#     logger.info(f"Training parameters: {training_args}")
    
#     # Set up output paths
#     current_time = datetime.now()
#     formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    
#     model_name_or_path = model_args.model_name_or_path
#     ckpt_dir = os.environ.get("CKPT_DIR", "./checkpoints")
#     ckpt_prefix = f"{ckpt_dir}/models/{model_name_or_path}"
    
#     if model_args.use_peft:
#         ckpt_postfix = f"offpolicy_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
#     else:
#         ckpt_postfix = f"offpolicy_full_{pt_args.model_post_train_type}_{pt_args.model_post_train_dataset_name}"
    
#     model_args.model_name_or_path = f"{ckpt_prefix}/base"
#     training_args.output_dir = f"{ckpt_prefix}/{ckpt_postfix}"
#     training_args.run_name = f"{model_name_or_path}_{ckpt_postfix}_{formatted_datetime}"
#     training_args.hub_model_id = f"{training_args.hub_model_id}/{model_name_or_path}"
    
#     # Determine system prompt
#     model_post_train_dataset_name = RL_POST_TRAIN_CONFIG_MAP[pt_args.model_post_train_dataset_name]
#     SYSTEM_PROMPT = (OPEN_RS_SYSTEM_PROMPT 
#                     if "open-rs" in model_post_train_dataset_name 
#                     else OPEN_R1_SYSTEM_PROMPT)
    
#     # Load and prepare dataset
#     logger.info("\nLoading and preparing dataset...")
#     train_dataset = prepare_dataset(pt_args, SYSTEM_PROMPT)
    
#     # Set up reward functions
#     RL_POST_TRAIN_REWARD_MAP = {
#         "accuracy": accuracy_reward
#     }
    
#     rl_reward_funcs = [
#         RL_POST_TRAIN_REWARD_MAP[func] 
#         for func in pt_args.rl_post_train_reward_funcs
#     ]
#     training_args.reward_weights = pt_args.rl_post_train_reward_weights
    
#     # Configure GRPO training arguments for distributed setting
#     # IMPORTANT: num_generations should match num_worker_nodes
#     training_args.num_generations = pt_args.num_worker_nodes
    
#     if not hasattr(training_args, 'max_prompt_length') or training_args.max_prompt_length is None:
#         training_args.max_prompt_length = 512
#     if not hasattr(training_args, 'max_completion_length') or training_args.max_completion_length is None:
#         training_args.max_completion_length = 1024
#     if not hasattr(training_args, 'temperature') or training_args.temperature is None:
#         training_args.temperature = 0.9
#     if not hasattr(training_args, 'beta') or training_args.beta is None:
#         training_args.beta = 0.04
    
#     logger.info(f"\n{'='*80}")
#     logger.info("Initializing Distributed Off-Policy GRPO System")
#     logger.info(f"Number of workers: {pt_args.num_worker_nodes}")
#     logger.info(f"Global dataset size: {len(train_dataset)}")
#     logger.info(f"Model: {model_args.model_name_or_path}")
#     logger.info(f"Using LoRA: {model_args.use_peft} (r={model_args.lora_r}, alpha={model_args.lora_alpha})")
#     logger.info(f"Output directory: {training_args.output_dir}")
#     logger.info(f"{'='*80}\n")
    
#     # Initialize distributed off-policy GRPO system
#     distributed_system = DistributedOffPolicyGRPO(
#         model_name_or_path=model_args.model_name_or_path,
#         global_dataset=train_dataset,
#         model_config=model_args,
#         training_config=training_args,
#         reward_funcs=rl_reward_funcs,
#         num_workers=pt_args.num_worker_nodes,
#         device="cuda" if training_args.n_gpu > 0 else "cpu"
#     )
    
#     # Calculate number of training iterations
#     if training_args.max_steps > 0:
#         num_iterations = training_args.max_steps
#     else:
#         # Estimate from epochs and dataset size
#         samples_per_epoch = len(train_dataset)
#         samples_per_step = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
#         steps_per_epoch = samples_per_epoch // samples_per_step
#         num_iterations = int(steps_per_epoch * training_args.num_train_epochs)
    
#     logger.info(f"Training for {num_iterations} iterations")
#     logger.info(f"Epochs: {training_args.num_train_epochs}")
#     logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
#     logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    
#     # Train the distributed system
#     logger.info("\n" + "="*80)
#     logger.info("Starting Training")
#     logger.info("="*80 + "\n")
    
#     distributed_system.train(
#         num_iterations=num_iterations,
#         prompts_per_iteration=training_args.per_device_train_batch_size,
#         sync_frequency=10,  # Sync workers every 10 iterations
#         save_frequency=training_args.save_steps,
#         output_dir=training_args.output_dir
#     )
    
#     logger.info("\n" + "="*80)
#     logger.info("Training Completed Successfully!")
#     logger.info(f"Final model saved to: {training_args.output_dir}/final_model")
#     logger.info("="*80 + "\n")
    
#     # Optional: Push to hub if configured
#     if training_args.push_to_hub:
#         logger.info("Pushing model to hub...")
#         distributed_system.parameter_server.trainer.push_to_hub(
#             commit_message=f"Off-policy GRPO training completed: {num_iterations} iterations"
#         )


# if __name__ == "__main__":
#     main()