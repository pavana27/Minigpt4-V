"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os

import torch
import torch.distributed as dist
from minigpt4.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from minigpt4.common.logger import MetricLogger, SmoothedValue
from minigpt4.common.registry import registry
from minigpt4.datasets.data_utils import prepare_sample
from minigpt4.conversation.conversation import CONV_VISION

import wandb
import openai
import ast
from dotenv import load_dotenv,find_dotenv
import os

openai.api_key_path = '/home/pavana/MiniGPT4-video/.env'

from utils import init_logger
import os
program = os.path.basename(__file__)
if os.path.exists(f"../../logs/{os.path.splitext(program)[0]}.log"):
    os.remove(f"../../logs/{os.path.splitext(program)[0]}.log")
logger = init_logger(program)

class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.cfg = ""
        
        

    @classmethod
    def setup_task(cls, **kwargs):
        
        return cls()
    

    def build_model(self, cfg):
        self.cfg = cfg
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            dataset['train'].name = name
            if 'sample_ratio' in dataset_config:
                dataset['train'].sample_ratio = dataset_config.sample_ratio

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return loss

    def valid_step(self, model, samples):
        if 'rppg' in samples and not (samples['rppg'].shape[0] == 1):
            answers = model.generate(
                images=samples['image'],
                texts=samples['instruction_input'],
                lengths=samples['length'],
                rppg=samples['rppg'],
            )
            return answers
        
        answers = model.generate(
            images=samples['image'],
            texts=samples['instruction_input'],
            lengths=samples['length'],
        )
        return answers

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))
        
    def chatgpt_eval(self,question, answer,pred):
        try:
            # Compute the correctness score
            completion = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                # model='gpt-4',
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                            "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            # response_message = completion["choices"][0]["message"]["content"]
            response_message = completion.choices[0].message.content
            response_dict = ast.literal_eval(response_message)
            return response_dict
        except Exception as e:
            logger.info(f"Error : {e}")
            return None
        
    def after_evaluation(self, val_result,epoch,**kwargs):
        scores=[]
        yes_count=0
        no_count=0
        # Load the .env file
        for res in val_result:
            gpt_response=self.chatgpt_eval(client,res['Q'],res['A'],res['pred'])
            if gpt_response is None:
                continue
            try:
                scores.append(float(gpt_response['score']))
                if 'yes' in gpt_response['pred'].lower():
                    yes_count+=1
                elif 'no' in gpt_response['pred'].lower():
                    no_count+=1
            except Exception as e:
                logger.info(f"GPT EVALUATION ERROR: {e}")
        avg_score=sum(scores)/len(scores)
        accuracy=(yes_count/(yes_count+no_count))*100
        logger.info(f"Epoch {epoch} chatgpt score: {avg_score} accuracy: {accuracy}")
        val_accuracy={"agg_metrics":accuracy,"best_epoch":epoch}
        # val_accuracy={"agg_metrics":50.2,"best_epoch":epoch}
        return val_accuracy

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        logger.info_freq = 10
        results = []
        logger.info(f"EVAL # SAMPLES - {len(data_loader)}")
        for samples in metric_logger.log_every(data_loader, logger.info_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            
            conv = CONV_VISION.copy()
            conv.system = ""
            conv.append_message(conv.roles[0],samples['instruction_input'][0])
            conv.append_message(conv.roles[1],None)
            samples['instruction_input'] = [conv.get_prompt()]
            samples['rppg'] = samples['rppg'].float() if 'rppg' in samples and not (samples['rppg'] == 0).all() else 0
            
            eval_output = self.valid_step(model=model, samples=samples)
            for i,pred in enumerate(eval_output):
                res={}
                res['video_name'] = samples['image_id'][i]
                res['Q'] = samples['instruction_input'][i].split('\n')[-1]
                res['A'] = samples['answer'][i]
                res['pred'] = pred
                results.append(res)
            # break
        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=lr_scheduler.iters_per_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logger.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)
            
        # loss = 999
        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = self.train_step(model=model, samples=samples)

            # after_train_step()
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if hasattr(model, 'visual_encoder'):
                    visual_encoder_params = model.visual_encoder.parameters()
                else:
                    visual_encoder_params = model.module.visual_encoder.parameters()

                if use_amp:
                    scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(visual_encoder_params,
                    #                                max_norm=0.3)  # apply gradient clipping on vit
                    scaler.step(optimizer)
                    scaler.update()                     
                else:
                    # torch.nn.utils.clip_grad_norm_(visual_encoder_params,
                    #                                max_norm=0.3)  # apply gradient clipping on vit
                    optimizer.step()
                optimizer.zero_grad()
                if self.cfg.run_cfg.rank==0:
                    wandb.log({"epoch": inner_epoch, "loss": loss})
                    
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            # break
        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: " + str(metric_logger.global_avg()))
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logger.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            logger.info("result file saved to %s" % final_result_file)

        return final_result_file
