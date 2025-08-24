# -*- coding: utf-8 -*-
"""Language model evaluator using lm_eval."""

import lm_eval
import lm_eval.models
from transformers import PreTrainedModel, PreTrainedTokenizer

from .base import LlmEvaluatorBase

__all__ = ["LmevalEvaluator"]


class LmevalEvaluator(LlmEvaluatorBase):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, batch_size: int = 1):
        super().__init__(model=model, tokenizer=tokenizer)
        self.lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size) # 看起来是按huggingface的格式加载的伪量化模型

    def filter_tasks(self, tasks: list[str]) -> list[str]:
        """Filter the tasks to only include supported tasks."""
        return tasks
    
    # 参考 https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py
    # 8.20 添加其他参数
    def evaluate(
        self,
        tasks: list[str],
        max_length: int | None = None,
        num_shot: int | None = None,
        fewshot_as_multiturn: bool = False,
        apply_chat_template: bool = False,
        max_new_tokens: int | None = None,
        confirm_run_unsafe_code: bool = False,
        parallelize: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Evaluate the model on the given tasks.

        Args:
            tasks (`list[str]`): List of tasks to evaluate on.
            max_length (`int`, optional, defaults to `None`): Maximum length for the model.

        Returns:
            dict[str, dict[str, dict[str, float]]]: Evaluation results `{"results": {"task": {"metric": score}}}`.
        """
        self.lm._max_length = max_length  # 这里传入最大长度
        result = lm_eval.evaluator.simple_evaluate(
            model=self.lm,
            tasks=tasks,
            verbosity="ERROR",
            num_fewshot=num_shot,
            fewshot_as_multiturn=fewshot_as_multiturn,
            apply_chat_template=apply_chat_template,
            confirm_run_unsafe_code = confirm_run_unsafe_code,
            model_args = {"parallelize":parallelize},
            gen_kwargs={"max_new_tokens": max_new_tokens},
            **kwargs,
        )
        self.lm._max_length = None
        result.pop("samples", None)
        result.pop("config", None)
        print(result)  # 测试lm_eval的评测格式输出的区别
        return result
