"""与大语言模型交互的辅助函数。

该模块负责：
1. 封装聊天补全接口，统一请求参数；
2. 生成咨询师回复；
3. 调用督导模型对回复进行打分与二次润色。"""

from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import requests

from .prompts import MC_SYSTEM, MC_USER_FMT
from .utils import strip_reasoning_prefix

LOGGER = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120


def _chat_completion(
    system_prompt: str,
    user_prompt: str,
    model: str,
    endpoint: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """直接调用聊天补全接口并返回文本结果。

    参数说明：
    * ``system_prompt`` / ``user_prompt``：系统与用户提示词；
    * ``model`` / ``endpoint``：模型名称与推理服务地址；
    * ``max_new_tokens``、``temperature``、``top_p``：生成控制参数；
    * ``timeout``：请求超时时长。

    返回值会在成功解析后去除首尾空白字符。"""
    # 构造符合 OpenAI ChatCompletion 兼容格式的请求体
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }

    response = requests.post(endpoint, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Invalid LLM response: {data}") from exc
    # 最终仅返回模型文本，不携带多余空白
    return content.strip()


def generate_reply(
    system_prompt: str,
    user_prompt: str,
    model: str,
    endpoint: str,
    max_new_tokens: int = 220,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> str:
    """Generate a counselor reply using the configured LLM endpoint.

    使用指定的大模型接口生成咨询师回复。"""

    # 生成前写入调试日志，方便排查接口调用问题

    LOGGER.debug("Generating reply via %s", endpoint)
    raw = _chat_completion(
        system_prompt,
        user_prompt,
        model,
        endpoint,
        max_new_tokens,
        temperature,
        top_p,
    )
    # 去除模型可能产生的 <think> 思维链包裹内容
    return strip_reasoning_prefix(raw)


def refine_with_judge(
    client_last: str,
    evidence: str,
    draft: str,
    judge_sys: str,
    judge_user_template: str,
    endpoint: str,
    model: str,
    min_pass: float,
    max_refine: int = 1,
    system_prompt: str | None = None,
    user_template: str | None = None,
    generation_kwargs: Optional[Dict] = None,
) -> str:
    """Evaluate and optionally refine a draft reply using a judge model.

    调用督导模型判断回复是否达标，必要时根据建议进行润色。"""

    # 默认生成参数在需要时补齐，避免调用方重复传参
    generation_kwargs = generation_kwargs or {}

    def call_judge(reply: str) -> Dict:
        """调用督导模型并将 JSON 文本安全解析为字典。"""

        judge_prompt = judge_user_template.format(
            client_last=client_last,
            evidence=evidence,
            reply=reply,
        )
        content = _chat_completion(
            judge_sys,
            judge_prompt,
            model,
            endpoint,
            generation_kwargs.get("max_new_tokens", 220),
            generation_kwargs.get("temperature", 0.7),
            generation_kwargs.get("top_p", 0.95),
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            LOGGER.warning("Judge response is not valid JSON: %s", content)
            return {"scores": {}, "advice": ""}

    scores_payload = call_judge(draft)
    scores = scores_payload.get("scores", {})
    advice = scores_payload.get("advice", "")
    # 只要有任一评分低于通过线，则认为需要润色
    need_refine = any(float(value) < float(min_pass) for value in scores.values()) if scores else False

    if not need_refine or max_refine <= 0:
        return draft

    system_prompt = system_prompt or MC_SYSTEM
    base_prompt = (user_template or MC_USER_FMT).format(client_last=client_last, evidence=evidence)
    current_reply = draft

    for attempt in range(max_refine):
        refine_instruction = (
            "【督导建议】\n"
            f"{advice or '请在保持真诚共情的同时补充督导指出的要素。'}\n"
            "请根据督导建议重写你的回复，并保留上一版中已经做得好的部分。\n"
            "【上一版回复】\n"
            f"{current_reply}"
        )
        refined_prompt = f"{base_prompt}\n\n{refine_instruction}"
        # 依据督导建议重新生成一版回复
        current_reply = generate_reply(
            system_prompt,
            refined_prompt,
            model,
            endpoint,
            generation_kwargs.get("max_new_tokens", 220),
            generation_kwargs.get("temperature", 0.7),
            generation_kwargs.get("top_p", 0.95),
        )

        if attempt + 1 >= max_refine:
            break

        # 针对新的回复再次进行打分，如满足要求则提前退出循环
        scores_payload = call_judge(current_reply)
        scores = scores_payload.get("scores", {})
        advice = scores_payload.get("advice", advice)
        need_refine = any(float(value) < float(min_pass) for value in scores.values()) if scores else False
        if not need_refine:
            break

    return current_reply

