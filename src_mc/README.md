## CloudCom 2025 情感咨询对话挑战实现

本项目实现了一个基于检索增强生成（RAG）的情绪咨询对话流水线：
读取 `data/Conversations_Long.jsonl` 中的对话历史，检索与来访者末轮发言最相关的上下文，调用心理咨询提示词驱动的 LLM 生成回复，并将结果写入 `outputs/Emotion_Conversatin_Result.jsonl`。

### 工作流程概览
1. **数据读取**：`src/utils.py` 的 `read_jsonl` 按行加载对话条目，每条包含 `id` 与 `conversation_history`。
2. **对话解析**：`last_client_turn` 从多轮文本中抽取来访者最新发言，用于检索与生成。
3. **语义检索**：`src/retriever.py` 中的 `Retriever` 会在首次运行时将历史切分为固定 Token 长度的片段，使用 `BAAI/bge-m3` 向量模型编码，并建立/加载 FAISS 索引；随后以来访者末轮发言作为查询，返回最相关的若干上下文。
4. **LLM 生成**：`src/llm.py` 的 `generate_reply` 依据 `src/prompts.py` 中的系统与用户提示词模板，调用配置文件指定的 OpenAI/vLLM 兼容接口生成心理咨询师回复。
5. **自我督导**（可选）：若 `config.yaml` 的 `self_judge.enable` 为 `true`，`refine_with_judge` 会使用督导提示对回复进行评分，若任一维度低于阈值则按建议改写一次。
6. **结果输出**：最终回复以 `{"id": <原样>, "predicted_response": <回复>}` 的形式写入目标 JSONL 文件，确保语言与来访者一致且长度符合要求。

### 配置与运行
默认配置位于 `src/config.yaml`，包含检索分块、嵌入模型、LLM 接口以及自评审阈值等参数，可通过命令行覆盖：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /data/zhangjingwei/LL-Doctor-qwen3-8b-Model \
    --host 0.0.0.0 --port 8000 \
    --gpu-memory-utilization 0.65 \
    --max-model-len 32768
```

> --gpu-memory-utilization根据memory占用情况修改

```bash
export HF_ENDPOINT=https://hf-mirror.com
python -m src_mc.runner_mc --config src_mc/config.yaml \
    --data data/Conversations_Long.jsonl \
    --output outputs/Emotion_Conversatin_Result.jsonl
```

首次运行会在 `indexes/` 目录生成 FAISS 索引与文本缓存；若需重新构建，可删除对应文件后再次执行。确保 `llm.endpoint` 指向可用的推理服务（默认 `http://127.0.0.1:8000/v1/chat/completions`），并满足 120–220 中文字或 80–150 英文词的回复长度约束。

### 依赖安装（uv）

```bash
uv sync
```

安装完成后即可使用 `uv run` 执行脚本，例如：

```bash
uv run python -m src.runner_mc
```

