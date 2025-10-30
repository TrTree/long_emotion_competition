# Long Emotion MC (ES) 流水线

本仓库实现了 CloudCom 2025 长文本情感咨询挑战的检索增强生成（RAG）流水线。系统会读取多轮对话、检索最相关的上下文，并生成具备治疗性的回复（可选自我评估改写）。

## 仓库结构
- `src_mc/`：情感咨询流水线的核心实现（检索器、提示词模板、LLM 编排与 CLI 入口）。原始的参赛说明文档现已移动至 [`src_mc/README.md`](src_mc/README.md)。
- `data/`：输入的 JSONL 对话数据，默认文件为 `Conversations_Long.jsonl`。
- `outputs/`：运行流水线后生成的回复 JSONL 文件。
- `main.py`：用于实验或集成的便捷入口脚本。
- `pyproject.toml` / `uv.lock`：使用 [uv](https://github.com/astral-sh/uv) 管理的 Python 依赖定义。

## 快速开始
### 1. 安装依赖
项目使用 `uv` 管理环境与依赖，可通过以下命令安装：

```bash
uv sync
```

后续的运行命令均可视需要添加 `uv run` 前缀，以在受控环境中执行。

### 2. 准备推理端点
流水线假设存在一个兼容 OpenAI Chat Completions 的接口。若本地部署 [vLLM](https://github.com/vllm-project/vllm)，可参考如下命令：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /data/zhangjingwei/LL-Doctor-qwen3-8b-Model \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 32768
```

根据实际硬件调整显存相关参数；若服务地址不同，请同步更新 `src_mc/config.yaml` 中的 `llm.endpoint`。

### 3. 运行情感咨询流水线
使用默认配置处理示例数据：

```bash
python -m src_mc.runner_mc \
  --config src_mc/config.yaml \
  --data data/Conversations_Long.jsonl \
  --output outputs/Emotion_Conversatin_Result.jsonl
```

首次运行会在 `indexes/` 目录下生成 FAISS 索引及文本缓存，如需重建可删除后重新执行。

## 配置说明
`src_mc/config.yaml` 用于控制检索分块大小、是否启用 FAISS、嵌入模型与 LLM 生成参数。调用 `runner_mc` 时可通过命令行覆盖配置路径以便实验。

启用 `self_judge` 配置块后，流水线会执行一次基于督导提示词的自我评估，并在得分低于阈值时进行改写。

## 开发提示
- JSONL 读写、对话末轮抽取等工具函数位于 `src_mc/utils.py`。
- 检索相关逻辑（切分、嵌入、FAISS 索引）实现于 `src_mc/retriever.py`。
- 心理咨询与督导的提示词模板定义在 `src_mc/prompts.py`。
- 与推理服务交互的辅助函数位于 `src_mc/llm.py`。

如需更详尽的流程说明，请查阅 [`src_mc/README.md`](src_mc/README.md)。
