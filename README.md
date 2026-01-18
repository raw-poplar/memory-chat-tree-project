# Memory Chat Tree (Standalone)

把 `infinite_context_pipeline/memory_chat_tree` 拆出来的可运行小项目：带持久化记忆树 + 向量 Top‑K 检索 + 槽位预算拼接（无需额外训练）。

## 项目简介

这个项目的目标是用“外部记忆”在不训练模型的前提下实现更长的有效上下文：

- `chat_tree.py`：交互式聊天 + 记忆落盘 + 检索注入 + 超预算自动压缩（生成或截断摘要）
- `memory_tree.py`：层级记忆树（叶子=原样对话；摘要=高层记忆，可用 `--summary_mode state` 提取用户事实/偏好/任务状态）
- `retrieval/`：向量检索（可选 `hnswlib` 持久化索引；否则内存 flat）
- `local_memory/`：本地记忆与向量缓存目录（可通过 `--memory_dir` 指定）

## 主要思路

把“长上下文”拆成两部分：
- **短期上下文**：最近几轮对话作为 `[HIST]`（或 chat messages）直接喂给模型
- **长期上下文**：把更早的内容落盘成“记忆树”，每轮按需检索出少量相关片段注入

一轮对话的大致流程：

1. **写入叶子节点**：上一轮 `(User, Assistant)` 作为 `leaf` 节点写入 `memory_tree.json`（原样对话，便于追溯）。
2. **向量化与建索引**：把可检索的记忆节点文本做 embedding，缓存到 `memory_vectors.json`；索引后端优先用 `hnswlib`（持久化到 `memory_hnsw.bin`），否则用内存 flat。
3. **检索与注入**：用“当前输入 + 最近 N 轮历史”构造检索 query，按 `--mem_select time|vector|hybrid` 选出 Top‑K 记忆；检索到摘要节点时可用 `--expand_children` 再补充部分子节点细节。
4. **提示词预算拼接**：把系统提示/记忆/历史/当前输入按预算拼成最终输入（`--pack_mode slot` 会做槽位预算；或使用模型自带 chat template）。
5. **超预算自动压缩**：如果仍超过 `prompt_budget`，就把最老的一批兄弟节点合并成 `summary` 节点（`--summary_mode truncate|llm|state`），在保留细节（叶子）同时保留高层信息（摘要）。

## 实现概览
简而言之，所有的叶子节点是具体的记忆，父节点是所有子节点的摘要压缩。
- **记忆树结构**：`memory_tree.json` 记录 root/leaf/summary 三类节点；leaf 保存原样对话，summary 聚合早期节点，并通过 `children/parent_id` 保留可回溯关系。
- **向量缓存与索引**：`memory_vectors.json` 缓存记忆向量；索引优先用 HNSW（持久化到 `memory_hnsw.bin`），维度或模型指纹不一致时自动重建。
- **检索与补细节**：用当前输入 + 最近历史构造 query，Top-K 检索后可选轻量重排；命中 summary 节点时可 `--expand_children` 展开部分子节点。
- **提示词拼接**：按槽位预算拼 `[SYS]/[MEM]/[HIST]/[USER]`，或直接走模型的 chat template。
- **超预算压缩**：当提示词超过预算，按 `summary_mode` 合并最老节点并生成摘要节点，兼顾可追溯与上下文压缩。

## 依赖

- Python 3.10+
- `torch`, `transformers`, `numpy`
- 可选：`hnswlib`（装了会自动用 HNSW 索引；否则回退为内存 flat 检索）

## 安装

在本目录执行：

```bash
pip install -r requirements.txt
```

## 运行

在本目录执行：

```bash
python chat_tree.py
```

默认会从项目下 `./model` 加载模型（可把本地 HuggingFace `save_pretrained` 输出放进去，包含 `config.json`、tokenizer 与权重文件等）。
也可以显式指定：

```bash
python chat_tree.py --model_dir <HF模型目录>
```

### 快速上手（推荐命令）

- 默认记忆 + 向量检索（推荐）：

```bash
python chat_tree.py --retrieval_min_score 0.2
```

- 高层摘要使用“用户事实/偏好/任务状态”（更像长期记忆）：

```bash
python chat_tree.py --summary_mode state --retrieval_min_score 0.2
```

- 想快速看到“自动压缩生成摘要节点”的效果（缩小上下文窗口触发压缩）：

```bash
python chat_tree.py --context_window 2048 --summary_mode state --retrieval_min_score 0.2 --debug
```

### 评估脚本（超出上下文时的召回/有效性）

项目内置一个简单的自动化评估脚本：会先写入一批“随机偏好标签”，再用大量无关文本把对话撑到超过设定的 `--context_window`，最后统计向量检索的召回率，并可选测试“无记忆 vs 注入记忆”回答是否变正确。

示例（模拟 10k tokens 上下文，用于“管线自检”）：

```bash
python eval_memory_recall.py --context_window 10240 --topk 5 --retrieval_query_turns 0 --min_score 0.0 --eval_generation
```

更接近“语义记忆”的自检（不使用 ID 式 key，且关闭重排，结果通常会更真实也更不稳定）：

```bash
python eval_memory_recall.py --task semantic --context_window 10240 --topk 5 --retrieval_query_turns 0 --min_score 0.0 --retrieval_rerank off
```

提示：
- Windows 路径里如果有空格，请给 `--model_dir` 加双引号，例如：`--model_dir "F:\path with space\model"`。
- 该评估里最近几轮都是“无关 filler”，所以建议 `--retrieval_query_turns 0`（只用当前问题做检索 query），否则 query 会被 filler 文本“带偏”，召回会明显下降。
- 该脚本默认启用 `--retrieval_rerank keyword`：它会从 query 里抽取字母数字“标识符”（例如 `K0_xxx`），对包含同样标识符的记忆做加权重排；在本脚本这种“键值查询”的任务上容易出现 `top1≈1.0 / acc≈1.0`，这更像“功能验证：超窗后记忆是否确实注入并生效”，不代表真实语义检索的上限。
- 如果你想看“纯向量检索”的效果（更接近语义检索难度），可以关掉重排：`python eval_memory_recall.py --context_window 10240 --topk 5 --retrieval_query_turns 0 --min_score 0.0 --retrieval_rerank off --eval_generation`。
- 对于 Qwen/Qwen3 这类 Instruct/Chat 模型，建议使用默认的 `--prompt_format auto`（会自动走 chat template），避免把 `[SYS]/[MEM]/[USER]` 这类“slot-tag 协议”当作普通文本导致模型复读标签。

### 纯对话测试（不含记忆）

如果你只想先确认模型能“正常对话/续写”，建议用这个脚本（不注入 `[MEM]`，也不做记忆落盘）：

```bash
python chat_plain.py
```

也可以显式指定模型目录：

```bash
python chat_plain.py --model_dir <HF模型目录>
```

如果你在仓库根目录运行，也可以：

```bash
python experiments/memory_chat_tree_project/chat_plain.py
```

说明：
- 若模型自带 chat template（常见于 Instruct/Chat 模型），`chat_plain.py` 会自动使用；否则回退为简单的 `User:`/`Assistant:` 拼接。

常用参数：

- `--mem_select vector|hybrid|time`：记忆选择策略（默认 `vector`）
- `--pack_mode slot|legacy`：上下文拼接（默认 `slot`，按槽位预算裁剪）
- `--expand_children N`：当检索到的是“摘要节点”时，额外注入其最多 N 个子节点（细节）
- `--summary_mode truncate|llm|state`：当提示词超预算时，如何把旧叶子合并成摘要节点；`state` 会提取“用户事实/偏好/任务状态”作为高层记忆
- `--retrieval_rerank off|keyword|token_overlap`：向量 Top‑K 之后再做一次轻量重排（默认 `keyword`，对包含 `Qwen3/K0_xxx` 这类字母数字“标识符”的问题，能显著提升 top1）
- `--prompt_format auto|chat|slot`：提示词格式；`auto` 会在可用时使用 chat template（推荐用于 Qwen/Qwen3 等），`slot` 强制使用 `[SYS]/[MEM]/[USER]`。
- `--chat_template auto|on|off`：当 `prompt_format=chat/auto` 时生效；控制是否使用 tokenizer 自带的 chat template。
- `--enable_thinking`：当 chat template 支持（如 Qwen3）时，允许输出 `<think>...</think>` 推理过程；默认关闭（更像“正常对话”）。
- `--slot_tag_lang en|zh`：当 `prompt_format=slot` 时生效；把 `[SYS]/[MEM]/[HIST]/[USER]` 切换为 `[系统]/[记忆]/[历史]/[用户]`（更像自然语言，也更易读）。
- `--mem_structured_hint`：给模型补一句提示：注入的记忆块是外部检索的结构化输入（可能有噪声），按需参考即可。
- `--memory_dir <path>`：记忆落盘目录（默认该项目下 `local_memory/`）
- `--retrieval_min_score 0.2`：提高检索相似度门槛，减少噪声

交互命令：

- `/mem_stats`：查看记忆树与索引状态
- `/mem_clear`：清空记忆树与检索索引（慎用）
- `/mem_tail [n]`：查看最近 n 条记忆节点（默认 10）
- `/mem_show <id>`：查看某条记忆节点的完整内容
- `/mem_search <kw>`：按关键词搜索记忆节点（最多显示 50 条）
- `/show_mem`：查看上一轮检索到并注入的记忆（含相似度分数）
- `/show_prompt`：查看上一轮实际喂给模型的输入（slot prompt 或渲染后的 chat template）
- `/debug [on|off]`：运行时开关调试输出（也可用 `--debug` 默认开启）

## 记忆树与“摘要父节点”是怎么落地的？

记忆树会同时保留两种粒度的信息：

- **叶子节点（leaf）**：每一轮对话 `(User, Assistant)` 原样写入，便于回溯与审计。
- **摘要节点（summary）**：当提示词超过预算时，把最早的一批兄弟节点合并成一个摘要节点；摘要节点会成为这些节点的“父节点”，并替换掉 root 下对应位置，从而形成层级结构。

你可以在 `local_memory/memory_tree.json` 里看到这些字段：
- `role: "leaf" | "summary"`
- `children: [...]`（摘要节点会记录被合并的子节点 id）
- 子节点的 `parent_id` 会被更新为该摘要节点 id

配合参数：
- `--summary_mode truncate|llm|state`：控制摘要生成方式（`state` 更偏“长期用户状态”）。
- `--expand_children N`：当检索命中摘要节点时，可额外注入最多 N 个子节点细节。

## 数据与隐私

- `local_memory/` 会落盘对话与向量缓存（例如 `memory_tree.json`、`memory_vectors.json`、`memory_hnsw.bin`），可能包含敏感信息；不建议提交到 GitHub。
- 清空本地记忆：运行时输入 `/mem_clear`（会删除本地落盘文件）。
- 如果你曾经把 `local_memory/` 误加入 git 跟踪，需要执行 `git rm --cached -r local_memory` 再提交一次（保留本地文件不受影响）。
- `model/` 用于放模型权重，默认已在 `.gitignore` 中忽略（只保留 `model/README.md`）。

## 常见问题

### 1) 为什么开启 `--enable_thinking` 后，输出一直停留在 `<think>...`，看不到 `</think>` 和最终回答？

`<think>...</think>` 只是模型生成的“思考文本”，不是程序逻辑。开启 `--enable_thinking` 后，模型可能先输出很长的思考内容；如果 `--max_new_tokens` 太小，就会在思考还没写完时被截断，于是看起来像“永远到不了 `</think>` 和最终回答”。

解决办法：
- 增大生成长度：例如 `--max_new_tokens 512`
- 或者不需要思考就不要开启：去掉 `--enable_thinking`（对 Qwen/Qwen3 这类带 chat template 的模型，通常会更像“直接回答”）

## License

Apache License 2.0. See `LICENSE`.
