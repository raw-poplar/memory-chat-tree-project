# 模型放置说明

本目录用于存放本地 HuggingFace `save_pretrained` 输出的模型文件（例如 `config.json`、tokenizer 文件、权重文件等）。

默认行为：
- 不传 `--model_dir` 时，脚本会默认从 `./model` 加载模型。
- 如果 `./model` 本身没有 `config.json`，但它**只包含一个**子目录且该子目录包含 `config.json`，脚本会自动使用这个唯一子目录。

示例：
- 直接把模型文件放在 `model/`：`python chat_tree.py`
- 多模型时放在 `model/qwen3-0.6b/`：`python chat_tree.py --model_dir model/qwen3-0.6b`

注意：
- 模型文件通常很大，本仓库默认会忽略 `model/` 下除本说明文件以外的所有内容（见 `.gitignore`）。
