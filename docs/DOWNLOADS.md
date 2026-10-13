# 外部数据、模型与本地素材

Git 保存代码和选定实验记录；原视频、权重、Conda 环境、视频预处理缓存及重复中间产物留在 Git 外。当前四数据集配置来自 `runs/research_pilot_20260923/config.json`。历史绝对路径和结果保持原样，迁移机器时生成新配置，不覆盖历史文件。

## 下载来源与期望目录

以下地址于 2026-10-13 核对。`assets.json` 保存对应下载 revision；这是当前可下载版本，原实验使用的完整资产 revision 未记录，不能声称字节级重建全部历史实验。

| 资产 | 下载入口 | 解压后的目录，相对资产根目录 |
|---|---|---|
| LVBench，当前必需 | [HF](https://huggingface.co/datasets/lmms-eval/LVBench)、[官方说明](https://github.com/zai-org/LVBench) | `datasets/LVBench/data/train-00000-of-00001.parquet`、`datasets/LVBench/video_chunks/*.mp4` |
| Video-MME，当前必需 | [HF](https://huggingface.co/datasets/lmms-eval/Video-MME)、[官方说明](https://github.com/MME-Benchmarks/Video-MME) | `datasets/Video-MME/videomme/test-00000-of-00001.parquet`、`datasets/Video-MME/data/*.mp4` |
| EgoSchema，当前使用公开 Subset | [HF](https://huggingface.co/datasets/lmms-eval/egoschema)、[项目](https://egoschema.github.io/) | `datasets/egoschema/Subset/test-00000-of-00001.parquet`、`datasets/egoschema/videos/*.mp4` |
| MLVU，当前使用 Dev 数据 | [HF：仓库名为 MVLU](https://huggingface.co/datasets/MLVU/MVLU)、[官方说明](https://github.com/JUNJIE99/MLVU) | `datasets/MLVU/MLVU/json/`、`datasets/MLVU/MLVU/video/` |
| LongVideoBench，关联但未启用 | [HF](https://huggingface.co/datasets/longvideobench/LongVideoBench)、[官方说明](https://github.com/longvideobench/LongVideoBench) | `datasets/LongVideoBench/lvb_val.json`、`subtitles/`、`videos/` |
| CG-Bench，关联但未启用 | [HF](https://huggingface.co/datasets/CG-Bench/CG-Bench)、[官方说明](https://github.com/CG-Bench/CG-Bench) | `datasets/CG-Bench/cgbench.json`、原视频/线索视频解压目录 |
| Planner / Observer / 视频 caption | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | `models/Qwen/Qwen3.5-9B/` |
| 文本检索 embedding | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | `models/Qwen/Qwen3-Embedding-0.6B/` |
| 多模态检索 embedding | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | `models/Qwen/Qwen3-VL-Embedding-2B/` |
| ASR，CTranslate2 权重 | [faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | `models/Systran/faster-whisper-large-v3/` |

LVBench、Video-MME、EgoSchema 的旧 `lmms-lab` 地址目前重定向到 `lmms-eval`。MLVU Dev 不等于答案未公开的 `MLVU/MLVU_Test`。当前 Video-MME 契约是 **w/o sub.**，下载字幕不会自动将实验变为带字幕评估。下载前阅读各数据源的许可；需要授权访问的资产先在网页申请，并使用自己的 HF 账号登录。

## 下载和解压

在仓库根目录执行；`VAS_ASSETS` 可以放在独立数据盘，不要放入 Git 仓库。

```bash
export VAS_ASSETS=/path/to/vas-assets
python -m pip install huggingface_hub
python tools/download_assets.py --root "$VAS_ASSETS" --group core --dry-run
# 实际下载四个当前数据集及四个模型，包含大量视频。
python tools/download_assets.py --root "$VAS_ASSETS" --group core
# 只下载模型，或只下载一个数据集：
python tools/download_assets.py --root "$VAS_ASSETS" --group models
python tools/download_assets.py --root "$VAS_ASSETS" --asset lvbench
# 可选关联数据：
python tools/download_assets.py --root "$VAS_ASSETS" --asset longvideobench --asset cgbench
```

下载脚本使用固定 revision，不自动解压或删除包。模型目录可直接使用；数据集按下列要求解压，先用 `unzip -l <包>` 查看内部路径，避免出现 `data/data/` 或 `videos/videos/`：

- LVBench：下载的 `video_chunks/videos_chunk_*.zip` 最终应形成 `LVBench/video_chunks/<video_id>.mp4`。
- Video-MME：`videos_chunked_*.zip` 最终应形成 `Video-MME/data/<videoID>.mp4`。包内已经带 `data/` 时解到 `Video-MME/`，只有文件名时解到 `Video-MME/data/`。
- EgoSchema：`videos_chunked_*.zip` 最终应形成 `egoschema/videos/<uuid>.mp4`；按包内是否已有 `videos/` 前缀选择目标目录。
- MLVU：下载仓库已提供 `MLVU/json` 和 `MLVU/video` 树，保持原任务子目录及大小写，不重命名视频。
- LongVideoBench：在其下载目录中运行 `cat videos.tar.part.* | tar -xf -`，并用 `tar -xf subtitles.tar` 解压字幕。拼接顺序按文件名排序，下载所有分卷后再解压。当前配置未启用此数据集。
- CG-Bench：区分 `video_chunk_*.zip` 原视频和 `clue_video_chunk_*.zip` 线索片段，不将线索片段替代原视频。当前配置未启用此数据集。

原机器本地数据约 805 GiB、模型约 26 GiB；这是已有文件的占用，不能当作新下载大小。压缩包与解压视频同时保留还需要额外空间。

## 环境、配置与检查

`environment.yml` 是原导出环境说明；`runtime_packages_20261013.txt` 记录当前环境的包版本，供排查差异，并非跨平台安装锁。依赖安装、CUDA/PyTorch 匹配和 ffmpeg 需要在目标机器单独验证。尚未完成干净环境的完整 GPU 复现。

```bash
python tools/prepare_local_config.py --assets-root "$VAS_ASSETS" --check-data
export MODEL_BASE_PATH="$VAS_ASSETS/models"
export PYTHONPATH=.:RL:evaluationVideo/Mix
export ONLINE_EVOLUTION=0
export MIX_PAIRED_ABLATIONS=0
python evaluationVideo/Mix/run_mix_batch.py \
  --config runtime/local_config.json --plan-only --max-samples 2 \
  --sample-uids mlvu:topic-reasoning:0:awa-6,mlvu:order:0:order-126
```

配置生成器使用历史配置的模型/推理条件，替换数据与 prompt 路径，将新输出写到 `runtime/fresh_baseline/`，缓存放到资产目录的 `cache/shared_preprocess/`。再次实验须指定新的 `--output` 和 `--run-root`。选样成功只验证输入装载；真正执行 GPU 实验时去掉 `--plan-only`，结果仍需独立核查。新机器冷缓存的耗时不能与原有共享缓存实验直接比较。

## 旧 MLVU 八样本子集

Git 保留 `mlvu_test_subset_20260714/subset_manifest.json` 和过滤标注，视频未上传。下载 MLVU Dev 后恢复对应视频：

```bash
python tools/restore_mlvu_subset.py --mlvu-root "$VAS_ASSETS/datasets/MLVU/MLVU" --dry-run
python tools/restore_mlvu_subset.py --mlvu-root "$VAS_ASSETS/datasets/MLVU/MLVU"
```

该子集是旧暂停实验中的八个已完成样本，不能替代 9 月的冻结开发套件或未见测试集。

## 没有公开下载地址的本地内容

共享缓存可从原视频重新生成，但原证据图像、运行时检查点及缓存条件影响精确历史重放。`local_only_artifacts.json` 列出未提交的中间产物；这些文件仍在原机器，本仓库不是全部本地文件的完整备份。它们没有已发布的私有下载链接，不能从数据源下载替代。

需要迁移精确重放素材时，应从原机器复制相关工作区的 `_shared_cache/` 和清单中指定的 checkpoint；保留相同目录结构，并记录文件摘要。不要发布原视频/帧缓存到 GitHub 来代替下载授权。Conda 目录、Codex 会话、工具缓存和账号凭据也不属于项目发布内容。可选工具的额外模型见 `RL/config/settings.py` 的路径属性；当前未启用的工具无需为最小流程下载全部模型。
