# Perfect-Shot

Local AI photo optimizer using Qwen2-VL for analysis and auto-edits.

## Recommended Models by VRAM

| VRAM      | Model                                      | Quantization | VRAM Usage | Env Vars to Set                          |
|-----------|--------------------------------------------|--------------|------------|------------------------------------------|
| 6-8 GB   | Qwen/Qwen2-VL-2B-Instruct                  | None         | ~4-5 GB    | MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct     |
| 8-12 GB  | Qwen/Qwen2-VL-7B-Instruct-AWQ              | AWQ          | ~6-8 GB    | MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct-AWQ<br>QUANTIZATION=awq |
| 12-16 GB | Qwen/Qwen2-VL-7B-Instruct                  | None         | ~12-14 GB  | MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct     |
| 16+ GB   | Qwen/Qwen2-VL-72B-Instruct-AWQ             | AWQ          | ~30+ GB    | MODEL_NAME=Qwen/Qwen2-VL-72B-Instruct-AWQ<br>QUANTIZATION=awq |

## All Environment Variables

| Variable                  | Default                          | Description                              |
|---------------------------|----------------------------------|------------------------------------------|
| MODEL_NAME                | Qwen/Qwen2-VL-7B-Instruct-AWQ   | Hugging Face model ID                    |
| QUANTIZATION              | awq                             | None or awq (only for AWQ models)        |
| MAX_MODEL_LEN             | 8192                            | Max context length                       |
| GPU_MEMORY_UTILIZATION    | 0.85                            | Fraction of GPU memory to use            |
| TEMPERATURE               | 0.3                             | Sampling temperature                     |
| MAX_TOKENS                | 512                             | Max output tokens                        |
| MINIO_ROOT_USER           | perfectshotadmin                | MinIO admin username                     |
| MINIO_ROOT_PASSWORD       | perfectshotpassword123          | MinIO admin password (change this!)      |

Set via `.env` file or docker-compose environment.

Example `.env` for 6 GB GPU:
```
MODEL_NAME=Qwen/Qwen2-VL-2B-Instruct
QUANTIZATION=
MAX_MODEL_LEN=8192
MINIO_ROOT_PASSWORD=your-strong-password
```

Run: `python -m src.process <image_or_directory>`