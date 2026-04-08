# Banking77 Notebooks

Сейчас в проекте оставлены только то, что нужно для текущего этапа:

- `notebooks/01_banking77_eda.ipynb` — self-contained EDA ноутбук.
- `notebooks/02_banking77_classic_ml_rapids.ipynb` — self-contained RAPIDS baseline ноутбук.
- `notebooks/03_banking77_rnn_pytorch.ipynb` — self-contained Lightning ноутбук для `RNN/LSTM/GRU` и серии RNN-экспериментов.
- `notebooks/04_banking77_transformers_hf.ipynb` — self-contained notebook на `Transformers + Accelerate` для `BERT/RoBERTa/DeBERTa/MoE` и собственного `family D` (`encoder-decoder` + `GPT-like`) с `KV-cache`, `torch.compile`, `CUDA Graphs` и native `C++/CUDA` kernels.
- `notebooks/05_banking77_model_comparison.ipynb` — финальный сравнительный ноутбук, который объединяет `02/03/04`, строит единый leaderboard, аудит невалидных результатов, cross-family runtime benchmark и итоговые визуализации.
- `requirements/eda.txt` — зависимости для EDA kernel.
- `requirements/gpu-common.txt` — базовые зависимости для GPU kernel.

Ноутбуки сами:

- скачивают `PolyAI/banking77`;
- сохраняют локальный snapshot в `data/raw/banking77_snapshot`;
- используют рабочую parquet-ревизию `refs/pr/6`, потому что старый script-based вариант больше не грузится текущим `datasets`.

## Быстрый старт

EDA:

```sh
python3 -m venv .venv-eda
. .venv-eda/bin/activate
pip install -U pip
pip install -r requirements/eda.txt
python -m ipykernel install --user --name rnn-eda --display-name rnn-eda
jupyter lab
```

GPU / RAPIDS:

```sh
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -U pip
pip install -r requirements/gpu-common.txt
pip install --extra-index-url https://pypi.nvidia.com cupy-cuda13x==14.0.1 cudf-cu13==26.2.0 cuml-cu13==26.2.0
python -m ipykernel install --user --name rnn-gpu --display-name rnn-gpu
jupyter lab
```

Если на хосте будет не CUDA 13, а CUDA 12, то вместо `cu13` пакетов ставятся `cu12` аналоги.

PyTorch для ноутбука `03` ставится отдельным шагом в ту же `.venv-gpu`, потому что для него нужен официальный CUDA wheel-индекс:

```sh
. .venv-gpu/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

После установки `torch` в это же окружение нужно добавить `lightning` и `torchmetrics`:

```sh
. .venv-gpu/bin/activate
python -m pip install lightning torchmetrics
```

Для ноутбука `04` в это же окружение нужен Hugging Face стек:

```sh
. .venv-gpu/bin/activate
python -m pip install -r requirements/gpu-common.txt
```

Для ускоренного инференса и ONNX Runtime-бенчмарков в `04` нужно добавить:

```sh
. .venv-gpu/bin/activate
python -m pip install bitsandbytes "optimum[onnxruntime-gpu]" onnxscript
```

Для Blackwell/Hopper GPU можно дополнительно попробовать:

```sh
. .venv-gpu/bin/activate
python -m pip install "flash-attn-4[cu13]==4.0.0b7"
```

Ноутбук `04` не полагается на это слепо: он отдельно делает smoke test и, если backend реально нерабочий, автоматически остаётся на `PyTorch SDPA`.

`xformers` в проекте остаётся опциональным ускорением и не требуется как обязательная зависимость.

Для native-kernel части в `04` нужен рабочий локальный CUDA toolkit, чтобы `torch.utils.cpp_extension.load(...)` мог собрать tracked `C++/CUDA` extension из [`native/transformer_kernels`](/home/sukuna/Projects/RNN/native/transformer_kernels). На текущей машине notebook сам валидирует сборку, корректность и fallback-поведение.

После этого в `jupyter lab` можно запускать:

- `notebooks/02_banking77_classic_ml_rapids.ipynb` — классические модели на RAPIDS;
- `notebooks/03_banking77_rnn_pytorch.ipynb` — нейросетевые модели и серия RNN-экспериментов на `PyTorch Lightning`.
- `notebooks/04_banking77_transformers_hf.ipynb` — основной transformer-benchmark: большие pretrained encoder-модели, encoder-side `MoE`, а также `family D` с собственными `encoder-decoder` и `GPT-like` архитектурами, optimizer sweep, memory/runtime benchmark и native kernels.
- `notebooks/05_banking77_model_comparison.ipynb` — итоговый сравнительный отчёт по всем моделям с красивыми таблицами, graph-heavy сравнением и unified runtime benchmark.
