# Banking77 Benchmark Suite

Репозиторий предназначен для воспроизводимого сравнения моделей на задаче многоклассовой классификации банковских текстовых обращений на датасете `PolyAI/banking77`.

Центр инженерный benchmark-контур:
- фиксированный data pipeline
- воспроизводимые baseline
- отдельный RNN блок
- отдельный transformer блок
- единый comparison слой
- контроль leakage и невалидных результатов

---

## 1. Что делает этот репозиторий

Задача:

`text -> intent_class -> route_target`

Где:
- `text` - пользовательское банковское обращение
- `intent_class` - один из 77 интентов
- `route_target` - прикладное направление обработки

Репозиторий нужен для ответа на прикладные вопросы:
- какой baseline реально силён на коротких банковских текстах
- даёт ли recurrent модель реальный прирост
- сколько дают transformers
- какой ценой по fit time и runtime достигается этот прирост
- какие результаты валидны, а какие нужно выбрасывать из-за leakage

---

## 2. Что репозиторий не делает

Это не production-сервис и не готовый API.

Здесь нет:
- готового HTTP inference сервиса
- orchestration
- model registry
- deployment manifests
- drift monitoring
- access control
- полной эксплуатационной обвязки

Это исследовательский и инженерный стенд, на базе которого уже можно принимать технические решения по выбору модели и следующему шагу в сторону прикладной routing-системы.

---

## 3. Структура репозитория

```text
.
├── notebooks/
│   ├── 01_banking77_eda.ipynb
│   ├── 02_banking77_classic_ml_rapids.ipynb
│   ├── 03_banking77_rnn_pytorch.ipynb
│   ├── 04_banking77_transformers_hf.ipynb
│   └── 05_banking77_model_comparison.ipynb
├── requirements/
│   ├── eda.txt
│   └── gpu-common.txt
├── data/
│   └── raw/
│       └── banking77_snapshot/
└── native/
    └── transformer_kernels/
```

### Назначение ноутбуков

#### `01_banking77_eda.ipynb`
Задачи:
- валидация датасета
- анализ длины текстов
- анализ распределения классов
- поиск точных и нормализованных дубликатов
- служебная нормализация текста

#### `02_banking77_classic_ml_rapids.ipynb`
Задачи:
- TF-IDF baseline
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- group-aware CV
- фиксация сильного classic baseline

#### `03_banking77_rnn_pytorch.ipynb`
Задачи:
- RNN / LSTM / GRU
- baseline recurrent конфигурации
- последовательный тюнинг архитектуры
- тюнинг регуляризации и оптимизации
- выбор итоговой recurrent конфигурации

#### `04_banking77_transformers_hf.ipynb`
Задачи:
- pretrained encoders
- tuned encoder configurations
- encoder-side MoE
- family D: encoder-decoder и GPT-like
- runtime и memory benchmark
- runtime build native kernels
- проверка fallback path

#### `05_banking77_model_comparison.ipynb`
Задачи:
- единый leaderboard
- quality vs cost
- межсемейное сравнение
- runtime benchmark
- аудит невалидных результатов
- финальная инженерная сводка

---

## 4. Данные

Источник:
- `PolyAI/banking77`

Тип задачи:
- `single-label multi-class text classification`

Число классов:
- `77`

Исходные split:
- `official_train = 10003`
- `official_test = 3080`

После очистки и контроля пересечений в проектном comparison слое используется рабочее разбиение:
- `train = 7999`
- `validation = 2000`
- `test = 3072`

### Что важно по данным
- тексты короткие
- классы близки семантически
- lexical baseline здесь объективно силён
- accuracy недостаточна как основная метрика
- leakage контроль обязателен

---

## 5. Data pipeline

Фактический pipeline в репозитории:

1. загрузка `PolyAI/banking77`
2. локальное сохранение snapshot в `data/raw/banking77_snapshot`
3. приведение схемы столбцов
4. проверка пропусков
5. удаление точных дубликатов
6. построение нормализованного текста
7. проверка пересечений между split
8. построение model-specific представления текста
9. обучение
10. выбор конфигурации по validation
11. финальная оценка на test
12. сведение результатов в comparison notebook

### Нормализованный текст нужен для
- дедупликации
- группировки
- leakage control
- контроля пересечений

Нормализованный текст не является универсальным модельным input для всех семейств моделей.  
Это техническая сущность пайплайна.

---

## 6. Методологические правила проекта

### 6.1. Роли выборок
- `train` только для обучения
- `validation` только для выбора конфигурации
- `test` только для финальной контрольной оценки

### 6.2. Основная метрика
Основная метрика выбора модели:

`validation_macro_f1`

Используемые метрики:
- `accuracy`
- `precision_macro`
- `recall_macro`
- `f1_macro`
- `precision_weighted`
- `recall_weighted`
- `f1_weighted`

### 6.3. Почему `macro_f1`
Потому что:
- классов много
- дисбаланс умеренный, но есть
- важно качество по всем классам, а не только на частых

### 6.4. Leakage control
Leakage проверяется на нескольких уровнях:
- точные дубликаты
- нормализованные дубликаты
- пересечения между split
- group-aware CV
- проверка корректности generative / GPT-like input policy

---

## 7. Classic ML блок

Используемые модели:
- `TF-IDF + Logistic Regression`
- `TF-IDF + Multinomial Naive Bayes`
- `Dense TF-IDF + Random Forest`

### Зачем этот блок
Classic ML здесь не декоративный baseline.  
Он нужен как реальная контрольная линия, которую deep модели должны обгонять честно, а не на фоне искусственно слабой стартовой точки.

### Лучшая classic конфигурация
- candidate: `lr_tfidf_40k_bigrams_c2`
- `ngram_range=(1,2)`
- `max_features=40000`
- `min_df=2`
- `LogisticRegression(C=2.0, max_iter=500)`

### Результат
- `validation_macro_f1 = 0.85345`
- `test_macro_f1 = 0.86763`

### Инженерный смысл
На коротких банковских текстах TF-IDF + linear classifier уже даёт сильный результат.  

---

## 8. RNN блок

Используемые архитектуры:
- `RNN`
- `LSTM`
- `GRU`
- `bidirectional variants`

Базовый вычислительный путь:

`text -> tokenization -> embedding -> recurrent backbone -> pooling -> classifier`

### Baseline recurrent конфигурация
- recurrent cell: `GRU`
- `bidirectional=True`
- `num_layers=2`
- `embedding_dim=128`
- `hidden_size=128`
- `dropout=0.3`
- `batch_size=128`
- `lr=0.001`
- `weight_decay=0.0001`
- `optimizer=AdamW`
- `scheduler=none`
- `max_epochs=20`
- `early_stopping_patience=3`
- `gradient_clip_val=1.0`
- `max_vocab_size=20000`
- `max_seq_len=48`

### Тюнинг выполнен сериями
- `A` - входной пайплайн
- `B` - архитектура
- `C` - регуляризация
- `D` - оптимизация
- `E` - финальные комбинации

### Лучшая tuned recurrent конфигурация
- experiment: `combo_01_a1_b1_c1_d1`
- cell: `BiGRU`
- `bidirectional=True`
- `num_layers=3`
- `embedding_dim=256`
- `hidden_size=256`
- `vocab_cap=1000`
- `max_seq_len=96`
- `dropout=0.5`
- `LayerNorm=Yes`
- `optimizer=AdamW`
- `scheduler=reduce_on_plateau`
- `weight_decay=0.0005`
- `label_smoothing=0.05`

### Результат
- `validation_macro_f1 = 0.89074`
- `test_macro_f1 = 0.89398`

### Инженерный смысл
Baseline recurrent модель не гарантирует выигрыш.  
Но после нормального тюнинга recurrent блок уже даёт практический прирост относительно strong lexical baseline и остаётся умеренным по вычислительной цене.

---

## 9. Transformer блок

Используемые линии:
- tuned pretrained encoders
- encoder-side MoE
- family D encoder-decoder
- family D GPT-like

### Лучший tuned encoder
- experiment: `deberta_v3_base__freeze2_reinit3_cosine_accum2`
- pretrained model: `microsoft/deberta-v3-base`
- `batch_size=8`
- `lr=0.000018`
- `max_epochs=16`
- `optimizer=adamw`
- `scheduler=cosine`
- `gradient_accumulation_steps=2`
- `label_smoothing=0.05`
- `freeze_backbone_epochs=2`
- `head_lr_multiplier=5.0`
- `layerwise_lr_decay=0.88`
- `reinit_top_layers=3`
- `dropout=0.1`

### Лучший encoder-side MoE
- experiment: `deberta_v3_base__freeze2_reinit3_cosine_accum2__moe_top2_e8_wide_freeze2`
- `num_experts=8`
- `expert_top_k=2`
- `expert_hidden_multiplier=1.75`

### Лучший valid family D candidate
- experiment: `family_d1b_seq2seq_tuned_wide_classification_head_adamw_fused_cosine`

### Результаты верхнего уровня
- best valid MoE by validation: `test_macro_f1 = 0.93710`
- best encoder by test: `test_macro_f1 = 0.93891`

### Инженерный смысл
Transformers задают верхнюю границу качества.  
Но их нужно рассматривать вместе с fit time, runtime и общей стоимостью, а не только по одной метрике `F1`.

---

## 10. Native kernels и runtime build

В `native/transformer_kernels` лежит tracked native extension для transformer части.

Состав:
- `bindings.cpp`
- `kernels_cpu.cpp`
- `kernels_cuda.cu`

### Что экспортируется
Судя по bindings:
- `fused_causal_masked_softmax`
- `kv_cache_append`

### Что реализовано
CPU path:
- causal masked softmax через masked fill + softmax
- конкатенация KV cache по временной оси

CUDA path:
- собственный CUDA kernel для causal masked softmax
- собственный CUDA kernel для append в KV cache
- dispatch через `AT_DISPATCH_FLOATING_TYPES_AND2`
- поддержка `Half` и `BFloat16`

### Как это собирается
В проекте нет отдельного standalone build pipeline через `setup.py` или `CMake` для native части.  
Сборка делается runtime-способом внутри notebook `04` через:

`torch.utils.cpp_extension.load(...)`

Это означает следующее:
- extension не собирается заранее как wheel
- сборка происходит на машине пользователя
- успех сборки зависит от локального CUDA toolkit, компилятора и совместимости с установленным PyTorch

### Что нужно для успешной сборки native path
Нужно:
- установленный `nvcc`
- локальный CUDA toolkit
- PyTorch wheel, совместимый с используемой CUDA
- рабочий C++ toolchain
- доступ к заголовкам PyTorch extension API

### Что происходит при проблеме
Судя по текущему устройству репозитория и описанию notebook `04`, native path не является безусловным hard requirement.  
Ноутбук выполняет smoke test и при нерабочем ускорении может откатиться на fallback path.

Это важно. Репозиторий не должен ломаться целиком только из-за того, что ускоряющая native часть не собралась.

---

## 11. Build и окружения

### EDA окружение

Используется `requirements/eda.txt`.

Состав:
- `jupyterlab==4.5.6`
- `ipykernel==7.1.0`
- `datasets==4.4.0`
- `pandas==2.3.3`
- `pyarrow==22.0.0`
- `matplotlib==3.10.7`
- `seaborn==0.13.2`
- `scikit-learn==1.7.2`
- `tqdm==4.67.1`
- `joblib==1.5.2`
- `packaging==25.0`

Команды:
```sh
python3 -m venv .venv-eda
. .venv-eda/bin/activate
pip install -U pip
pip install -r requirements/eda.txt
python -m ipykernel install --user --name rnn-eda --display-name rnn-eda
jupyter lab
```

### GPU окружение

Используется `requirements/gpu-common.txt`.

Состав:
- всё из общего data / plotting / sklearn набора
- `transformers==4.57.6`
- `accelerate==1.13.0`
- `peft==0.18.1`
- `sentencepiece==0.2.1`
- `safetensors==0.7.0`
- `evaluate==0.4.6`

Команды:
```sh
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -U pip
pip install -r requirements/gpu-common.txt
pip install --extra-index-url https://pypi.nvidia.com cupy-cuda13x==14.0.1 cudf-cu13==26.2.0 cuml-cu13==26.2.0
python -m ipykernel install --user --name rnn-gpu --display-name rnn-gpu
jupyter lab
```

Если хост использует CUDA 12, нужно ставить `cu12` аналоги вместо `cu13`.

### Дополнительно для notebook 03
```sh
. .venv-gpu/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install lightning torchmetrics
```

### Дополнительно для notebook 04
```sh
. .venv-gpu/bin/activate
python -m pip install -r requirements/gpu-common.txt
python -m pip install bitsandbytes "optimum[onnxruntime-gpu]" onnxscript
```

### Опциональные ускорения
```sh
. .venv-gpu/bin/activate
python -m pip install "flash-attn-4[cu13]==4.0.0b7"
```

`xformers` остаётся опциональным ускорением и не должен считаться обязательной зависимостью.

---

## 12. Runbook

Рекомендуемый порядок запуска:
1. `notebooks/01_banking77_eda.ipynb`
2. `notebooks/02_banking77_classic_ml_rapids.ipynb`
3. `notebooks/03_banking77_rnn_pytorch.ipynb`
4. `notebooks/04_banking77_transformers_hf.ipynb`
5. `notebooks/05_banking77_model_comparison.ipynb`

### Почему именно так
Потому что это фактический pipeline:
- сначала проверяется и понимается датасет
- затем фиксируется strong baseline
- затем строится recurrent uplift
- затем измеряется transformer ceiling
- затем все результаты приводятся в единый comparison слой

---

## 13. Ключевые результаты

### Рабочее финальное разбиение
- `train = 7999`
- `validation = 2000`
- `test = 3072`

### Classic baseline
- `lr_tfidf_40k_bigrams_c2`
- `validation_macro_f1 = 0.85345`
- `test_macro_f1 = 0.86763`

### Baseline RNN
- `bilstm_2layer`
- `validation_macro_f1 = 0.85373`
- `test_macro_f1 = 0.85499`

### Tuned RNN
- `combo_01_a1_b1_c1_d1`
- `validation_macro_f1 = 0.89074`
- `test_macro_f1 = 0.89398`

### Best valid family D
- `family_d1b_seq2seq_tuned_wide_classification_head_adamw_fused_cosine`
- `validation_macro_f1 = 0.88716`
- `test_macro_f1 = 0.89489`

### Best valid MoE by validation
- `deberta_v3_base__freeze2_reinit3_cosine_accum2__moe_top2_e8_wide_freeze2`
- `validation_macro_f1 = 0.93818`
- `test_macro_f1 = 0.93710`

### Best encoder by test
- `deberta_v3_base__freeze2_reinit3_cosine_accum2`
- `validation_macro_f1 = 0.93572`
- `test_macro_f1 = 0.93891`

---

## 14. Leakage-case

В проекте есть важный технический кейс, который нельзя скрывать.

Одна из GPT-like classification_head конфигураций показывала почти идеальные метрики:
- `train_macro_f1 ~ 1.00000`
- `val_macro_f1 ~ 0.99962`
- `test_macro_f1 ~ 0.99968`

Причина была не в "магически сильной модели", а в ошибке постановки:
- истинный `label_token` попадал во входную последовательность

После исправления:
- `train_macro_f1 ~ 0.99633`
- `val_macro_f1 ~ 0.86423`
- `test_macro_f1 ~ 0.87882`

Этот кейс нужен в README как инженерное предупреждение:
- benchmark без leakage audit ненадёжен
- красивые метрики могут быть мусором
- comparison слой обязан фильтровать такие случаи

---

## 15. Практический смысл результатов

Если нужен:
- **сильный и дешёвый baseline** -> бери `TF-IDF + Logistic Regression`
- **разумный рабочий deep candidate** -> бери tuned recurrent model
- **максимум качества без жёсткой экономии ресурсов** -> бери tuned DeBERTa encoder / encoder-side MoE

---

## 16. Ограничения репозитория

- нет production API
- нет полного build / release цикла
- native kernels собираются runtime-способом, а не отдельным пакетированием
- датасет открытый и англоязычный
- routing слой пока логический, а не сервисный
- часть transformer конфигураций слишком дорога для дешёвого ежедневного inference

---

## 17. Текущее состояние

Репозиторий уже содержит всё необходимое для инженерного и дипломного анализа:
- reproducible data pipeline
- strong classic baseline
- tuned recurrent block
- transformer benchmark
- runtime и quality comparison
- leakage audit
- comparison notebook как единый аналитический слой
