# Banking77 Benchmark Suite

Полноценный исследовательский и инженерный benchmark-suite для задачи автоматической классификации и маршрутизации текстовых обращений на датасете **PolyAI/Banking77**.

Проект объединяет:
- разведочный анализ данных
- сильные классические baseline-модели
- серию рекуррентных нейросетевых экспериментов
- transformer-бенчмарк
- единый сравнительный отчёт по качеству, стоимости обучения и корректности эксперимента

---

## 1. Описание проекта

**Тема проекта**  
Разработка системы автоматической классификации и маршрутизации текстовых обращений на основе методов машинного обучения.

**Предметная область**  
Обработка естественного языка, intent classification, автоматическая маршрутизация пользовательских обращений, сравнение baseline и deep learning подходов.

**Ключевая идея**  
Проект рассматривает не только задачу классификации текста, но и её прикладную интерпретацию. После определения класса обращения система может вернуть направление маршрутизации в нужный контур обработки:
- card support
- transfer support
- cash operations
- payments
- fraud / security
- KYC / verification
- account information

---

## 2. Постановка задачи

На вход подаётся короткое текстовое обращение пользователя.  
Необходимо:
1. определить его класс среди **77 банковских интентов**
2. оценить качество предсказания
3. сравнить несколько семейств моделей
4. проанализировать trade-off между качеством и вычислительной стоимостью
5. использовать предсказанный класс как основу для прикладной маршрутизации

Формально задача задаётся как многоклассовая классификация:

\[
f(x) \rightarrow y,\quad y \in \{1,2,\dots,77\}
\]

где:
- `x` — текст обращения
- `y` — предсказанный класс

После этого применяется функция маршрутизации:

\[
r(y) \rightarrow route\_target
\]

Итоговый прикладной выход системы:
- `predicted_label`
- `route_target`
- `confidence` или распределение вероятностей

---

## 3. Почему именно Banking77

В проекте используется открытый датасет **PolyAI/Banking77**, предназначенный для fine-grained intent classification в банковской предметной области.

Преимущества датасета:
- открытый и воспроизводимый источник
- прикладная тематика, близкая к реальным обращениям
- достаточно сложная постановка из-за большого числа классов
- короткие тексты с сильной лексической сигнализацией
- полезен как для baseline-моделей, так и для нейросетевых архитектур

Особенности постановки:
- **77 классов**
- тексты, как правило, короткие
- многие классы семантически близки
- accuracy недостаточна как единственная метрика
- важно контролировать устойчивость по всем классам

---

## 4. Цели проекта

Проект решает несколько взаимосвязанных задач.

### 4.1. Исследовательская цель
Построить честный и воспроизводимый benchmark-suite для сравнения разных подходов к классификации банковских текстовых обращений.

### 4.2. Прикладная цель
Показать, какая модель или семейство моделей является наиболее разумным выбором для системы маршрутизации обращений.

### 4.3. Методологическая цель
Не просто получить высокую метрику, а обеспечить:
- корректную постановку эксперимента
- контроль утечек данных
- сопоставимость результатов
- анализ стоимости обучения и инференса
- аудит подозрительно хороших результатов

---

## 5. Структура репозитория

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

### Смысл блоков

#### `01_banking77_eda.ipynb`
Разведочный анализ данных:
- проверка структуры датасета
- дубликаты
- длина текстов
- распределение классов
- анализ аномалий
- подготовка служебной нормализации текста

#### `02_banking77_classic_ml_rapids.ipynb`
Классические baseline-модели:
- TF-IDF
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- group-aware cross-validation
- базовые таблицы качества

#### `03_banking77_rnn_pytorch.ipynb`
Рекуррентные модели:
- RNN / LSTM / GRU
- baseline-конфигурации
- серии тюнинга
- подбор архитектуры, регуляризации и оптимизации
- итоговый tuned recurrent candidate

#### `04_banking77_transformers_hf.ipynb`
Transformer-блок:
- pretrained encoder-модели
- усиленный fine-tuning
- encoder-side MoE
- family D: encoder-decoder и GPT-like
- runtime / memory benchmark
- дополнительные оптимизации и native-kernel path

#### `05_banking77_model_comparison.ipynb`
Итоговый сравнительный отчёт:
- единый leaderboard
- cross-family comparison
- quality vs cost
- runtime benchmark
- аудит невалидных результатов
- финальные графики и сводные выводы

---

## 6. Используемые подходы

В проекте рассматриваются три основные линии.

### 6.1. Classic ML
Используются лексические признаки и линейные или вероятностные модели:
- TF-IDF + Logistic Regression
- TF-IDF + Multinomial Naive Bayes
- Dense TF-IDF + Random Forest

Роль этого блока:
- задать сильный baseline
- проверить, сколько качества можно получить без глубоких моделей
- использовать как контрольную точку для последующих улучшений

### 6.2. Recurrent Neural Networks
Используются модели последовательностей:
- RNN
- LSTM
- GRU
- bidirectional варианты
- tuning архитектуры и протокола обучения

Базовая идея:
`text -> tokenization -> embedding -> recurrent backbone -> pooling -> classifier`

### 6.3. Transformers
Используются более мощные и дорогие модели:
- pretrained encoder classifiers
- tuned encoder configurations
- encoder-side MoE
- собственные архитектурные ветки family D

Этот блок нужен не только ради рекорда по метрике, но и для понимания:
- насколько велик реальный выигрыш по качеству
- оправдан ли он вычислительно
- где проходит граница между разумным рабочим решением и дорогим исследовательским максимумом

---

## 7. Подготовка данных

Все ноутбуки работают с открытым датасетом `PolyAI/banking77`.

### Общий пайплайн
1. Загрузка official split
2. Локальное сохранение snapshot
3. Приведение имён столбцов и форматов
4. Проверка пропусков
5. Поиск точных и нормализованных дубликатов
6. Нормализация текста для служебного контроля
7. Формирование независимых train / validation / test
8. Построение представления текста для конкретного семейства моделей

### Важный момент
Нормализованный текст используется не как единственное модельное представление, а как служебная форма для:
- дедупликации
- группировки
- контроля leakage
- устойчивой проверки пересечений между split

---

## 8. Контроль корректности эксперимента

Один из принципиальных пунктов этого репозитория — **не доверять метрикам слепо**.

В проекте контролируются:
- точные и нормализованные дубликаты
- пересечения между выборками
- leakage на уровне fold'ов
- корректность входной последовательности в generative / GPT-like постановках
- валидность итогового leaderboard

### Почему это важно
Высокая метрика сама по себе не гарантирует, что эксперимент поставлен правильно.  
В ходе работы был обнаружен невалидный leakage-case в одном из GPT-like classification_head сценариев, когда истинная метка оказывалась во входной последовательности. До исправления это давало почти идеальные значения. После исправления метрики вернулись к реалистичному уровню.

Это не побочный эпизод, а важная часть проекта.  
Здесь качество исследования определяется не тем, насколько красива цифра, а тем, насколько эта цифра заслуживает доверия.

---

## 9. Метрики

Основная метрика проекта:
- **Macro F1**

Дополнительно анализируются:
- Accuracy
- Precision Macro
- Recall Macro
- Precision Weighted
- Recall Weighted
- F1 Weighted

### Почему Macro F1
Для задачи с 77 классами и умеренным дисбалансом accuracy может быть слишком грубой.  
Macro F1 лучше отражает:
- устойчивость по классам
- качество на редких интентах
- пригодность модели для прикладной маршрутизации

---

## 10. Протокол оценки

Во всём проекте соблюдается единый принцип:
- `train` используется для обучения
- `validation` используется для выбора конфигурации
- `test` используется как финальная контрольная оценка

### Почему это важно
Если выбирать победителя по `test`, финальная метрика перестаёт быть честной.  
Поэтому в итоговом сравнении модели должны оцениваться либо:
- по `val macro-F1` как основной метрике отбора
- либо отдельно отмечаться как post hoc analysis, если сравнение идёт по `test`

---

## 11. Ключевые результаты

Ниже приведены ключевые опорные точки, которые используются в итоговом сравнении.

### 11.1. Рабочее финальное разбиение
- Train: **7 999**
- Validation: **2 000**
- Test: **3 072**

### 11.2. Лучший classic baseline
- Model: `lr_tfidf_40k_bigrams_c2`
- Val Macro F1: **0.85345**
- Test Macro F1: **0.86763**

### 11.3. Лучший baseline RNN
- Model: `bilstm_2layer`
- Val Macro F1: **0.85373**
- Test Macro F1: **0.85499**

### 11.4. Лучший tuned RNN
- Model: `combo_01_a1_b1_c1_d1`
- Val Macro F1: **0.89074**
- Test Macro F1: **0.89398**

### 11.5. Лучший valid family D candidate
- Model: `family_d1b_seq2seq_tuned_wide_classification_head_adamw_fused_cosine`
- Val Macro F1: **0.88716**
- Test Macro F1: **0.89489**

### 11.6. Лучший valid MoE по выбору через validation
- Model: `deberta_v3_base__freeze2_reinit3_cosine_accum2__moe_top2_e8_wide_freeze2`
- Val Macro F1: **0.93818**
- Test Macro F1: **0.93710**

### 11.7. Лучший tuned encoder по test
- Model: `deberta_v3_base__freeze2_reinit3_cosine_accum2`
- Val Macro F1: **0.93572**
- Test Macro F1: **0.93891**

---

## 12. Как интерпретировать результаты

Из полученных результатов следуют несколько важных выводов.

### 12.1. TF-IDF baseline сильнее, чем может показаться
На коротких банковских текстах линейная модель с хорошими признаками даёт очень сильный baseline.  
Это означает, что:
- слабая нейросеть не гарантирует улучшения
- любая более сложная модель должна оправдать себя цифрами

### 12.2. Рекуррентные модели дают реальный прирост только после нормального тюнинга
Baseline recurrent models не дают автоматического выигрыша над сильным classic ML baseline.  
Но последовательная настройка архитектуры и протокола обучения позволяет получить уже заметный прирост.

### 12.3. Transformers задают верхнюю границу качества
Лучшие encoder / MoE решения дают максимальный Macro F1.  
Но это происходит ценой:
- большего времени обучения
- большего memory footprint
- более тяжёлого inference path

### 12.4. Quality и cost нужно рассматривать вместе
Сама по себе лучшая метрика ещё не означает, что именно эта модель является лучшим практическим выбором.  
Для прикладной системы маршрутизации часто важнее:
- стабильность
- разумная стоимость обучения
- умеренный inference budget
- прозрачность поведения модели

---

## 13. Какой вывод по выбору модели

Если нужен **сильный и дешёвый baseline**, разумный выбор:
- TF-IDF + Logistic Regression

Если нужен **основной прикладной кандидат с хорошим балансом качества и стоимости**, разумный выбор:
- tuned recurrent model

Если нужен **максимум качества без жёсткой экономии ресурсов**, разумный выбор:
- tuned DeBERTa encoder / encoder-side MoE

---

## 14. Практическая интерпретация: routing layer

Репозиторий в первую очередь решает задачу классификации, но результаты можно напрямую использовать для построения маршрутизационного слоя.

### Пример
```json
{
  "text": "My transfer is still pending, what should I do?",
  "predicted_label": "pending_transfer",
  "route_target": "transfer_support",
  "confidence": 0.94
}
```

### Простейшая логика маршрутизации
- card-related -> `card_support`
- transfer-related -> `transfer_support`
- cash-related -> `cash_operations`
- payment-related -> `payments`
- identity-related -> `kyc_verification`
- security-related -> `fraud_security`
- balance-related -> `account_info`

---

## 15. Ограничения проекта

Этот репозиторий не следует трактовать как полностью готовую production-систему.  
Текущий статус проекта:
- исследовательский benchmark-suite
- notebook-centric pipeline
- сравнительный контур для выбора модели

Ограничения:
- англоязычный открытый датасет
- данные короче и чище, чем реальные корпоративные обращения
- часть моделей слишком дорога для повседневного продового inference
- routing layer показан как логическое продолжение, а не как готовая интеграция

---

## 16. Воспроизведение окружения

### EDA environment
```sh
python3 -m venv .venv-eda
. .venv-eda/bin/activate
pip install -U pip
pip install -r requirements/eda.txt
python -m ipykernel install --user --name rnn-eda --display-name rnn-eda
jupyter lab
```

### GPU environment
```sh
python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install -U pip
pip install -r requirements/gpu-common.txt
pip install --extra-index-url https://pypi.nvidia.com cupy-cuda13x==14.0.1 cudf-cu13==26.2.0 cuml-cu13==26.2.0
python -m ipykernel install --user --name rnn-gpu --display-name rnn-gpu
jupyter lab
```

Если на хосте используется CUDA 12, необходимо заменить `cu13` пакеты на `cu12` аналоги.

### PyTorch for notebook 03
```sh
. .venv-gpu/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install lightning torchmetrics
```

### HF stack for notebook 04
```sh
. .venv-gpu/bin/activate
python -m pip install -r requirements/gpu-common.txt
python -m pip install bitsandbytes "optimum[onnxruntime-gpu]" onnxscript
```

### Optional acceleration
```sh
. .venv-gpu/bin/activate
python -m pip install "flash-attn-4[cu13]==4.0.0b7"
```

`xformers` остаётся опциональным ускорением.

### Native kernels
Для части notebook 04 требуется локальный CUDA toolkit, чтобы `torch.utils.cpp_extension.load(...)` мог собрать extension из `native/transformer_kernels`.

---

## 17. Порядок запуска ноутбуков

Рекомендуемая последовательность:
1. `01_banking77_eda.ipynb`
2. `02_banking77_classic_ml_rapids.ipynb`
3. `03_banking77_rnn_pytorch.ipynb`
4. `04_banking77_transformers_hf.ipynb`
5. `05_banking77_model_comparison.ipynb`

### Почему именно так
Такая последовательность отражает реальную исследовательскую логику:
- сначала понять данные
- затем построить baseline
- затем усилить его recurrent-моделями
- затем проверить transformer upper bound
- затем свести всё в единый comparison report

---

## 18. Что именно делает comparison notebook

Итоговый comparison notebook нужен не как красивая витрина, а как главный аналитический слой проекта.  
Он выполняет:
- унификацию результатов из classic, RNN и transformer ноутбуков
- построение leaderboard
- удаление или маркировку невалидных результатов
- сравнение quality vs fit time
- сравнение throughput / runtime
- построение итоговых графиков
- формирование финальных выводов

---

## 19. Для кого полезен этот репозиторий

Репозиторий полезен:
- студентам, которым нужен честный NLP дипломный проект
- исследователям, которым важен воспроизводимый baseline-to-transformer comparison
- инженерам, которые хотят понять, оправдывает ли сложная модель свою стоимость
- тем, кто строит intent classification / routing pipeline и хочет не абстрактные советы, а конкретный сравнительный контур

---

## 20. Возможные направления развития

Следующие логичные шаги развития проекта:
- явная реализация API для inference
- отдельный модуль confidence-aware routing
- calibration и abstention
- multilingual path
- русскоязычные банковские обращения
- hierarchical classification
- multilabel routing
- production-style packaging вместо notebook-centric формата
- автоматическая генерация reproducible reports

---

## 21. Рабочий статус проекта

Текущее состояние проекта:
- benchmark-suite собран
- основные семейства моделей исследованы
- единый comparison notebook присутствует
- leakage audit выполнен
- результаты пригодны для академического анализа и дипломной работы

---

## 22. Репозиторий и источник данных

- Dataset: `PolyAI/banking77`
- Repository: `SUKUNA-AI/banking77-benchmark-suite`

---

## 23. Список литературы

1. Casanueva I., Temčinas T., Gerz D., Henderson M., Vulić I. Efficient Intent Detection with Dual Sentence Encoders // Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI. Association for Computational Linguistics, 2020. P. 38-45.

2. PolyAI. Banking77 Dataset Card.  
   https://huggingface.co/datasets/PolyAI/banking77

3. Lhoest Q., Villanova del Moral A., Jernite Y. et al. Datasets: A Community Library for Natural Language Processing.  
   https://arxiv.org/abs/2109.02846

4. Pedregosa F., Varoquaux G., Gramfort A. et al. Scikit-learn: Machine Learning in Python // Journal of Machine Learning Research. 2011. Vol. 12. P. 2825-2830.

5. PyTorch Documentation.  
   https://docs.pytorch.org/docs/stable/index.html

6. Lightning AI. PyTorch Lightning Documentation.  
   https://lightning.ai/docs/pytorch/stable/index.html

7. Hugging Face Transformers Documentation.  
   https://huggingface.co/docs/transformers/index

8. Hochreiter S., Schmidhuber J. Long Short-Term Memory // Neural Computation. 1997. Vol. 9, No. 8. P. 1735-1780.

9. Devlin J., Chang M.-W., Lee K., Toutanova K. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding // NAACL-HLT. 2019.

10. Vaswani A., Shazeer N., Parmar N. et al. Attention Is All You Need // NeurIPS. 2017.

11. He P., Liu X., Gao J., Chen W. DeBERTa: Decoding-enhanced BERT with Disentangled Attention // ICLR. 2021.

---


