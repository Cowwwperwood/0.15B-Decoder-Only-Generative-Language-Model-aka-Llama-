# 0.15B Decoder Only Generative Language Model aka Llama

## Отчет по проекту: Реализация и обучение модели Decoder-Only Language Model

### Введение

В данном проекте была разработана и обучена модель типа Decoder-Only Language Model, аналогичная архитектуре LLaMA.

### Архитектура модели

Модель состоит из следующих основных компонентов:

RMSNorm: Метод нормализации скрытых состояний для стабилизации обучения.

SwiGLU: Улучшенный вариант нелинейной активации для MLP-блока.

TransformerDecoderBlock: Основной блок трансформера, включающий механизмы внимания(Flash-Attention) и MLP(SwiGLU).

Llama: Итоговая модель, включающая несколько слоев TransformerDecoderBlock и выходной слой для генерации токенов.

### Особенности реализации

Использование RMSNorm вместо LayerNorm.

Применение SwiGLU в блоке MLP для повышения экспрессивности модели.

Flash Attention для ускоренной вычислительной эффективности механизма внимания: За счет оптимизация механизма внимания удалось не только увеличить эффективность модели, но и оптимизировать затраты по памяти. С изначальным MHA памяти хватало лишь на батч из 32 элементов, а после добавления Flash-Attention я смог поместить в батч 50 элементов.

Rotary Embeddings для улучшенного кодирования позиций токенов.

Оптимизация вычислений за счет bfloat16: Благодаря тому, что все компоненты модели считаются в bloat16(за исключением нормализаций) удалось добиться ускорения обучения и инференса модели в два раза(Причем использование bfloat16 вместо float16 значительно, так как при использовании float16 результаты вычислений в DecodeBlock оказывались численно нестабильными и часто выбивали NaN).

### Обучение

Модель обучалась на части данных из датасета OpenWebText(на полной версии датасета обучался GPT2)  с использованием CrossEntropyLoss.В качестве целевых значений были использованы токены сдвинутой версии входных данных. Обучение производилось в течении 4-6 часов на видеокарте NVIDIA RTX A4000 16GB, на 256000 токенах, с 12800 токенами в одном батче. Было принято решение взять разогрев 5%(повышение LR с 0 до 1e-4 в течении 5% времени обучения), а после Cosine Learning Rate. Минимальное значение CrossEntropyLoss которого удалось достигнуть - 3.8. График обучения можно увидеть ниже.
   ![Train loss](https://github.com/Cowwwperwood/0.15B-Decoder-Only-Generative-Language-Model-aka-Llama-/blob/main/loss_graph.jpg)

### Результаты

По итогам обучения мне удалось получить модель, которая генерирует связнные по смыслу текста, но, к сожалению, не более. Пример генерации модели при промпте "California is ":

California is the richest state in America.

In a state that includes the U.S. Constitution, the state of Minnesota on the state’s second-largest federal law, has been on the rise in the past decade and has been largely associated with the state the state.

The state’s legislature has made a state-of-the-art plan to eliminate all the state’s state tax credits and state-of-the-art ordinance rules, with many states approving a $2.5-billion amendment allowing a state of the state to use a state-state solution of state-controlled state-of-the-art state laws.

That’s also the first in a state that needs more state-of-the-art legislation to change its bill for nonprofits and state governments under the age 10-21 law.

### Выводы

Разработанная архитектура соответствует современным стандартам моделей типа decoder-only и может использоваться для задач генерации текста. Дальнейшие улучшения могут включать масштабирование модели(увеличение числа параметров), дообучение на 1024 токенах(Position Interpolation), использование большего датасета. 
