# TODO

[ ] EDA
    <!-- 
    1) Количество ненормализованных ко всем (символы/слова и тд.)
    2) В скольких фразах вообще встречаются ненормализованные
     -->
    [x] Цифры
    [x] Некириллические символы
    [ ] Технические символы
    [ ] Обозначения (самостоятельные знаки типа $)
    [x] Сокращения (не аббревиатуры)
    [ ] Необычные знаки препинания
        [ ] Отдельно посмотреть на скобочку в конце предложения (не смайлик)
        [ ] Отдельно разобрать, что к чему с долгими (3+) точками,
            со скобками, которые не были закрыты, смайликами и прочим
        <!-- точка, запятая, восклицательный знак, вопросительный знак, двойная кавычка, апостроф, точка с запятой, многоточие, двоеточие, дефис, тире, кавычки "ёлочки". -->
    [x] Аббревиатуры
    [ ] Междометия – надо как-то решить проблему того, что мм и мммм должны как-то матчиться
    [x] Эмодзи
    [ ] Записи капсом (отделить от аббревиатур)
[ ] Бинарный классификатор "нормализован/не нормализован"
    [ ] Rule-based
    [x] Попробовать эмбеддинги (multilingual-e5-large) + классификатор
    [ ] Сделать простой классификатор на CountVectorizer()
    [ ] Попробовать [тэггер норм-не норм](https://huggingface.co/RUNorm/RUNorm-tagger) и обвес на него
[ ] Нормализатор текста
    <!-- Датасеты: 
    1) https://www.kaggle.com/competitions/text-normalization-challenge-russian-language/rules
    2) https://huggingface.co/datasets/ruscorpora/normalization
    3)  -->
    [ ] Rule-based
    [ ] Готовая RUNorm-normalizer-small
    [ ] ByT5-small fine-tuning


<!-- 
Одновременно в EDA EntryStatistics() и в классификатор со своими векторами

Группа A, меняет произношение:

n_digits, digit_ratio, has_time (\d{1,2}:\d{2}), has_date, has_phone, n_latin, latin_ratio,
has_homoglyph (латиница внутри кириллического слова), n_roman, has_ordinal_digit (3-й),
n_short_forms, n_abbrev_caps, has_currency,
has_percent_degree_no, has_unit (км, ГБ, руб), has_url_email

Группа B, типографика:

n_zero_width, n_double_space, space_before_punct, comma_no_space, max_punct_repeat, has_paren,
n_emoji, has_curly_quotes, unbalanced_quotes, is_interjection, trailing_ws, len_chars, n_words
-->

Вопрос вот в чем по архитектуре: наверное, нет смысла брать этот класс и поновой его переписывать, стоит просто сделать более подробную статистику, ведь мы в feature векторах как раз и описываем то, какие features должны быть учтены.

Но тут надо смотреть конкретно
