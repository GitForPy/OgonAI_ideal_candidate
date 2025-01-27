from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.output_parser import StrOutputParser
import pandas as pd
import json
import spacy


def extract_pron_with_parameters(text):
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    pron = []
    
    for token in doc:
        if token.pos_ == "PRON":
            pron.append({
                "pron": token.text,
                "lemma": token.lemma_,
                "case": token.morph.get("Case"),
                "person": token.morph.get("Person"),
                "number": token.morph.get("Number")
            })
    return pron

def extract_verbs_with_parameters(text):
    nlp = spacy.load("ru_core_news_sm")
    doc = nlp(text)
    verbs = []
    
    for token in doc:
        if token.pos_ == "VERB":
            verbs.append({
                "verb": token.text,
                "lemma": token.lemma_,
                "tense": token.morph.get("Tense"),
                "aspect": token.morph.get("Aspect"),
                "mood": token.morph.get("Mood"),
                "voice": token.morph.get("Voice"),
                "person": token.morph.get("Person"),
                "number": token.morph.get("Number")
            })
    return verbs


def get_metric_activity_vs_passivity(verbs, prons):
    #активность-пассивность
    
    active_sing = sum(1 for v in verbs if v['voice'] == ['Act'] and v['person'] == ['First'] and v['number'] == ['Sing'])
    active_plur = sum(1 for v in verbs if v['voice'] == ['Act'] and v['person'] == ['First'] and v['number'] == ['Plur'])
    tense_fut_pres = sum(1 for v in verbs if 'Fut' in v['tense'] or 'Pres' in v['tense'])
    
    metric_activity_vs_passivity = (active_sing + active_plur) / tense_fut_pres if tense_fut_pres > 0 else 0

    return metric_activity_vs_passivity



def get_metric_team_vs_individual(verbs, prons):
    #Одиночка-командный игрок
    sing_count = sum(1 for v in verbs if v['number'] == ['Sing'])
    plur_count = sum(1 for v in verbs if v['number'] == ['Plur'])
    
    metric_team_vs_individual = plur_count / (plur_count + sing_count)

    return metric_team_vs_individual


def get_metric_process_vs_result(verbs, prons):
    #Процесс - результат 
    perf_count = sum(1 for v in verbs if 'Perf' in v['aspect'])
    imp_count = sum(1 for v in verbs if 'Imp' in v['aspect'])
    
    metric_process_vs_result = perf_count / (perf_count + imp_count)

    return metric_process_vs_result


def check_key_existence(input_dict):
    """
    Вспомогательная функция проверки существующего ключа в словаре.
    Используется для старых критериев (например, References, Similarity, IntentionAvoidance),
    где мы смотрим, какой из ключей присутствует.
    """
    if 'outside_references_amount' in input_dict:
        return 'outside_references_amount'
    elif 'similarity_amount' in input_dict:
        return 'similarity_amount'
    elif 'intention_amount' in input_dict:
        return 'intention_amount'
    elif 'people' in input_dict:
        return 'people'


def calculate_metric(input_quantity_dict, numerator_column):
    """
    Модифицированная функция для игнорирования нечисловых полей
    """
    denominator = 0
    numerator = 0
    
    for key, item in input_quantity_dict.items():
        # Пропускаем нечисловые значения
        if isinstance(item, (int, float)):
            denominator += item
            if key == numerator_column:
                numerator = item
                
    return numerator / denominator if denominator > 0 else 0


def test_metrics_get_description(
    analyzer_extractor,
    files_texts,
    types_of_criteria,
    system_prompts,
    user_prompts,
    pydantic_classes,
    metrics,
    n
):
    """
    Пример функции, в которой несколько раз вызывается универсальный метод анализа
    для различных критериев и собираются результаты в DataFrame.

    Параметры:
      analyzer_extractor: экземпляр класса InterviewAnalyzerExtractor (см. ниже)
      files_texts: словарь { 'References': 'текст...', 'Similarity': '...', ... }
      types_of_criteria: словарь, определяющий тип каждого критерия (quantitative, qualitative, custom)
      system_prompts, user_prompts: словари с промптами
      pydantic_classes: словарь { 'References': ReferenceResults, ... }
      metrics: список критериев, по которым нужно собрать метрики
      n: сколько раз повторять анализ (например, для расчёта средних)

    Возвращает:
      df_metrics: DataFrame с сырыми метриками
      df_description: DataFrame-описание (параметры count, mean, std и т.д.)
    """
    df_metrics = pd.DataFrame({})

    for criteria, text_criteria in files_texts.items():
        # Проверяем, что критерий в списке metrics и он количественный
        if criteria in metrics and types_of_criteria[criteria] == 'quantitative':
            sample = []
            for i in range(n):
                # Универсальный метод для анализа
                extracted_answer = analyzer_extractor.analyze_criterion(
                    transcript=text_criteria,
                    system_prompt=system_prompts[criteria],
                    user_prompt=user_prompts[criteria],
                    pydantic_class=pydantic_classes[criteria],
                    criterion_type='quantitative'  # сообщаем, что критерий количественный
                )

                # Вычисляем метрику
                numerator = check_key_existence(extracted_answer)
                metric = calculate_metric(extracted_answer, numerator)
                sample.append(metric)

            # Записываем результаты в DataFrame
            temp_df = pd.DataFrame({criteria: sample})
            if df_metrics.empty:
                df_metrics = temp_df
            else:
                df_metrics = pd.concat([df_metrics, temp_df], axis=1)

    df_description = df_metrics.describe()
    return df_metrics, df_description


def write_to_excel_test_results(metrics, description, path):
    """
    Пример функции записи результатов в Excel.
    Записывает два листа:
      1) 'Метрики' — DataFrame с результатами
      2) 'Описание' — describe() по DataFrame
    """
    dataframes = {
        'Метрики': metrics,
        'Описание': description
    }

    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print('Результаты тестирования успешно записаны в файл', path)


class InterviewAnalyzerExtractor:
    """
    Универсальный класс, позволяющий работать с одним методом: analyze_criterion().
    На основе параметра criterion_type ('quantitative', 'qualitative' или 'custom')
    вызывает соответствующий приватный метод.

    Логика:
      - quantitative (количественный):
            1) Получаем "сырой" ответ от модели (текст).
            2) Извлекаем структуру JSON и валидируем через pydantic.
      - qualitative (качественный):
            1) Сразу просим модель вернуть ответ в формате JSON (через pydantic).
      - custom (пользовательский):
            1) Можно добавить любой произвольный функционал:
               парсинг spacy, расчет внешних метрик и т.д.
            2) Сформировать промпт
            3) Спарсить результат через pydantic
    """

    def __init__(self, api_key):


            
        self.llm = ChatOpenAI(
        api_key='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',
        model="gpt-4o-mini",
        temperature=0,
        streaming=True )

        
    def analyze_criterion(
        self, 
        criteria,
        transcript: str,
        system_prompt: str,
        user_prompt: str,
        pydantic_class,
        criterion_type: str = "quantitative",
    ):
        """
        Параметры:
          transcript (str): текст для анализа (например, расшифровка ответа).
          system_prompt (str): системный prompt (инструкции для модели).
          user_prompt (str): пользовательский prompt, в котором есть placeholders.
          pydantic_class: класс pydantic, в который будет парситься итоговый JSON.
          criterion_type (str): 'quantitative', 'qualitative' или 'custom'.

        Возвращает:
          dict-объект (JSON), прошедший валидацию pydantic.
        """

        if criterion_type == "quantitative":
            return self._analyze_quantitative(criteria, transcript, system_prompt, user_prompt, pydantic_class)

        elif criterion_type == "characteristics":
            return self._analyze_qualitative(transcript, system_prompt, user_prompt, pydantic_class)

        elif criterion_type == "custom":
            # Здесь ваша специальная логика (например, spacy-парсинг)
            return self._analyze_custom_criterion(transcript, system_prompt, user_prompt, pydantic_class, criteria)

        else:
            raise ValueError(f"Неизвестный тип критерия: {criterion_type}")

    # ==========================================================================
    # ============== Приватные методы для каждого типа критерия ================
    # ==========================================================================

    def _analyze_quantitative(self, criteria, transcript, system_prompt, user_prompt, pydantic_class):
        """
        Обработка количественных критериев:
          1) Получаем "сырой" ответ (строку).
          2) Извлекаем структуру (pydantic).
        """

        user_prompt = f'Оцени результат критерия. {transcript}'
        # user_prompt = user_prompt.format(transcript)
        # 1. Генерация "сырого" ответа модели
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
        chain_raw = prompt_template | self.llm | StrOutputParser()
        raw_answer = chain_raw.invoke({})  # строка с анализом

        # 2. Извлечение структуры (pydantic) из ответа
        parser = JsonOutputParser(pydantic_object=pydantic_class)
        prompt_extraction = PromptTemplate(
            template="""
            Извлеки структурированные данные из анализа критерия (количественный).

            Ответ верни в формате JSON:
            {format_instructions}

            Текст анализа:
            {query}
            """,
            input_variables=["query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        chain_extract = prompt_extraction | self.llm | parser
        structured_data = chain_extract.invoke({"query": raw_answer})

        return structured_data

    def _analyze_qualitative(self, transcript, system_prompt, user_prompt, pydantic_class):
        """
        Обработка качественных критериев:
          1) Сразу просим модель вернуть результат в формате JSON.
          2) Парсим pydantic-моделью.
        """
        parser = JsonOutputParser(pydantic_object=pydantic_class)

        prompt_json = PromptTemplate(
            template="""
            {system_prompt}
            
            {user_prompt}

            Ответ верни в формате JSON:
            {format_instructions}

            Текст анализа:
            {query}
            """,
            input_variables=["query", "system_prompt", "user_prompt"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )
        chain_json = prompt_json | self.llm | parser

        result_json = chain_json.invoke({
            "query": transcript,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt
        })
        return result_json

    def _analyze_custom_criterion(self, transcript, system_prompt, user_prompt, pydantic_class, criteria):
        prons = extract_pron_with_parameters(transcript)
        verbs = extract_verbs_with_parameters(transcript)

        result_metric = 0
        if criteria == 'ProcessVsResult':
            result_metric = get_metric_process_vs_result(verbs, prons)
        elif criteria == 'TeamVsIndividual':
            result_metric = get_metric_team_vs_individual(verbs, prons)
        elif criteria == 'ActivityVsPassivity':
            result_metric = get_metric_activity_vs_passivity(verbs, prons)



        # Формируем переменные (убрана переменная test и компактный JSON)
        template_vars = {
            "transcript": transcript,
            "verbs": json.dumps(verbs, ensure_ascii=False, separators=(",", ":")),  # Без переносов
            "prons": json.dumps(prons, ensure_ascii=False, separators=(",", ":")),
            "result_metric": result_metric
        }

        # Экранируем случайные скобки в промптах
        def escape_braces(text):
            return text.replace("{", "{{").replace("}", "}}").replace("{{transcript}}", "{transcript}").replace("{{verbs}}", "{verbs}").replace("{{prons}}", "{prons}").replace("{{result_metric}}", "{result_metric}")

        # Подготавливаем промпты
        system_escaped = escape_braces(system_prompt)
        user_escaped = escape_braces(user_prompt)

        # Собираем цепочку
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_escaped),
            ("user", user_escaped)
        ])

        # Выполняем запрос
        chain_raw = prompt_template | self.llm | StrOutputParser()
        raw_answer = chain_raw.invoke(template_vars)
       
        # Дальнейший парсинг остается без изменений
        parser = JsonOutputParser(pydantic_object=pydantic_class)
        prompt_extraction = PromptTemplate(
            template="""
            Извлеки структурированные данные из анализа критерия (custom).
            Ответ верни в формате JSON:
            {format_instructions}
            Текст анализа:
            {query}
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        chain_extract = prompt_extraction | self.llm | parser
        structured_data = chain_extract.invoke({"query": raw_answer})

        return structured_data