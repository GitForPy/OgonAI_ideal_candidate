import os
import json
import pandas as pd
import concurrent.futures
from datetime import datetime, timezone

from prompts import (
    criteria_system_prompts,
    criteria_user_prompts,
    criteria_types
)

from transcribation.video_to_text import video2text
from modules import (
    InterviewAnalyzerExtractor,
    calculate_metric,
    check_key_existence
)
from pydantic_strusture import (
    extraction_classes
)


# Сборка общего поплайна

def main():
    """
    Основная функция для анализа интервью.
    Возвращает два JSON-объекта с количественными и качественными метриками.
    """


    DEEPGRAM_API_KEY = "2eddd749bd80fe842cd57402a2d3be6d7a3e1367"
    SOURCE_DIR = "../uploads_video"     # Папка с исходными файлами
    OUTPUT_DIR = "../transcribed_text"  # Папка для готовых транскрипций
    TEMP_DIR = "./transcribation/temp"   # Временная папка для MP3


    # 1. Транскрибация (если надо)
    video2text(
        deepgram_api_key=DEEPGRAM_API_KEY,
        source_dir=SOURCE_DIR,
        temp_dir=TEMP_DIR,
        output_dir=OUTPUT_DIR
    )


    # Папка с транскрибированными текстами
    folder_path = "../transcribed_text"
    file_data = {}

    # Читаем все .txt файлы
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            prefix = file_name.split("_")[0]
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            file_data[prefix] = content

    # Инициализируем анализатор
    analyzer_extractor = InterviewAnalyzerExtractor(HYPERBOLIC_API_KEY)

    # Словари для хранения метрик (количественные и качественные метрики)
    quantitative_metrics = {}
    qualitative_metrics = {}

    # Параллельная обработка критериев
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_criteria = {}
        
        # Создаем задачи для каждого критерия
        for criteria, text_criteria in file_data.items():
            if criteria in criteria_system_prompts:
                future = executor.submit(
                    analyzer_extractor.analyze_criterion,
                    criteria,
                    transcript=text_criteria,
                    system_prompt=criteria_system_prompts[criteria],
                    user_prompt=criteria_user_prompts[criteria],
                    pydantic_class=extraction_classes[criteria],
                    criterion_type=criteria_types[criteria]
                )
                future_to_criteria[future] = criteria

        # Обрабатываем результаты
        for future in concurrent.futures.as_completed(future_to_criteria):
            crit = future_to_criteria[future]
            try:
                extracted_answer = future.result()
                
                # Выводим промежуточный результат
                print(f"Результат для {crit}:\n", 
                      json.dumps(extracted_answer, ensure_ascii=False, indent=4))

                # Определяем тип критерия
                criterion_type = criteria_types[crit]

                # Обработка количественных и custom метрик
                if criterion_type in ['quantitative', 'custom']:
                    if isinstance(extracted_answer, dict):
                        # Для количественных метрик
                        if 'score' in extracted_answer:
                            score = round(float(extracted_answer['score']), 2)
                        else:
                            numerator = check_key_existence(extracted_answer)
                            score = round(calculate_metric(extracted_answer, numerator), 2)

                        quantitative_metrics[crit] = {
                            "score": score,
                            "description": extracted_answer.get("description", 
                                         extracted_answer.get("info", "Нет описания"))
                        }

                # Обработка качественных и custom метрик
                if criterion_type in ['qualitative', 'custom'] or crit in [
                    "LocusControl", 
                    "ProcessesAndPossibilities", 
                    "Motivation", 
                    "ProcessVsResult",
                    "TeamVsIndividual",
                    "ActivityVsPassivity"
                ]:
                    if isinstance(extracted_answer, dict):
                        # Получаем описание из разных возможных полей
                        if 'description' in extracted_answer:
                            description = extracted_answer['description']
                        elif 'info' in extracted_answer:
                            description = extracted_answer['info']
                        else:
                            # Для метрик со сложной структурой
                            if crit == "LocusControl" and 'locus_control' in extracted_answer:
                                description = extracted_answer['locus_control'][0] if extracted_answer['locus_control'] else ""
                            
                            elif crit == "ProcessesAndPossibilities":
                                processes = extracted_answer.get('processes', [])
                                possibilities = extracted_answer.get('possibilities', [])
                                description = (
                                    f"Процессы: {', '.join(processes)}. "
                                    f"Возможности: {', '.join(possibilities)}"
                                )
                            
                            elif crit == "Motivation":
                                interests = extracted_answer.get('interest_and_self_development', [])
                                financial = extracted_answer.get('financial_and_material_side', [])
                                wellbeing = extracted_answer.get('physical_and_personal_well_being', [])
                                description = (
                                    f"Интересы и саморазвитие: {', '.join(interests)}. "
                                    f"Финансовая сторона: {', '.join(financial)}. "
                                    f"Благополучие: {', '.join(wellbeing)}"
                                )
                            else:
                                description = str(extracted_answer)
                    else:
                        description = str(extracted_answer)
                    
                    qualitative_metrics[crit] = {
                        "description": description
                    }

            except Exception as e:
                print(f"Ошибка при обработке критерия {crit}: {e}")
                continue

    # Создаем директорию для результатов
    os.makedirs("outputs", exist_ok=True)
    
    # Сохраняем результаты в JSON-файлы
    with open("outputs/quantitative_metrics.json", "w", encoding="utf-8") as f:
        json.dump(quantitative_metrics, f, ensure_ascii=False, indent=2)
    
    with open("outputs/qualitative_metrics.json", "w", encoding="utf-8") as f:
        json.dump(qualitative_metrics, f, ensure_ascii=False, indent=2)

    # Выводим информацию о созданных файлах
    print("\n" + "="*50)
    print("Созданы файлы результатов:")
    print(f"1. quantitative_metrics.json ({len(quantitative_metrics)} критериев)")
    print(f"2. qualitative_metrics.json ({len(qualitative_metrics)} критериев)")
    print("Путь: ./outputs/")
    print("="*50)

    return {
        "quantitative_metrics": quantitative_metrics,
        "qualitative_metrics": qualitative_metrics
    }

if __name__ == "__main__":
    # Информация о текущей сессии
    
    print("-" * 50)

    # Запуск основного анализа
    result = main()
    
    # Выводим результаты для проверки
    print("\nКоличественные метрики:")
    print(json.dumps(result["quantitative_metrics"], indent=2, ensure_ascii=False))
    print("\nКачественные метрики:")
    print(json.dumps(result["qualitative_metrics"], indent=2, ensure_ascii=False))
    
    print("\nAnalysis completed successfully!")