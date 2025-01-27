import os
import subprocess
from datetime import datetime
from deepgram import DeepgramClient, PrerecordedOptions

def convert_to_mp3(input_file: str, temp_dir: str) -> str:
    """
    Конвертирует входной аудио/видео-файл в MP3 с заданными параметрами.
    Возвращает путь к конвертированному MP3-файлу.
    """
    print(f"Начало конвертации файла {input_file} в MP3...")
    output_file = os.path.join(
        temp_dir, os.path.splitext(os.path.basename(input_file))[0] + ".mp3"
    )
    subprocess.run(
        [
            "ffmpeg", "-i", input_file, 
            "-vn",         # Отключить видео
            "-ar", "16000",  # Частота дискретизации 16 kHz
            "-ac", "1",      # Один канал (моно)
            "-b:a", "128k",  # Битрейт
            output_file
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    print(f"Файл успешно конвертирован в {output_file}")
    return output_file

def save_transcription(
    transcription_data,
    original_filename: str,
    output_dir: str
) -> (str, str):
    """
    Сохраняет результаты транскрибации в текстовый файл.
    Возвращает путь к сохранённому текстовому файлу и сам транскрипт.
    """
    print(f"Сохранение результатов для файла {original_filename}...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.splitext(os.path.basename(original_filename))[0]
    output_filename = f"{base_filename}_transcription_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)

    # Получаем сам текст транскрибации
    transcript = transcription_data.results.channels[0].alternatives[0].transcript

    # Записываем в файл
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(transcript)

    print(f"Транскрипт сохранен: {output_path}")
    return output_path, transcript

def process_audio_file(
    input_file: str,
    deepgram_api_key: str,
    temp_dir: str,
    output_dir: str
) -> str:
    """
    Обрабатывает один аудио/видео-файл:
    1) Конвертирует в MP3,
    2) Отправляет на транскрибацию в Deepgram,
    3) Сохраняет результат,
    4) Удаляет временный MP3-файл.

    Возвращает финальный транскрибированный текст.
    """
    print(f"\nОбработка файла: {input_file}")
    print("-" * 50)

    # 1) Конвертация в MP3
    mp3_file = convert_to_mp3(input_file, temp_dir)

    # 2) Инициализация клиента Deepgram и чтение MP3
    print("Инициализация клиента Deepgram...")
    deepgram = DeepgramClient(deepgram_api_key)
    with open(mp3_file, "rb") as file:
        buffer_data = file.read()

    if not buffer_data:
        raise ValueError(f"Файл {input_file} пуст или не может быть прочитан.")

    # Подготовка данных и опций для Deepgram
    payload = {"buffer": buffer_data}
    options = PrerecordedOptions(
        model="nova-2",  # Модель Nova-2 (поддерживает русский язык)
        language="ru",   # Язык
        punctuate=True,  # Включить пунктуацию
    )

    # 3) Отправка файла на транскрибацию
    print("Отправка файла на транскрибацию...")
    try:
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    except Exception as e:
        print(f"Ошибка при транскрибации: {e}")
        raise

    # 4) Сохраняем результаты
    txt_path, transcript = save_transcription(response, input_file, output_dir)
    print(f"Результаты успешно сохранены для файла: {txt_path}")

    # Удаление временного MP3-файла
    os.remove(mp3_file)
    print(f"Временный файл {mp3_file} удалён.")
    print("-" * 50)

    return transcript

def process_multiple_files(
    file_list: list,
    deepgram_api_key: str,
    temp_dir: str,
    output_dir: str
) -> dict:
    """
    Обрабатывает список аудио/видео-файлов.
    Возвращает словарь {путь_к_файлу: транскрибированный_текст}.
    """
    results = {}
    total_files = len(file_list)
    print(f"\nНачало обработки {total_files} файлов")
    print("=" * 50)

    for index, file in enumerate(file_list, 1):
        print(f"\nОбработка файла {index}/{total_files}: {file}")
        try:
            results[file] = process_audio_file(file, deepgram_api_key, temp_dir, output_dir)
        except Exception as e:
            print(f"Ошибка при обработке файла {file}: {e}")

    return results

def video2text(deepgram_api_key: str, source_dir: str, temp_dir: str, output_dir: str) -> dict:
    """
    Находит все файлы в папке source_dir, 
    транскрибирует их через Deepgram и сохраняет текст в output_dir.
    Возвращает словарь {путь_к_файлу: транскрибированный_текст}.
    """
    # Создаём папки, если их нет
    for directory in [output_dir, temp_dir, source_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Создана папка: {directory}")

    # Собираем список файлов в source_dir
    files_to_process = [
        os.path.join(source_dir, file)
        for file in os.listdir(source_dir)
        if os.path.isfile(os.path.join(source_dir, file))
    ]

    if not files_to_process:
        print(f"В папке {source_dir} нет файлов для обработки.")
        return {}

    print(f"Найдено файлов для обработки: {len(files_to_process)}")
    print("=" * 50)

    # Транскрибируем все файлы
    results = process_multiple_files(files_to_process, deepgram_api_key, temp_dir, output_dir)

    # Вывод результатов в консоль (по желанию)
    print("\nРезультаты транскрибации:")
    print("=" * 50)
    for file_path, text in results.items():
        print(f"\nФайл: {file_path}")
        print("-" * 30)
        print(text)
        print("-" * 50)

    return results