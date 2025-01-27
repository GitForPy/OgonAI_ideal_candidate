# 🚀 OgonAI Ideal Candidate

**OgonAI Ideal Candidate** — это система для анализа интервью кандидатов с использованием автоматической транскрибации видео/аудио и анализа речи на основе современных моделей AI.

---

## 🎯 Цели и задачи проекта

- 🗻 **Автоматическая транскрипция** видео-интервью кандидатов.
- 📊 Оценка кандидатов по нескольким критериям (мотивирующие факторы, референция, локус контроля, активность/пассивность и др.).
- 🤖 Использование AI для генерации количественных и качественных метрик, которые помогают принимать решения.

---

## ⚙️ Установка и настройка

Для работы с проектом выполните следующие шаги:

### 1 Клонируйте репозиторий:

```bash
git clone git@github.com:MVolobueva/ideal_candidate.git
cd ideal_candidate
```

---

### 2  Создайте и активируйте виртуальное окружение:

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

---

### 3 Установите зависимости:

```bash
pip install -r requirements.txt
```

---

### 4  Настройте ключи API:

- Откройте файл `main.py`.
- Замените значения переменных `DEEPGRAM_API_KEY` и `HYPERBOLIC_API_KEY` на ваши ключи API.

---

### 5⃣ Убедитесь, что на вашем устройстве установлен ffmpeg:

- macOS:
  ```bash
  brew install ffmpeg
  ```
- Ubuntu:
  ```bash
  sudo apt install ffmpeg
  ```
- Windows: Скачайте и установите [ffmpeg](https://ffmpeg.org/download.html).

---

## 🚀 Пример использования

1. Поместите видеофайлы в папку `uploads_video/` 🎥.
2. Запустите проект командой:

   ```bash
   python main.py
   ```

3. После выполнения вы найдёте:
   - 📜 Расшифровки в папке `transcribed_text/`.
   - 📊 Результаты анализа в папке `outputs/`:
     - `quantitative_metrics.json` — количественные метрики.
     - `qualitative_metrics.json` — качественные метрики.

---

## 📋 Требования

- Python 3.9 или выше 🐍
- ffmpeg 🛠️
- API-ключи для:
  - [Deepgram](https://deepgram.com/) 🔑
  - [OpenAI](https://openai.com/) 🔐

---

## ✉️ Контакты

Если у вас есть вопросы или предложения, свяжитесь с нами через [GitHub Issues]([https://github.com](https://github.com/GitForPy)/)  📧.

---

🎉 **Спасибо, что используете OgonAI Ideal Candidate! Удачи в анализе интервью!** 🎉

