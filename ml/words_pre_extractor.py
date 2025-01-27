import pandas as pd
import numpy as np
import spacy


# python -m spacy download ru_core_news_sm


interview = """
Может ли он выполнять задачи из своего стека технологий? Объективная метрика это сколько он зарабатывает. И какое положение в сообществе того же языка он занимает. Там различные награды, тэйги, типа контрипутеры, мейнтейнеры. Ну и по количеству проектов, какие самые сложные проекты он делал, показывает какой у него уровень. Какие самые сложные проекты он делал? 
ставлю таймер пишу название задачи и начинаю ее делать это если локальная задача самом начале дня просто перечень задач и дальше побежал каждая из них.
...
"""

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

print(extract_pron_with_parameters(interview))
