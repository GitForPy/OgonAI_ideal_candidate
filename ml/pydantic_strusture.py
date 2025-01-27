from pydantic import BaseModel, Field
from typing import Optional, List

# quantity metrics:

class ReferenceResults(BaseModel):
    outside_references_amount: int = Field(description="Количество пунктов про внешние референции", default=0)
    inside_references_amount: int = Field(description="Количество пунктов про внутренние референции", default=0)
    info: str = Field(description="Обоснуй итоговый тип референции: внутреняя, внешняя, смешанная и краткое пояснение почему. Если есть, ыведи соотношение тех и других референций из текста")


class IntentionAvoidanceResults(BaseModel):
    intention_amount: int = Field(description="Количество пунктов про стремление", default=0)
    avoidance_amount: int = Field(description="Количество пунктов про избегание", default=0)
    info: str = Field(description="Обоснуй превалирования итогого типа критерия и краткое пояснение почему. Выведи соотношение количества пунктов, если есть") # добавлено от 27.01

class SimilarityDifferenceResults(BaseModel):
    similarity_amount: int = Field(description="Количество пунктов про сходство", default=0)
    difference_amount: int = Field(description="Количество пунктов про различие", default=0)
    info: str = Field(description="Обоснуй превалирования итогого типа критерия и краткое пояснение почему.Выведи соотношение количества пунктов, если есть") # добавлено от 27.01

class WorkAndColleaguesResults(BaseModel):
    content: int = Field(description="Количество пунктов про содержание", default=0)
    people: int = Field(description="Количество пунктов про окружение", default=0)
    info: str = Field(description="Обоснуй превалирования итогого типа критерия и краткое пояснение почему. Выведи соотношение количества пунктов, если есть") # добавлено от 27.01


# charecteristic merics


class ReferencesCategoriesResults(BaseModel):
    interest_and_self_development: Optional[List[str]] = Field(description="Список мотивирующих факторов из текста", default=None)
    financial_and_material_side: Optional[List[str]] = Field(description="Список мотивирующих факторов из текста", default=None)
    physical_and_personal_well_being: Optional[List[str]] = Field(description="Список мотивирующих факторов из текста", default=None)


class LocusControlResults(BaseModel):
  locus_control: List[str] = Field(description="Итоговая характеристика по кандидату о внешнем и внутреннем локус контроле:", default=None)


class ProcessesAndPossibilitiesResults(BaseModel):
    possibilities: List[str] = Field(description="список потенциальных возможностей", default='Нет упоминания')
    processes: List[str] = Field(description="список текущих процедур (процессов)", default='Нет упоминания')



# # 3 показателя новых

class ProcessVsResult(BaseModel):
    score: float = Field(description="Твоя оценка активности от 0 до 1, где 1 = кандидат нацелен на активную деятельность, 0 = кандидат склонен к пассивности.", default=None)
    morph_score: float = Field(description='{внешняя оценка}')
    average_score: float = Field(description='{среднее арифметическое}')
    description: float = Field(description=  'Твой вывод на основе расчетов дай оценку по харктеристике процесс - результат, дополни вывод списком глаголов (глагол вид)',  default='Нет упоминания') 

class TeamVsIndividual(BaseModel):
    score: float = Field(description="Твоя оценка активности от 0 до 1, где 1 = кандидат нацелен на активную деятельность, 0 = кандидат склонен к пассивности.", default=None)
    morph_score: float = Field(description='внешняя оценка', default=None)
    pronouns_score: float = Field(description='{[мы]/[я]}', default=None)
    description: float = Field(description=  'Твой вывод на основе расчетов дай оценку по харктеристике одиночка-командный игрок',  default='Нет упоминания') 


class ActivityVsPassivity(BaseModel):
    score: float = Field(description="Твоя оценка активности от 0 до 1, где 1 = кандидат нацелен на активную деятельность, 0 = кандидат склонен к пассивности.", default=None)
    description: str = Field(description="Твой вывод на основе расчетов дай оценку по харктеристике активность", default='Нет упоминания')



extraction_classes = {
    'References': ReferenceResults,
    'Similarity': SimilarityDifferenceResults,
    'IntentionAvoidance': IntentionAvoidanceResults,
    'WorkAndColleagues': WorkAndColleaguesResults,
    'Motivation': ReferencesCategoriesResults,
    'ProcessesAndPossibilities': ProcessesAndPossibilitiesResults,
    'LocusControl': LocusControlResults,
    'ProcessVsResult': ProcessVsResult,
    'TeamVsIndividual': TeamVsIndividual,
    'ActivityVsPassivity': ActivityVsPassivity
}



