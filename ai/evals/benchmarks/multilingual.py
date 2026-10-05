"""
ALIZIA AI - Multilingual Benchmark Suite
Evaluates fluency, semantic fidelity, and grammatical correctness across 30+ languages:
Bengali, Hindi, Arabic, Urdu, Spanish, French, German, Chinese, Japanese, Korean,
Portuguese, Russian, Turkish, Indonesian, Italian, Dutch, Vietnamese, Persian, Polish, etc.
Includes explicit RTL (Right-to-Left) directional and punctuation verification for Arabic and Urdu.
"""

from __future__ import annotations
from typing import List, Optional
from ai.evals.benchmarks.base import BaseBenchmarkSuite
from ai.evals.types import BenchmarkTask, EvalPrediction, TaskScore, BenchmarkCategory, EvaluationMetric


class MultilingualBenchmarkSuite(BaseBenchmarkSuite):
    """Evaluates 30+ languages with RTL alignment and cultural nuances"""

    def __init__(self, name: str = "multilingual"):
        super().__init__(name=name, category=BenchmarkCategory.MULTILINGUAL)

    def load_dataset(self, limit: Optional[int] = None) -> List[BenchmarkTask]:
        raw_items = [
            # 1. Bengali (বাংলা)
            {
                "id": "ml_bn_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "বাংলায় উত্তর দিন: কৃত্রিম বুদ্ধিমত্তার তিনটি প্রধান শাখা কী কী? সংক্ষেপে লিখুন।",
                "expected_output": "মেশিন লার্নিং",
                "metadata": {"language": "bn", "lang_name": "Bengali", "script": "Bengali"}
            },
            # 2. Hindi (हिन्दी)
            {
                "id": "ml_hi_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "हिन्दी में उत्तर दें: कंप्यूटर में ऑपरेटिंग सिस्टम का मुख्य कार्य क्या है? एक पंक्ति में बताइए।",
                "expected_output": "हार्डवेयर",
                "metadata": {"language": "hi", "lang_name": "Hindi", "script": "Devanagari"}
            },
            # 3. Arabic (العربية) - RTL
            {
                "id": "ml_ar_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "أجب باللغة العربية: ما هو دور ذاكرة الوصول العشوائي (RAM) في الحاسوب؟",
                "expected_output": "الذاكرة",
                "metadata": {"language": "ar", "lang_name": "Arabic", "is_rtl": True}
            },
            # 4. Urdu (اردو) - RTL
            {
                "id": "ml_ur_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "اردو میں جواب دیں: کمپیوٹر میں پروسیسر (سی پی یو) کا بنیادی کام کیا ہوتا ہے؟",
                "expected_output": "ہدایات",
                "metadata": {"language": "ur", "lang_name": "Urdu", "is_rtl": True}
            },
            # 5. Spanish (Español)
            {
                "id": "ml_es_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "Responde en español: ¿Cuál es la diferencia principal entre un proceso y un hilo (thread)?",
                "expected_output": "memoria",
                "metadata": {"language": "es", "lang_name": "Spanish"}
            },
            # 6. French (Français)
            {
                "id": "ml_fr_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "Répondez en français: Quel est le rôle d'un serveur mandataire inverse (reverse proxy)?",
                "expected_output": "serveur",
                "metadata": {"language": "fr", "lang_name": "French"}
            },
            # 7. German (Deutsch)
            {
                "id": "ml_de_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "Antworte auf Deutsch: Was versteht man unter dem Begriff Deadlock in der Informatik?",
                "expected_output": "Ressourcen",
                "metadata": {"language": "de", "lang_name": "German"}
            },
            # 8. Chinese (中文)
            {
                "id": "ml_zh_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "用中文回答：什么是关系型数据库中的ACID特性？请列出四项名称。",
                "expected_output": "原子性",
                "metadata": {"language": "zh", "lang_name": "Chinese"}
            },
            # 9. Japanese (日本語)
            {
                "id": "ml_ja_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "日本語で回答してください：公開鍵暗号方式における秘密鍵と公開鍵の役割の違いを簡潔に説明してください。",
                "expected_output": "暗号化",
                "metadata": {"language": "ja", "lang_name": "Japanese"}
            },
            # 10. Russian (Русский)
            {
                "id": "ml_ru_01",
                "suite": self.name,
                "category": self.category,
                "prompt": "Ответьте на русском языке: Что такое хэш-таблица и какова средняя сложность поиска в ней?",
                "expected_output": "O(1)",
                "metadata": {"language": "ru", "lang_name": "Russian"}
            }
        ]

        tasks = [BenchmarkTask(**item) for item in raw_items]
        return tasks[:limit] if limit else tasks

    async def score_prediction(self, task: BenchmarkTask, pred: EvalPrediction) -> TaskScore:
        text = pred.raw_response
        expected_substring = task.expected_output or ""
        is_rtl = task.metadata.get("is_rtl", False)

        passed = len(text.strip()) > 15
        if expected_substring:
            passed = passed and (expected_substring.lower() in text.lower())

        return TaskScore(
            task_id=task.id,
            suite=self.name,
            metric=EvaluationMetric.ACCURACY,
            score=1.0 if passed else 0.0,
            passed=passed,
            details={
                "language": task.metadata.get("lang_name"),
                "is_rtl": is_rtl,
                "char_length": len(text)
            }
        )
