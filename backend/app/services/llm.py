import re

from pydantic import BaseModel

from app.core.config import Settings
from app.models.schemas import SourceChunk, TokenUsage
from app.services.vector_store import tokenize


class LLMResult(BaseModel):
    answer: str
    usage: TokenUsage


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate_answer(self, question: str, sources: list[SourceChunk]) -> LLMResult:
        if self.settings.openai_api_key:
            try:
                return self._generate_with_openai(question, sources)
            except Exception:
                return self._generate_local_result(question, sources)
        return self._generate_local_result(question, sources)

    def _generate_local_result(self, question: str, sources: list[SourceChunk]) -> LLMResult:
        return LLMResult(
            answer=self._generate_local_answer(question, sources),
            usage=TokenUsage(),
        )

    def _generate_with_openai(self, question: str, sources: list[SourceChunk]) -> LLMResult:
        from openai import OpenAI

        source_block = "\n\n".join(
            f"[{index}] {source.document_name}"
            f"{', page ' + str(source.page) if source.page else ''}\n{source.text}"
            for index, source in enumerate(sources, start=1)
        )

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer only from the provided sources. "
                        "If the sources do not contain enough evidence, say so clearly. "
                        "Keep the answer concise and cite source numbers when useful."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question:\n{question}\n\nSources:\n{source_block}",
                },
            ],
            temperature=0.2,
        )
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens
        return LLMResult(
            answer=response.choices[0].message.content or "",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=self._estimate_cost(prompt_tokens, completion_tokens),
            ),
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * self.settings.openai_input_cost_per_1m_tokens
        output_cost = (
            completion_tokens / 1_000_000
        ) * self.settings.openai_output_cost_per_1m_tokens
        return round(input_cost + output_cost, 6)

    def _generate_local_answer(self, question: str, sources: list[SourceChunk]) -> str:
        if not sources:
            return "I could not find enough evidence in the indexed documents to answer this question."

        query_terms = set(tokenize(question))
        candidates: list[tuple[int, str, SourceChunk]] = []
        for source in sources:
            for sentence in _sentences(source.text):
                sentence_terms = set(tokenize(sentence))
                overlap = len(query_terms & sentence_terms)
                if overlap > 0:
                    candidates.append((overlap, sentence.strip(), source))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = candidates[:4]

        if not selected:
            top_source = sources[0]
            return (
                "The retrieved documents contain related context, but the evidence is weak. "
                f"The closest source is {top_source.document_name}."
            )

        lines = [
            "Based on the indexed documents, the strongest evidence points to these findings:"
        ]
        seen: set[str] = set()
        for _, sentence, source in selected:
            if sentence in seen:
                continue
            seen.add(sentence)
            page = f", page {source.page}" if source.page else ""
            lines.append(f"- {sentence} ({source.document_name}{page})")
        return "\n".join(lines)


def _sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    cleaned: list[str] = []
    for chunk in chunks:
        sentence = chunk.strip().lstrip("-* ")
        if sentence.startswith("#"):
            continue
        if len(sentence) > 20:
            cleaned.append(sentence)
    return cleaned
