"""Split oversized paragraphs while preserving source metadata."""

from dataclasses import replace
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data.block_normalizer import Block
from src.data.token_counter import count_tokens


def split_large_paragraph(
    block: Block,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
    hard_max_tokens: int = 800,
) -> List[Block]:
    if count_tokens(block.content) <= hard_max_tokens:
        return [block]

    if target_tokens <= 0 or hard_max_tokens <= 0:
        raise ValueError("Token limits must be positive")

    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be smaller than target_tokens")

    # Use the actual tokenizer rather than estimating characters per token.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=min(target_tokens, hard_max_tokens),
        chunk_overlap=overlap_tokens,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        keep_separator=True,
    )

    parts = splitter.split_text(block.content)

    result = []
    for i, part in enumerate(parts, 1):
        if not part.strip():
            continue

        tokens = count_tokens(part)
        if tokens > hard_max_tokens:
            raise ValueError(
                f"Paragraph split failed: {block.block_id}, {tokens} tokens"
            )

        result.append(
            replace(
                block,
                block_id=f"{block.block_id}_sub_{i}",
                content=part,
                token_count=tokens,
                element_ids=list(block.element_ids),
            )
        )

    if not result:
        raise ValueError(f"Empty paragraph after splitting: {block.block_id}")

    return result
