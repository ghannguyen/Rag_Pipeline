"""Row-aware table splitting with strict token limits."""

from dataclasses import replace
from typing import List

from src.data.block_normalizer import Block
from src.data.token_counter import count_tokens


def split_large_table(
    block: Block,
    target_tokens: int = 600,
    hard_max_tokens: int = 800,
) -> List[Block]:
    if count_tokens(block.content) <= hard_max_tokens:
        return [block]

    lines = block.content.splitlines()
    header = []
    rows = []
    table_header_count = 0
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_row = stripped.startswith("|") and stripped.count("|") >= 2

        if is_table_row:
            in_table = True
            if table_header_count < 2:
                header.append(line)
                table_header_count += 1
            else:
                rows.append(line)
        elif not in_table:
            header.append(line)
        else:
            rows.append(line)

    if not rows:
        raise ValueError(
            f"Cannot safely split table without rows: {block.block_id}"
        )

    prefix = "\n".join(header).rstrip()
    if prefix:
        prefix += "\n"

    if count_tokens(prefix) >= hard_max_tokens:
        raise ValueError(
            f"Table header exceeds token budget: {block.block_id}"
        )

    chunks = []
    current = []

    def build_text(selected_rows):
        return prefix + "\n".join(selected_rows)

    def emit():
        if not current:
            return
        content = build_text(current)
        tokens = count_tokens(content)
        if tokens > hard_max_tokens:
            raise ValueError(
                f"Table chunk exceeds limit: {block.block_id}, {tokens}"
            )
        chunks.append(
            replace(
                block,
                block_id=f"{block.block_id}_sub_{len(chunks) + 1}",
                content=content,
                token_count=tokens,
                element_ids=list(block.element_ids),
            )
        )

    for row in rows:
        candidate = current + [row]

        if count_tokens(build_text(candidate)) <= min(
            target_tokens, hard_max_tokens
        ):
            current = candidate
            continue

        if current:
            emit()
            current = []

        if count_tokens(build_text([row])) > hard_max_tokens:
            raise ValueError(
                f"Single table row exceeds limit: {block.block_id}"
            )

        current = [row]

    emit()

    if not chunks:
        raise ValueError(f"No chunks generated for table: {block.block_id}")

    return chunks
