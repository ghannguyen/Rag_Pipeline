"""Structure-aware block packing with a strict final token limit."""

import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List

from src.data.block_normalizer import Block
from src.data.paragraph_splitter import split_large_paragraph
from src.data.table_splitter import split_large_table
from src.data.token_counter import count_tokens


@dataclass
class PackedChunk:
    chunk_index: int
    section: str
    section_path: str
    content: str
    token_count: int
    page_start: int
    page_end: int
    doc_id: str = "default_doc"
    element_ids: List[str] = field(default_factory=list)
    element_types: List[str] = field(default_factory=list)
    table_ids: List[str] = field(default_factory=list)
    image_ids: List[str] = field(default_factory=list)
    image_paths: List[str] = field(default_factory=list)


SENTENCE_END = re.compile(r"[.!?;:]\s+")


def tail_text(text: str, max_tokens: int) -> str:
    """Last <= max_tokens tokens of text, cut on word (preferably sentence) boundaries."""
    words = text.split()
    tail = []
    for word in reversed(words):
        if count_tokens(" ".join([word] + tail)) > max_tokens:
            break
        tail.insert(0, word)

    if len(tail) == len(words):
        return text.strip()

    joined = " ".join(tail)
    # Start at a sentence boundary when that keeps at least half the overlap.
    match = SENTENCE_END.search(joined)
    if match:
        candidate = joined[match.end():]
        if count_tokens(candidate) * 2 >= max_tokens:
            return candidate
    return joined


def same_split_parent(a: Block, b: Block) -> bool:
    # split_large_paragraph names pieces <block_id>_sub_<n> and already overlaps them.
    return (
        "_sub_" in a.block_id
        and "_sub_" in b.block_id
        and a.block_id.rsplit("_sub_", 1)[0] == b.block_id.rsplit("_sub_", 1)[0]
    )


def make_overlap_block(previous: Block, following: Block, overlap_tokens: int):
    if overlap_tokens <= 0 or previous.block_type != "paragraph":
        return None
    if is_heading(previous) or same_split_parent(previous, following):
        return None

    tail = tail_text(previous.content, overlap_tokens)
    if not tail:
        return None

    return replace(
        previous,
        block_id=f"{previous.block_id}_overlap",
        content=tail,
        token_count=count_tokens(tail),
    )


MIN_SECTION_TOKENS = 50
MAX_SECTION_LABEL_TOKENS = 120


def is_heading(block: Block) -> bool:
    # block_normalizer renders heading elements as "### <text>" paragraph blocks.
    return block.block_type == "paragraph" and block.content.startswith("### ")


def is_tiny_group(group: List[Block]) -> bool:
    heading_only = all(is_heading(b) for b in group)
    return heading_only or sum(count_tokens(b.content) for b in group) < MIN_SECTION_TOKENS


def section_label(group: List[Block]) -> str:
    paths = list(dict.fromkeys(b.section_path for b in group))
    # Keep the most specific parts when a long heading chain would not fit.
    while len(paths) > 1 and count_tokens(" > ".join(paths)) > MAX_SECTION_LABEL_TOKENS:
        paths.pop(0)
    return " > ".join(paths)


def merge_tiny_groups(groups: List[List[Block]]) -> List[List[Block]]:
    """
    Docling headings come out flat, so "### Chương I" forms its own section
    right before "Điều 1". Carry heading-only / tiny sections forward into the
    next section of the same document instead of emitting them as chunks.
    """
    merged: List[List[Block]] = []
    pending: List[Block] = []

    def flush_pending():
        # No following section in this document: attach to the previous one.
        if merged and merged[-1][0].doc_id == pending[0].doc_id:
            merged[-1].extend(pending)
        else:
            merged.append(list(pending))

    for group in groups:
        if pending and pending[0].doc_id != group[0].doc_id:
            flush_pending()
            pending = []

        group = pending + group
        pending = []

        if is_tiny_group(group):
            pending = group
        else:
            merged.append(group)

    if pending:
        flush_pending()

    return merged


def pack_blocks(
    blocks: List[Block],
    target_tokens: int = 600,
    overlap_tokens: int = 100,
    hard_max_tokens: int = 800,
) -> List[PackedChunk]:
    if not blocks:
        return []

    if not (0 < target_tokens <= hard_max_tokens):
        raise ValueError("Invalid target_tokens or hard_max_tokens")

    groups = []
    current_group = []

    for block in blocks:
        if (
            current_group
            and (
                current_group[-1].doc_id != block.doc_id
                or current_group[-1].section_path != block.section_path
            )
        ):
            groups.append(current_group)
            current_group = []

        current_group.append(block)

    if current_group:
        groups.append(current_group)

    groups = merge_tiny_groups(groups)

    result = []

    for group in groups:
        section_path = section_label(group)
        section = group[-1].section
        prefix = f"Section: {section_path}\n---\n"

        prefix_tokens = count_tokens(prefix)
        available = hard_max_tokens - prefix_tokens - 16

        if available <= 0:
            raise ValueError(
                f"Section header too long: {section_path}"
            )

        effective_target = min(target_tokens, available)
        effective_overlap = min(overlap_tokens, effective_target - 1)

        processed = []

        for block in group:
            actual_tokens = count_tokens(block.content)
            block = replace(block, token_count=actual_tokens)

            if actual_tokens <= available:
                processed.append(block)
                continue

            if block.block_type == "paragraph":
                pieces = split_large_paragraph(
                    block,
                    target_tokens=effective_target,
                    overlap_tokens=effective_overlap,
                    hard_max_tokens=available,
                )
            elif block.block_type == "table":
                pieces = split_large_table(
                    block,
                    target_tokens=effective_target,
                    hard_max_tokens=available,
                )
            elif block.block_type == "image":
                # Preserve image metadata while splitting long captions.
                text_block = replace(
                    block,
                    block_type="paragraph",
                )
                split_texts = split_large_paragraph(
                    text_block,
                    target_tokens=effective_target,
                    overlap_tokens=effective_overlap,
                    hard_max_tokens=available,
                )
                pieces = [
                    replace(
                        piece,
                        block_type="image",
                        image_data=block.image_data,
                    )
                    for piece in split_texts
                ]
            else:
                raise ValueError(
                    f"Unsupported block type: {block.block_type}"
                )

            processed.extend(pieces)

        current = []

        def compose(selected):
            return prefix + "\n\n".join(b.content for b in selected)

        def emit(selected):
            if not selected:
                return

            content = compose(selected)
            tokens = count_tokens(content)

            if tokens > hard_max_tokens:
                raise ValueError(
                    f"Chunk exceeds {hard_max_tokens} tokens: "
                    f"{selected[0].block_id}, {tokens}"
                )

            pages = [b.page for b in selected if b.page]
            element_ids = list(
                dict.fromkeys(
                    element_id
                    for b in selected
                    for element_id in b.element_ids
                )
            )

            image_blocks = [
                b for b in selected if b.block_type == "image"
            ]

            result.append(
                PackedChunk(
                    chunk_index=len(result) + 1,
                    section=section,
                    section_path=section_path,
                    content=content,
                    token_count=tokens,
                    page_start=min(pages) if pages else 1,
                    page_end=max(pages) if pages else 1,
                    doc_id=selected[0].doc_id,
                    element_ids=element_ids,
                    element_types=sorted(
                        set(b.block_type for b in selected)
                    ),
                    table_ids=[
                        b.block_id
                        for b in selected
                        if b.block_type == "table"
                    ],
                    image_ids=[
                        b.block_id for b in image_blocks
                    ],
                    image_paths=list(
                        dict.fromkeys(
                            b.image_data["image_path"]
                            for b in image_blocks
                            if b.image_data
                            and b.image_data.get("image_path")
                        )
                    ),
                )
            )

        # Pack up to target_tokens; the gap up to hard_max_tokens holds the
        # overlap carried over from the previous chunk of the same section.
        for block in processed:
            if not current:
                current = [block]
            elif count_tokens(compose(current + [block])) <= target_tokens:
                current = current + [block]
            else:
                # A heading belongs with the content after it, not at a chunk's end.
                split_at = len(current)
                while split_at > 0 and is_heading(current[split_at - 1]):
                    split_at -= 1
                headings = current[split_at:]

                if headings and count_tokens(compose(headings + [block])) <= hard_max_tokens:
                    emit(current[:split_at])
                    current = headings + [block]
                else:
                    emit(current)
                    previous = current[-1]
                    current = [block]

                    overlap = make_overlap_block(previous, block, effective_overlap)
                    if overlap and count_tokens(compose([overlap, block])) <= hard_max_tokens:
                        current = [overlap, block]

            if count_tokens(compose(current)) > hard_max_tokens:
                raise ValueError(
                    f"Single block exceeds hard limit: {block.block_id}"
                )

        emit(current)

    return result
