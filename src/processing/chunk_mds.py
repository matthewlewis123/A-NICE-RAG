# --- Imports ---

import glob
import json
import os
import re
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoTokenizer


class MarkdownChunker:
    """Chunks NICE guideline Markdown files using a hierarchical approach"""

    def __init__(self, max_tokens: int = 500, min_tokens: int = 200, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("voyageai/voyage-3-large")

    def extract_guideline_number(self, file_path: str) -> str:
        """Extract the guideline number from the file path."""

        filename = os.path.basename(file_path)
        match = re.match(r"^([A-Z]{1,3}\d+)", filename)
        if match:
            return match.group(1)
        return os.path.splitext(filename)[0]

    def count_tokens(self, text: str) -> int:
        """Count tokens using the voyage tokenizer."""

        return len(self.tokenizer.encode(text))

    def parse_headings(self, content: str) -> List[Dict]:
        """Parse markdown content to extract headings and their levels."""

        heading_pattern = r"^(#{1,6})\s+(.+)$"
        headings = []

        lines = content.split("\n")
        for i, line in enumerate(lines):
            match = re.match(heading_pattern, line.strip())
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                headings.append(
                    {"level": level, "title": title, "line_number": i, "full_line": line}
                )
        return headings

    def extract_section_content(
        self, content: str, start_heading: Dict, next_heading: Dict = None
    ) -> str:
        """Extract content between two headings."""

        lines = content.split("\n")

        if next_heading:
            section_lines = lines[start_heading["line_number"] : next_heading["line_number"]]
        else:
            section_lines = lines[start_heading["line_number"] :]

        return "\n".join(section_lines).strip()

    def is_priority_section(self, title: str) -> bool:
        """Check if a section should be prioritized as a primary chunk boundary."""

        priority_sections = [
            "overview",
            "background",
            "context",
            "terms used in this guideline",
            "update information",
            "introduction",
            "scope",
            "methodology",
        ]
        return title.lower() in priority_sections

    def should_keep_table_with_context(self, content: str) -> bool:
        """Check if content contains tables that should be kept with surrounding text."""

        return bool(re.search(r"\|.*\|.*\|", content, re.MULTILINE))

    def get_top_level_sections(self, headings: List[Dict]) -> List[Dict]:
        """Get top-level sections (H1 and H2) for initial chunking."""

        return [h for h in headings if h["level"] <= 2]

    def find_optimal_split_points(self, content: str) -> List[Dict]:
        """Find optimal split points in content, prioritizing subsections, then paragraphs."""

        lines = content.split("\n")
        split_points = []

        for i, line in enumerate(lines):
            if re.match(r"^#{3,6}\s+", line.strip()):
                split_points.append(
                    {"line_number": i, "priority": 1, "type": "subsection", "content": line.strip()}
                )

            elif line.strip() == "" and i > 0 and i < len(lines) - 1:
                if lines[i - 1].strip() != "" and lines[i + 1].strip() != "":
                    split_points.append(
                        {"line_number": i, "priority": 2, "type": "paragraph", "content": ""}
                    )

        return split_points

    def get_overlap_content(
        self, lines: List[str], split_point: int, direction: str = "before"
    ) -> str:
        """Get overlap content from around a split point."""

        if direction == "before":
            text_before = "\n".join(lines[max(0, split_point - 10) : split_point])
            sentences = re.split(r"(?<=[.!?])\s+", text_before)
            overlap_sentences = sentences[-3:] if len(sentences) >= 3 else sentences
            return " ".join(overlap_sentences).strip()

        else:
            text_after = "\n".join(lines[split_point : min(len(lines), split_point + 10)])
            sentences = re.split(r"(?<=[.!?])\s+", text_after)
            overlap_sentences = sentences[:3] if len(sentences) >= 3 else sentences
            return " ".join(overlap_sentences).strip()

    def smart_split_content(self, content: str, base_title: str) -> List[Dict]:
        """Split content using optimal split points and balanced token distribution."""

        lines = content.split("\n")
        total_tokens = self.count_tokens(content)

        target_chunks = max(2, (total_tokens + self.max_tokens - 1) // self.max_tokens)
        target_tokens_per_chunk = total_tokens // target_chunks

        split_points = self.find_optimal_split_points(content)

        if not split_points:
            return self.split_long_content_fallback(content, base_title)

        line_tokens = []
        cumulative_tokens = 0

        for line in lines:
            line_token_count = self.count_tokens(line)
            line_tokens.append(line_token_count)
            cumulative_tokens += line_token_count

        selected_splits = []
        current_tokens = 0

        for i in range(1, target_chunks):
            target_position = i * target_tokens_per_chunk

            best_split = None
            best_score = float("inf")

            for split_point in split_points:
                split_tokens = sum(line_tokens[: split_point["line_number"]])

                if split_tokens - current_tokens < 50:
                    continue

                distance_score = abs(split_tokens - target_position)
                priority_score = split_point["priority"] * 50
                balance_score = abs(split_tokens - current_tokens - target_tokens_per_chunk)

                total_score = distance_score + priority_score + balance_score

                if total_score < best_score:
                    best_score = total_score
                    best_split = split_point

            if best_split:
                selected_splits.append(best_split)
                current_tokens = sum(line_tokens[: best_split["line_number"]])

        chunks = []
        start_line = 0

        for i, split_point in enumerate(selected_splits + [None]):
            if split_point:
                end_line = split_point["line_number"]
            else:
                end_line = len(lines)

            chunk_lines = lines[start_line:end_line]
            chunk_content = "\n".join(chunk_lines).strip()

            if not chunk_content or self.count_tokens(chunk_content) < 10:
                start_line = end_line
                continue

            if i > 0 and self.overlap_tokens > 0:
                overlap_before = self.get_overlap_content(lines, start_line, "before")
                if overlap_before and self.count_tokens(overlap_before) <= self.overlap_tokens:
                    chunk_content = f"[...{overlap_before}]\n\n{chunk_content}"

            if split_point and self.overlap_tokens > 0:
                overlap_after = self.get_overlap_content(lines, end_line, "after")
                if overlap_after and self.count_tokens(overlap_after) <= self.overlap_tokens:
                    chunk_content = f"{chunk_content}\n\n[...{overlap_after}]"

            chunk_tokens = self.count_tokens(chunk_content)
            part_number = len(chunks) + 1

            chunk_title = (
                f"{base_title} - Part {part_number}" if len(selected_splits) > 0 else base_title
            )

            chunks.append(
                {
                    "title": chunk_title,
                    "content": chunk_content,
                    "tokens": chunk_tokens,
                    "part_of_split": part_number > 1,
                    "split_type": split_point["type"] if split_point else "end",
                    "part_number": part_number,
                    "total_parts": len(selected_splits) + 1,
                }
            )

            start_line = end_line

        if not chunks:
            return self.split_long_content_fallback(content, base_title)

        for chunk in chunks:
            chunk["total_parts"] = len(chunks)

        return chunks

    def split_long_content_fallback(self, content: str, base_title: str) -> List[Dict]:
        """Fallback splitting method when no good split points are found."""

        chunks = []
        paragraphs = content.split("\n\n")
        current_chunk = []
        current_tokens = 0
        part_number = 1

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            para_tokens = self.count_tokens(paragraph)

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                chunk_content = "\n\n".join(current_chunk)

                if part_number > 1 and self.overlap_tokens > 0:
                    overlap = (
                        current_chunk[-1][-100:]
                        if len(current_chunk[-1]) > 100
                        else current_chunk[-1]
                    )
                    if self.count_tokens(overlap) <= self.overlap_tokens:
                        chunk_content = f"[...{overlap}]\n\n{chunk_content}"

                chunk_title = (
                    f"{base_title} - Part {part_number}" if part_number > 1 else base_title
                )

                chunks.append(
                    {
                        "title": chunk_title,
                        "content": chunk_content,
                        "tokens": current_tokens,
                        "part_of_split": part_number > 1,
                        "split_type": "paragraph",
                        "part_number": part_number,
                    }
                )

                current_chunk = []
                current_tokens = 0
                part_number += 1

            current_chunk.append(paragraph)
            current_tokens += para_tokens

        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            chunk_title = f"{base_title} - Part {part_number}" if part_number > 1 else base_title

            chunks.append(
                {
                    "title": chunk_title,
                    "content": chunk_content,
                    "tokens": current_tokens,
                    "part_of_split": part_number > 1,
                    "split_type": "paragraph",
                    "part_number": part_number,
                }
            )

        return chunks

    def merge_consecutive_small_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Merge consecutive small chunks together while respecting max token limit."""

        if not chunks:
            return chunks

        merged_chunks = []

        with tqdm(total=len(chunks), desc="Merging consecutive small chunks", leave=False) as pbar:
            i = 0
            while i < len(chunks):
                current_chunk = chunks[i].copy()

                if current_chunk["tokens"] < self.min_tokens:
                    chunks_to_merge = [current_chunk]
                    total_tokens = current_chunk["tokens"]
                    j = i + 1

                    while j < len(chunks) and total_tokens < self.max_tokens:
                        next_chunk = chunks[j]

                        if total_tokens + next_chunk["tokens"] > self.max_tokens:
                            break

                        if next_chunk["tokens"] < self.min_tokens or total_tokens < self.min_tokens:
                            chunks_to_merge.append(next_chunk)
                            total_tokens += next_chunk["tokens"]
                            j += 1
                        else:
                            if total_tokens >= self.min_tokens:
                                break
                            chunks_to_merge.append(next_chunk)
                            total_tokens += next_chunk["tokens"]
                            j += 1

                    if len(chunks_to_merge) > 1:
                        merged_content = "\n\n".join(
                            [chunk["content"] for chunk in chunks_to_merge]
                        )
                        merged_titles = [chunk["title"] for chunk in chunks_to_merge]

                        title_parts = []
                        source_prefix = None
                        for title in merged_titles:
                            if "_" in title:
                                source, title_part = title.split("_", 1)
                                if source_prefix is None:
                                    source_prefix = source
                                title_parts.append(title_part)
                            else:
                                title_parts.append(title)

                        if len(title_parts) <= 3:
                            merged_title_part = " & ".join(title_parts)
                        else:
                            merged_title_part = (
                                f"{title_parts[0]} & {len(title_parts)-1} more sections"
                            )

                        if source_prefix:
                            merged_title = f"{source_prefix}_{merged_title_part}"
                        else:
                            merged_title = merged_title_part

                        min_heading_level = min(
                            chunk.get("heading_level", 6) for chunk in chunks_to_merge
                        )

                        merged_chunk = {
                            "title": merged_title,
                            "content": merged_content,
                            "tokens": total_tokens,
                            "source_file": current_chunk["source_file"],
                            "source": current_chunk["source"],
                            "heading_level": min_heading_level,
                            "merged_sections": merged_titles,
                            "merged_count": len(chunks_to_merge),
                        }

                        parent_sections = [
                            chunk.get("parent_section")
                            for chunk in chunks_to_merge
                            if "parent_section" in chunk
                        ]
                        if parent_sections and all(
                            ps == parent_sections[0] for ps in parent_sections
                        ):
                            merged_chunk["parent_section"] = parent_sections[0]

                        merged_chunks.append(merged_chunk)
                        pbar.update(len(chunks_to_merge))
                        i = j
                    else:
                        merged_chunks.append(current_chunk)
                        pbar.update(1)
                        i += 1

                else:
                    if i + 1 < len(chunks):
                        next_chunk = chunks[i + 1]
                        if (
                            next_chunk["tokens"] < self.min_tokens
                            and current_chunk["tokens"] + next_chunk["tokens"] <= self.max_tokens
                        ):

                            merged_content = (
                                current_chunk["content"] + "\n\n" + next_chunk["content"]
                            )
                            current_title = current_chunk["title"]
                            next_title = next_chunk["title"]

                            if "_" in current_title:
                                source_prefix, current_title_part = current_title.split("_", 1)
                            else:
                                source_prefix = current_chunk["source"]
                                current_title_part = current_title

                            if "_" in next_title:
                                _, next_title_part = next_title.split("_", 1)
                            else:
                                next_title_part = next_title

                            merged_title = (
                                f"{source_prefix}_{current_title_part} & {next_title_part}"
                            )

                            current_chunk["title"] = merged_title
                            current_chunk["content"] = merged_content
                            current_chunk["tokens"] = current_chunk["tokens"] + next_chunk["tokens"]
                            current_chunk["merged_sections"] = [current_title, next_title]
                            current_chunk["merged_count"] = 2

                            merged_chunks.append(current_chunk)
                            pbar.update(2)
                            i += 2
                        else:
                            merged_chunks.append(current_chunk)
                            pbar.update(1)
                            i += 1
                    else:
                        merged_chunks.append(current_chunk)
                        pbar.update(1)
                        i += 1

        return merged_chunks

    def chunk_markdown_file(self, file_path: str) -> List[Dict]:
        """Main chunking method that processes a single markdown file."""

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        headings = self.parse_headings(content)
        chunks = []
        guideline_number = self.extract_guideline_number(file_path)

        if not headings:
            tokens = self.count_tokens(content)
            if tokens <= self.max_tokens:
                title = os.path.basename(file_path).replace(".md", "")
                chunks.append(
                    {
                        "title": f"{guideline_number}_{title}",
                        "content": content,
                        "tokens": tokens,
                        "source_file": file_path,
                        "source": guideline_number,
                    }
                )
            else:
                split_chunks = self.smart_split_content(
                    content, os.path.basename(file_path).replace(".md", "")
                )
                for chunk in split_chunks:
                    chunk["source_file"] = file_path
                    chunk["source"] = guideline_number
                    chunk["title"] = f"{guideline_number}_{chunk['title']}"
                    chunks.append(chunk)
            return chunks

        top_level_headings = self.get_top_level_sections(headings)

        with tqdm(top_level_headings, desc=f"Processing top-level sections", leave=False) as pbar:
            for i, heading in enumerate(pbar):
                pbar.set_postfix(
                    {
                        "section": (
                            heading["title"][:30] + "..."
                            if len(heading["title"]) > 30
                            else heading["title"]
                        )
                    }
                )

                next_heading = (
                    top_level_headings[i + 1] if i + 1 < len(top_level_headings) else None
                )
                section_content = self.extract_section_content(content, heading, next_heading)

                tokens = self.count_tokens(section_content)

                base_title = f"{heading['title']}"
                unique_title = f"{guideline_number}_{base_title}"

                if self.is_priority_section(heading["title"]):
                    if tokens <= self.max_tokens:
                        chunks.append(
                            {
                                "title": unique_title,
                                "content": section_content,
                                "tokens": tokens,
                                "source_file": file_path,
                                "source": guideline_number,
                                "heading_level": heading["level"],
                            }
                        )
                    else:
                        split_chunks = self.smart_split_content(section_content, base_title)
                        for chunk in split_chunks:
                            chunk["source_file"] = file_path
                            chunk["source"] = guideline_number
                            chunk["heading_level"] = heading["level"]
                            chunk["title"] = f"{guideline_number}_{chunk['title']}"
                            chunks.append(chunk)

                elif tokens <= self.max_tokens:
                    chunks.append(
                        {
                            "title": unique_title,
                            "content": section_content,
                            "tokens": tokens,
                            "source_file": file_path,
                            "source": guideline_number,
                            "heading_level": heading["level"],
                        }
                    )
                else:
                    sub_headings = []
                    section_lines = section_content.split("\n")

                    for j, line in enumerate(section_lines):
                        sub_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
                        if sub_match and len(sub_match.group(1)) > heading["level"]:
                            sub_headings.append(
                                {
                                    "level": len(sub_match.group(1)),
                                    "title": sub_match.group(2).strip(),
                                    "line_number": j,
                                    "full_line": line,
                                }
                            )

                    if sub_headings:
                        child_level = min(sub_heading["level"] for sub_heading in sub_headings)
                        direct_children = [sh for sh in sub_headings if sh["level"] == child_level]

                        for k, sub_heading in enumerate(direct_children):
                            next_sub_heading = (
                                direct_children[k + 1] if k + 1 < len(direct_children) else None
                            )

                            if next_sub_heading:
                                sub_content_lines = section_lines[
                                    sub_heading["line_number"] : next_sub_heading["line_number"]
                                ]
                            else:
                                sub_content_lines = section_lines[sub_heading["line_number"] :]

                            sub_content = "\n".join(sub_content_lines).strip()
                            sub_tokens = self.count_tokens(sub_content)

                            sub_base_title = f"{heading['title']}: {sub_heading['title']}"
                            sub_unique_title = f"{guideline_number}_{sub_base_title}"

                            if sub_tokens <= self.max_tokens:
                                chunks.append(
                                    {
                                        "title": sub_unique_title,
                                        "content": sub_content,
                                        "tokens": sub_tokens,
                                        "source_file": file_path,
                                        "source": guideline_number,
                                        "heading_level": sub_heading["level"],
                                        "parent_section": heading["title"],
                                    }
                                )
                            else:
                                split_chunks = self.smart_split_content(sub_content, sub_base_title)
                                for chunk in split_chunks:
                                    chunk["source_file"] = file_path
                                    chunk["source"] = guideline_number
                                    chunk["heading_level"] = sub_heading["level"]
                                    chunk["parent_section"] = heading["title"]
                                    chunk["title"] = f"{guideline_number}_{chunk['title']}"
                                    chunks.append(chunk)
                    else:
                        split_chunks = self.smart_split_content(section_content, base_title)
                        for chunk in split_chunks:
                            chunk["source_file"] = file_path
                            chunk["source"] = guideline_number
                            chunk["heading_level"] = heading["level"]
                            chunk["title"] = f"{guideline_number}_{chunk['title']}"
                            chunks.append(chunk)

        unique_chunks = []
        seen_content = set()

        for chunk in chunks:
            content_preview = chunk["content"][:500].strip()
            content_words = set(content_preview.lower().split())

            is_duplicate = False
            for existing_content in seen_content:
                existing_words = set(existing_content.lower().split())

                intersection = content_words.intersection(existing_words)
                union = content_words.union(existing_words)

                if len(union) > 0:
                    similarity = len(intersection) / len(union)
                    if similarity > 0.85:
                        is_duplicate = True
                        break

            if not is_duplicate:
                seen_content.add(content_preview)
                unique_chunks.append(chunk)

        merged_chunks = self.merge_consecutive_small_chunks(unique_chunks)

        return merged_chunks

    def chunk_all_markdown_files(self, input_dir: str, output_file: str = None) -> List[Dict]:
        """Process all markdown files in the input directory."""

        if not os.path.exists(input_dir):
            print(f"Error: Input directory '{input_dir}' does not exist.")
            return []

        md_pattern = os.path.join(input_dir, "*.md")
        md_files = glob.glob(md_pattern)

        if not md_files:
            print(f"No markdown files found in '{input_dir}'.")
            return []

        print(f"Found {len(md_files)} markdown files to chunk.")

        all_chunks = []

        with tqdm(md_files, desc="Processing markdown files", unit="file") as file_pbar:
            for md_file in file_pbar:
                file_pbar.set_postfix({"file": os.path.basename(md_file)})

                try:
                    file_chunks = self.chunk_markdown_file(md_file)
                    all_chunks.extend(file_chunks)

                    tokens = [chunk["tokens"] for chunk in file_chunks]
                    merged_count = sum(1 for chunk in file_chunks if "merged_sections" in chunk)
                    split_count = sum(
                        1 for chunk in file_chunks if chunk.get("part_of_split", False)
                    )

                    file_pbar.set_postfix(
                        {
                            "chunks": len(file_chunks),
                            "merged": merged_count,
                            "split": split_count,
                            "avg_tokens": f"{sum(tokens)/len(tokens):.0f}" if tokens else "0",
                        }
                    )

                except Exception as e:
                    file_pbar.set_postfix({"error": str(e)[:30]})
                    print(f"\n Error processing {md_file}: {e}")

        if output_file:
            print(f"\nSaving chunks to {output_file}...")
            with tqdm(total=1, desc="Saving JSON file") as save_pbar:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(all_chunks, f, indent=2, ensure_ascii=False)
                save_pbar.update(1)
            print(f"Saved {len(all_chunks)} chunks to {output_file}")

        print(f"\n{'='*60}")
        print("CHUNKING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {len(md_files)}")
        print(f"Total chunks created: {len(all_chunks)}")

        if all_chunks:
            tokens = [chunk["tokens"] for chunk in all_chunks]
            print(f"Token statistics:")
            print(f"  - Range: {min(tokens)}-{max(tokens)}")
            print(f"  - Average: {sum(tokens)/len(tokens):.1f}")
            print(f"  - Median: {sorted(tokens)[len(tokens)//2]}")

            priority_chunks = sum(1 for chunk in all_chunks if chunk.get("heading_level", 0) <= 2)
            split_chunks = sum(1 for chunk in all_chunks if chunk.get("part_of_split", False))
            merged_chunks = sum(1 for chunk in all_chunks if "merged_sections" in chunk)
            multi_merged_chunks = sum(1 for chunk in all_chunks if chunk.get("merged_count", 0) > 2)

            priority_only = sum(
                1
                for chunk in all_chunks
                if chunk.get("heading_level", 0) <= 2
                and not chunk.get("part_of_split", False)
                and "merged_sections" not in chunk
            )

            split_only = sum(
                1
                for chunk in all_chunks
                if chunk.get("part_of_split", False)
                and chunk.get("heading_level", 0) > 2
                and "merged_sections" not in chunk
            )

            merged_only = sum(
                1
                for chunk in all_chunks
                if "merged_sections" in chunk
                and not chunk.get("part_of_split", False)
                and chunk.get("heading_level", 0) > 2
            )

            priority_split = sum(
                1
                for chunk in all_chunks
                if chunk.get("heading_level", 0) <= 2
                and chunk.get("part_of_split", False)
                and "merged_sections" not in chunk
            )

            priority_merged = sum(
                1
                for chunk in all_chunks
                if chunk.get("heading_level", 0) <= 2
                and "merged_sections" in chunk
                and not chunk.get("part_of_split", False)
            )

            split_merged = sum(
                1
                for chunk in all_chunks
                if chunk.get("part_of_split", False)
                and "merged_sections" in chunk
                and chunk.get("heading_level", 0) > 2
            )

            priority_split_merged = sum(
                1
                for chunk in all_chunks
                if chunk.get("heading_level", 0) <= 2
                and chunk.get("part_of_split", False)
                and "merged_sections" in chunk
            )

            regular_chunks = (
                len(all_chunks)
                - priority_only
                - split_only
                - merged_only
                - priority_split
                - priority_merged
                - split_merged
                - priority_split_merged
            )

            print(f"Chunk types (overlapping categories):")
            print(f"  - Priority sections: {priority_chunks}")
            print(f"  - Split chunks: {split_chunks}")
            print(f"  - Merged chunks: {merged_chunks}")
            print(f"  - Multi-merged chunks (3+ sections): {multi_merged_chunks}")

            print(f"\nChunk types (non-overlapping breakdown):")
            print(f"  - Priority only: {priority_only}")
            print(f"  - Split only: {split_only}")
            print(f"  - Merged only: {merged_only}")
            print(f"  - Priority + Split: {priority_split}")
            print(f"  - Priority + Merged: {priority_merged}")
            print(f"  - Split + Merged: {split_merged}")
            print(f"  - Priority + Split + Merged: {priority_split_merged}")
            print(f"  - Regular chunks: {regular_chunks}")
            print(
                f"  - Total: {priority_only + split_only + merged_only + priority_split + priority_merged + split_merged + priority_split_merged + regular_chunks}"
            )

            small_chunks = sum(1 for chunk in all_chunks if chunk["tokens"] < 200)
            medium_chunks = sum(1 for chunk in all_chunks if 200 <= chunk["tokens"] < 600)
            large_chunks = sum(1 for chunk in all_chunks if chunk["tokens"] >= 600)

            print(f"Token distribution:")
            print(f"  - Small (<200): {small_chunks}")
            print(f"  - Medium (200-599): {medium_chunks}")
            print(f"  - Large (≥600): {large_chunks}")

            if split_chunks > 0:
                split_by_type = {}
                for chunk in all_chunks:
                    if chunk.get("part_of_split", False):
                        split_type = chunk.get("split_type", "unknown")
                        split_by_type[split_type] = split_by_type.get(split_type, 0) + 1

                print(f"Split types:")
                for split_type, count in split_by_type.items():
                    print(f"  - {split_type}: {count}")

            if merged_chunks > 0:
                avg_merge_count = (
                    sum(
                        chunk.get("merged_count", 1)
                        for chunk in all_chunks
                        if "merged_sections" in chunk
                    )
                    / merged_chunks
                )
                print(f"Merging efficiency:")
                print(f"  - Average sections per merged chunk: {avg_merge_count:.1f}")

        return all_chunks


def main():
    """Main function to run the chunking process."""

    input_dir = "data/NICE_Guidelines_MD"
    output_file = "data/chunked_guidelines.json"

    chunker = MarkdownChunker(max_tokens=600, min_tokens=200, overlap_tokens=50)
    chunks = chunker.chunk_all_markdown_files(input_dir, output_file)

    return chunks


if __name__ == "__main__":
    main()
