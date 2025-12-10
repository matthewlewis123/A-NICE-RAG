# --- Imports ---
import glob
import os
import re
import xml.etree.ElementTree as ET
from html import unescape
from xml.etree.ElementTree import ParseError


class FinalXmlToMarkdownConverter:
    """A class to convert NICE guideline XML files into Markdown."""

    def __init__(self):
        self.markdown_content = []

    def _clean_text(self, text):
        """Clean up whitespace and decode entities in a string."""

        if not text:
            return ""
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _html_to_markdown(self, html_content, section_title_to_remove=None):
        """Converts a string of HTML content into Markdown format."""

        if not html_content:
            return ""

        html_content = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", html_content, flags=re.DOTALL)

        if section_title_to_remove:
            escaped_title = re.escape(self._clean_text(section_title_to_remove))
            html_content = re.sub(
                r"<h\d[^>]*>\s*" + escaped_title + r"\s*</h\d>",
                "",
                html_content,
                flags=re.IGNORECASE,
            )

        html_content = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n\n", html_content, flags=re.DOTALL)
        html_content = re.sub(
            r'<h(\d)[^>]*class="recommendation__number"[^>]*>(.*?)</h\d>',
            r"\n\n#### \2\n\n",
            html_content,
            flags=re.DOTALL,
        )
        html_content = re.sub(
            r"<h(\d)[^>]*>(.*?)</h\d>",
            lambda m: "\n\n" + "#" * int(m.group(1)) + " " + self._clean_text(m.group(2)) + "\n\n",
            html_content,
            flags=re.DOTALL,
        )
        html_content = re.sub(
            r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r"[\2](\1)", html_content, flags=re.DOTALL
        )
        html_content = re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", html_content, flags=re.DOTALL)
        html_content = re.sub(r"<(em|i)>(.*?)</\1>", r"*\2*", html_content, flags=re.DOTALL)
        html_content = re.sub(r"<br\s*/?>", "\n", html_content)
        html_content = re.sub(
            r"<li[^>]*>(.*?)</li>",
            lambda m: "- "
            + self._clean_text(re.sub(r"<p[^>]*>(.*?)</p>", r"\1", m.group(1)))
            + "\n",
            html_content,
            flags=re.DOTALL,
        )
        html_content = re.sub(r"</(ul|ol)>", "\n", html_content)

        html_content = self._convert_html_tables_to_markdown(html_content)

        html_content = re.sub(r"<[^>]+>", "", html_content)

        html_content = re.sub(r"(\n\s*){3,}", "\n\n", html_content)
        return html_content.strip()

    def _convert_html_tables_to_markdown(self, html_content):
        """Convert HTML tables to properly formatted Markdown tables."""

        table_pattern = r"<table[^>]*>(.*?)</table>"

        def convert_single_table(match):
            table_html = match.group(1)

            caption_match = re.search(r"<caption[^>]*>(.*?)</caption>", table_html, re.DOTALL)
            caption_text = ""
            if caption_match:
                caption_content = self._clean_text(re.sub(r"<[^>]+>", "", caption_match.group(1)))
                caption_text = f"\n**{caption_content}**\n"
                table_html = table_html.replace(caption_match.group(0), "")

            table_html = re.sub(r"</?tbody[^>]*>", "", table_html)
            table_html = re.sub(r"</?thead[^>]*>", "", table_html)
            table_html = re.sub(r"</?tfoot[^>]*>", "", table_html)

            row_pattern = r"<tr[^>]*>(.*?)</tr>"
            rows = re.findall(row_pattern, table_html, re.DOTALL)

            if not rows:
                return "\n\n*[Table content could not be parsed]*\n\n"

            markdown_rows = []

            for i, row_html in enumerate(rows):
                cell_pattern = r"<(?:th|td)[^>]*>(.*?)</(?:th|td)>"
                cells = re.findall(cell_pattern, row_html, re.DOTALL)

                if not cells:
                    continue

                cleaned_cells = []
                for cell in cells:
                    cell_content = re.sub(r"<p[^>]*>(.*?)</p>", r"\1", cell, flags=re.DOTALL)
                    cell_content = re.sub(r"<[^>]+>", "", cell_content)
                    cell_content = self._clean_text(cell_content)
                    cell_content = re.sub(r"\s+", " ", cell_content).strip()
                    cell_content = cell_content.replace("|", "\\|")
                    cleaned_cells.append(cell_content)

                markdown_row = "| " + " | ".join(cleaned_cells) + " |"
                markdown_rows.append(markdown_row)

                if i == 0:
                    separator = "|" + "---|" * len(cleaned_cells)
                    markdown_rows.append(separator)

            return caption_text + "\n".join(markdown_rows) + "\n\n"

        html_content = re.sub(table_pattern, convert_single_table, html_content, flags=re.DOTALL)

        return html_content

    def _process_node(self, node, level):
        """Recursively processes an XML node and its children."""

        title_node = node.find("Title")
        content_node = node.find("Content")
        sections_node = node.find("Sections")

        title = (
            self._clean_text(title_node.text) if title_node is not None and title_node.text else ""
        )

        if title:
            self.markdown_content.append(f"{'#' * level} {title}\n\n")

        if content_node is not None and content_node.text:
            markdown_text = self._html_to_markdown(content_node.text, section_title_to_remove=title)
            if markdown_text:
                self.markdown_content.append(f"{markdown_text}\n\n")

        if sections_node is not None:
            for section in sections_node:
                self._process_node(section, level + 1)

    def convert(self, xml_file_path, output_file_path):
        """Converts the entire XML file to a Markdown file."""

        self.markdown_content = []
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        doc_title = root.find("Title")
        if doc_title is not None and doc_title.text:
            self.markdown_content.append(f"# {self._clean_text(doc_title.text)}\n\n")

        guidance_num = root.find("GuidanceNumber")
        if guidance_num is not None and guidance_num.text:
            self.markdown_content.append(
                f"**Guidance Number:** {self._clean_text(guidance_num.text)}\n"
            )

        last_mod = root.find("LastModified")
        if last_mod is not None and last_mod.text:
            self.markdown_content.append(
                f"**Last Modified:** {self._clean_text(last_mod.text)}\n\n"
            )

        chapters_node = root.find("Chapters")
        if chapters_node is not None:
            for chapter in chapters_node.findall("Chapter"):
                self._process_node(chapter, level=2)

        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        final_text = "".join(self.markdown_content)
        final_text = re.sub(r"\n{3,}", "\n\n", final_text).strip()

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(final_text)


def convert_all_xmls_to_markdown():
    """Converts all XML files in the NICE_Guidelines_XML directory to Markdown files."""

    xml_input_dir = "../../data/NICE_Guidelines_XML"
    md_output_dir = "../../data/NICE_Guidelines_MD"

    os.makedirs(md_output_dir, exist_ok=True)

    if not os.path.exists(xml_input_dir):
        print(f"Error: Input directory '{xml_input_dir}' does not exist.")
        print("Please run the download script first to get the XML files.")
        return

    xml_pattern = os.path.join(xml_input_dir, "*.xml")
    xml_files = glob.glob(xml_pattern)

    if not xml_files:
        print(f"No XML files found in '{xml_input_dir}'.")
        return

    print(f"Found {len(xml_files)} XML files to convert.")

    converter = FinalXmlToMarkdownConverter()

    successful_conversions = 0
    failed_conversions = 0

    for i, xml_file in enumerate(xml_files, 1):
        filename = os.path.basename(xml_file)
        base_name = os.path.splitext(filename)[0]

        md_filename = f"{base_name}.md"
        md_filepath = os.path.join(md_output_dir, md_filename)

        print(f"\n--- Converting {i}/{len(xml_files)}: {filename} ---")

        try:
            converter.convert(xml_file, md_filepath)
            print(f"Successfully converted to {md_filepath}")
            successful_conversions += 1

        except ParseError as e:
            print(f"XML parsing error in {filename}: {e}")
            failed_conversions += 1
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            failed_conversions += 1
        except Exception as e:
            print(f"Unexpected error converting {filename}: {e}")
            failed_conversions += 1

    print(f"Total XML files processed: {len(xml_files)}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    print(f"Markdown files saved to: {md_output_dir}")


# --- Execute ---
convert_all_xmls_to_markdown()
