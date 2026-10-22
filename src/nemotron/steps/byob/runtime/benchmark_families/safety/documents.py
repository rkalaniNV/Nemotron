# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turn one policy file into plain lines, locally and without a model.

Every format becomes the same shape: a list of lines, each with an optional heading
level (from a DOCX heading style) and an optional page number (from a PDF). The span
builder then treats every format alike.
"""

import codecs
import io
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

FORMATS = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".text": "text",
    ".docx": "docx",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".pdf": "pdf",
}
STRUCTURED_FORMATS = frozenset({"yaml", "json"})
MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_DOCX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_PDF_PAGES = 2000


@dataclass(frozen=True)
class Line:
    text: str
    heading_level: int | None = None
    page: int | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    format: str
    lines: tuple[Line, ...]
    markdown_headings: bool


class DocumentError(Exception):
    """A file cannot be read as a supported, text-bearing policy document."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def detect_format(relative: str) -> str | None:
    return FORMATS.get(PurePosixPath(relative).suffix.lower())


def extract(data: bytes, fmt: str) -> ExtractedDocument:
    """Read the bytes of one file. Raises DocumentError with a stable reason code."""
    if len(data) > MAX_SOURCE_BYTES:
        raise DocumentError("file_too_large", f"files larger than {MAX_SOURCE_BYTES} bytes are not read")
    if fmt == "docx":
        lines = _docx_lines(data)
    elif fmt == "pdf":
        lines = _pdf_lines(data)
    elif fmt in {"markdown", "text", "yaml", "json"}:
        lines = tuple(Line(text) for text in decode_text(data).split("\n"))
    else:
        raise DocumentError("unsupported_format", f"format {fmt!r} is not supported")
    if not any(line.text.strip() for line in lines):
        raise DocumentError("empty_document", "the file contains no text")
    return ExtractedDocument(fmt, lines, markdown_headings=fmt in {"markdown", "text"})


def decode_text(data: bytes) -> str:
    """UTF-8, or UTF-16 with a byte-order mark. Other encodings are refused, not guessed."""
    try:
        if data.startswith(codecs.BOM_UTF8):
            text = data[len(codecs.BOM_UTF8) :].decode("utf-8")
        elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            text = data.decode("utf-16")
        else:
            text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentError(
            "unsupported_encoding", "the file is not UTF-8, or UTF-16 with a byte-order mark; save it as UTF-8"
        ) from exc
    if "\x00" in text:
        raise DocumentError("binary_content", "the file contains NUL bytes and is not a text document")
    return normalize_text(text)


def normalize_text(text: str) -> str:
    """One newline style and Unicode NFC, so offsets and span IDs do not depend on the editor."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    return unicodedata.normalize("NFC", text)


def _docx_lines(data: bytes) -> tuple[Line, ...]:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as exc:
        raise DocumentError("parser_unavailable", "DOCX support needs python-docx; install the byob extra") from exc
    _check_docx_archive(data)
    lines: list[Line] = []
    try:
        document = docx.Document(io.BytesIO(data))
        for child in document.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                paragraph = Paragraph(child, document)
                _append_split(lines, paragraph.text, _paragraph_heading_level(paragraph))
            elif tag == "tbl":
                for row in Table(child, document).rows:
                    cells: list[str] = []
                    for cell in row.cells:
                        text = " ".join(cell.text.split())
                        if not cells or cells[-1] != text:
                            cells.append(text)
                    if any(cells):
                        lines.append(Line("| " + " | ".join(cells) + " |"))
    # python-docx and lxml raise many unrelated exception types on a damaged package.
    except Exception as exc:
        raise DocumentError("unreadable_document", f"the DOCX file cannot be parsed: {type(exc).__name__}") from exc
    return tuple(lines)


def _check_docx_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            expanded = sum(info.file_size for info in archive.infolist())
    except (zipfile.BadZipFile, OSError) as exc:
        raise DocumentError("unreadable_document", "the DOCX file is not a valid ZIP package") from exc
    if "word/document.xml" not in names:
        raise DocumentError("unreadable_document", "the DOCX package has no word/document.xml")
    if expanded > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise DocumentError(
            "file_too_large", f"the DOCX package expands to more than {MAX_DOCX_UNCOMPRESSED_BYTES} bytes"
        )


def _paragraph_heading_level(paragraph: object) -> int | None:
    """Heading depth from the paragraph outline, then from the style id or name.

    A localized style such as "Tiêu đề 1" still carries ``w:outlineLvl`` and the built-in
    ``Heading1`` style id. The English style name remains a last fallback.
    """
    direct = _outline_level(getattr(paragraph, "_p", None))
    if direct is not None:
        return direct
    style = getattr(paragraph, "style", None)
    seen: set[int] = set()
    while style is not None and id(style) not in seen:
        seen.add(id(style))
        by_id = _heading_level(getattr(style, "style_id", "") or "")
        if by_id is not None:
            return by_id
        named = _heading_level(getattr(style, "name", "") or "")
        if named is not None:
            return named
        found = _outline_level(getattr(style, "element", None))
        if found is not None:
            return found
        style = getattr(style, "base_style", None)
    return None


def _outline_level(element: object) -> int | None:
    find = getattr(element, "find", None)
    if find is None:
        return None
    try:
        from docx.oxml.ns import qn
    except ImportError:
        return None
    properties = find(qn("w:pPr"))
    if properties is None:
        return None
    outline = properties.find(qn("w:outlineLvl"))
    if outline is None:
        return None
    raw = outline.get(qn("w:val"))
    if not isinstance(raw, str) or not raw.isdigit():
        return None
    value = int(raw)
    return value + 1 if 0 <= value <= 8 else None


def _heading_level(style: str) -> int | None:
    name = style.strip().casefold()
    if name == "title":
        return 1
    if name.startswith("heading"):
        suffix = name[len("heading") :].strip()
        return int(suffix) if suffix.isdigit() and 1 <= int(suffix) <= 9 else 1
    return None


def _append_split(lines: list[Line], text: str, heading_level: int | None, page: int | None = None) -> None:
    for index, part in enumerate(normalize_text(text).replace("\t", " ").replace("\v", "\n").split("\n")):
        lines.append(Line(part, heading_level if index == 0 else None, page))


def _pdf_lines(data: bytes) -> tuple[Line, ...]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentError("parser_unavailable", "PDF support needs pypdf; install the byob extra") from exc
    lines: list[Line] = []
    # pypdf raises many unrelated exception types on a damaged or hostile file.
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not _decrypt_without_password(reader):
            raise DocumentError("encrypted_document", "the PDF is password protected; supply an unprotected copy")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise DocumentError("file_too_large", f"PDFs with more than {MAX_PDF_PAGES} pages are not read")
        for number, page in enumerate(reader.pages, start=1):
            _append_split(lines, page.extract_text() or "", None, number)
    except DocumentError:
        raise
    except Exception as exc:
        raise DocumentError("unreadable_document", f"the PDF file cannot be parsed: {type(exc).__name__}") from exc
    if not any(line.text.strip() for line in lines):
        raise DocumentError("no_text_layer", "the PDF has no text layer; supply a text-based PDF or the source text")
    return tuple(lines)


def _decrypt_without_password(reader: object) -> bool:
    try:
        return bool(reader.decrypt(""))  # type: ignore[attr-defined]
    except Exception:
        return False
