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

"""Local privacy scan and redaction. Nothing here touches the network.

Two kinds of finding are treated differently:

- A credential (API key, private key, access token, or a token assigned to a secret
  field) means the document is internal material. The whole file is quarantined; a
  credential is never redacted and passed on. An assignment counts only when the value
  looks like a token, so a password-policy sentence stays in the document.
- Personal data (email, payment card, phone number, identity document, bank account) is
  replaced by a typed placeholder such as ``REDACTED_EMAIL_1``. The same value keeps the
  same placeholder inside one document. The redacted text is scanned again, and any
  remaining finding quarantines the file.

A finding records its kind and line number, never the matched value.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

CREDENTIAL = "credential"
PERSONAL_DATA = "personal_data"
SECRET_PATTERN = "secret_pattern"
REDACTION_INCOMPLETE = "redaction_incomplete"

_WORD = r"A-Za-z0-9"
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:api[_ -]?key|secret[_ -]?key|client[_ -]?secret|access[_ -]?token|auth[_ -]?token"
    r"|password|passwd|pwd|mật khẩu)\b[ \t]*[:=][ \t]*[\"']?([^\s\"'<>]{8,})"
)


def _assigned_secret(match: re.Match[str]) -> bool:
    """True when the assigned value looks like a token, not like a password-policy sentence.

    A symbol is enough (``Matkhau2024!``). Letters and digits alone count only when the value
    is long and mixed-case, so ``MustBe8chars`` in a policy stays in the document.
    """
    value = match.group(1)
    has_letter = any(char.isalpha() for char in value)
    has_digit = any(char.isdigit() for char in value)
    if not (has_letter and has_digit):
        return False
    if any(not char.isalnum() for char in value):
        return True
    return len(value) >= 20 and any(char.isupper() for char in value) and any(char.islower() for char in value)


_CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str], Callable[[re.Match[str]], bool] | None], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"), None),
    ("api_key", re.compile(rf"(?<![{_WORD}])(?:sk|rk|pk)[-_](?:live|test)[-_][{_WORD}]{{3,}}"), None),
    ("api_key", re.compile(rf"(?<![{_WORD}])sk-(?:proj-|ant-)?[{_WORD}_-]{{20,}}"), None),
    ("api_key", re.compile(rf"(?<![{_WORD}])nvapi-[{_WORD}_-]{{20,}}"), None),
    ("api_key", re.compile(rf"(?<![{_WORD}])hf_[{_WORD}]{{30,}}"), None),
    ("api_key", re.compile(rf"(?<![{_WORD}])AIza[{_WORD}_-]{{35}}"), None),
    ("cloud_access_key", re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Z0-9])"), None),
    ("access_token", re.compile(rf"(?<![{_WORD}])gh[pousr]_[{_WORD}]{{36,}}"), None),
    ("access_token", re.compile(rf"(?<![{_WORD}])xox[abprs]-[{_WORD}-]{{10,}}"), None),
    (
        "access_token",
        re.compile(rf"(?<![{_WORD}_-])eyJ[{_WORD}_-]{{10,}}\.eyJ[{_WORD}_-]{{10,}}\.[{_WORD}_-]{{10,}}"),
        None,
    ),
    ("access_token", re.compile(r"\bBearer[ \t]+[A-Za-z0-9._~+/-]{20,}=*"), None),
    ("secret_assignment", _SECRET_ASSIGNMENT, _assigned_secret),
)


def _luhn(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for position, digit in enumerate(reversed(digits)):
        if position % 2:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _iban(value: str) -> bool:
    """ISO 13616 MOD-97 validation after removing presentation spaces."""
    compact = "".join(value.split()).upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for char in rearranged:
        encoded = char if char.isdigit() else str(ord(char) - ord("A") + 10)
        for digit in encoded:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


@dataclass(frozen=True)
class _PersonalPattern:
    kind: str
    pattern: re.Pattern[str]
    group: int = 0
    accept: Callable[[str], bool] | None = None


_PERSONAL_PATTERNS: tuple[_PersonalPattern, ...] = (
    _PersonalPattern(
        "email",
        re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])"),
    ),
    _PersonalPattern("card_number", re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"), accept=_luhn),
    _PersonalPattern(
        "national_id",
        re.compile(
            r"(?i)\b(?:CCCD|CMND|CMT|căn cước(?: công dân)?|chứng minh nhân dân|hộ chiếu|passport(?: number| no\.?)?"
            r"|national id(?: number)?|SSN)\b[^\w\n]{0,5}(?:số[^\w\n]{0,3})?(\d(?:[ \-]?\d){6,14})"
        ),
        group=1,
    ),
    _PersonalPattern(
        "bank_account",
        re.compile(
            r"(?i)\b(?:số tài khoản|STK|tài khoản số|account (?:number|no\.?))[^\w\n]{0,5}(\d(?:[ -]?\d){4,22})"
        ),
        group=1,
    ),
    _PersonalPattern(
        "iban",
        re.compile(r"(?i)(?<![A-Z0-9])[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}(?![A-Z0-9])"),
        accept=_iban,
    ),
    _PersonalPattern(
        "date_of_birth",
        re.compile(
            r"(?i)(?:date of birth|birth date|DOB|ngày sinh|fecha de nacimiento|date de naissance|geburtsdatum"
            r"|出生日期|生年月日|생년월일)[ \t]*[:：]?[ \t]*"
            r"(\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
        ),
        group=1,
    ),
    _PersonalPattern("phone_number", re.compile(r"(?<![\w+])(?:\+84|0084|0)[ .-]?[35789](?:[ .-]?\d){8}(?!\d)")),
    _PersonalPattern("phone_number", re.compile(r"(?<![\w+])\+[1-9]\d{0,2}(?:[ .-]?\d){7,12}(?!\d)")),
)


@dataclass(frozen=True)
class Finding:
    kind: str
    category: str
    line: int

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "category": self.category, "line": self.line}


@dataclass(frozen=True)
class PrivacyResult:
    """Outcome of the local scan. ``lines`` is the redacted text when ``safe`` is true."""

    safe: bool
    reason: str | None
    lines: tuple[str, ...]
    findings: tuple[Finding, ...]
    redactions: dict[str, int] = field(default_factory=dict)


def assess(lines: Sequence[str]) -> PrivacyResult:
    """Scan one document, given as lines, and redact personal data line by line.

    Redaction never adds or removes a line, so structure and section numbers survive it.
    """
    credentials = credential_findings(lines)
    if credentials:
        return PrivacyResult(False, SECRET_PATTERN, (), tuple(credentials))
    placeholders: dict[tuple[str, str], str] = {}
    counters: dict[str, int] = {}
    redactions: dict[str, int] = {}
    findings: list[Finding] = []
    redacted: list[str] = []
    for number, line in enumerate(lines, start=1):
        for spec in _PERSONAL_PATTERNS:
            line, hits = _redact(line, spec, placeholders, counters)
            if hits:
                redactions[spec.kind] = redactions.get(spec.kind, 0) + hits
                findings.append(Finding(spec.kind, PERSONAL_DATA, number))
        redacted.append(line)
    remaining = credential_findings(redacted) + [
        Finding(spec.kind, PERSONAL_DATA, number)
        for number, line in enumerate(redacted, start=1)
        for spec in _PERSONAL_PATTERNS
        if any(_accepted(spec, match) for match in spec.pattern.finditer(line))
    ]
    if remaining:
        return PrivacyResult(False, REDACTION_INCOMPLETE, (), tuple(findings + remaining))
    return PrivacyResult(True, None, tuple(redacted), tuple(findings), dict(sorted(redactions.items())))


def credential_findings(lines: Sequence[str]) -> list[Finding]:
    """Credentials anywhere in the document, including ones that span lines."""
    text = "\n".join(lines)
    findings = {
        (kind, text.count("\n", 0, match.start()) + 1)
        for kind, pattern, accept in _CREDENTIAL_PATTERNS
        for match in pattern.finditer(text)
        if accept is None or accept(match)
    }
    return [Finding(kind, CREDENTIAL, line) for kind, line in sorted(findings, key=lambda item: (item[1], item[0]))]


def _accepted(spec: _PersonalPattern, match: re.Match[str]) -> bool:
    return spec.accept is None or spec.accept(match.group(spec.group))


def _redact(
    line: str, spec: _PersonalPattern, placeholders: dict[tuple[str, str], str], counters: dict[str, int]
) -> tuple[str, int]:
    pieces: list[str] = []
    cursor = 0
    hits = 0
    for match in spec.pattern.finditer(line):
        if not _accepted(spec, match):
            continue
        start, end = match.span(spec.group)
        key = (spec.kind, re.sub(r"[\s.-]", "", match.group(spec.group)).casefold())
        if key not in placeholders:
            counters[spec.kind] = counters.get(spec.kind, 0) + 1
            placeholders[key] = f"REDACTED_{spec.kind.upper()}_{counters[spec.kind]}"
        pieces.append(line[cursor:start])
        pieces.append(placeholders[key])
        cursor = end
        hits += 1
    if not hits:
        return line, 0
    pieces.append(line[cursor:])
    return "".join(pieces), hits
