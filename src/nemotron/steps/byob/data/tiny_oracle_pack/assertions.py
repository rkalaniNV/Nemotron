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

"""Assertions for the tiny_library oracle pack."""

from __future__ import annotations

from typing import Any

ASSERTIONS: dict[str, Any] = {}


def assert_book_status_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    slots = task.get("slots") or {}
    book_ids = []
    if "book_id" in slots:
        book_ids.append(slots["book_id"])
    for key in ("book_a", "book_b"):
        if key in slots:
            book_ids.append(slots[key])
    if not book_ids:
        raise AssertionError("no book slot bound for assert_book_status_reported")

    results = [
        item.get("result")
        for item in trace
        if item.get("tool") == "get_book_status" and isinstance(item.get("result"), dict)
    ]
    reported = {result.get("book_id") for result in results if "error" not in result}
    for book_id in book_ids:
        if book_id not in reported:
            raise AssertionError(f"missing get_book_status result for {book_id}")


def assert_checkout_awaiting_then_committed(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    slots = task.get("slots") or {}
    book_id = slots.get("book_id")
    loans = state.get("loans") or []
    if not any(loan.get("book_id") == book_id and loan.get("status") == "active" for loan in loans):
        raise AssertionError(f"expected active loan for {book_id}")


def assert_book_now_on_loan(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    slots = task.get("slots") or {}
    book_id = slots.get("book_id")
    books = {book["book_id"]: book for book in state.get("books", [])}
    book = books.get(book_id)
    if book is None or book.get("status") != "on_loan":
        raise AssertionError(f"expected book {book_id} status on_loan")


def assert_no_tool_called(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    if trace:
        called = sorted({str(item.get("tool")) for item in trace})
        raise AssertionError(f"expected no tool call, got {', '.join(called)}")


ASSERTIONS = {
    "assert_book_status_reported": assert_book_status_reported,
    "assert_checkout_awaiting_then_committed": assert_checkout_awaiting_then_committed,
    "assert_book_now_on_loan": assert_book_now_on_loan,
    "assert_no_tool_called": assert_no_tool_called,
}

ASSERTION_CAPABILITIES = {
    "assert_book_status_reported": {
        "trace": True,
        "executable": True,
        "category": "result",
    },
    "assert_checkout_awaiting_then_committed": {
        "trace": False,
        "executable": True,
        "category": "state",
    },
    "assert_book_now_on_loan": {
        "trace": False,
        "executable": True,
        "category": "state",
    },
    "assert_no_tool_called": {
        "trace": True,
        "executable": True,
        "category": "path",
    },
}
