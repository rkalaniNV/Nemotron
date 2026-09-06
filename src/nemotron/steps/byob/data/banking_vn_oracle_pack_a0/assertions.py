"""Assertions for the banking_vn oracle pack."""

from __future__ import annotations

from typing import Any


def _slots(task: dict) -> dict:
    return task.get("slots") or {}


def _results(trace: list, tool: str) -> list[dict]:
    return [
        item.get("result")
        for item in trace
        if item.get("tool") == tool and isinstance(item.get("result"), dict)
    ]


def assert_account_balance_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    account_id = _slots(task).get("account_id")
    for result in _results(trace, "get_account_balance"):
        if "error" not in result and result.get("account_id") == account_id:
            fixture = next(a for a in state["accounts"] if a["account_id"] == account_id)
            if result.get("balance_vnd") != fixture["balance_vnd"]:
                raise AssertionError("balance_vnd mismatch")
            return
    raise AssertionError(f"missing get_account_balance for {account_id}")


def assert_card_limit_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    card_id = _slots(task).get("card_id")
    for result in _results(trace, "get_card_limit"):
        if "error" not in result and result.get("card_id") == card_id:
            return
    raise AssertionError(f"missing get_card_limit for {card_id}")


def assert_transaction_status_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    txn_id = _slots(task).get("transaction_id")
    for result in _results(trace, "get_transaction_status"):
        if "error" not in result and result.get("transaction_id") == txn_id:
            return
    raise AssertionError(f"missing get_transaction_status for {txn_id}")


def assert_recent_transactions_listed(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    account_id = _slots(task).get("account_id")
    for result in _results(trace, "list_recent_transactions"):
        if "error" not in result and result.get("account_id") == account_id and "transactions" in result:
            return
    raise AssertionError(f"missing list_recent_transactions for {account_id}")


def assert_transfer_fee_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    for result in _results(trace, "get_transfer_fee"):
        if "error" not in result and "fee_vnd" in result:
            return
    raise AssertionError("missing get_transfer_fee result")


def assert_transfer_committed(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    slots = _slots(task)
    from_id = slots.get("from_account_id")
    amount = int(slots.get("amount_vnd"))
    transfers = state.get("transfers") or []
    if not any(
        t.get("from_account_id") == from_id
        and t.get("amount_vnd") == amount
        and t.get("status") == "succeeded"
        for t in transfers
    ):
        raise AssertionError("expected succeeded transfer row in final state")


def assert_only_corrected_amount_transferred(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    """The corrected amount moves once and the amount the user withdrew never moves."""
    slots = _slots(task)
    from_id = slots.get("from_account_id")
    corrected = int(slots.get("amount_vnd"))
    superseded = int((task.get("slots_initial") or {}).get("amount_vnd", corrected))
    mine = [t for t in (state.get("transfers") or []) if t.get("from_account_id") == from_id]
    if any(t.get("amount_vnd") == superseded for t in mine):
        raise AssertionError(f"transferred the superseded amount {superseded}")
    succeeded = [t for t in mine if t.get("amount_vnd") == corrected and t.get("status") == "succeeded"]
    if len(succeeded) != 1:
        raise AssertionError(f"expected exactly one succeeded transfer of {corrected}, found {len(succeeded)}")


def assert_vietqr_status_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    ref = _slots(task).get("payment_ref")
    for result in _results(trace, "get_vietqr_payment_status"):
        if "error" not in result and result.get("payment_ref") == ref:
            return
    raise AssertionError(f"missing get_vietqr_payment_status for {ref}")


def assert_dispute_status_reported(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    dispute_id = _slots(task).get("dispute_id")
    for result in _results(trace, "get_dispute_status"):
        if "error" not in result and result.get("dispute_id") == dispute_id:
            return
    raise AssertionError(f"missing get_dispute_status for {dispute_id}")


def assert_status_checked_from_listed_transaction(
    *, state: dict, trace: list, task: dict, ctx: Any
) -> None:
    listed = _results(trace, "list_recent_transactions")
    if not listed:
        raise AssertionError("missing list_recent_transactions result")
    available = {
        row.get("transaction_id")
        for result in listed
        for row in (result.get("transactions") or [])
    }
    checked = [
        result.get("transaction_id")
        for result in _results(trace, "get_transaction_status")
        if "error" not in result
    ]
    if not checked:
        raise AssertionError("missing get_transaction_status result")
    invented = sorted(str(txn_id) for txn_id in checked if txn_id not in available)
    if invented:
        raise AssertionError(f"status checked for ids that were never listed: {invented}")


def assert_transaction_not_found(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    txn_id = _slots(task).get("transaction_id")
    for result in _results(trace, "get_transaction_status"):
        error = result.get("error") or {}
        if error.get("code") == "not_found" and error.get("id") == txn_id:
            return
    raise AssertionError(f"expected a not_found error for {txn_id}")


def assert_transfer_rejected_for_funds(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    slots = _slots(task)
    from_id = slots.get("from_account_id")
    amount = int(slots.get("amount_vnd"))
    rejected = any(
        result.get("status") == "rejected_insufficient_funds"
        and result.get("transfer_id") is None
        for result in _results(trace, "create_transfer")
    )
    if not rejected:
        raise AssertionError("expected create_transfer to reject for insufficient funds")
    for transfer in state.get("transfers") or []:
        if transfer.get("from_account_id") == from_id and transfer.get("amount_vnd") == amount:
            raise AssertionError("rejected transfer must not be written to state")


def assert_no_tool_called(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    if trace:
        called = sorted({str(item.get("tool")) for item in trace})
        raise AssertionError(f"expected no tool call, saw {called}")


def assert_dispute_opened(*, state: dict, trace: list, task: dict, ctx: Any) -> None:
    txn_id = _slots(task).get("transaction_id")
    disputes = state.get("disputes") or []
    if not any(d.get("transaction_id") == txn_id and d.get("status") == "open" for d in disputes):
        raise AssertionError(f"expected open dispute for {txn_id}")


ASSERTIONS = {
    "assert_account_balance_reported": assert_account_balance_reported,
    "assert_card_limit_reported": assert_card_limit_reported,
    "assert_transaction_status_reported": assert_transaction_status_reported,
    "assert_recent_transactions_listed": assert_recent_transactions_listed,
    "assert_transfer_fee_reported": assert_transfer_fee_reported,
    "assert_transfer_committed": assert_transfer_committed,
    "assert_only_corrected_amount_transferred": assert_only_corrected_amount_transferred,
    "assert_vietqr_status_reported": assert_vietqr_status_reported,
    "assert_dispute_status_reported": assert_dispute_status_reported,
    "assert_dispute_opened": assert_dispute_opened,
    "assert_status_checked_from_listed_transaction": assert_status_checked_from_listed_transaction,
    "assert_transaction_not_found": assert_transaction_not_found,
    "assert_transfer_rejected_for_funds": assert_transfer_rejected_for_funds,
    "assert_no_tool_called": assert_no_tool_called,
}
