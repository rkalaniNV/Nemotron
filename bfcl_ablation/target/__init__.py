"""A5 — target-model evaluation across wordings.

A0 through A4 measure benchmark *content*: how much it costs to author, how diverse it
is, whether its assertions check anything. None of them runs a model against the
benchmark, so none can say whether a benchmark *conclusion* is stable.

This arm does. It takes the same 33 tasks in two wordings — A0's authored sentence and
A2's paraphrase of it — and scores one target model on both. Because `task_id` is hashed
over (pack, template, fixture refs, slot bindings, variant index) and *not* over the
surface, the two arms carry identical ids, so every task is its own control and the
comparison is paired rather than between-groups.
"""
