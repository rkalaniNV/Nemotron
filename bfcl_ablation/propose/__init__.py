"""A3 — LLM task proposal under a system-controlled policy distribution.

The model proposes what a task *means*: which tools compose, which fixture records
the slots bind, how the slots relate, which of the pack's existing assertions pin the
outcome. It never proposes what the benchmark is *made of*: the (category x policy)
target is sampled by `sampler`, the backend supplies ground truth, and `validate`
refuses anything the pipeline would later have to guess about.

The split exists because an arm that let the model pick its own policy mix could not
measure selection bias. A model that quietly avoids `correction` and `dependent_call`
would return a benchmark that looks healthy and tests less, and the arm would report
its own blind spot as a result.
"""
