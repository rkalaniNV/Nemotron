"""Prove the live episode simulator runs through NeMo Data Designer's real engine.

Runs ``DataDesigner.preview()`` on a config whose only generator column is the
``episode-simulator`` plugin, with the model endpoint AND the retriever mocked (no
API key / no network). Success = DD discovered the plugin via entry point, resolved
its ``column_type``, instantiated the generator, and drove it to a populated
``structured_messages`` with ``trajectory_status=True`` and no compress artifacts.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("NVIDIA_API_KEY", "placeholder-key")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402
import data_designer.config as dd  # noqa: E402
from data_designer.interface import DataDesigner  # noqa: E402
from data_designer.engine.models.facade import ModelFacade  # noqa: E402

import mtsdg.generator as gen  # noqa: E402
from fixtures import FakeFacade, FakeRetriever, make_query  # noqa: E402
from mtsdg.generator_config import EpisodeSimulatorConfig  # noqa: E402
from mtsdg.model_configs import default_model_configs, nvidia_provider  # noqa: E402

query = make_query(turn_budget=8)
_fake = FakeFacade()

# Mock the model endpoint and the retriever (no network, no key).
ModelFacade.completion = lambda self, messages, **kw: _fake.completion(messages, **kw)  # type: ignore
gen.RetrieverClient = lambda *a, **k: FakeRetriever()  # type: ignore

seed_df = pd.DataFrame([{"episode_input": query.model_dump_json()}])
seed_path = os.path.join(os.path.dirname(__file__), "..", "output", "mock_episode_input.jsonl")
os.makedirs(os.path.dirname(seed_path), exist_ok=True)
seed_df.to_json(seed_path, orient="records", lines=True, force_ascii=False)

model_configs = default_model_configs()
for mc in model_configs:
    mc.skip_health_check = True

builder = dd.DataDesignerConfigBuilder(model_configs=model_configs)
builder.with_seed_dataset(dd.LocalFileSeedSource(path=seed_path), sampling_strategy=dd.SamplingStrategy.ORDERED)
builder.add_column(
    EpisodeSimulatorConfig(
        name="conversation", episode_input_column="episode_input",
        context_token_threshold=1200, run_trajectory_judge=True, majority_vote_n=1,
    )
)

designer = DataDesigner(model_providers=[nvidia_provider()])
row = designer.preview(builder, num_records=1).dataset.iloc[0]

print("=" * 60)
print("DD ENGINE RAN THE LIVE PLUGIN. Row output:")
print("trajectory_status:", row["trajectory_status"])
msgs = json.loads(row["structured_messages"])
print("n structured_messages:", len(msgs))
print("roles:", [m["role"] for m in msgs])
meta = json.loads(row["episode_metadata"])
print("compaction_events:", meta.get("compaction_events"))
print("retrieved chunks:", meta.get("n_retrieved_chunks"))
print("validation:", row["trajectory_validation"][:300])
assert row["trajectory_status"] in (True, "true", "True"), "trajectory failed"
assert len(msgs) > 5
assert not any(m.get("name") == "context.compress" for m in msgs), "compress leaked into chat"
print("SUCCESS: live episode-simulator is a working Data Designer plugin.")
