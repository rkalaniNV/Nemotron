# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""retrieval_sdg — the multi-turn retrieval-grounded conversation-generation engine.

Given a JSONL list of queries and a retrieval service, this package deduplicates,
clusters, and samples the queries, then generates grounded multi-turn / multi-step
tool-calling conversations (trajectory shape sampled per row) with an embedded
judge — ready for SFT. The retrieval service and the upstream query producer are
external; this package wraps the retrieval service thinly and consumes the queries.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
