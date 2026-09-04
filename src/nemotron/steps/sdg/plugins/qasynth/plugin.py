# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point descriptor for the QASynth Data Designer plugin."""

from data_designer.plugins.plugin import Plugin, PluginType

qasynth_mcq = Plugin(
    impl_qualified_name="nemotron.steps.sdg.plugins.qasynth.generator.QASynthMCQGenerator",
    config_qualified_name="nemotron.steps.sdg.plugins.qasynth.config.QASynthMCQConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
