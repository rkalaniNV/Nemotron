"""Data Designer plugin entry point for the generic live episode simulator."""

from data_designer.plugins.plugin import Plugin, PluginType

episode_simulator = Plugin(
    impl_qualified_name="mtsdg.generator.EpisodeSimulatorGenerator",
    config_qualified_name="mtsdg.generator_config.EpisodeSimulatorConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
