"""Data Designer plugin registration."""

from data_designer.plugins.plugin import Plugin, PluginType

long_context_episode_simulator = Plugin(
    impl_qualified_name="long_context_sdg.generator.LongContextEpisodeGenerator",
    config_qualified_name="long_context_sdg.generator_config.LongContextEpisodeConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)

persona_query_generator = Plugin(
    impl_qualified_name="long_context_sdg.query_generation.generator.PersonaQueryGenerator",
    config_qualified_name="long_context_sdg.query_generation.generator_config.SyntheticQueryConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
