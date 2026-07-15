"""Data Designer plugin registration for the deep-research RAG simulator."""

from data_designer.plugins.plugin import Plugin, PluginType

deep_research_simulator = Plugin(
    impl_qualified_name="agentic_rag.generator.DeepResearchSimulatorGenerator",
    config_qualified_name="agentic_rag.config.DeepResearchSimulatorConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
