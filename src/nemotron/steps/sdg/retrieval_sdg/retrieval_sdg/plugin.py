"""Data Designer plugin registration for the conversation generator."""

from data_designer.plugins.plugin import Plugin, PluginType

conversation_simulator = Plugin(
    impl_qualified_name="retrieval_sdg.conversation.generator.ConversationSimulatorGenerator",
    config_qualified_name="retrieval_sdg.conversation.config.ConversationSimulatorConfig",
    plugin_type=PluginType.COLUMN_GENERATOR,
)
