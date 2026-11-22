use crate::{AgentTool, Thread, ToolCallEventStream, ToolInput};
use agent_client_protocol::v2::schema as acp;
use anyhow::Result;
use gpui::{App, SharedString, Task, WeakEntity};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Browse this thread's retained conversation, including pre-compaction history.
/// Search is case-insensitive over visible text. Reasoning and provider-native
/// compaction payloads are never returned. Pages contain at most 20 messages,
/// each truncated to 2000 characters. This does not read files or authorize edits.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConversationToolInput {
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub limit: Option<usize>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub newest_first: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ConversationToolOutput(pub serde_json::Value);

impl From<ConversationToolOutput> for LanguageModelToolResultContent {
    fn from(output: ConversationToolOutput) -> Self {
        output.0.to_string().into()
    }
}

pub struct ConversationTool {
    thread: WeakEntity<Thread>,
}

impl ConversationTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

impl AgentTool for ConversationTool {
    type Input = ConversationToolInput;
    type Output = ConversationToolOutput;
    const NAME: &'static str = "conversation";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Read
    }

    fn initial_title(
        &self,
        _input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        "Browse retained conversation".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        _events: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |cx| {
            let result = async {
                let input = input.recv().await?;
                self.thread.read_with(cx, |thread, _| {
                    thread.conversation_page(
                        input.offset,
                        input.limit.unwrap_or(10),
                        input.query.as_deref(),
                        input.newest_first,
                    )
                })
            }
            .await;
            match result {
                Ok(output) => Ok(ConversationToolOutput(output)),
                Err(error) => Err(ConversationToolOutput(
                    serde_json::json!({ "error": error.to_string() }),
                )),
            }
        })
    }
}
