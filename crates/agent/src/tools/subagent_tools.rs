use super::spawn_agent_tool::{SpawnAgentToolInput, send_with_timeout_and_cancel};
use crate::{AgentTool, ThreadEnvironment, ToolCallEventStream, ToolInput};
use acp_thread::{
    AgentModelId, SUBAGENT_SESSION_CARDS_META_KEY, SubagentSessionCard, SubagentSessionCardStatus,
    SubagentSessionInfo,
};
use agent_client_protocol::v2::schema as acp;
use anyhow::{Result, anyhow, ensure};
use futures::channel::oneshot;
use gpui::{App, AsyncApp, SharedString, Task};
use language_model::LanguageModelToolResultContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::{Rc, Weak},
    sync::Arc,
    time::{Duration, Instant},
};

/// Start independent subagents in parallel and retain their jobs across tool calls.
/// Spawn waits for the first agent result (up to 30 seconds). SpawnGroups starts
/// several groups and keeps this tool call pending until the first entire group
/// finishes, returning its outputs AND statuses for every retained group. A failed
/// agent is terminal for group completion. Other jobs keep running.
/// Call WaitGroups again for the next unreported group-completion boundary;
/// use Wait for individual agent results or completion of a specific group.
/// No model turn is started automatically after your user turn has ended: keep a
/// tool call pending to resume through its result, or call wait again explicitly.
/// Jobs persist for this parent session, not across application restarts.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum SubagentsToolInput {
    Spawn {
        agents: Vec<SpawnAgentToolInput>,
    },
    /// Start all groups concurrently, then return the first fully finished group.
    SpawnGroups {
        /// Disjoint groups of new agents: 1–16 agents per group, at most 128
        /// retained jobs across this parent session. All groups start concurrently.
        groups: Vec<Vec<SpawnAgentToolInput>>,
        /// Omit to wait until a group finishes. Zero returns immediately.
        /// This limits only this wait; it does not cancel jobs or their own budgets.
        #[serde(default)]
        wait_timeout_ms: Option<u64>,
    },
    /// Wait for the next fully finished, not-yet-reported group. All responses
    /// include statuses for every retained group, including groups outside this set.
    WaitGroups {
        /// Omit or pass an empty list to consider every retained group.
        #[serde(default)]
        group_ids: Vec<String>,
        /// Omit to keep the tool pending until a boundary. Zero polls current state.
        #[serde(default)]
        wait_timeout_ms: Option<u64>,
    },
    Wait {
        group_id: String,
        #[serde(default)]
        mode: SubagentWaitMode,
        /// Maximum time to wait, not a job timeout. Jobs continue after this wait expires.
        #[serde(default)]
        wait_timeout_ms: Option<u64>,
    },
    Inspect {
        session_id: acp::SessionId,
    },
    List {},
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SubagentWaitMode {
    #[default]
    Any,
    All,
}

const SUBAGENT_CARDS_OUTPUT_KEY: &str = "_zed_subagent_session_cards";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SubagentsToolOutput(pub serde_json::Value);

impl SubagentsToolOutput {
    fn with_cards(mut value: serde_json::Value, cards: &[SubagentSessionCard]) -> Self {
        if let Some(object) = value.as_object_mut() {
            object.insert(SUBAGENT_CARDS_OUTPUT_KEY.into(), serde_json::json!(cards));
        }
        Self(value)
    }
}

impl SubagentsToolOutput {
    fn model_facing_text(mut self) -> String {
        if let Some(object) = self.0.as_object_mut() {
            object.remove(SUBAGENT_CARDS_OUTPUT_KEY);
        }
        self.0.to_string()
    }
}

impl From<SubagentsToolOutput> for LanguageModelToolResultContent {
    fn from(output: SubagentsToolOutput) -> Self {
        output.model_facing_text().into()
    }
}

const MAX_RETAINED_OUTPUT_CHARACTERS: usize = 16_384;
const MAX_RETAINED_ERROR_CHARACTERS: usize = 4096;

struct RetainedJobOutput {
    text: String,
    truncated: bool,
}

impl RetainedJobOutput {
    fn new(mut text: String, limit: usize) -> Self {
        text.shrink_to_fit();
        let end = text.char_indices().nth(limit).map(|(index, _)| index);
        match end {
            Some(end) => Self {
                text: text[..end].to_owned(),
                truncated: true,
            },
            None => Self {
                text,
                truncated: false,
            },
        }
    }
}

struct Job {
    session_id: Option<acp::SessionId>,
    handle: Option<Rc<dyn crate::thread::SubagentHandle>>,
    label: String,
    result: RefCell<Option<Result<RetainedJobOutput, RetainedJobOutput>>>,
    cancellation: RefCell<Option<oneshot::Sender<()>>>,
    delivered: RefCell<bool>,
    completed_at: Cell<Option<Instant>>,
    message_start_index: usize,
    message_end_index: Cell<Option<usize>>,
    task: RefCell<Option<Task<()>>>,
}

#[derive(Default)]
struct Jobs {
    groups: RefCell<BTreeMap<String, Vec<Rc<Job>>>>,
    reported_groups: RefCell<BTreeSet<String>>,
}

impl Job {
    fn card(&self) -> Option<SubagentSessionCard> {
        let session_id = self.session_id.clone()?;
        let result = self.result.borrow();
        let (status, error) = match result.as_ref() {
            None => (SubagentSessionCardStatus::Running, None),
            Some(Ok(_)) => (SubagentSessionCardStatus::Completed, None),
            Some(Err(error)) if error.text.to_lowercase().contains("cancel") => (
                SubagentSessionCardStatus::Canceled,
                Some(error.text.clone()),
            ),
            Some(Err(error)) => (SubagentSessionCardStatus::Failed, Some(error.text.clone())),
        };
        Some(SubagentSessionCard {
            session_info: SubagentSessionInfo {
                session_id,
                message_start_index: self.message_start_index,
                message_end_index: self.message_end_index.get(),
            },
            label: self.label.clone(),
            status,
            error,
        })
    }
}

impl Jobs {
    fn cancel(&self) {
        for job in self.groups.borrow().values().flatten() {
            if let Some(sender) = job.cancellation.borrow_mut().take()
                && sender.send(()).is_err()
            {
                log::debug!("Subagent job already stopped before parent cancellation");
            }
        }
    }
}

pub struct SubagentsTool {
    environment: Rc<dyn ThreadEnvironment>,
    jobs: Rc<Jobs>,
    parent_subscription: Option<gpui::Subscription>,
}

impl SubagentsTool {
    pub fn new(environment: Rc<dyn ThreadEnvironment>) -> Self {
        Self {
            environment,
            jobs: Rc::new(Jobs::default()),
            parent_subscription: None,
        }
    }

    pub fn with_parent(mut self, parent: gpui::Entity<crate::Thread>, cx: &mut App) -> Self {
        let jobs = Rc::downgrade(&self.jobs);
        self.parent_subscription = Some(cx.subscribe(
            &parent,
            move |_, _: &crate::thread::SubagentsCancelled, _| {
                if let Some(jobs) = jobs.upgrade() {
                    jobs.cancel();
                }
            },
        ));
        self
    }

    fn publish_cards(jobs: &[Weak<Job>], events: &ToolCallEventStream) {
        let cards = jobs
            .iter()
            .filter_map(Weak::upgrade)
            .filter_map(|job| job.card())
            .collect::<Vec<_>>();
        events.update_fields(|update| {
            update.meta(acp::Meta::from_iter([(
                SUBAGENT_SESSION_CARDS_META_KEY.into(),
                serde_json::json!(cards),
            )]))
        });
    }

    fn snapshot(
        &self,
        group_id: &str,
        acknowledge: bool,
        cx: &mut App,
    ) -> Result<serde_json::Value> {
        let groups = self.jobs.groups.borrow();
        let jobs = groups
            .get(group_id)
            .ok_or_else(|| anyhow!("Unknown group {group_id}"))?;
        let mut finished = Vec::new();
        let mut remaining = Vec::new();
        for job in jobs {
            if let Some(result) = job.result.borrow().as_ref() {
                finished.push(match result {
                    Ok(output) => serde_json::json!({
                        "session_id": job.session_id, "label": job.label, "status": "completed",
                        "output": output.text,
                        "output_truncated": output.truncated,
                    }),
                    Err(error) => serde_json::json!({
                        "session_id": job.session_id, "label": job.label, "status": "failed",
                        "error": error.text,
                        "error_truncated": error.truncated,
                    }),
                });
                if acknowledge {
                    job.delivered.replace(true);
                }
            } else {
                let activity = job.handle.as_ref().map(|handle| handle.activity(cx));
                remaining.push(serde_json::json!({
                    "session_id": job.session_id, "label": job.label, "status": "running",
                    "activity": activity,
                }));
            }
        }
        if acknowledge && remaining.is_empty() {
            self.jobs
                .reported_groups
                .borrow_mut()
                .insert(group_id.to_owned());
        }
        Ok(
            serde_json::json!({ "group_id": group_id, "finished": finished, "remaining": remaining }),
        )
    }

    fn group_statuses(&self, cx: &App) -> Vec<serde_json::Value> {
        let groups = self.jobs.groups.borrow();
        let reported = self.jobs.reported_groups.borrow();
        groups.iter().map(|(group_id, jobs)| {
            let completed = jobs.iter().all(|job| job.result.borrow().is_some());
            let jobs: Vec<_> = jobs.iter().map(|job| {
                let result = job.result.borrow();
                let status = match result.as_ref() {
                    None => "running",
                    Some(Ok(_)) => "completed",
                    Some(Err(_)) => "failed",
                };
                serde_json::json!({
                    "session_id": job.session_id, "label": job.label, "status": status,
                    "activity": if result.is_none() { job.handle.as_ref().map(|handle| handle.activity(cx)) } else { None },
                })
            }).collect();
            serde_json::json!({
                "group_id": group_id, "status": if completed { "completed" } else { "running" },
                "completion_reported": reported.contains(group_id), "jobs": jobs,
            })
        }).collect()
    }

    async fn wait_group(
        &self,
        group_id: &str,
        mode: SubagentWaitMode,
        wait_timeout_ms: Option<u64>,
        events: &ToolCallEventStream,
        cx: &AsyncApp,
    ) -> Result<serde_json::Value> {
        let executor = cx.background_executor().clone();
        let started_at = executor.now();
        let budget = Duration::from_millis(wait_timeout_ms.unwrap_or(30_000).min(60_000));
        loop {
            let ready = {
                let groups = self.jobs.groups.borrow();
                let jobs = groups
                    .get(group_id)
                    .ok_or_else(|| anyhow!("Unknown group {group_id}"))?;
                match mode {
                    SubagentWaitMode::Any => {
                        jobs.iter()
                            .any(|job| job.result.borrow().is_some() && !*job.delivered.borrow())
                            || jobs.iter().all(|job| job.result.borrow().is_some())
                    }
                    SubagentWaitMode::All => jobs.iter().all(|job| job.result.borrow().is_some()),
                }
            };
            let pending_request = cx.update(|cx| events.pending_subagent_request(cx));
            let remaining =
                budget.saturating_sub(executor.now().saturating_duration_since(started_at));
            if ready || remaining.is_zero() || pending_request {
                return cx.update(|cx| {
                    let mut snapshot = self.snapshot(group_id, !pending_request, cx)?;
                    snapshot["wait_expired"] = serde_json::json!(!ready && !pending_request);
                    snapshot["yielded_for_new_request"] = serde_json::json!(pending_request);
                    snapshot["groups"] = serde_json::json!(self.group_statuses(cx));
                    Ok(snapshot)
                });
            }
            executor
                .timer(remaining.min(Duration::from_millis(10)))
                .await;
        }
    }

    async fn wait_groups(
        &self,
        group_ids: Vec<String>,
        wait_timeout_ms: Option<u64>,
        events: &ToolCallEventStream,
        cx: &AsyncApp,
    ) -> Result<serde_json::Value> {
        let group_ids = if group_ids.is_empty() {
            self.jobs.groups.borrow().keys().cloned().collect()
        } else {
            group_ids
        };
        ensure!(group_ids.len() <= 128, "At most 128 groups may be selected");
        ensure!(
            group_ids
                .iter()
                .all(|group_id| self.jobs.groups.borrow().contains_key(group_id)),
            "Unknown group in group_ids"
        );
        let executor = cx.background_executor().clone();
        let started_at = executor.now();
        let budget = wait_timeout_ms.map(Duration::from_millis);
        loop {
            let (boundary, all_reported) = {
                let groups = self.jobs.groups.borrow();
                let reported = self.jobs.reported_groups.borrow();
                let boundary = group_ids
                    .iter()
                    .filter(|group_id| !reported.contains(*group_id))
                    .filter_map(|group_id| {
                        let jobs = groups.get(group_id)?;
                        if !jobs.iter().all(|job| job.result.borrow().is_some()) {
                            return None;
                        }
                        let completed_at =
                            jobs.iter().filter_map(|job| job.completed_at.get()).max()?;
                        Some((group_id.clone(), completed_at))
                    })
                    .min_by_key(|(_, completed_at)| *completed_at)
                    .map(|(group_id, _)| group_id);
                (
                    boundary,
                    group_ids.iter().all(|group_id| reported.contains(group_id)),
                )
            };
            let remaining = budget.map(|budget| {
                budget.saturating_sub(executor.now().saturating_duration_since(started_at))
            });
            let expired = remaining.is_some_and(|remaining| remaining.is_zero());
            let pending_request = cx.update(|cx| events.pending_subagent_request(cx));
            if boundary.is_some() || all_reported || expired || pending_request {
                return cx.update(|cx| {
                    let completed_groups = boundary.as_ref().map(|group_id| self.snapshot(group_id, !pending_request, cx)).transpose()?.into_iter().collect::<Vec<_>>();
                    Ok(serde_json::json!({
                        "group_ids": group_ids,
                        "completed_groups": completed_groups,
                        "groups": self.group_statuses(cx),
                        "wait_expired": boundary.is_none() && !all_reported && expired && !pending_request,
                        "yielded_for_new_request": pending_request,
                        "all_requested_groups_reported": group_ids.iter().all(|group_id| self.jobs.reported_groups.borrow().contains(group_id)),
                    }))
                });
            }
            executor
                .timer(
                    remaining
                        .unwrap_or(Duration::from_millis(10))
                        .min(Duration::from_millis(10)),
                )
                .await;
        }
    }

    #[cfg(test)]
    fn spawn_groups(
        &self,
        groups: Vec<Vec<SpawnAgentToolInput>>,
        events: &ToolCallEventStream,
        cx: &mut App,
    ) -> Result<Vec<String>> {
        self.spawn_groups_scoped(groups, events, Rc::new(RefCell::new(Vec::new())), cx)
    }

    fn spawn_groups_scoped(
        &self,
        groups: Vec<Vec<SpawnAgentToolInput>>,
        events: &ToolCallEventStream,
        scope: Rc<RefCell<Vec<Weak<Job>>>>,
        cx: &mut App,
    ) -> Result<Vec<String>> {
        ensure!(
            !groups.is_empty() && groups.len() <= 128,
            "Submit 1 to 128 groups"
        );
        let retained_jobs: usize = self.jobs.groups.borrow().values().map(Vec::len).sum();
        let new_jobs: usize = groups.iter().map(Vec::len).sum();
        ensure!(
            retained_jobs + new_jobs <= 128,
            "This parent session retains at most 128 jobs"
        );
        // Validate the entire submission before starting any of its groups.
        for group in &groups {
            ensure!(
                !group.is_empty() && group.len() <= 16,
                "Each group must contain 1 to 16 agents"
            );
            for agent in group {
                ensure!(
                    agent.session_id.is_none(),
                    "Group spawning requires new sessions"
                );
                agent.prompt()?;
            }
        }
        let group_ids = groups
            .into_iter()
            .map(|agents| self.spawn_group_scoped(agents, events, scope.clone(), cx))
            .collect::<Result<Vec<_>>>()?;
        Self::publish_cards(&scope.borrow(), events);
        Ok(group_ids)
    }

    #[cfg(test)]
    fn spawn_group(
        &self,
        agents: Vec<SpawnAgentToolInput>,
        events: &ToolCallEventStream,
        cx: &mut App,
    ) -> Result<String> {
        self.spawn_group_scoped(agents, events, Rc::new(RefCell::new(Vec::new())), cx)
    }

    fn spawn_group_scoped(
        &self,
        agents: Vec<SpawnAgentToolInput>,
        events: &ToolCallEventStream,
        scope: Rc<RefCell<Vec<Weak<Job>>>>,
        cx: &mut App,
    ) -> Result<String> {
        ensure!(
            !agents.is_empty() && agents.len() <= 16,
            "A group must contain 1 to 16 agents"
        );
        let retained_jobs: usize = self.jobs.groups.borrow().values().map(Vec::len).sum();
        ensure!(
            retained_jobs + agents.len() <= 128,
            "This parent session already retains too many jobs (limit 128)"
        );
        // Validate all prompts before creating sessions; no job is started on validation errors.
        let prompts = agents
            .iter()
            .map(SpawnAgentToolInput::prompt)
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            agents.iter().all(|agent| agent.session_id.is_none()),
            "Group spawning requires new sessions; use spawn_agent for follow-ups"
        );
        let group_id = uuid::Uuid::new_v4().to_string();
        let mut jobs = Vec::new();
        for (agent, prompt) in agents.into_iter().zip(prompts) {
            let (handle, result) = match self.environment.create_subagent(
                agent.label.clone(),
                agent.model.clone().map(AgentModelId::from),
                cx,
            ) {
                Ok(handle) => (Some(handle), None),
                Err(error) => (
                    None,
                    Some(Err(RetainedJobOutput::new(
                        error.to_string(),
                        MAX_RETAINED_ERROR_CHARACTERS,
                    ))),
                ),
            };
            let (sender, receiver) = oneshot::channel();
            let job = Rc::new(Job {
                session_id: handle.as_ref().map(|handle| handle.id()),
                handle: handle.clone(),
                label: agent.label.chars().take(256).collect(),
                completed_at: Cell::new(result.as_ref().map(|_| cx.background_executor().now())),
                message_start_index: handle.as_ref().map_or(0, |handle| handle.num_entries(cx)),
                message_end_index: Cell::new(None),
                result: RefCell::new(result),
                cancellation: RefCell::new(handle.as_ref().map(|_| sender)),
                delivered: RefCell::new(false),
                task: RefCell::new(None),
            });
            if let Some(handle) = handle {
                let weak_job = Rc::downgrade(&job);
                let events = events.clone();
                let scope = scope.clone();
                let task = cx.spawn(async move |cx| {
                    let result = send_with_timeout_and_cancel(
                        handle,
                        prompt,
                        agent.timeout_ms,
                        agent.timeout_mode.unwrap_or_default(),
                        Some(receiver),
                        cx,
                    )
                    .await
                    .map(|output| RetainedJobOutput::new(output, MAX_RETAINED_OUTPUT_CHARACTERS))
                    .map_err(|error| {
                        RetainedJobOutput::new(error.to_string(), MAX_RETAINED_ERROR_CHARACTERS)
                    });
                    if let Some(job) = weak_job.upgrade() {
                        job.cancellation.borrow_mut().take();
                        job.completed_at.set(Some(cx.background_executor().now()));
                        job.message_end_index.set(job.handle.as_ref().map(|handle| {
                            cx.update(|cx| handle.num_entries(cx)).saturating_sub(1)
                        }));
                        job.result.replace(Some(result));
                        events.update_fields(|update| {
                            update.meta(acp::Meta::from_iter([(
                                SUBAGENT_SESSION_CARDS_META_KEY.into(),
                                serde_json::json!(
                                    scope
                                        .borrow()
                                        .iter()
                                        .filter_map(Weak::upgrade)
                                        .filter_map(|job| job.card())
                                        .collect::<Vec<_>>()
                                ),
                            )]))
                        });
                    }
                });
                job.task.replace(Some(task));
            }
            jobs.push(job);
        }
        scope.borrow_mut().extend(jobs.iter().map(Rc::downgrade));
        self.jobs.groups.borrow_mut().insert(group_id.clone(), jobs);
        Ok(group_id)
    }
}

impl AgentTool for SubagentsTool {
    type Input = SubagentsToolInput;
    type Output = SubagentsToolOutput;
    const NAME: &'static str = "subagents";

    fn input_schema() -> schemars::Schema {
        let mut schema = language_model::tool_schema::root_schema_for::<Self::Input>();
        let Some(serde_json::Value::Array(variants)) = schema.ensure_object().remove("oneOf")
        else {
            return schema;
        };
        let mut properties = serde_json::Map::new();
        let mut operations = Vec::new();
        for variant in variants {
            let Some(variant_properties) = variant
                .get("properties")
                .and_then(serde_json::Value::as_object)
            else {
                continue;
            };
            let operation_schema = variant_properties.get("operation");
            let variant_operations: Vec<_> = operation_schema
                .and_then(|operation| operation.get("enum"))
                .and_then(serde_json::Value::as_array)
                .into_iter()
                .flatten()
                .chain(operation_schema.and_then(|operation| operation.get("const")))
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect();
            operations.extend(variant_operations.iter().cloned());
            let required = variant
                .get("required")
                .and_then(serde_json::Value::as_array);
            for (name, property) in variant_properties {
                if name == "operation" {
                    continue;
                }
                let mut property = property.clone();
                if required.is_some_and(|required| {
                    required
                        .iter()
                        .any(|field| field.as_str() == Some(name.as_str()))
                }) && let Some(object) = property.as_object_mut()
                {
                    let description = object
                        .get("description")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or_default();
                    object.insert(
                        "description".into(),
                        serde_json::json!(format!(
                            "Required for operation {}. {description}",
                            variant_operations.join(" or ")
                        )),
                    );
                }
                properties.entry(name.clone()).or_insert(property);
            }
        }
        properties.insert("operation".into(), serde_json::json!({
            "type": "string", "enum": operations,
            "description": "Choose the operation. Supply agents for spawn, groups for spawn_groups, group_id for wait, or session_id for inspect. wait_groups and list have no additional required fields.",
        }));
        if let Some(timeout) = properties
            .get_mut("wait_timeout_ms")
            .and_then(serde_json::Value::as_object_mut)
        {
            timeout.insert("description".into(), serde_json::json!(
                "Wait limit only; expiry does not cancel jobs. For spawn_groups and wait_groups, omission waits until a whole group finishes. For wait, omission means 30000 ms and the limit is capped at 60000 ms. Zero polls immediately."
            ));
        }
        // Providers require object-root parameters and do not uniformly support
        // root unions. The tagged Input enum still validates each operation's fields.
        let object = schema.ensure_object();
        object.insert("type".into(), serde_json::json!("object"));
        object.insert("properties".into(), serde_json::Value::Object(properties));
        object.insert("required".into(), serde_json::json!(["operation"]));
        schema
    }

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        _input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        "Coordinate subagents".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        events: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<Self::Output, Self::Output>> {
        cx.spawn(async move |cx| {
            let scope = Rc::new(RefCell::new(Vec::<Weak<Job>>::new()));
            let result: Result<serde_json::Value> = async {
                match input.recv().await? {
                    SubagentsToolInput::Spawn { agents } => {
                        let events = events.clone();
                        let group_id = cx.update(|cx| {
                            let group_id =
                                self.spawn_group_scoped(agents, &events, scope.clone(), cx)?;
                            Self::publish_cards(&scope.borrow(), &events);
                            Ok::<_, anyhow::Error>(group_id)
                        })?;
                        self.wait_group(&group_id, SubagentWaitMode::Any, None, &events, cx)
                            .await
                    }
                    SubagentsToolInput::SpawnGroups {
                        groups,
                        wait_timeout_ms,
                    } => {
                        let events = events.clone();
                        let group_ids = cx.update(|cx| {
                            self.spawn_groups_scoped(groups, &events, scope.clone(), cx)
                        })?;
                        self.wait_groups(group_ids, wait_timeout_ms, &events, cx)
                            .await
                    }
                    SubagentsToolInput::WaitGroups {
                        group_ids,
                        wait_timeout_ms,
                    } => {
                        self.wait_groups(group_ids, wait_timeout_ms, &events, cx)
                            .await
                    }
                    SubagentsToolInput::Wait {
                        group_id,
                        mode,
                        wait_timeout_ms,
                    } => {
                        self.wait_group(&group_id, mode, wait_timeout_ms, &events, cx)
                            .await
                    }
                    SubagentsToolInput::Inspect { session_id } => cx.update(|cx| {
                        self.environment
                            .resume_subagent(session_id, cx)?
                            .inspect(cx)
                    }),
                    SubagentsToolInput::List {} => {
                        cx.update(|cx| Ok(serde_json::json!({ "groups": self.group_statuses(cx) })))
                    }
                }
            }
            .await;
            let cards = scope
                .borrow()
                .iter()
                .filter_map(Weak::upgrade)
                .filter_map(|job| job.card())
                .collect::<Vec<_>>();
            match result {
                Ok(output) if !cards.is_empty() => {
                    Ok(SubagentsToolOutput::with_cards(output, &cards))
                }
                Ok(output) => Ok(SubagentsToolOutput(output)),
                Err(error) => Err(SubagentsToolOutput(
                    serde_json::json!({ "error": error.to_string() }),
                )),
            }
        })
    }

    fn replay(
        &self,
        _input: Self::Input,
        output: Self::Output,
        events: ToolCallEventStream,
        _cx: &mut App,
    ) -> Result<()> {
        let cards = output.0.get(SUBAGENT_CARDS_OUTPUT_KEY).cloned();
        let content = output.model_facing_text();
        events.update_fields(|update| {
            let update = update.content(vec![content.into()]);
            match cards {
                Some(cards) => update.meta(acp::Meta::from_iter([(
                    SUBAGENT_SESSION_CARDS_META_KEY.into(),
                    cards,
                )])),
                None => update,
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::thread::SubagentHandle;
    use futures::channel::oneshot;
    use gpui::{AsyncApp, TestAppContext};
    use serde_json::json;

    struct ControlledHandle {
        id: acp::SessionId,
        receiver: RefCell<Option<oneshot::Receiver<String>>>,
        cancellation: RefCell<Option<oneshot::Sender<()>>>,
        sends: Rc<std::cell::Cell<usize>>,
        entries: Rc<Cell<usize>>,
    }

    impl SubagentHandle for ControlledHandle {
        fn id(&self) -> acp::SessionId {
            self.id.clone()
        }
        fn num_entries(&self, _cx: &App) -> usize {
            self.entries.get()
        }
        fn cancel(&self, _cx: &mut App) -> Task<()> {
            if let Some(sender) = self.cancellation.borrow_mut().take()
                && sender.send(()).is_err()
            {
                log::debug!("Controlled subagent already stopped");
            }
            Task::ready(())
        }
        fn send(&self, _message: String, cx: &AsyncApp) -> Task<Result<String>> {
            use futures::FutureExt as _;
            self.sends.set(self.sends.get() + 1);
            let receiver = self.receiver.borrow_mut().take().expect("one send per job");
            let (sender, cancelled) = oneshot::channel();
            self.cancellation.replace(Some(sender));
            let entries = self.entries.clone();
            cx.spawn(async move |_| {
                futures::select_biased! {
                    _ = cancelled.fuse() => Err(anyhow!("User canceled")),
                    result = receiver.fuse() => {
                        let output = result?;
                        entries.set(entries.get() + 1);
                        Ok(output)
                    },
                }
            })
        }
    }

    #[derive(Default)]
    struct Environment {
        sends: Rc<std::cell::Cell<usize>>,
        senders: RefCell<Vec<oneshot::Sender<String>>>,
        models: RefCell<Vec<Option<AgentModelId>>>,
    }

    impl ThreadEnvironment for Environment {
        fn create_terminal(
            &self,
            _command: String,
            _environment: Vec<(String, String)>,
            _cwd: Option<std::path::PathBuf>,
            _limit: Option<u64>,
            _sandbox: Option<acp_thread::SandboxWrap>,
            _cx: &mut AsyncApp,
        ) -> Task<Result<Rc<dyn crate::TerminalHandle>>> {
            Task::ready(Err(anyhow!("not supported")))
        }
        fn create_subagent(
            &self,
            label: String,
            model: Option<AgentModelId>,
            _cx: &mut App,
        ) -> Result<Rc<dyn SubagentHandle>> {
            ensure!(label != "unavailable", "Model unavailable");
            let (sender, receiver) = oneshot::channel();
            self.senders.borrow_mut().push(sender);
            self.models.borrow_mut().push(model);
            Ok(Rc::new(ControlledHandle {
                id: acp::SessionId::new(label),
                receiver: RefCell::new(Some(receiver)),
                cancellation: RefCell::new(None),
                sends: self.sends.clone(),
                entries: Rc::new(Cell::new(0)),
            }))
        }
    }

    #[test]
    fn provider_schema_has_meaningful_object_properties() {
        let mut schema = SubagentsTool::input_schema().to_value();
        language_model::tool_schema::normalize_tool_schema(&mut schema);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], json!(["operation"]));
        assert!(schema.get("oneOf").is_none());
        let properties = &schema["properties"];
        assert_eq!(properties["operation"]["type"], "string");
        assert_eq!(
            properties["operation"]["enum"],
            json!([
                "spawn",
                "spawn_groups",
                "wait_groups",
                "wait",
                "inspect",
                "list"
            ])
        );
        assert_eq!(properties["agents"]["type"], "array");
        assert_eq!(properties["groups"]["type"], "array");
        assert_eq!(properties["groups"]["items"]["type"], "array");
        assert_eq!(
            properties["groups"]["items"]["items"]["properties"]["message"]["type"],
            "string"
        );
        for field in ["agents", "groups", "group_id", "session_id"] {
            assert!(
                properties[field]["description"]
                    .as_str()
                    .expect("conditional requirement")
                    .starts_with("Required for operation ")
            );
        }
        for field in ["group_ids", "wait_timeout_ms", "mode"] {
            assert!(
                properties.get(field).is_some(),
                "missing operation field {field}"
            );
        }
        assert_eq!(properties["mode"]["enum"], json!(["any", "all"]));
    }

    #[test]
    fn tagged_input_preserves_operation_specific_validation() {
        for input in [
            json!({"operation": "spawn"}),
            json!({"operation": "spawn_groups"}),
            json!({"operation": "wait"}),
            json!({"operation": "inspect"}),
            json!({"operation": "unknown"}),
            json!({"operation": "spawn_groups", "groups": "not an array"}),
            json!({"operation": "wait", "group_id": "group", "mode": "unknown"}),
        ] {
            assert!(
                serde_json::from_value::<SubagentsToolInput>(input.clone()).is_err(),
                "accepted invalid operation input: {input}"
            );
        }
        for input in [
            json!({"operation": "spawn", "agents": [{"label": "task", "message": "prompt"}]}),
            json!({"operation": "spawn_groups", "groups": [[{"label": "task", "message": "prompt"}]]}),
            json!({"operation": "wait_groups"}),
            json!({"operation": "wait", "group_id": "group"}),
            json!({"operation": "inspect", "session_id": "session"}),
            json!({"operation": "list"}),
        ] {
            assert!(
                serde_json::from_value::<SubagentsToolInput>(input.clone()).is_ok(),
                "rejected valid operation input: {input}"
            );
        }
    }

    fn agent(label: &str) -> SpawnAgentToolInput {
        SpawnAgentToolInput {
            label: label.into(),
            message: label.into(),
            ..Default::default()
        }
    }

    fn cards_in_update(update: acp::ToolCallUpdate) -> Vec<SubagentSessionCard> {
        update
            .meta
            .value()
            .and_then(|meta| meta.get(SUBAGENT_SESSION_CARDS_META_KEY))
            .cloned()
            .and_then(|cards| serde_json::from_value(cards).ok())
            .expect("card metadata in tool update")
    }

    #[gpui::test]
    async fn output_replay_restores_cards_without_exposing_them_to_the_model(
        cx: &mut TestAppContext,
    ) {
        let cards = vec![SubagentSessionCard {
            session_info: SubagentSessionInfo {
                session_id: acp::SessionId::new("child"),
                message_start_index: 3,
                message_end_index: Some(5),
            },
            label: "Review".into(),
            status: SubagentSessionCardStatus::Completed,
            error: None,
        }];
        let output = SubagentsToolOutput::with_cards(json!({"result": "done"}), &cards);
        let replayed: SubagentsToolOutput =
            serde_json::from_value(serde_json::to_value(output).expect("persist output"))
                .expect("replay output");
        assert_eq!(replayed.0[SUBAGENT_CARDS_OUTPUT_KEY][0]["label"], "Review");
        let tool = SubagentsTool::new(Rc::new(Environment::default()));
        let (events, mut receiver) = ToolCallEventStream::test();
        cx.update(|cx| {
            tool.replay(SubagentsToolInput::List {}, replayed.clone(), events, cx)
                .expect("replay succeeds");
        });
        let replayed_cards = cards_in_update(receiver.expect_update_fields().await);
        assert_eq!(replayed_cards.len(), 1);
        assert_eq!(replayed_cards[0].label, "Review");
        assert_eq!(replayed_cards[0].session_info.message_end_index, Some(5));
        let model_text = replayed.model_facing_text();
        assert!(model_text.contains("done"));
        assert!(!model_text.contains(SUBAGENT_CARDS_OUTPUT_KEY));
    }

    #[gpui::test]
    async fn dropping_coordinator_releases_retained_subagent_jobs(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = SubagentsTool::new(environment.clone());
        let group_id = cx.update(|cx| {
            tool.spawn_group(vec![agent("pending")], &ToolCallEventStream::test().0, cx)
                .expect("spawn child")
        });
        let weak_job = Rc::downgrade(
            tool.jobs
                .groups
                .borrow()
                .get(&group_id)
                .and_then(|jobs| jobs.first())
                .expect("retained job"),
        );
        cx.run_until_parked();
        assert_eq!(environment.sends.get(), 1);
        assert!(weak_job.upgrade().is_some());

        drop(tool);
        cx.run_until_parked();
        assert!(weak_job.upgrade().is_none());
    }

    #[gpui::test]
    async fn spawn_cards_are_scoped_and_update_independently(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = Arc::new(SubagentsTool::new(environment.clone()));
        let (events, mut receiver) = ToolCallEventStream::test();
        let first_spawn = cx.update(|cx| {
            tool.clone().run(
                ToolInput::resolved(SubagentsToolInput::SpawnGroups {
                    groups: vec![vec![agent("first"), agent("second")]],
                    wait_timeout_ms: Some(0),
                }),
                events,
                cx,
            )
        });
        cx.run_until_parked();
        let initial_update = receiver.expect_update_fields().await;
        let initial_cards = cards_in_update(initial_update);
        assert_eq!(initial_cards.len(), 2);
        assert!(
            initial_cards
                .iter()
                .all(|card| matches!(card.status, SubagentSessionCardStatus::Running))
        );
        let persisted = serde_json::to_value(first_spawn.await.expect("spawn result").0)
            .expect("persisted output");
        assert_eq!(
            persisted[SUBAGENT_CARDS_OUTPUT_KEY]
                .as_array()
                .expect("persisted cards")
                .len(),
            2
        );

        environment
            .senders
            .borrow_mut()
            .remove(0)
            .send("first result".into())
            .expect("child receiver");
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(10));
        cx.run_until_parked();
        let updated_cards = cards_in_update(receiver.expect_update_fields().await);
        assert_eq!(updated_cards.len(), 2);
        assert!(matches!(
            updated_cards[0].status,
            SubagentSessionCardStatus::Completed
        ));
        assert_eq!(updated_cards[0].session_info.message_end_index, Some(0));
        assert!(matches!(
            updated_cards[1].status,
            SubagentSessionCardStatus::Running
        ));

        let (second_events, mut second_receiver) = ToolCallEventStream::test();
        let mut second_spawn = cx.update(|cx| {
            tool.clone().run(
                ToolInput::resolved(SubagentsToolInput::Spawn {
                    agents: vec![agent("third")],
                }),
                second_events,
                cx,
            )
        });
        cx.run_until_parked();
        let second_cards = cards_in_update(second_receiver.expect_update_fields().await);
        assert_eq!(second_cards.len(), 1);
        assert_eq!(second_cards[0].label, "third");
        assert!(futures::poll!(&mut second_spawn).is_pending());

        tool.jobs.cancel();
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn spawn_groups_returns_first_whole_group_and_all_other_statuses(
        cx: &mut TestAppContext,
    ) {
        let environment = Rc::new(Environment::default());
        let tool = Arc::new(SubagentsTool::new(environment.clone()));
        let older = cx.update(|cx| {
            tool.spawn_group(vec![agent("older")], &ToolCallEventStream::test().0, cx)
                .expect("older group")
        });
        let mut spawned = cx.update(|cx| {
            tool.clone().run(
                ToolInput::resolved(SubagentsToolInput::SpawnGroups {
                    groups: vec![
                        vec![agent("slow-first"), agent("slow-second")],
                        vec![agent("fast")],
                    ],
                    wait_timeout_ms: None,
                }),
                ToolCallEventStream::test().0,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(futures::poll!(&mut spawned).is_pending());
        let fast = environment.senders.borrow_mut().pop().expect("fast sender");
        environment
            .senders
            .borrow_mut()
            .pop()
            .expect("slow second sender")
            .send("partial slow group".into())
            .expect("receiver");
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(10));
        cx.run_until_parked();
        assert!(
            futures::poll!(&mut spawned).is_pending(),
            "one agent does not complete a two-agent group"
        );
        fast.send("fast group output".into()).expect("receiver");
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(10));
        cx.run_until_parked();
        let spawned = spawned.await.expect("first whole-group boundary").0;
        let completed = spawned["completed_groups"]
            .as_array()
            .expect("completed groups");
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0]["finished"][0]["output"], "fast group output");
        let statuses = spawned["groups"].as_array().expect("all statuses");
        assert_eq!(statuses.len(), 3);
        assert_eq!(
            statuses
                .iter()
                .filter(|group| group["status"] == "running")
                .count(),
            2
        );
        assert!(
            statuses
                .iter()
                .any(|group| group["group_id"] == older && group["status"] == "running")
        );
        let partial = cx
            .update(|cx| {
                tool.clone().run(
                    ToolInput::resolved(SubagentsToolInput::Wait {
                        group_id: spawned["group_ids"][0]
                            .as_str()
                            .expect("slow group id")
                            .to_owned(),
                        mode: SubagentWaitMode::Any,
                        wait_timeout_ms: Some(0),
                    }),
                    ToolCallEventStream::test().0,
                    cx,
                )
            })
            .await
            .expect("legacy wait reports all groups")
            .0;
        assert_eq!(partial["groups"].as_array().expect("all statuses").len(), 3);
        assert_eq!(
            partial["finished"]
                .as_array()
                .expect("partial results")
                .len(),
            1
        );
        assert_eq!(
            partial["remaining"]
                .as_array()
                .expect("running slow agent")
                .len(),
            1
        );
        let mut next = cx.update(|cx| {
            tool.clone().run(
                ToolInput::resolved(SubagentsToolInput::WaitGroups {
                    group_ids: vec![],
                    wait_timeout_ms: None,
                }),
                ToolCallEventStream::test().0,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(
            futures::poll!(&mut next).is_pending(),
            "do not repeat the already-reported fast group"
        );
        environment
            .senders
            .borrow_mut()
            .pop()
            .expect("slow first sender")
            .send("slow group output".into())
            .expect("receiver");
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(10));
        cx.run_until_parked();
        let next = next.await.expect("next boundary").0;
        assert_eq!(
            next["completed_groups"][0]["finished"]
                .as_array()
                .expect("slow outputs")
                .len(),
            2
        );
        assert_eq!(
            next["groups"]
                .as_array()
                .expect("statuses")
                .iter()
                .filter(|group| group["status"] == "running")
                .count(),
            1
        );
        assert_eq!(
            environment.sends.get(),
            4,
            "completion does not start additional jobs"
        );
        tool.jobs.cancel();
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn legacy_spawn_waits_and_includes_other_groups(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = Arc::new(SubagentsTool::new(environment.clone()));
        cx.update(|cx| {
            tool.spawn_group(vec![agent("older")], &ToolCallEventStream::test().0, cx)
                .expect("older group")
        });
        let mut spawned = cx.update(|cx| {
            tool.clone().run(
                ToolInput::resolved(SubagentsToolInput::Spawn {
                    agents: vec![agent("new")],
                }),
                ToolCallEventStream::test().0,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(futures::poll!(&mut spawned).is_pending());
        environment
            .senders
            .borrow_mut()
            .pop()
            .expect("new sender")
            .send("new output".into())
            .expect("receiver");
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(10));
        cx.run_until_parked();
        let result = spawned.await.expect("first agent boundary").0;
        assert_eq!(result["finished"][0]["output"], "new output");
        assert_eq!(result["groups"].as_array().expect("all groups").len(), 2);
        tool.jobs.cancel();
        cx.run_until_parked();
    }

    #[gpui::test]
    fn spawn_groups_validates_all_groups_before_starting(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = SubagentsTool::new(environment.clone());
        assert!(
            cx.update(|cx| tool.spawn_groups(
                vec![vec![agent("valid")], vec![]],
                &ToolCallEventStream::test().0,
                cx
            ))
            .is_err()
        );
        assert!(tool.jobs.groups.borrow().is_empty());
        assert!(environment.senders.borrow().is_empty());
    }

    #[test]
    fn retained_outputs_are_bounded_at_storage_time() {
        let output = RetainedJobOutput::new("雪".repeat(100), 10);
        assert_eq!(output.text.chars().count(), 10);
        assert!(output.truncated);
        assert!(!RetainedJobOutput::new("exact".into(), 5).truncated);
    }

    #[gpui::test]
    async fn group_cancellation_covers_jobs_that_have_not_started(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = SubagentsTool::new(environment.clone());
        let agents = vec![SpawnAgentToolInput {
            label: "pending".into(),
            message: "prompt".into(),
            ..Default::default()
        }];
        let group_id = cx.update(|cx| {
            tool.spawn_group(agents, &ToolCallEventStream::test().0, cx)
                .expect("spawn")
        });
        tool.jobs.cancel();
        cx.run_until_parked();
        assert_eq!(environment.sends.get(), 0);
        let snapshot = cx.update(|cx| tool.snapshot(&group_id, false, cx).expect("snapshot"));
        assert_eq!(snapshot["finished"][0]["error"], "User canceled");
        assert_eq!(snapshot["remaining"], json!([]));
    }

    #[gpui::test]
    async fn group_cancellation_drains_running_jobs(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = SubagentsTool::new(environment.clone());
        let agents = vec![SpawnAgentToolInput {
            label: "running".into(),
            ..Default::default()
        }];
        let group_id = cx.update(|cx| {
            tool.spawn_group(agents, &ToolCallEventStream::test().0, cx)
                .expect("spawn")
        });
        cx.run_until_parked();
        assert_eq!(environment.sends.get(), 1);
        tool.jobs.cancel();
        cx.run_until_parked();
        let snapshot = cx.update(|cx| tool.snapshot(&group_id, false, cx).expect("snapshot"));
        assert_eq!(snapshot["finished"][0]["error"], "User canceled");
        assert_eq!(snapshot["remaining"], json!([]));
    }

    #[gpui::test]
    async fn group_creation_failures_are_retained_per_job(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = SubagentsTool::new(environment);
        let agents = vec![
            SpawnAgentToolInput {
                label: "unavailable".into(),
                ..Default::default()
            },
            SpawnAgentToolInput {
                label: "pending".into(),
                ..Default::default()
            },
        ];
        let group_id = cx.update(|cx| {
            tool.spawn_group(agents, &ToolCallEventStream::test().0, cx)
                .expect("partial group")
        });
        let snapshot = cx.update(|cx| tool.snapshot(&group_id, false, cx).expect("snapshot"));
        assert_eq!(snapshot["finished"][0]["status"], "failed");
        assert_eq!(
            snapshot["finished"][0]["session_id"],
            serde_json::Value::Null
        );
        assert_eq!(
            snapshot["remaining"].as_array().expect("remaining").len(),
            1
        );
        tool.jobs.cancel();
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn groups_persist_and_wait_returns_finished_and_remaining(cx: &mut TestAppContext) {
        let environment = Rc::new(Environment::default());
        let tool = Arc::new(SubagentsTool::new(environment.clone()));
        let agents: Vec<SpawnAgentToolInput> = serde_json::from_value(json!([
            {"label": "first", "message": "first task", "model": "provider/explicit"},
            {"label": "second", "message": "second task"}
        ]))
        .expect("valid inputs");
        let group_id = cx.update(|cx| {
            tool.spawn_group(agents, &ToolCallEventStream::test().0, cx)
                .expect("spawn group")
        });
        let initial = cx.update(|cx| tool.snapshot(&group_id, false, cx).expect("group exists"));
        assert_eq!(initial["remaining"].as_array().expect("remaining").len(), 2);
        assert_eq!(
            environment.models.borrow().first(),
            Some(&Some(AgentModelId::from("provider/explicit")))
        );
        cx.run_until_parked();
        environment
            .senders
            .borrow_mut()
            .pop()
            .expect("second sender")
            .send("second output".into())
            .expect("running receiver");
        cx.run_until_parked();
        let any = cx
            .update(|cx| {
                tool.clone().run(
                    ToolInput::resolved(SubagentsToolInput::Wait {
                        group_id: group_id.clone(),
                        mode: SubagentWaitMode::Any,
                        wait_timeout_ms: Some(0),
                    }),
                    ToolCallEventStream::test().0,
                    cx,
                )
            })
            .await
            .expect("wait any");
        assert_eq!(any.0["finished"].as_array().expect("finished").len(), 1);
        assert_eq!(any.0["remaining"].as_array().expect("remaining").len(), 1);
        assert_eq!(any.0["finished"][0]["output"], "second output");
        let mut all = cx.update(|cx| {
            tool.clone().run(
                ToolInput::resolved(SubagentsToolInput::Wait {
                    group_id: group_id.clone(),
                    mode: SubagentWaitMode::All,
                    wait_timeout_ms: Some(1000),
                }),
                ToolCallEventStream::test().0,
                cx,
            )
        });
        cx.run_until_parked();
        assert!(futures::poll!(&mut all).is_pending());
        environment
            .senders
            .borrow_mut()
            .pop()
            .expect("first sender")
            .send("first output".into())
            .expect("running receiver");
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_millis(10));
        cx.run_until_parked();
        let all = all.await.expect("wait all");
        assert_eq!(all.0["finished"].as_array().expect("finished").len(), 2);
        assert_eq!(all.0["remaining"], json!([]));
        assert_eq!(all.0["wait_expired"], false);
    }
}
