use std::{
    cell::RefCell,
    collections::VecDeque,
    future::Future,
    pin::Pin,
    rc::Rc,
    sync::Arc,
    task::{Context, Poll, Waker},
    time::Instant,
};

use agent_client_protocol::v2::schema as acp_v2;
use agent_settings::AgentSettings;
use anyhow::Result;
use futures::{FutureExt as _, channel::oneshot};
use gpui::{App, Entity, Global, SharedString, Task, WeakEntity};
use project::Project;
use settings::Settings as _;

use super::terminal_tool::{SandboxedTerminalToolInput, run_queued_terminal_tool};
use crate::thread::{ThreadActivityGuard, ThreadActivityRegistry};
use crate::{AgentTool, Thread, ThreadEnvironment, ToolCallEventStream, ToolInput};

mod global_queue;

#[derive(Clone, Default)]
struct BuildQueue(Rc<RefCell<QueueState>>);

impl Global for BuildQueue {}

#[derive(Default)]
struct QueueState {
    active: bool,
    cleanup_error: Option<String>,
    pending: VecDeque<QueueEntry>,
    cleanup_waiters: Vec<std::rc::Weak<RefCell<Option<Waker>>>>,
}

struct QueueEntry {
    ticket: Rc<()>,
    #[cfg(test)]
    session_id: acp_v2::SessionId,
    waker: Option<Waker>,
}

impl BuildQueue {
    fn enqueue(&self, _session_id: acp_v2::SessionId) -> QueueWaiter {
        let ticket = Rc::new(());
        self.0.borrow_mut().pending.push_back(QueueEntry {
            ticket: ticket.clone(),
            #[cfg(test)]
            session_id: _session_id,
            waker: None,
        });
        QueueWaiter {
            queue: self.clone(),
            ticket,
            activity_guard: None,
            executor: None,
        }
    }

    #[cfg(test)]
    fn enqueue_tracked(&self, session_id: acp_v2::SessionId, cx: &App) -> Result<QueueWaiter> {
        let activity_guard = ThreadActivityRegistry::queue_wait(&session_id, cx)?;
        let mut waiter = self.enqueue(session_id);
        waiter.activity_guard = Some(activity_guard);
        waiter.executor = Some(cx.background_executor().clone());
        Ok(waiter)
    }

    async fn cleanup_failure(&self) -> String {
        let waker = Rc::new(RefCell::new(None));
        {
            let mut state = self.0.borrow_mut();
            state
                .cleanup_waiters
                .retain(|waiter| waiter.strong_count() > 0);
            state.cleanup_waiters.push(Rc::downgrade(&waker));
        }
        futures::future::poll_fn(|context| {
            if let Some(error) = self.0.borrow().cleanup_error.clone() {
                return Poll::Ready(error);
            }
            *waker.borrow_mut() = Some(context.waker().clone());
            Poll::Pending
        })
        .await
    }

    fn wake_front(&self) {
        let waker = self
            .0
            .borrow()
            .pending
            .front()
            .and_then(|entry| entry.waker.clone());
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}

/// True only while this session has a job waiting for the process-wide build slot.
/// Running builds do not pause the orchestration timeout.
#[cfg(test)]
fn is_queue_waiting(session_id: &acp_v2::SessionId, cx: &App) -> bool {
    cx.try_global::<BuildQueue>().is_some_and(|queue| {
        queue
            .0
            .borrow()
            .pending
            .iter()
            .any(|entry| &entry.session_id == session_id)
    })
}

struct QueueWaiter {
    queue: BuildQueue,
    ticket: Rc<()>,
    activity_guard: Option<ThreadActivityGuard>,
    executor: Option<gpui::BackgroundExecutor>,
}

impl Future for QueueWaiter {
    type Output = Result<QueuePermit, String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.queue.0.borrow_mut();
        if let Some(error) = state.cleanup_error.clone() {
            state
                .pending
                .retain(|entry| !Rc::ptr_eq(&entry.ticket, &self.ticket));
            drop(state);
            self.activity_guard.take();
            return Poll::Ready(Err(error));
        }
        if !state.active
            && state
                .pending
                .front()
                .is_some_and(|entry| Rc::ptr_eq(&entry.ticket, &self.ticket))
        {
            state.pending.pop_front();
            state.active = true;
            drop(state);
            self.activity_guard.take();
            let started_at = self.executor.as_ref().map(|executor| executor.now());
            return Poll::Ready(Ok(QueuePermit(self.queue.clone(), started_at, None)));
        }
        if let Some(entry) = state
            .pending
            .iter_mut()
            .find(|entry| Rc::ptr_eq(&entry.ticket, &self.ticket))
        {
            entry.waker = Some(cx.waker().clone());
        }
        Poll::Pending
    }
}

impl Drop for QueueWaiter {
    fn drop(&mut self) {
        self.queue
            .0
            .borrow_mut()
            .pending
            .retain(|entry| !Rc::ptr_eq(&entry.ticket, &self.ticket));
        self.queue.wake_front();
    }
}

struct QueuePermit(BuildQueue, Option<Instant>, Option<GlobalPermit>);

struct GlobalPermit {
    ticket: Option<global_queue::Ticket>,
    executor: gpui::BackgroundExecutor,
    remove_on_drop: bool,
}

impl QueuePermit {
    async fn quarantine_global(
        &mut self,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<(), String> {
        let Some(global) = self.2.as_mut() else {
            return Ok(());
        };
        let Some(ticket) = global.ticket.take() else {
            return Err("Missing build queue lease during quarantine".into());
        };
        let worker_executor = executor.clone();
        let (ticket, result) = executor
            .spawn(async move {
                let result = loop {
                    match ticket.try_quarantine() {
                        Ok(true) => break Ok(()),
                        Ok(false) => {
                            worker_executor
                                .timer(std::time::Duration::from_millis(25))
                                .await
                        }
                        Err(error) => {
                            break Err(format!("Cannot quarantine host-wide build queue: {error}"));
                        }
                    }
                };
                (ticket, result)
            })
            .await;
        global.ticket = Some(ticket);
        result
    }

    fn confirm_global_exit(&mut self) {
        if let Some(permit) = self.2.as_mut() {
            permit.remove_on_drop = true;
        }
    }
}

impl Drop for GlobalPermit {
    fn drop(&mut self) {
        if let Some(ticket) = self.ticket.take() {
            let executor = self.executor.clone();
            let remove_on_drop = self.remove_on_drop;
            self.executor
                .spawn(async move {
                    if !remove_on_drop {
                        // A worker dropped by shutdown or unwinding cannot vouch
                        // for its terminal. Keep the active marker for recovery.
                        log::error!(
                            "Build queue worker dropped without confirmed terminal cleanup"
                        );
                        drop(ticket);
                        return;
                    }
                    loop {
                        match ticket.try_remove() {
                            Ok(true) => break,
                            Ok(false) => executor.timer(std::time::Duration::from_millis(25)).await,
                            Err(error) => {
                                // A stale queued ticket is recoverable. An admitted
                                // ticket instead fails closed if removal failed.
                                log::error!("Cannot remove build queue ticket: {error}");
                                break;
                            }
                        }
                    }
                    drop(ticket);
                })
                .detach();
        }
    }
}

async fn acquire_global(
    executor: gpui::BackgroundExecutor,
    directory: std::path::PathBuf,
) -> Result<GlobalPermit, String> {
    let (cancellation_sender, cancellation_receiver) = oneshot::channel::<()>();
    let (result_sender, result_receiver) = oneshot::channel();
    let worker_executor = executor.clone();
    executor.spawn(async move {
        let acquisition = async {
            let queue = global_queue::Queue::open(directory)?;
            let ticket = loop {
                if let Some(ticket) = queue.register()? {
                    break ticket;
                }
                worker_executor.timer(std::time::Duration::from_millis(25)).await;
            };
            let permit = GlobalPermit {
                ticket: Some(ticket),
                executor: worker_executor.clone(),
                remove_on_drop: true,
            };
            let ticket = permit.ticket.as_ref()
                .ok_or_else(|| std::io::Error::other("Missing build queue lease"))?;
            loop {
                if ticket.try_acquire()? {
                    return Ok::<_, std::io::Error>(permit);
                }
                worker_executor.timer(std::time::Duration::from_millis(25)).await;
            }
        };
        let result = futures::select_biased! {
            _ = cancellation_receiver.fuse() => return,
            result = acquisition.fuse() => result.map_err(|error| format!("Cannot acquire host-wide build queue: {error}")),
        };
        // A dropped receiver drops the permit, which schedules ticket cleanup.
        if result_sender.send(result).is_err() {
            log::debug!("Build queue admission caller was cancelled");
        }
    }).detach();
    let result = result_receiver.await.map_err(|error| error.to_string())?;
    drop(cancellation_sender);
    result
}

impl Drop for QueuePermit {
    fn drop(&mut self) {
        {
            let mut state = self.0.0.borrow_mut();
            state.active = false;
            state.cleanup_error = None;
        }
        self.0.wake_front();
    }
}

pub struct BuildQueueTool {
    project: Entity<Project>,
    environment: Rc<dyn ThreadEnvironment>,
    thread: WeakEntity<Thread>,
}

impl BuildQueueTool {
    pub fn new(
        project: Entity<Project>,
        environment: Rc<dyn ThreadEnvironment>,
        thread: WeakEntity<Thread>,
    ) -> Self {
        Self {
            project,
            environment,
            thread,
        }
    }
}

impl AgentTool for BuildQueueTool {
    // This is a superset of the legacy single-command terminal input, including
    // the sandbox permission requests rather than inventing a separate policy.
    type Input = SandboxedTerminalToolInput;
    type Output = String;

    const NAME: &'static str = "build_queue";

    fn description() -> SharedString {
        format!(
            "Submit one build command to the same-user host-wide FIFO build queue and wait for its result. \
             Use this for resource-intensive builds that must not overlap other queued builds. \
             Available by default without configuration; users can disable it in global settings. \
             Queued execution requires a local Linux terminal. Commands must remain in their \
             original process group: do not daemonize, use setsid, or create job-control groups. \
             timeout_ms defaults to \
             the global agent.build_queue_default_timeout_ms and must be positive and no greater \
             than agent.build_queue_max_timeout_ms. The deadline starts when the slot is acquired \
             and covers authorization, sandbox preparation, startup, and execution, but not queue wait. \
             Output is byte-bounded even when head_lines or tail_lines is set. Cancelling a running \
             job kills it and holds the queue slot until exit is confirmed. If termination cannot \
             be confirmed within five seconds, the call returns an error and the queue rejects jobs \
             until background cleanup confirms exit.\n\n{}",
            super::terminal_tool::SandboxedTerminalTool::description(),
        ).into()
    }

    fn kind() -> acp_v2::ToolKind {
        acp_v2::ToolKind::Execute
    }

    fn allow_in_restricted_mode() -> bool {
        false
    }

    fn initial_title(
        &self,
        input: Result<Self::Input, serde_json::Value>,
        _cx: &mut App,
    ) -> SharedString {
        input.map(|input| input.command).unwrap_or_default().into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        event_stream: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        cx.spawn(async move |cx| {
            let mut input = input.recv().await.map_err(|error| error.to_string())?;
            let (session_id, activity_guard, queue) = cx.update(|cx| {
                let settings = AgentSettings::get_global(cx);
                if !settings.build_queue_enabled {
                    return Err("The global build queue was disabled in global user settings (agent.build_queue_enabled).".to_string());
                }
                input.timeout_ms = Some(settings.build_queue_timeout_ms(input.timeout_ms)
                    .map_err(|error| error.to_string())?);
                let session_id = self.thread.read_with(cx, |thread, _| thread.id().clone())
                    .map_err(|error| error.to_string())?;
                if cx.try_global::<BuildQueue>().is_none() {
                    cx.set_global(BuildQueue::default());
                }
                let activity_guard = ThreadActivityRegistry::queue_wait(&session_id, cx)
                    .map_err(|error| format!("Cannot track build queue wait: {error:#}"))?;
                Ok((session_id, activity_guard, cx.global::<BuildQueue>().clone()))
            })?;

            let executor = cx.background_executor().clone();
            let mut global_permit = futures::select_biased! {
                _ = event_stream.cancelled_by_user().fuse() => return Err("Queued build cancelled.".into()),
                error = queue.cleanup_failure().fuse() => return Err(error),
                permit = acquire_global(executor.clone(), global_queue::default_directory()
                    .map_err(|error| error.to_string())?).fuse() => permit?,
            };
            // Global admission orders local admission as well, so a local head
            // can never wait for a global ticket behind its own successor.
            let waiter = cx.update(|cx| cx.global::<BuildQueue>().enqueue(session_id));
            let mut permit = futures::select_biased! {
                _ = event_stream.cancelled_by_user().fuse() => return Err("Queued build cancelled.".into()),
                permit = waiter.fuse() => permit?,
            };
            global_permit.remove_on_drop = false;
            permit.2 = Some(global_permit);
            permit.1 = Some(executor.now());
            drop(activity_guard);
            let (cancellation_sender, cancellation_receiver) = oneshot::channel();
            let (result_sender, result_receiver) = oneshot::channel();
            // Cancellation drops only the caller. The worker must keep the slot
            // until terminal kill has been followed by a confirmed process exit.
            cx.spawn(async move |cx| {
                let result = execute_queued_job(
                    self.project.clone(),
                    self.environment.clone(),
                    input,
                    event_stream,
                    cancellation_receiver,
                    permit,
                    cx,
                ).await;
                if result_sender.send(result).is_err() {
                    log::debug!("Build queue caller was dropped before terminal cleanup completed");
                }
            }).detach();
            let result = result_receiver.await.map_err(|error| error.to_string())?;
            drop(cancellation_sender);
            result
        })
    }
}

async fn execute_queued_job(
    project: Entity<Project>,
    environment: Rc<dyn ThreadEnvironment>,
    input: SandboxedTerminalToolInput,
    event_stream: ToolCallEventStream,
    cancellation: oneshot::Receiver<()>,
    mut permit: QueuePermit,
    cx: &mut gpui::AsyncApp,
) -> Result<String, String> {
    let mut outcome = run_queued_terminal_tool(
        project,
        environment,
        input,
        event_stream,
        cancellation,
        permit.1.unwrap_or_else(|| cx.background_executor().now()),
        cx,
    )
    .await;
    if let Some(cleanup) = outcome.cleanup {
        let error = outcome.result.as_ref().err().cloned()
            .unwrap_or_else(|| "Build termination is unconfirmed; the global build queue is temporarily unavailable.".into());
        log::error!("{error}");
        let wakers =
            {
                let mut state = permit.0.0.borrow_mut();
                state.cleanup_error = Some(error);
                state
                    .pending
                    .iter()
                    .filter_map(|entry| entry.waker.clone())
                    .chain(state.cleanup_waiters.iter().filter_map(|waiter| {
                        waiter.upgrade().and_then(|waker| waker.borrow().clone())
                    }))
                    .collect::<Vec<_>>()
            };
        for waker in wakers {
            waker.wake();
        }
        if let Err(error) = permit.quarantine_global(cx.background_executor()).await {
            log::error!("{error}");
            let original = outcome.result.err().unwrap_or_default();
            outcome.result = Err(format!("{original}\n{error}"));
        }
        cx.spawn(async move |_cx| {
            cleanup.await;
            permit.confirm_global_exit();
            drop(permit);
        })
        .detach();
    } else {
        permit.confirm_global_exit();
        drop(permit);
    }
    outcome.result
}

#[cfg(test)]
mod tests {
    use super::*;

    use gpui::AppContext as _;

    #[gpui::test]
    async fn global_wait_is_cancellable_without_blocking_local_work(cx: &mut gpui::TestAppContext) {
        let directory = tempfile::tempdir().expect("isolated queue directory");
        let path = directory.path().join("queue");
        let executor = cx.background_executor.clone();
        let first = acquire_global(executor.clone(), path.clone())
            .await
            .expect("first lease");
        let second = executor.spawn(acquire_global(executor.clone(), path.clone()));
        cx.run_until_parked();
        // A parked remote waiter must not monopolize a background thread or
        // GPUI's foreground dispatcher, and cancellation must remove its ticket.
        assert_eq!(executor.spawn(async { 42 }).await, 42);
        drop(second);
        cx.run_until_parked();
        let entries = executor
            .spawn({
                let path = path.clone();
                async move { std::fs::read_dir(path).expect("queue directory").count() }
            })
            .await;
        assert_eq!(entries, 2, "only coordinator and active ticket remain");
        drop(first);
        cx.run_until_parked();
        let third = acquire_global(executor.clone(), path.clone())
            .await
            .expect("lease after cancellation");
        drop(third);
        cx.run_until_parked();
        let entries = executor
            .spawn(async move { std::fs::read_dir(path).expect("queue directory").count() })
            .await;
        assert_eq!(entries, 1, "coordinator inode must remain");
    }

    #[gpui::test]
    async fn global_lease_follows_local_permit_lifetime(cx: &mut gpui::TestAppContext) {
        let directory = tempfile::tempdir().expect("isolated queue directory");
        let path = directory.path().join("queue");
        let executor = cx.background_executor.clone();
        let queue = BuildQueue::default();
        let mut permit = queue
            .enqueue(acp_v2::SessionId::new("first"))
            .await
            .expect("local permit");
        permit.2 = Some(
            acquire_global(executor.clone(), path.clone())
                .await
                .expect("global permit"),
        );
        permit.2.as_mut().expect("global permit").remove_on_drop = false;
        let mut successor = executor.spawn(acquire_global(executor.clone(), path));
        cx.run_until_parked();
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(10));
        cx.run_until_parked();
        assert!((&mut successor).now_or_never().is_none());
        // execute_queued_job moves this same permit into detached exit cleanup.
        permit.confirm_global_exit();
        drop(permit);
        cx.run_until_parked();
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(25));
        drop(successor.await.expect("successor after confirmed cleanup"));
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn global_quarantine_retains_lease_until_confirmed_cleanup(
        cx: &mut gpui::TestAppContext,
    ) {
        let directory = tempfile::tempdir().expect("isolated queue directory");
        let path = directory.path().join("queue");
        let executor = cx.background_executor.clone();
        let queue = BuildQueue::default();
        let mut permit = queue
            .enqueue(acp_v2::SessionId::new("first"))
            .await
            .expect("local permit");
        let mut global = acquire_global(executor.clone(), path.clone())
            .await
            .expect("global permit");
        global.remove_on_drop = false;
        permit.2 = Some(global);
        let waiting = executor.spawn(acquire_global(executor.clone(), path.clone()));
        cx.run_until_parked();
        permit
            .quarantine_global(&executor)
            .await
            .expect("publish quarantine");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(25));
        assert!(
            waiting
                .await
                .err()
                .expect("reject existing waiter")
                .contains("temporarily unavailable")
        );
        assert!(
            acquire_global(executor.clone(), path.clone())
                .await
                .err()
                .expect("reject new waiter")
                .contains("temporarily unavailable")
        );
        assert!(permit.2.as_ref().expect("global permit").ticket.is_some());
        permit.confirm_global_exit();
        drop(permit);
        cx.run_until_parked();
        drop(
            acquire_global(executor, path)
                .await
                .expect("recover after cleanup"),
        );
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn abandoned_worker_keeps_active_marker(cx: &mut gpui::TestAppContext) {
        let directory = tempfile::tempdir().expect("isolated queue directory");
        let path = directory.path().join("queue");
        let executor = cx.background_executor.clone();
        let mut permit = acquire_global(executor.clone(), path.clone())
            .await
            .expect("lease");
        permit.remove_on_drop = false;
        drop(permit);
        cx.run_until_parked();
        let error = acquire_global(executor, path)
            .await
            .err()
            .expect("fail closed");
        assert!(error.contains("unconfirmed build"));
    }

    struct DelayedExitTerminal {
        terminal: crate::tests::FakeTerminalHandle,
        kill_requested: std::cell::Cell<bool>,
        kill_fails: bool,
        tracking_failures: std::cell::Cell<usize>,
    }

    impl crate::TerminalHandle for DelayedExitTerminal {
        fn id(&self, cx: &gpui::AsyncApp) -> Result<acp_v2::TerminalId> {
            self.terminal.id(cx)
        }

        fn current_output(&self, cx: &gpui::AsyncApp) -> Result<acp_thread::ProcessOutput> {
            self.terminal.current_output(cx)
        }

        fn wait_for_exit(
            &self,
            cx: &gpui::AsyncApp,
        ) -> Result<futures::future::Shared<Task<acp_v2::TerminalExitStatus>>> {
            let remaining = self.tracking_failures.get();
            if remaining > 0 {
                self.tracking_failures.set(remaining - 1);
                anyhow::bail!("simulated transient exit tracking failure");
            }
            self.terminal.wait_for_exit(cx)
        }

        fn kill(&self, _cx: &gpui::AsyncApp) -> Result<()> {
            self.kill_requested.set(true);
            anyhow::ensure!(!self.kill_fails, "simulated kill failure");
            Ok(())
        }

        fn was_stopped_by_user(&self, cx: &gpui::AsyncApp) -> Result<bool> {
            self.terminal.was_stopped_by_user(cx)
        }
    }

    struct DelayedExitEnvironment(Rc<DelayedExitTerminal>);

    impl ThreadEnvironment for DelayedExitEnvironment {
        fn create_terminal(
            &self,
            _command: String,
            _extra_env: Vec<(String, String)>,
            _cwd: Option<std::path::PathBuf>,
            _output_byte_limit: Option<u64>,
            _sandbox_wrap: Option<acp_thread::SandboxWrap>,
            _cx: &mut gpui::AsyncApp,
        ) -> Task<Result<Rc<dyn crate::TerminalHandle>>> {
            Task::ready(Ok(self.0.clone()))
        }

        fn create_subagent(
            &self,
            _label: String,
            _model: Option<acp_thread::AgentModelId>,
            _cx: &mut App,
        ) -> Result<Rc<dyn crate::SubagentHandle>> {
            anyhow::bail!("not used by build queue tests")
        }
    }

    #[gpui::test]
    async fn timeout_and_caller_cancellation_hold_slot_until_confirmed_exit(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/root", serde_json::json!({})).await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.tool_permissions.default = settings::ToolPermissionMode::Allow;
            settings
                .tool_permissions
                .tools
                .remove(super::super::terminal_tool::TerminalTool::NAME);
            AgentSettings::override_global(settings, cx);
        });

        for (cancel_caller, kill_fails) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let queue = BuildQueue::default();
            let blocking_permit = queue
                .enqueue(acp_v2::SessionId::new("blocking"))
                .await
                .expect("initial permit");
            let first = queue.enqueue(acp_v2::SessionId::new("first"));
            let mut second = Box::pin(queue.enqueue(acp_v2::SessionId::new("second")));
            let terminal = Rc::new(cx.update(|cx| DelayedExitTerminal {
                terminal: crate::tests::FakeTerminalHandle::new_never_exits(cx),
                kill_requested: std::cell::Cell::new(false),
                kill_fails,
                tracking_failures: std::cell::Cell::new(0),
            }));
            let environment = Rc::new(DelayedExitEnvironment(terminal.clone()));
            let (event_stream, _events) = ToolCallEventStream::test();
            let (cancellation_sender, cancellation_receiver) = oneshot::channel();
            let task = cx.update(|cx| {
                let project = project.clone();
                cx.spawn(async move |cx| {
                    let permit = first.await.expect("first permit");
                    execute_queued_job(
                        project,
                        environment,
                        SandboxedTerminalToolInput {
                            command: "echo build".into(),
                            cd: "root".into(),
                            timeout_ms: Some(100),
                            ..Default::default()
                        },
                        event_stream,
                        cancellation_receiver,
                        permit,
                        cx,
                    )
                    .await
                })
            });
            cx.run_until_parked();
            cx.executor()
                .advance_clock(std::time::Duration::from_secs(1));
            cx.run_until_parked();
            assert!(!terminal.kill_requested.get());
            drop(blocking_permit);
            cx.run_until_parked();
            assert!(!terminal.kill_requested.get());
            if cancel_caller {
                drop(cancellation_sender);
            } else {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(100));
            }
            cx.run_until_parked();
            assert!(terminal.kill_requested.get());
            assert!(second.as_mut().now_or_never().is_none());
            terminal.terminal.signal_exit();
            let result = task.await;
            assert!(
                result.is_err(),
                "deadline and cancellation must report an error"
            );
            drop(second.await.expect("next permit after confirmed exit"));
            assert!(!queue.0.borrow().active);
        }
    }

    #[gpui::test]
    async fn queue_activity_guard_ends_at_acquisition_or_cancellation(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/root", serde_json::json!({})).await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let project_context = cx.new(|_| prompt_store::ProjectContext::default());
        let context_server_store =
            project.read_with(cx, |project, _| project.context_server_store());
        let registry =
            cx.new(|cx| crate::tools::ContextServerRegistry::new(context_server_store, cx));
        let thread = cx.new(|cx| {
            Thread::new(
                project,
                project_context,
                registry,
                crate::Templates::new(),
                None,
                cx,
            )
        });
        let session_id = thread.read_with(cx, |thread, _| thread.id().clone());
        let queue = BuildQueue::default();
        let blocking_permit = queue
            .enqueue(acp_v2::SessionId::new("blocking"))
            .await
            .expect("blocking permit");
        let mut waiter = Box::pin(cx.update(|cx| {
            queue
                .enqueue_tracked(session_id.clone(), cx)
                .expect("registered session")
        }));
        assert_eq!(
            thread.read_with(cx, |thread, cx| thread
                .subagent_activity(cx)
                .queued_build_waits),
            1
        );
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(3));
        drop(blocking_permit);
        let permit = waiter
            .as_mut()
            .now_or_never()
            .expect("ready waiter")
            .expect("permit");
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(2));
        let activity = thread.read_with(cx, |thread, cx| thread.subagent_activity(cx));
        assert_eq!(activity.queued_build_waits, 0);
        assert_eq!(activity.queue_wait, std::time::Duration::from_secs(3));
        let cancelled = cx.update(|cx| {
            queue
                .enqueue_tracked(session_id.clone(), cx)
                .expect("registered session")
        });
        cx.executor()
            .advance_clock(std::time::Duration::from_secs(1));
        drop(cancelled);
        let activity = thread.read_with(cx, |thread, cx| thread.subagent_activity(cx));
        assert_eq!(activity.queued_build_waits, 0);
        assert_eq!(activity.queue_wait, std::time::Duration::from_secs(4));
        drop(permit);
        assert!(
            cx.update(|cx| queue.enqueue_tracked(acp_v2::SessionId::new("unknown"), cx))
                .is_err()
        );
        assert!(queue.0.borrow().pending.is_empty());
    }

    #[gpui::test]
    async fn unconfirmed_cleanup_reports_errors_and_recovers_after_exit(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/root", serde_json::json!({})).await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.tool_permissions.default = settings::ToolPermissionMode::Allow;
            settings
                .tool_permissions
                .tools
                .remove(super::super::terminal_tool::TerminalTool::NAME);
            AgentSettings::override_global(settings, cx);
        });
        for tracking_failures in [0, 2] {
            let queue = BuildQueue::default();
            let permit = queue
                .enqueue(acp_v2::SessionId::new("first"))
                .await
                .expect("initial permit");
            let mut waiting = Box::pin(queue.enqueue(acp_v2::SessionId::new("waiting")));
            assert!(waiting.as_mut().now_or_never().is_none());
            let terminal = Rc::new(cx.update(|cx| DelayedExitTerminal {
                terminal: crate::tests::FakeTerminalHandle::new_never_exits(cx),
                kill_requested: std::cell::Cell::new(false),
                kill_fails: true,
                tracking_failures: std::cell::Cell::new(tracking_failures),
            }));
            let environment = Rc::new(DelayedExitEnvironment(terminal.clone()));
            let (event_stream, _events) = ToolCallEventStream::test();
            let (cancellation_sender, cancellation_receiver) = oneshot::channel();
            let task = cx.update(|cx| {
                let project = project.clone();
                cx.spawn(async move |cx| {
                    execute_queued_job(
                        project,
                        environment,
                        SandboxedTerminalToolInput {
                            command: "echo build".into(),
                            cd: "root".into(),
                            timeout_ms: Some(100),
                            ..Default::default()
                        },
                        event_stream,
                        cancellation_receiver,
                        permit,
                        cx,
                    )
                    .await
                })
            });
            cx.run_until_parked();
            cx.executor()
                .advance_clock(std::time::Duration::from_millis(100));
            cx.run_until_parked();
            assert!(terminal.kill_requested.get());
            if tracking_failures > 0 {
                terminal.terminal.signal_exit();
                cx.executor()
                    .advance_clock(std::time::Duration::from_secs(1));
                cx.run_until_parked();
                let error = task.await.expect_err("tracking failure should be reported");
                assert!(error.contains("simulated transient exit tracking failure"));
                assert!(error.contains("Process exit was subsequently confirmed"));
                drop(
                    waiting
                        .await
                        .expect("transient tracking failure must not block the queue"),
                );
                assert!(!queue.0.borrow().active);
                drop(cancellation_sender);
                continue;
            }
            cx.executor()
                .advance_clock(std::time::Duration::from_secs(5));
            cx.run_until_parked();
            let error = task.await.expect_err("bounded cleanup must report failure");
            assert!(error.contains("within 5 seconds"));
            assert!(error.contains("temporarily unavailable"));
            assert!(waiting.await.is_err());
            assert!(queue.enqueue(acp_v2::SessionId::new("new")).await.is_err());
            assert!(queue.0.borrow().active);
            terminal.terminal.signal_exit();
            cx.run_until_parked();
            let recovered = queue
                .enqueue(acp_v2::SessionId::new("recovered"))
                .await
                .expect("queue should reopen after confirmed exit");
            drop(recovered);
            assert!(!queue.0.borrow().active);
            drop(cancellation_sender);
        }
    }

    struct DelayedCreationEnvironment {
        terminal: Rc<DelayedExitTerminal>,
        started: std::cell::Cell<bool>,
        creation: RefCell<Option<oneshot::Receiver<bool>>>,
    }

    impl ThreadEnvironment for DelayedCreationEnvironment {
        fn create_terminal(
            &self,
            _command: String,
            _extra_env: Vec<(String, String)>,
            _cwd: Option<std::path::PathBuf>,
            _output_byte_limit: Option<u64>,
            _sandbox_wrap: Option<acp_thread::SandboxWrap>,
            cx: &mut gpui::AsyncApp,
        ) -> Task<Result<Rc<dyn crate::TerminalHandle>>> {
            self.started.set(true);
            let creation = self
                .creation
                .borrow_mut()
                .take()
                .expect("one startup attempt");
            let terminal = self.terminal.clone();
            cx.spawn(async move |_cx| {
                anyhow::ensure!(
                    creation.await?,
                    "simulated startup failure without a process"
                );
                Ok(terminal as Rc<dyn crate::TerminalHandle>)
            })
        }

        fn create_subagent(
            &self,
            _label: String,
            _model: Option<acp_thread::AgentModelId>,
            _cx: &mut App,
        ) -> Result<Rc<dyn crate::SubagentHandle>> {
            anyhow::bail!("not used by build queue tests")
        }
    }

    #[gpui::test]
    async fn authorization_is_bounded_by_default_deadline_and_cancellation(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/root", serde_json::json!({})).await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.build_queue_default_timeout_ms = 100;
            settings.tool_permissions.default = settings::ToolPermissionMode::Confirm;
            settings
                .tool_permissions
                .tools
                .remove(super::super::terminal_tool::TerminalTool::NAME);
            AgentSettings::override_global(settings, cx);
        });
        for cancel in [false, true] {
            let queue = BuildQueue::default();
            let permit = queue
                .enqueue(acp_v2::SessionId::new("authorization"))
                .await
                .expect("initial permit");
            let mut waiting = Box::pin(queue.enqueue(acp_v2::SessionId::new("next")));
            let environment = Rc::new(crate::tests::FakeThreadEnvironment::default());
            let (event_stream, mut events) = ToolCallEventStream::test();
            let (cancellation_sender, cancellation_receiver) = oneshot::channel();
            let mut cancellation_sender = Some(cancellation_sender);
            let task = cx.update(|cx| {
                let project = project.clone();
                let environment = environment.clone();
                cx.spawn(async move |cx| {
                    execute_queued_job(
                        project,
                        environment,
                        SandboxedTerminalToolInput {
                            command: "echo build".into(),
                            cd: "root".into(),
                            ..Default::default()
                        },
                        event_stream,
                        cancellation_receiver,
                        permit,
                        cx,
                    )
                    .await
                })
            });
            let authorization = events.expect_authorization().await;
            assert!(waiting.as_mut().now_or_never().is_none());
            if cancel {
                drop(cancellation_sender.take());
            } else {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(100));
            }
            cx.run_until_parked();
            let error = task
                .now_or_never()
                .expect("authorization must not hold the queue indefinitely")
                .expect_err("job must stop");
            assert!(error.contains(if cancel { "cancelled" } else { "timed out" }));
            assert_eq!(environment.terminal_creation_count(), 0);
            drop(
                waiting
                    .await
                    .expect("no process was launched, so queue must reopen"),
            );
            assert!(!queue.0.borrow().active);
            drop(authorization);
        }
    }

    #[gpui::test]
    async fn deadline_is_measured_from_acquisition_not_worker_start(cx: &mut gpui::TestAppContext) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/root", serde_json::json!({})).await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        let queue = BuildQueue::default();
        let mut waiter = queue.enqueue(acp_v2::SessionId::new("delayed worker"));
        waiter.executor = Some(cx.executor());
        let permit = waiter.await.expect("initial permit");
        cx.executor()
            .advance_clock(std::time::Duration::from_millis(100));
        let environment = Rc::new(crate::tests::FakeThreadEnvironment::default());
        let (event_stream, _events) = ToolCallEventStream::test();
        let (cancellation_sender, cancellation_receiver) = oneshot::channel();
        let task = cx.update(|cx| {
            let environment = environment.clone();
            cx.spawn(async move |cx| {
                execute_queued_job(
                    project,
                    environment,
                    SandboxedTerminalToolInput {
                        command: "echo build".into(),
                        cd: "root".into(),
                        timeout_ms: Some(100),
                        ..Default::default()
                    },
                    event_stream,
                    cancellation_receiver,
                    permit,
                    cx,
                )
                .await
            })
        });
        let error = task
            .await
            .expect_err("worker scheduling delay must consume the deadline");
        assert!(error.contains("before authorization or startup"));
        assert_eq!(environment.terminal_creation_count(), 0);
        assert!(!queue.0.borrow().active);
        drop(cancellation_sender);
    }

    #[gpui::test]
    async fn startup_is_retained_after_cancellation_or_deadline_until_no_process_can_remain(
        cx: &mut gpui::TestAppContext,
    ) {
        crate::tests::init_test(cx);
        let fs = fs::FakeFs::new(cx.executor());
        fs.insert_tree("/root", serde_json::json!({})).await;
        let project = Project::test(fs, ["/root".as_ref()], cx).await;
        cx.update(|cx| {
            let mut settings = AgentSettings::get_global(cx).clone();
            settings.tool_permissions.default = settings::ToolPermissionMode::Allow;
            settings
                .tool_permissions
                .tools
                .remove(super::super::terminal_tool::TerminalTool::NAME);
            AgentSettings::override_global(settings, cx);
        });
        for (cancel, startup_fails) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let queue = BuildQueue::default();
            let permit = queue
                .enqueue(acp_v2::SessionId::new("startup"))
                .await
                .expect("initial permit");
            let mut waiting = Box::pin(queue.enqueue(acp_v2::SessionId::new("next")));
            assert!(waiting.as_mut().now_or_never().is_none());
            let terminal = Rc::new(cx.update(|cx| DelayedExitTerminal {
                terminal: crate::tests::FakeTerminalHandle::new_never_exits(cx),
                kill_requested: std::cell::Cell::new(false),
                kill_fails: false,
                tracking_failures: std::cell::Cell::new(0),
            }));
            let (creation_sender, creation_receiver) = oneshot::channel();
            let environment = Rc::new(DelayedCreationEnvironment {
                terminal: terminal.clone(),
                started: std::cell::Cell::new(false),
                creation: RefCell::new(Some(creation_receiver)),
            });
            let (event_stream, _events) = ToolCallEventStream::test();
            let (cancellation_sender, cancellation_receiver) = oneshot::channel();
            let mut cancellation_sender = Some(cancellation_sender);
            let task = cx.update(|cx| {
                let project = project.clone();
                let environment = environment.clone();
                cx.spawn(async move |cx| {
                    execute_queued_job(
                        project,
                        environment,
                        SandboxedTerminalToolInput {
                            command: "echo build".into(),
                            cd: "root".into(),
                            timeout_ms: Some(100),
                            ..Default::default()
                        },
                        event_stream,
                        cancellation_receiver,
                        permit,
                        cx,
                    )
                    .await
                })
            });
            cx.run_until_parked();
            assert!(environment.started.get());
            if cancel {
                drop(cancellation_sender.take());
            } else {
                cx.executor()
                    .advance_clock(std::time::Duration::from_millis(100));
            }
            cx.run_until_parked();
            assert!(waiting.as_mut().now_or_never().is_none());
            cx.executor()
                .advance_clock(std::time::Duration::from_secs(5));
            cx.run_until_parked();
            let error = task
                .now_or_never()
                .expect("startup must have a bounded response")
                .expect_err("startup is unresolved");
            assert!(error.contains(if cancel { "cancelled" } else { "timed out" }));
            assert!(error.contains("temporarily unavailable"));
            assert!(waiting.await.is_err());
            assert!(queue.enqueue(acp_v2::SessionId::new("new")).await.is_err());
            assert!(queue.0.borrow().active);
            creation_sender
                .send(!startup_fails)
                .expect("startup task must still be owned and polled");
            cx.run_until_parked();
            if !startup_fails {
                assert!(terminal.kill_requested.get());
                assert!(
                    queue.0.borrow().active,
                    "kill request alone must not allow another build"
                );
                assert!(
                    queue
                        .enqueue(acp_v2::SessionId::new("still blocked"))
                        .await
                        .is_err()
                );
            }
            terminal.terminal.signal_exit();
            cx.run_until_parked();
            let recovered = queue
                .enqueue(acp_v2::SessionId::new("recovered"))
                .await
                .expect("queue must recover after startup failure or confirmed exit");
            drop(recovered);
            assert!(!queue.0.borrow().active);
        }
    }

    #[test]
    fn fifo_and_queued_cancellation() {
        let queue = BuildQueue::default();
        let mut first = Box::pin(queue.enqueue(acp_v2::SessionId::new("first")));
        let mut cancelled = Box::pin(queue.enqueue(acp_v2::SessionId::new("cancelled")));
        let mut last = Box::pin(queue.enqueue(acp_v2::SessionId::new("last")));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(last.as_mut().poll(&mut context).is_pending());
        let Poll::Ready(Ok(permit)) = first.as_mut().poll(&mut context) else {
            panic!("first job should acquire the slot");
        };
        assert!(cancelled.as_mut().poll(&mut context).is_pending());
        drop(cancelled);
        assert!(last.as_mut().poll(&mut context).is_pending());
        drop(permit);
        assert!(last.as_mut().poll(&mut context).is_ready());
        assert!(queue.0.borrow().pending.is_empty());
        assert!(!queue.0.borrow().active);
    }

    #[gpui::test]
    fn wait_tracking_counts_all_jobs_for_a_session(cx: &mut App) {
        let queue = BuildQueue::default();
        cx.set_global(queue.clone());
        let session_id = acp_v2::SessionId::new("session");
        let first = queue.enqueue(session_id.clone());
        let second = queue.enqueue(session_id.clone());
        assert!(is_queue_waiting(&session_id, cx));
        drop(first);
        assert!(is_queue_waiting(&session_id, cx));
        drop(second);
        assert!(!is_queue_waiting(&session_id, cx));
    }

    #[test]
    fn cancelled_front_does_not_block_following_job() {
        let queue = BuildQueue::default();
        let first = queue.enqueue(acp_v2::SessionId::new("first"));
        let mut second = Box::pin(queue.enqueue(acp_v2::SessionId::new("second")));
        drop(first);
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(second.as_mut().poll(&mut context).is_ready());
    }
}
