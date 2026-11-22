---
title: AI Agent Tools - Zed
description: Built-in tools for Zed's AI agent including file editing, code search, terminal commands, web search, skills, and diagnostics.
---

# Tools

Zed's built-in agent has access to these tools for reading, searching, and editing your codebase. These tools are used in the [Agent Panel](./agent-panel.md) during conversations with AI agents.

The exact tool list can vary by [Agent Profile](./agent-profiles.md), selected model provider, and Zed version.

You can configure permissions for tool actions, including situations where they are automatically approved, automatically denied, or require your confirmation on a case-by-case basis. See [Tool Permissions](./tool-permissions.md) for the list of permission-gated tools and details.

To add custom tools beyond these built-in ones, see [MCP servers](./mcp.md).

To choose which built-in tools and MCP tools are available in a Zed Agent thread, use [Agent Profiles](./agent-profiles.md). Profiles control tool availability; tool permissions control allow, deny, and confirm behavior.

The terminal tool can also run with additional OS-level restrictions when [Zed Agent sandboxing](./sandboxing.md) is enabled.

## Read & Search Tools

### `diagnostics`

Gets errors and warnings for either a specific file or the entire project, useful after making edits to determine if further changes are needed.
When a path is provided, shows all diagnostics for that specific file.
When no path is provided, shows a summary of error and warning counts for all files in the project.

**Example:** After editing `src/parser.rs`, call `diagnostics` with that path to check for type errors immediately. After a larger refactor touching many files, call it without a path to see a project-wide count of errors before deciding what to fix next.

### `fetch`

Fetches a URL and returns the content as Markdown. Useful for providing docs as context.

`fetch` is governed by tool permissions, agent profiles, and project trust. It is not run inside the terminal OS sandbox, so terminal sandbox network grants such as `allow_hosts` and `allow_all_hosts` do not apply to it.

**Example:** Fetching a library's changelog page to check whether a breaking API change was introduced in a recent version before writing integration code.

### `find_path`

Quickly finds files by matching glob patterns (like "\*_/_.js"), returning matching file paths alphabetically.

### `grep`

Searches file contents across the project using regular expressions, preferred for finding symbols in code without knowing exact file paths.

**Example:** To find every call site of a function before renaming it, search for `parse_config\(` — the regex matches the function name followed by an opening parenthesis, filtering out comments or variable names that happen to contain the string.

### `list_directory`

Lists files and directories in a given path, providing an overview of filesystem contents.

### `read_file`

Reads the content of a specified file in the project, allowing access to file contents.

### `read_files`

Reads 1–20 files in one call. Each entry accepts the same `path`, `start_line`, and `end_line` fields as `read_file`. Results stay in request order, including image content and individual errors. One failed read does not discard successful reads.

```json
{
  "files": [
    { "path": "my-project/src/parser.rs", "start_line": 1, "end_line": 80 },
    { "path": "my-project/src/config.rs" }
  ]
}
```

### `conversation`

Browses the current thread's retained messages, including messages omitted from model context by compaction. Use `query` for case-insensitive text search, `offset` and `limit` for pagination, and `newest_first` to reverse the order. Pages contain at most 20 messages, with each message limited to 2,000 characters. Hidden reasoning is excluded. Browsing old code does not count as reading its current file contents.

## Web Tools

### `search_web`

Searches the web for information, providing results with snippets and links from relevant web pages, useful for accessing real-time information.

**Example:** Looking up whether a known bug in a dependency has been patched in a recent release, or finding the current API signature for a third-party library when the local docs are out of date.

> **Note:** The built-in `search_web` tool is only available to [Zed Pro](https://zed.dev/pricing) subscribers using the Zed provider. If you're on a free plan or using a different provider, you can get equivalent functionality by connecting an MCP server that provides web search capabilities. See [MCP servers](./mcp.md) for details.

## Edit Tools

### `copy_path`

Copies a file or directory recursively in the project, more efficient than manually reading and writing files when duplicating content.

### `create_directory`

Creates a new directory at the specified path within the project, creating all necessary parent directories (similar to `mkdir -p`).

### `delete_path`

Deletes a file or directory (including contents recursively) at the specified path and confirms the deletion.

### `edit_file`

Edits files by replacing specific text with new content.

**Example:** Updating a function signature — the agent identifies the exact lines to replace and provides the updated version, leaving the surrounding code untouched. For widespread renames, it pairs this with `grep` to find every occurrence first.

### `move_path`

Moves or renames a file or directory in the project, performing a rename if only the filename differs.

### `write_file`

Creates a new file or overwrites an existing file with completely new contents.

### `write_files`

Creates or overwrites 1–20 files in one call, using a `files` array of `path` and `content` objects. Files are processed in order, with the same permissions and review flow as `write_file`. Results report each success or error separately. This is not an atomic transaction: successful writes remain if another file fails.

Overwriting an existing file with either write tool requires a successful read in the same agent thread. Forwarded startup snapshots and conversation-history results do not satisfy this requirement. Read again if the file changes or after overwriting it. Creating a new file does not require a prior read.

### `terminal`

Executes shell commands and returns the combined output, creating a new shell process for each invocation.

**Example:** After editing a Rust file, run `cargo test --package my_crate 2>&1 | tail -30` to confirm the changes don't break existing tests. Or run `git diff --stat` to review which files have been modified before wrapping up a task.

### `build_queue`

Submits one command to a FIFO queue shared by native-agent threads across all participating Zed processes for the same operating-system user. Separate windows, projects, release channels, and custom user-data folders share admission. Only one queued command can run at a time. Agents choose this tool instead of `terminal` to serialize heavy work; ordinary terminal commands and external-agent commands are not serialized.

The queue is available without adding any settings. See [Build Queue](./agent-settings.md#build-queue) for its built-in timeout defaults and optional customization. The command accepts the terminal tool's command, working folder, output-limit, and sandbox-permission fields. It uses the same terminal permission rules and sandbox policy.

Every submission has a positive timeout. The timeout starts after both local and cross-process admission and covers authorization, startup, and execution—not time waiting for a slot. Cancellation or expiry retains the queue lease until the command's process group is confirmed empty. If termination cannot be confirmed, participating processes reject queued jobs rather than start overlapping builds. The queue recovers when exit is confirmed.

Queued execution currently supports local Linux terminals. Commands must stay in their original process group: do not daemonize, use `setsid`, or enable separate job-control groups. Ordinary terminal calls keep their existing behavior.

Admission uses operating-system file locks and private ticket files in `/tmp/zed-build-queue-<effective-uid>`. Processes must share that filesystem namespace; separate containers with private `/tmp` mounts do not coordinate. Tickets from dead queued callers are recovered automatically. If a Zed process dies after admission, its build might survive, so its active ticket fails closed. The error identifies that ticket: verify that no build remains before removing it. Never delete the queue's coordinator file or folder while Zed processes are using it.

## Other Tools

### `skill`

Loads instructions from an available [Skill](./skills.md) so the agent can follow project-specific or workflow-specific guidance. Skills can also be invoked by you directly with slash commands.

**Example:** When a repository has a skill for release-note writing, the agent can load that skill before drafting release notes so it follows the local format.

### `spawn_agent`

Spawns a subagent with its own context window to perform a delegated task. Useful for running parallel investigations, completing self-contained tasks, or performing research where only the outcome matters. Each subagent has access to the same tools as the parent agent.

To configure a subagent's model, reasoning effort, and speed, see [Subagent Models](./agent-settings.md#subagent-models).

`timeout_ms` optionally bounds a subagent turn. Set `timeout_mode` to `"total"` to count the whole turn, or `"tool_calls"` to count only time spent executing tools, excluding model generation and streamed-input waits. Both modes pause while that subagent waits for the build queue. On expiry, Zed cancels the turn and returns its session ID for follow-up. Omitting the timeout preserves unlimited execution.

Use `startup_context` to forward code already known to the parent as an array of `{ "path": "...", "content": "..." }` snapshots. Zed labels these as untrusted, possibly stale context. They are limited to 32 snapshots and 128 KiB in total and do not count as fresh file reads. A subagent must read existing files itself before overwriting them.

### `subagents`

Coordinates multiple subagents and inspects their current work. It supports these operations:

- `spawn_groups`: starts several groups concurrently and waits until a whole group finishes. Its response contains that group's outputs and the current status of every remaining group and agent.
- `wait_groups`: waits for the next unreported whole-group completion. Pass `group_ids` to select groups, or omit it to consider every retained group.
- `spawn`: starts one group and waits for its first agent result, for up to 30 seconds.
- `wait`: waits for any new agent result or all results in a specific group, using `mode: "any"` or `"all"`.
- `inspect`: reads a directly owned subagent's visible output, active tool information, and activity counters using its `session_id`. It does not expose hidden reasoning or unrelated threads.
- `list`: lists retained groups and job statuses.

For example, this starts two groups and returns when the first group finishes, while the other keeps working:

```json
{
  "operation": "spawn_groups",
  "groups": [
    [{ "label": "Review parser", "message": "Review my-project/src/parser.rs for error-handling bugs. Do not edit.", "timeout_ms": 120000 }],
    [{ "label": "Review tests", "message": "Review my-project/tests for missing parser edge cases. Do not edit.", "timeout_ms": 180000 }]
  ]
}
```

The calling model resumes through the pending tool result. It can work on a completed group's findings, then call `wait_groups` again for the next boundary. No unsolicited model turn starts after the user turn ends. Optional `wait_timeout_ms` limits a wait without cancelling the jobs; `0` polls immediately. Individual jobs can use the same model, timeout, and startup-context options as `spawn_agent`.

Groups contain 1–16 agents, with up to 128 retained jobs per parent session. Jobs survive tool calls and ordinary parent turns, but not application restarts. Cancelling the parent stops its running jobs. Returned outputs are bounded and indicate truncation.

See [Agent Profiles](./agent-profiles.md) to control which tools your agents can use.
