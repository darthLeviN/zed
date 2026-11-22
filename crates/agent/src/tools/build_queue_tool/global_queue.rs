use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read as _, Seek as _, Write as _},
    path::{Path, PathBuf},
};

/// The coordinator inode must never be removed, even when the queue is empty.
/// Ticket files, in contrast, are removed only while holding that coordinator.
pub(super) struct Queue {
    directory: PathBuf,
    coordinator: File,
}

pub(super) struct Ticket {
    queue: Queue,
    number: u64,
    lease: File,
}

fn try_lock(file: &File) -> io::Result<bool> {
    match file.try_lock() {
        Ok(()) => Ok(true),
        Err(std::fs::TryLockError::WouldBlock) => Ok(false),
        Err(std::fs::TryLockError::Error(error)) => Err(error),
    }
}

impl Queue {
    pub(super) fn open(directory: PathBuf) -> io::Result<Self> {
        #[cfg(unix)]
        {
            use std::os::unix::fs::{DirBuilderExt as _, MetadataExt as _};
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
            let metadata = fs::symlink_metadata(&directory)?;
            if !metadata.is_dir() || metadata.uid() != user_id() || metadata.mode() & 0o077 != 0 {
                return Err(io::Error::other(
                    "Build queue directory must be private and owned by the current user",
                ));
            }
        }
        #[cfg(not(unix))]
        fs::create_dir_all(&directory)?;
        let coordinator = open_checked(&directory.join("coordinator"), Creation::Shared)?;
        if coordinator.metadata()?.len() != 0 {
            return Err(io::Error::other("Malformed build queue coordinator"));
        }
        Ok(Self {
            directory,
            coordinator,
        })
    }

    fn try_coordinate(&self) -> io::Result<bool> {
        let path = self.directory.join("coordinator");
        validate_open_file(&path, &self.coordinator)?;
        if !try_lock(&self.coordinator)? {
            return Ok(false);
        }
        if let Err(error) = validate_open_file(&path, &self.coordinator).and_then(|()| {
            if self.coordinator.metadata()?.len() != 0 {
                return Err(io::Error::other("Malformed build queue coordinator"));
            }
            Ok(())
        }) {
            self.coordinator.unlock()?;
            return Err(error);
        }
        Ok(true)
    }

    /// None means contention, not an error. Never wait on an OS lock.
    pub(super) fn register(&self) -> io::Result<Option<Ticket>> {
        if !self.try_coordinate()? {
            return Ok(None);
        }
        let result = (|| {
            let live = self.live_tickets()?;
            let number = live
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| io::Error::other("Build queue ticket numbers exhausted"))?;
            let path = self.directory.join(number.to_string());
            let lease = open_checked(&path, Creation::New)?;
            if !try_lock(&lease)? {
                return Err(io::Error::other("New build queue ticket is already locked"));
            }
            Ok(Some(Ticket {
                queue: Queue::open(self.directory.clone())?,
                number,
                lease,
            }))
        })();
        self.coordinator.unlock()?;
        result
    }

    fn live_tickets(&self) -> io::Result<Vec<u64>> {
        let mut live = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry.file_name();
            if name == "coordinator" {
                continue;
            }
            let number = name
                .to_str()
                .and_then(|name| name.parse::<u64>().ok())
                .ok_or_else(|| io::Error::other("Unexpected file in build queue directory"))?;
            let lease = open_checked(&entry.path(), Creation::Existing)?;
            let state = read_state(&lease)?;
            if try_lock(&lease)? {
                if state != State::Queued {
                    return Err(io::Error::other(format!(
                        "Build queue owner exited with an unconfirmed build. Verify no build remains, then remove ticket {} before retrying",
                        entry.path().display()
                    )));
                }
                // No live owner can acquire this inode: registration also holds
                // the coordinator, and never reopens an existing ticket.
                fs::remove_file(entry.path())?;
            } else {
                if state == State::Blocked {
                    return Err(io::Error::other(
                        "Build termination is unconfirmed; the host-wide build queue is temporarily unavailable until cleanup confirms exit",
                    ));
                }
                live.push(number);
            }
        }
        live.sort_unstable();
        Ok(live)
    }
}

impl Ticket {
    pub(super) fn try_acquire(&self) -> io::Result<bool> {
        if !self.queue.try_coordinate()? {
            return Ok(false);
        }
        let result = (|| {
            if self.queue.live_tickets()?.first() != Some(&self.number) {
                return Ok(false);
            }
            // A dead queued owner is recoverable, but a dead running owner may
            // have left a child alive. Persist this distinction before admission.
            self.write_state(b'A')?;
            Ok(true)
        })();
        self.queue.coordinator.unlock()?;
        result
    }

    pub(super) fn try_quarantine(&self) -> io::Result<bool> {
        if !self.queue.try_coordinate()? {
            return Ok(false);
        }
        let result = (|| {
            if read_state(&self.lease)? == State::Queued {
                return Err(io::Error::other(
                    "Cannot quarantine an unadmitted build ticket",
                ));
            }
            self.write_state(b'B')
        })();
        self.queue.coordinator.unlock()?;
        result.map(|()| true)
    }

    fn write_state(&self, state: u8) -> io::Result<()> {
        validate_open_file(
            &self.queue.directory.join(self.number.to_string()),
            &self.lease,
        )?;
        let mut lease = &self.lease;
        lease.rewind()?;
        lease.write_all(&[state])?;
        lease.sync_all()
    }

    pub(super) fn try_remove(&self) -> io::Result<bool> {
        if !self.queue.try_coordinate()? {
            return Ok(false);
        }
        let path = self.queue.directory.join(self.number.to_string());
        let result = validate_open_file(&path, &self.lease).and_then(|()| fs::remove_file(path));
        self.queue.coordinator.unlock()?;
        result.map(|()| true)
    }
}

#[derive(PartialEq)]
enum State {
    Queued,
    Active,
    Blocked,
}

fn read_state(file: &File) -> io::Result<State> {
    if file.metadata()?.len() > 1 {
        return Err(io::Error::other("Malformed build queue ticket status"));
    }
    let mut file = file;
    file.rewind()?;
    let mut status = [0];
    match file.read(&mut status)? {
        0 => Ok(State::Queued),
        1 if status[0] == b'A' => Ok(State::Active),
        1 if status[0] == b'B' => Ok(State::Blocked),
        _ => Err(io::Error::other("Malformed build queue ticket status")),
    }
}

enum Creation {
    Existing,
    Shared,
    New,
}

fn validate_metadata(metadata: &fs::Metadata) -> io::Result<()> {
    if !metadata.is_file() {
        return Err(io::Error::other(
            "Build queue files must be regular files, not symlinks or special files",
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.uid() != user_id() || metadata.mode() & 0o7777 != 0o600 || metadata.nlink() != 1
        {
            return Err(io::Error::other(
                "Build queue files must be private, singly linked, and owned by the current user",
            ));
        }
    }
    Ok(())
}

fn same_file(first: &fs::Metadata, second: &fs::Metadata) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if first.dev() != second.dev() || first.ino() != second.ino() {
            return Err(io::Error::other("Build queue file identity changed"));
        }
    }
    #[cfg(not(unix))]
    let _ = (first, second);
    Ok(())
}

fn validate_open_file(path: &Path, file: &File) -> io::Result<()> {
    let current = fs::symlink_metadata(path)?;
    let opened = file.metadata()?;
    validate_metadata(&current)?;
    validate_metadata(&opened)?;
    same_file(&current, &opened)
}

fn open_checked(path: &Path, creation: Creation) -> io::Result<File> {
    let before = match fs::symlink_metadata(path) {
        Ok(metadata) => {
            validate_metadata(&metadata)?;
            Some(metadata)
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    match creation {
        Creation::Existing => {}
        Creation::Shared => {
            options.create(true).truncate(false);
        }
        Creation::New => {
            options.create_new(true);
        }
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        // NONBLOCK also closes the lstat/open race with a substituted FIFO.
        options
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC);
    }
    let file = options.open(path)?;
    validate_open_file(path, &file)?;
    if let Some(before) = before {
        same_file(&before, &file.metadata()?)?;
    }
    Ok(file)
}

#[cfg(unix)]
fn user_id() -> u32 {
    // geteuid has no preconditions and does not modify process state.
    unsafe { libc::geteuid() }
}

pub(super) fn default_directory() -> io::Result<PathBuf> {
    #[cfg(unix)]
    {
        // Do not use TMPDIR, XDG_RUNTIME_DIR, Zed's data directory, or the
        // release channel: each can differ between participating processes.
        Ok(PathBuf::from("/tmp").join(format!("zed-build-queue-{}", user_id())))
    }
    #[cfg(not(unix))]
    {
        #[allow(deprecated)]
        let home = std::env::home_dir()
            .ok_or_else(|| io::Error::other("Cannot locate build queue user directory"))?;
        Ok(home.join(".zed-build-queue"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        process::{Child, Command},
        thread,
        time::{Duration, Instant},
    };

    const CHILD_ENV: &str = "ZED_BUILD_QUEUE_TEST_CHILD";

    fn wait_for(mut condition: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !condition() {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for queue subprocess"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }

    struct Harness {
        directory: PathBuf,
        children: Vec<Child>,
    }

    impl Harness {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let directory = std::env::temp_dir().join(format!(
                "zed-queue-test-{}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            fs::create_dir(&directory).unwrap();
            Self {
                directory,
                children: Vec::new(),
            }
        }

        fn spawn(&mut self, name: &str) {
            self.spawn_mode(name, "build");
            self.wait(name, "queued");
        }

        fn spawn_mode(&mut self, name: &str, mode: &str) {
            let child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    &format!("{}::subprocess", module_path!().split_once("::").unwrap().1),
                    "--nocapture",
                ])
                .env(CHILD_ENV, &self.directory)
                .env("ZED_BUILD_QUEUE_TEST_NAME", name)
                .env("ZED_BUILD_QUEUE_TEST_MODE", mode)
                .spawn()
                .unwrap();
            self.children.push(child);
            self.wait(name, "started");
        }

        fn path(&self, name: &str, state: &str) -> PathBuf {
            self.directory.join(format!("{name}-{state}"))
        }

        fn wait(&self, name: &str, state: &str) {
            wait_for(|| self.path(name, state).exists());
        }

        fn signal(&self, name: &str, state: &str) {
            fs::write(self.path(name, state), []).unwrap();
        }

        fn finish(&mut self) {
            for child in &mut self.children {
                wait_for(|| child.try_wait().unwrap().is_some());
                assert!(child.wait().unwrap().success());
            }
        }
    }

    impl Drop for Harness {
        fn drop(&mut self) {
            for child in &mut self.children {
                if child.try_wait().unwrap().is_none() {
                    child.kill().unwrap();
                }
                child.wait().unwrap();
            }
            fs::remove_dir_all(&self.directory).unwrap();
        }
    }

    #[test]
    fn subprocess() {
        let Some(directory) = std::env::var_os(CHILD_ENV).map(PathBuf::from) else {
            return;
        };
        let name = std::env::var("ZED_BUILD_QUEUE_TEST_NAME").unwrap();
        let status = |state| directory.join(format!("{name}-{state}"));
        let mode = std::env::var("ZED_BUILD_QUEUE_TEST_MODE").unwrap();
        fs::write(status("started"), []).unwrap();
        let queue = Queue::open(directory.join("queue"));
        if mode == "probe" {
            let mut error = None;
            wait_for(|| {
                match &queue {
                    Ok(queue) => match queue.register() {
                        Ok(None) => return false,
                        Ok(Some(_)) => {
                            panic!("admitted a build from a malformed or quarantined queue")
                        }
                        Err(rejection) => error = Some(rejection.to_string()),
                    },
                    Err(rejection) => error = Some(rejection.to_string()),
                }
                true
            });
            fs::write(status("rejected"), error.unwrap()).unwrap();
            return;
        }
        let queue = queue.unwrap();
        let mut ticket = None;
        wait_for(|| {
            ticket = queue.register().unwrap();
            ticket.is_some()
        });
        let ticket = ticket.unwrap();
        fs::write(status("queued"), []).unwrap();
        let mut cancelled = false;
        wait_for(|| {
            if status("cancel").exists() {
                cancelled = true;
                true
            } else {
                match ticket.try_acquire() {
                    Ok(acquired) => acquired,
                    Err(error) if mode == "rejecting_waiter" => {
                        fs::write(status("rejected"), error.to_string()).unwrap();
                        cancelled = true;
                        true
                    }
                    Err(error) => panic!("unexpected admission failure: {error}"),
                }
            }
        });
        if !cancelled {
            // A create_new sentinel detects real cross-process overlap, not
            // just the order in which test status messages happen to arrive.
            let sentinel = directory.join("running");
            File::create_new(&sentinel).unwrap();
            fs::write(status("running"), []).unwrap();
            wait_for(|| status("release").exists() || status("quarantine").exists());
            if status("quarantine").exists() {
                wait_for(|| ticket.try_quarantine().unwrap());
                fs::write(status("blocked"), []).unwrap();
                wait_for(|| status("release").exists());
            }
            fs::remove_file(sentinel).unwrap();
        }
        wait_for(|| ticket.try_remove().unwrap());
        fs::write(status("done"), []).unwrap();
    }

    #[test]
    fn multiprocess_fifo_and_cancellation() {
        let mut harness = Harness::new();
        harness.spawn("first");
        harness.wait("first", "running");
        harness.spawn("cancelled");
        harness.spawn("second");
        harness.spawn("third");
        harness.signal("cancelled", "cancel");
        harness.wait("cancelled", "done");
        assert!(!harness.path("second", "running").exists());
        harness.signal("first", "release");
        harness.wait("second", "running");
        assert!(!harness.path("third", "running").exists());
        harness.signal("second", "release");
        harness.wait("third", "running");
        harness.signal("third", "release");
        harness.finish();
    }

    #[test]
    fn multiprocess_dead_queued_owner_is_recovered() {
        let mut harness = Harness::new();
        harness.spawn("first");
        harness.wait("first", "running");
        harness.spawn("dead");
        harness.spawn("last");
        let dead = harness.children.get_mut(1).unwrap();
        dead.kill().unwrap();
        dead.wait().unwrap();
        harness.signal("first", "release");
        harness.wait("last", "running");
        harness.signal("last", "release");
        for index in [0, 2] {
            let child = harness.children.get_mut(index).unwrap();
            wait_for(|| child.try_wait().unwrap().is_some());
            assert!(child.wait().unwrap().success());
        }
    }

    #[test]
    fn dead_active_owner_fails_closed() {
        let mut harness = Harness::new();
        harness.spawn("first");
        harness.wait("first", "running");
        let child = harness.children.first_mut().unwrap();
        child.kill().unwrap();
        child.wait().unwrap();
        let queue = Queue::open(harness.directory.join("queue")).unwrap();
        assert!(
            queue
                .register()
                .err()
                .unwrap()
                .to_string()
                .contains("unconfirmed build")
        );
    }

    #[test]
    fn multiprocess_quarantine_rejects_new_and_existing_waiters_then_recovers() {
        let mut harness = Harness::new();
        harness.spawn("first");
        harness.wait("first", "running");
        harness.spawn_mode("waiting", "rejecting_waiter");
        harness.wait("waiting", "queued");
        harness.signal("first", "quarantine");
        harness.wait("first", "blocked");
        harness.wait("waiting", "rejected");
        harness.spawn_mode("late", "probe");
        harness.wait("late", "rejected");
        for name in ["waiting", "late"] {
            let message = fs::read_to_string(harness.path(name, "rejected")).unwrap();
            assert!(message.contains("temporarily unavailable"), "{message}");
        }
        assert!(
            harness.directory.join("running").exists(),
            "quarantine must retain the active build"
        );
        harness.signal("first", "release");
        harness.wait("first", "done");
        harness.spawn("recovered");
        harness.wait("recovered", "running");
        harness.signal("recovered", "release");
        harness.finish();
    }

    #[cfg(unix)]
    #[test]
    fn multiprocess_malformed_files_fail_closed_without_blocking() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};
        for name in ["1", "coordinator"] {
            for malformed in [
                "fifo",
                "symlink",
                "hardlink",
                "permissions",
                "oversized",
                "unknown",
                "directory",
            ] {
                let mut harness = Harness::new();
                let directory = harness.directory.join("queue");
                drop(Queue::open(directory.clone()).unwrap());
                let path = directory.join(name);
                if name == "coordinator" {
                    fs::remove_file(&path).unwrap();
                }
                match malformed {
                    "fifo" => {
                        use std::os::unix::ffi::OsStrExt as _;
                        let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
                        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
                    }
                    "symlink" => symlink("/dev/zero", &path).unwrap(),
                    "hardlink" => {
                        let outside = harness.directory.join("outside");
                        drop(open_checked(&outside, Creation::New).unwrap());
                        fs::hard_link(outside, &path).unwrap();
                    }
                    "permissions" => {
                        drop(open_checked(&path, Creation::New).unwrap());
                        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
                    }
                    "oversized" => {
                        let file = open_checked(&path, Creation::New).unwrap();
                        file.set_len(1 << 40).unwrap();
                    }
                    "unknown" => {
                        let mut file = open_checked(&path, Creation::New).unwrap();
                        file.write_all(b"?").unwrap();
                    }
                    "directory" => fs::create_dir(&path).unwrap(),
                    _ => unreachable!(),
                }
                harness.spawn_mode("probe", "probe");
                harness.wait("probe", "rejected");
                harness.finish();
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn replaced_coordinator_is_rejected_instead_of_splitting_lock_domains() {
        let harness = Harness::new();
        let directory = harness.directory.join("queue");
        let queue = Queue::open(directory.clone()).unwrap();
        fs::rename(
            directory.join("coordinator"),
            harness.directory.join("old-coordinator"),
        )
        .unwrap();
        drop(Queue::open(directory).unwrap());
        assert!(
            queue
                .register()
                .err()
                .unwrap()
                .to_string()
                .contains("identity changed")
        );
    }

    #[test]
    fn coordinator_contention_never_blocks_and_inode_is_retained() {
        let harness = Harness::new();
        let directory = harness.directory.join("queue");
        let first = Queue::open(directory.clone()).unwrap();
        let second = Queue::open(directory.clone()).unwrap();
        first.coordinator.lock().unwrap();
        assert!(second.register().unwrap().is_none());
        first.coordinator.unlock().unwrap();
        let ticket = second.register().unwrap().unwrap();
        assert!(ticket.try_acquire().unwrap());
        assert!(ticket.try_remove().unwrap());
        drop(ticket);
        assert!(directory.join("coordinator").exists());
        first.coordinator.lock().unwrap();
        assert!(second.register().unwrap().is_none());
        first.coordinator.unlock().unwrap();
    }
}
