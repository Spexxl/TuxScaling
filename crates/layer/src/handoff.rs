use ash::vk;
use std::panic::{AssertUnwindSafe, catch_unwind};

pub(crate) fn handoff(run: impl FnOnce(&mut Option<vk::Semaphore>)) -> Option<vk::Semaphore> {
    let mut submitted = None;
    let _ = catch_unwind(AssertUnwindSafe(|| run(&mut submitted)));
    submitted
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle;
    #[test]
    fn panic_before_submit_preserves_original_waits() {
        assert!(handoff(|_| panic!("record failure")).is_none());
    }
    #[test]
    fn panic_after_submit_preserves_replacement_wait() {
        let semaphore = vk::Semaphore::from_raw(42);
        assert_eq!(
            handoff(|submitted| {
                *submitted = Some(semaphore);
                panic!("post-submit failure");
            }),
            Some(semaphore)
        );
    }
}
