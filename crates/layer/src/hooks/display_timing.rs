use ash::vk;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PresentationTiming {
    pub(crate) present_id: u32,
    pub(crate) desired_present_time: u64,
    pub(crate) actual_present_time: u64,
    pub(crate) earliest_present_time: u64,
    pub(crate) present_margin: u64,
}

impl PresentationTiming {
    pub(crate) fn from_vk(timing: vk::PastPresentationTimingGOOGLE) -> Self {
        Self {
            present_id: timing.present_id,
            desired_present_time: timing.desired_present_time,
            actual_present_time: timing.actual_present_time,
            earliest_present_time: timing.earliest_present_time,
            present_margin: timing.present_margin,
        }
    }

    pub(crate) const fn to_vk(self) -> vk::PastPresentationTimingGOOGLE {
        vk::PastPresentationTimingGOOGLE {
            present_id: self.present_id,
            desired_present_time: self.desired_present_time,
            actual_present_time: self.actual_present_time,
            earliest_present_time: self.earliest_present_time,
            present_margin: self.present_margin,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TimingEnumeration {
    required_count: u32,
    written_count: u32,
    timings: Vec<PresentationTiming>,
    result: vk::Result,
}

impl TimingEnumeration {
    pub(crate) const fn required_count(&self) -> u32 {
        self.required_count
    }

    pub(crate) const fn written_count(&self) -> u32 {
        self.written_count
    }

    pub(crate) fn timings(&self) -> &[PresentationTiming] {
        &self.timings
    }

    #[cfg(test)]
    pub(crate) fn ids(&self) -> Vec<u32> {
        self.timings
            .iter()
            .map(|timing| timing.present_id)
            .collect()
    }

    pub(crate) const fn result(&self) -> vk::Result {
        self.result
    }
}

pub(crate) fn merge_timing_results(
    generations: &[Result<Vec<PresentationTiming>, vk::Result>],
    capacity: Option<usize>,
) -> Result<TimingEnumeration, vk::Result> {
    let mut merged = BTreeMap::new();
    for generation in generations {
        for timing in generation.as_ref().map_err(|error| *error)? {
            merged.insert(timing.present_id, *timing);
        }
    }
    let timings = merged.into_values().collect::<Vec<_>>();
    let required_count = timings.len() as u32;
    let timings = match capacity {
        None => Vec::new(),
        Some(capacity) => timings.into_iter().take(capacity).collect::<Vec<_>>(),
    };
    let written_count = timings.len() as u32;
    let result = if capacity.is_some_and(|capacity| capacity < required_count as usize) {
        vk::Result::INCOMPLETE
    } else {
        vk::Result::SUCCESS
    };
    Ok(TimingEnumeration {
        required_count,
        written_count,
        timings,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::{PresentationTiming, merge_timing_results};
    use ash::vk;

    fn timing(id: u32, desired: u64) -> PresentationTiming {
        PresentationTiming {
            present_id: id,
            desired_present_time: desired,
            actual_present_time: desired + 1,
            earliest_present_time: desired + 2,
            present_margin: 3,
        }
    }

    #[test]
    fn empty_and_single_generations_have_spec_shaped_results() {
        let empty = merge_timing_results(&[Ok(Vec::new())], None).unwrap();
        assert_eq!(empty.required_count(), 0);
        assert_eq!(empty.written_count(), 0);

        let one = merge_timing_results(&[Ok(vec![timing(4, 100)])], Some(1)).unwrap();
        assert_eq!(one.required_count(), 1);
        assert_eq!(one.written_count(), 1);
        assert_eq!(one.result(), vk::Result::SUCCESS);
    }

    #[test]
    fn overlapping_ids_prefer_newer_generation_and_are_sorted() {
        let old = vec![timing(9, 900), timing(2, 200)];
        let new = vec![timing(9, 999), timing(5, 500)];
        let merged = merge_timing_results(&[Ok(old), Ok(new)], Some(3)).unwrap();

        assert_eq!(merged.ids(), vec![2, 5, 9]);
        assert_eq!(merged.timings()[2].desired_present_time, 999);
    }

    #[test]
    fn undersized_data_query_returns_written_count_and_incomplete() {
        let merged = merge_timing_results(
            &[Ok(vec![timing(1, 10), timing(2, 20), timing(3, 30)])],
            Some(2),
        )
        .unwrap();

        assert_eq!(merged.required_count(), 3);
        assert_eq!(merged.written_count(), 2);
        assert_eq!(merged.result(), vk::Result::INCOMPLETE);
    }

    #[test]
    fn count_only_query_does_not_construct_output_records() {
        let merged = merge_timing_results(&[Ok(vec![timing(1, 10), timing(2, 20)])], None).unwrap();

        assert_eq!(merged.required_count(), 2);
        assert_eq!(merged.written_count(), 0);
        assert!(merged.timings().is_empty());
    }

    #[test]
    fn downstream_failure_is_returned_without_partial_merge() {
        assert_eq!(
            merge_timing_results(
                &[Ok(vec![timing(1, 10)]), Err(vk::Result::ERROR_DEVICE_LOST)],
                Some(4),
            ),
            Err(vk::Result::ERROR_DEVICE_LOST)
        );
    }
}
