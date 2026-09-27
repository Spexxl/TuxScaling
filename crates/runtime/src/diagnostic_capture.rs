use ash::vk;
use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::mpsc::{SyncSender, TrySendError, sync_channel},
    thread,
    thread::JoinHandle,
    time::{SystemTime, UNIX_EPOCH},
};
use tuxscaling_temporal::GuidanceAblations;
use tuxscaling_vulkan::{Buffer, image_barrier, transfer_memory_barrier};

const DEFAULT_MAX_FRAMES: usize = 120;
const MAX_CAPTURE_FRAMES: usize = 10_000;
#[cfg(test)]
const MAX_IN_FLIGHT_SLOTS: usize = 8;

#[derive(Debug, Clone)]
pub struct DiagnosticCaptureConfig {
    root: PathBuf,
    max_frames: usize,
    start_frame: u64,
    end_frame: Option<u64>,
}

impl DiagnosticCaptureConfig {
    pub fn from_root(root: Option<PathBuf>) -> Option<Self> {
        let root = root?;
        (!root.as_os_str().is_empty()).then_some(Self {
            root,
            max_frames: DEFAULT_MAX_FRAMES,
            start_frame: 0,
            end_frame: None,
        })
    }

    pub fn from_env() -> Option<Self> {
        let root = std::env::var_os("TUXSCALING_CAPTURE_DIR")
            .and_then(|value| Self::from_root(Some(value.into())).map(|config| config.root))?;
        if !is_writable_directory(&root) {
            eprintln!(
                "TuxScaling: diagnostic capture disabled; directory is not writable: {}",
                root.display()
            );
            return None;
        }
        let max_frames = std::env::var("TUXSCALING_CAPTURE_MAX_FRAMES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_FRAMES)
            .clamp(1, MAX_CAPTURE_FRAMES);
        let start_frame = std::env::var("TUXSCALING_CAPTURE_START_FRAME")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let end_frame = std::env::var("TUXSCALING_CAPTURE_END_FRAME")
            .ok()
            .and_then(|value| value.parse::<u64>().ok());
        if end_frame.is_some_and(|end| end < start_frame) {
            eprintln!("TuxScaling: diagnostic capture disabled; frame range is invalid");
            return None;
        }
        Some(Self {
            root,
            max_frames,
            start_frame,
            end_frame,
        })
    }

    #[cfg(test)]
    fn for_test(max_frames: usize) -> Self {
        Self {
            root: PathBuf::from("target/diagnostic-capture-test"),
            max_frames: max_frames.clamp(1, MAX_CAPTURE_FRAMES),
            start_frame: 0,
            end_frame: None,
        }
    }

    #[cfg(test)]
    fn for_test_range(start_frame: u64, end_frame: u64) -> Self {
        Self {
            root: PathBuf::from("target/diagnostic-capture-test"),
            max_frames: 3,
            start_frame,
            end_frame: Some(end_frame),
        }
    }
}

fn is_writable_directory(root: &Path) -> bool {
    if !root.is_dir() {
        return false;
    }
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos());
    let probe = root.join(format!(
        ".tuxscaling-capture-probe-{}-{stamp}",
        std::process::id()
    ));
    match OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(_) => fs::remove_file(probe).is_ok(),
        Err(_) => false,
    }
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotState {
    Free,
    Pending,
}

#[derive(Debug)]
struct CaptureState {
    enabled: bool,
    max_frames: usize,
    completed_frames: usize,
    start_frame: u64,
    end_frame: Option<u64>,
    #[cfg(test)]
    backend: String,
    slots: Vec<Option<u64>>,
}

impl CaptureState {
    #[cfg(test)]
    fn new(config: DiagnosticCaptureConfig) -> Self {
        let slot_count = config.max_frames.clamp(1, MAX_IN_FLIGHT_SLOTS);
        Self::with_slot_count(config, slot_count)
    }

    fn with_slot_count(config: DiagnosticCaptureConfig, slot_count: usize) -> Self {
        Self {
            enabled: true,
            max_frames: config.max_frames,
            completed_frames: 0,
            start_frame: config.start_frame,
            end_frame: config.end_frame,
            #[cfg(test)]
            backend: "Unknown".into(),
            slots: vec![None; slot_count],
        }
    }

    #[cfg(test)]
    fn arm(&mut self, frame_id: u64) -> bool {
        if !self.can_capture(frame_id) {
            return false;
        }
        let slot = self.slot_for(frame_id);
        if self.slots[slot].is_some() {
            return false;
        }
        self.slots[slot] = Some(frame_id);
        true
    }

    fn arm_at(&mut self, frame_id: u64, slot: usize) -> bool {
        if !self.can_capture(frame_id) {
            return false;
        }
        let slot = slot % self.slots.len();
        if self.slots[slot].is_some() {
            return false;
        }
        self.slots[slot] = Some(frame_id);
        true
    }

    fn can_capture(&self, frame_id: u64) -> bool {
        self.enabled
            && self.completed_frames < self.max_frames
            && frame_id >= self.start_frame
            && self.end_frame.is_none_or(|end| frame_id <= end)
    }

    fn complete(&mut self, frame_id: u64) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .find(|slot| slot.is_some_and(|id| id == frame_id))
        {
            *slot = None;
            self.completed_frames = self.completed_frames.saturating_add(1);
        }
    }

    fn resize(&mut self, slot_count: usize) {
        self.slots = vec![None; slot_count.max(1)];
    }

    #[cfg(test)]
    fn set_backend(&mut self, backend: &str) {
        self.backend = backend.into();
    }

    fn disable_for_io_error(&mut self) {
        self.enabled = false;
        self.slots.fill(None);
    }

    fn mark_device_lost(&mut self) {
        self.disable_for_io_error();
    }

    fn shutdown(&mut self) {
        self.disable_for_io_error();
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[cfg(test)]
    fn completed_frames(&self) -> usize {
        self.completed_frames
    }

    #[cfg(test)]
    fn pending(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }

    #[cfg(test)]
    fn slot_for(&self, frame_id: u64) -> usize {
        frame_id as usize % self.slots.len()
    }

    #[cfg(test)]
    fn slot_count(&self) -> usize {
        self.slots.len()
    }

    #[cfg(test)]
    fn backend(&self) -> &str {
        &self.backend
    }

    #[cfg(test)]
    fn slot_state(&self, slot: usize) -> SlotState {
        self.slots
            .get(slot % self.slots.len())
            .map_or(SlotState::Free, |slot| {
                slot.map_or(SlotState::Free, |_| SlotState::Pending)
            })
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DiagnosticImage {
    pub(crate) name: &'static str,
    pub(crate) image: vk::Image,
    pub(crate) extent: vk::Extent2D,
    pub(crate) format: vk::Format,
    pub(crate) layout: vk::ImageLayout,
}

impl DiagnosticImage {
    pub(crate) const fn new(
        name: &'static str,
        image: vk::Image,
        extent: vk::Extent2D,
        format: vk::Format,
        layout: vk::ImageLayout,
    ) -> Self {
        Self {
            name,
            image,
            extent,
            format,
            layout,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiagnosticFsrInputs {
    pub(crate) motion: String,
    pub(crate) confidence: String,
    pub(crate) depth: String,
    pub(crate) exposure: String,
    pub(crate) reactive: String,
    pub(crate) composition: String,
    pub(crate) jitter: String,
}

impl Default for DiagnosticFsrInputs {
    fn default() -> Self {
        Self {
            motion: "NotReported".into(),
            confidence: "NotReported".into(),
            depth: "NotReported".into(),
            exposure: "NotReported".into(),
            reactive: "NotReported".into(),
            composition: "NotReported".into(),
            jitter: "NotReported".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiagnosticFrameMetadata {
    pub(crate) frame_id: u64,
    pub(crate) generation_id: u64,
    pub(crate) slot_index: u32,
    pub(crate) slot_count: u32,
    pub(crate) motion_slot_index: u32,
    pub(crate) timestamp_ns: u64,
    pub(crate) frame_delta_raw_ns: u64,
    pub(crate) frame_delta_validated_ns: u64,
    pub(crate) frame_delta_smoothed_ns: u64,
    pub(crate) guidance_scale: f32,
    pub(crate) motion_quality: String,
    pub(crate) numeric_encoding: String,
    pub(crate) game_extent: [u32; 2],
    pub(crate) guidance_extent: [u32; 2],
    pub(crate) output_extent: [u32; 2],
    pub(crate) viewport: [f32; 4],
    pub(crate) reset_reason: String,
    pub(crate) backend: String,
    pub(crate) guidance_mode: String,
    pub(crate) ablations: GuidanceAblations,
    pub(crate) sharpening_enabled: bool,
    pub(crate) sharpness: f32,
    pub(crate) fsr_inputs: DiagnosticFsrInputs,
    pub(crate) history_age: u64,
    pub(crate) gpu_timings_ms: [f32; 15],
}

#[derive(Debug, Clone)]
struct ResourceLayout {
    name: &'static str,
    extent: vk::Extent2D,
    format: vk::Format,
    offset: u64,
    size: usize,
}

#[derive(Debug, Clone)]
struct CapturedResource {
    name: &'static str,
    extent: vk::Extent2D,
    format: vk::Format,
    offset: u64,
    size: usize,
}

struct PendingCapture {
    frame_id: u64,
    metadata: DiagnosticFrameMetadata,
    resources: Vec<CapturedResource>,
}

struct CaptureWriteTask {
    root: PathBuf,
    frame_id: u64,
    resources: Vec<(String, Vec<u8>)>,
    metadata: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EnqueueCapture {
    Queued,
    Full,
    Disconnected,
}

fn try_enqueue_capture(
    writer: &SyncSender<CaptureWriteTask>,
    task: CaptureWriteTask,
) -> EnqueueCapture {
    match writer.try_send(task) {
        Ok(()) => EnqueueCapture::Queued,
        Err(TrySendError::Full(_)) => EnqueueCapture::Full,
        Err(TrySendError::Disconnected(_)) => EnqueueCapture::Disconnected,
    }
}

fn write_capture_task(task: CaptureWriteTask) -> io::Result<()> {
    let frame_prefix = format!("frame-{:08}", task.frame_id);
    for (file, bytes) in task.resources {
        write_atomic(&task.root, &file, &bytes)?;
    }
    write_atomic(&task.root, &format!("{frame_prefix}.json"), &task.metadata)
}

struct CaptureWriter {
    sender: Option<SyncSender<CaptureWriteTask>>,
    thread: Option<JoinHandle<()>>,
}

impl CaptureWriter {
    fn enqueue(&self, task: CaptureWriteTask) -> EnqueueCapture {
        self.sender
            .as_ref()
            .map_or(EnqueueCapture::Disconnected, |sender| {
                try_enqueue_capture(sender, task)
            })
    }

    fn shutdown(&mut self) {
        self.sender.take();
        if let Some(worker) = self.thread.take()
            && worker.join().is_err()
        {
            eprintln!("TuxScaling: diagnostic capture writer thread panicked");
        }
    }
}

impl Drop for CaptureWriter {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn capture_writer() -> Result<CaptureWriter, vk::Result> {
    let (sender, receiver) = sync_channel::<CaptureWriteTask>(4);
    let worker = thread::Builder::new()
        .name("tuxscaling-capture-writer".into())
        .spawn(move || {
            while let Ok(task) = receiver.recv() {
                let frame_id = task.frame_id;
                if let Err(error) = write_capture_task(task) {
                    eprintln!(
                        "TuxScaling: diagnostic capture writer failed for frame {frame_id}: {error}"
                    );
                }
            }
        })
        .map_err(|_| vk::Result::ERROR_INITIALIZATION_FAILED)?;
    Ok(CaptureWriter {
        sender: Some(sender),
        thread: Some(worker),
    })
}

unsafe fn allocate_capture_resources(
    device: &ash::Device,
    memory: &vk::PhysicalDeviceMemoryProperties,
    game_extent: vk::Extent2D,
    guidance_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    output_format: vk::Format,
    image_count: usize,
) -> Result<(Vec<ResourceLayout>, Vec<DiagnosticSlot>), vk::Result> {
    let layouts = resource_layouts(game_extent, guidance_extent, output_extent, output_format)?;
    let total_size = layouts
        .iter()
        .map(|layout| layout.offset.saturating_add(layout.size as u64))
        .max()
        .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
    let usage = vk::BufferUsageFlags::TRANSFER_DST;
    let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let mut slots = Vec::with_capacity(image_count.max(1));
    for _ in 0..image_count.max(1) {
        slots.push(DiagnosticSlot {
            staging: unsafe { Buffer::new(device, memory, total_size, usage, flags) }?,
            pending: None,
        });
    }
    Ok((layouts, slots))
}

impl PendingCapture {
    #[cfg(test)]
    const fn readback_phase() -> &'static str {
        "after-owning-fence"
    }
}

struct DiagnosticSlot {
    staging: Buffer,
    pending: Option<PendingCapture>,
}

pub(crate) struct DiagnosticCapture {
    device: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    config: DiagnosticCaptureConfig,
    layouts: Vec<ResourceLayout>,
    slots: Vec<DiagnosticSlot>,
    state: CaptureState,
    writer: CaptureWriter,
}

impl DiagnosticCapture {
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        game_extent: vk::Extent2D,
        guidance_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        output_format: vk::Format,
        image_count: usize,
        config: DiagnosticCaptureConfig,
    ) -> Result<Self, vk::Result> {
        let (layouts, slots) = unsafe {
            allocate_capture_resources(
                device,
                memory,
                game_extent,
                guidance_extent,
                output_extent,
                output_format,
                image_count,
            )
        }?;
        let writer = capture_writer()?;
        Ok(Self {
            device: device.clone(),
            memory: *memory,
            config: config.clone(),
            layouts,
            slots,
            state: CaptureState::with_slot_count(config, image_count.max(1)),
            writer,
        })
    }

    pub(crate) fn config(&self) -> &DiagnosticCaptureConfig {
        &self.config
    }

    pub(crate) unsafe fn record_frame(
        &mut self,
        command: vk::CommandBuffer,
        slot_index: usize,
        metadata: DiagnosticFrameMetadata,
        images: &[DiagnosticImage],
    ) {
        if !self.state.arm_at(metadata.frame_id, slot_index) {
            return;
        }
        let index = slot_index % self.slots.len();
        let mut resources = Vec::with_capacity(images.len());
        for image in images {
            if let Some(layout) = self.layouts.iter().find(|layout| layout.name == image.name)
                && layout.extent == image.extent
                && layout.format == image.format
                && image.image != vk::Image::null()
            {
                unsafe {
                    self.record_image_copy(command, &self.slots[index].staging, layout, *image)
                };
                resources.push(CapturedResource {
                    name: layout.name,
                    extent: layout.extent,
                    format: layout.format,
                    offset: layout.offset,
                    size: layout.size,
                });
            }
        }
        if resources.is_empty() {
            self.state.complete(metadata.frame_id);
            return;
        }
        unsafe { transfer_memory_barrier(&self.device, command) };
        self.slots[index].pending = Some(PendingCapture {
            frame_id: metadata.frame_id,
            metadata,
            resources,
        });
    }

    pub(crate) unsafe fn record_additional(
        &mut self,
        command: vk::CommandBuffer,
        slot_index: usize,
        image: DiagnosticImage,
    ) {
        let index = slot_index % self.slots.len();
        let Some(layout) = self.layouts.iter().find(|layout| layout.name == image.name) else {
            return;
        };
        if layout.extent != image.extent
            || layout.format != image.format
            || image.image == vk::Image::null()
        {
            return;
        }
        if self.slots[index].pending.is_none() {
            return;
        }
        unsafe { self.record_image_copy(command, &self.slots[index].staging, layout, image) };
        let pending = self.slots[index]
            .pending
            .as_mut()
            .expect("pending checked above");
        pending
            .resources
            .retain(|resource| resource.name != image.name);
        pending.resources.push(CapturedResource {
            name: layout.name,
            extent: layout.extent,
            format: layout.format,
            offset: layout.offset,
            size: layout.size,
        });
        unsafe { transfer_memory_barrier(&self.device, command) };
    }

    unsafe fn record_image_copy(
        &self,
        command: vk::CommandBuffer,
        staging: &Buffer,
        layout: &ResourceLayout,
        image: DiagnosticImage,
    ) {
        unsafe {
            image_barrier(
                &self.device,
                command,
                image.image,
                image.layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );
            self.device.cmd_copy_image_to_buffer(
                command,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.handle,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(layout.offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: layout.extent.width,
                        height: layout.extent.height,
                        depth: 1,
                    })],
            );
            image_barrier(
                &self.device,
                command,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image.layout,
            );
        }
    }

    pub(crate) unsafe fn service_completed(&mut self, fences: &[vk::Fence]) {
        if !self.state.is_enabled() {
            return;
        }
        for index in 0..self.slots.len() {
            let Some(fence) = fences.get(index).copied() else {
                continue;
            };
            if fence == vk::Fence::null() {
                continue;
            }
            let ready = match unsafe { self.device.get_fence_status(fence) } {
                Ok(value) => value,
                Err(vk::Result::ERROR_DEVICE_LOST) => {
                    self.mark_device_lost();
                    return;
                }
                Err(error) => {
                    eprintln!("TuxScaling: diagnostic capture fence query failed: {error:?}");
                    continue;
                }
            };
            if !ready || self.slots[index].pending.is_none() {
                continue;
            }
            let pending = self.slots[index]
                .pending
                .take()
                .expect("pending checked above");
            match unsafe { self.queue_completed(index, &pending) } {
                Ok(EnqueueCapture::Queued) => self.state.complete(pending.frame_id),
                Ok(EnqueueCapture::Full) => {
                    eprintln!(
                        "TuxScaling: dropping diagnostic capture frame {}; writer queue is full",
                        pending.frame_id
                    );
                    self.state.complete(pending.frame_id);
                }
                Ok(EnqueueCapture::Disconnected) | Err(_) => {
                    eprintln!(
                        "TuxScaling: diagnostic capture disabled after frame {} writer became unavailable",
                        pending.frame_id
                    );
                    self.state.disable_for_io_error();
                    return;
                }
            }
        }
    }

    unsafe fn queue_completed(
        &self,
        index: usize,
        pending: &PendingCapture,
    ) -> io::Result<EnqueueCapture> {
        let frame_prefix = format!("frame-{:08}", pending.frame_id);
        let mut staging_bytes = vec![0_u8; self.slots[index].staging.size as usize];
        unsafe { self.slots[index].staging.read(&mut staging_bytes) }
            .map_err(|error| io::Error::other(format!("staging readback: {error:?}")))?;
        let mut resources = Vec::with_capacity(pending.resources.len());
        for resource in &pending.resources {
            let file = format!("{frame_prefix}-{}.bin", resource.name);
            let start = resource.offset as usize;
            let end = start.saturating_add(resource.size);
            let bytes = staging_bytes.get(start..end).ok_or_else(|| {
                io::Error::other(format!("readback {file}: staging range is invalid"))
            })?;
            resources.push((file, bytes.to_vec()));
        }
        let report = metadata_json(pending, &frame_prefix).into_bytes();
        Ok(self.writer.enqueue(CaptureWriteTask {
            root: self.config.root.clone(),
            frame_id: pending.frame_id,
            resources,
            metadata: report,
        }))
    }

    pub(crate) unsafe fn resize(
        &mut self,
        game_extent: vk::Extent2D,
        guidance_extent: vk::Extent2D,
        output_extent: vk::Extent2D,
        output_format: vk::Format,
        image_count: usize,
    ) -> Result<(), vk::Result> {
        let (layouts, slots) = unsafe {
            allocate_capture_resources(
                &self.device,
                &self.memory,
                game_extent,
                guidance_extent,
                output_extent,
                output_format,
                image_count,
            )
        }?;
        self.layouts = layouts;
        self.slots = slots;
        self.state.resize(image_count.max(1));
        Ok(())
    }

    pub(crate) fn mark_device_lost(&mut self) {
        for slot in &mut self.slots {
            slot.pending = None;
        }
        self.state.mark_device_lost();
    }

    pub(crate) fn shutdown(&mut self) {
        for slot in &mut self.slots {
            slot.pending = None;
        }
        self.state.shutdown();
        self.writer.shutdown();
    }
}

fn resource_layouts(
    game_extent: vk::Extent2D,
    guidance_extent: vk::Extent2D,
    output_extent: vk::Extent2D,
    output_format: vk::Format,
) -> Result<Vec<ResourceLayout>, vk::Result> {
    let definitions = [
        ("source", game_extent, output_format),
        ("reconstructed", output_extent, output_format),
        ("spatial_off", output_extent, output_format),
        ("comparison", output_extent, output_format),
        ("motion", guidance_extent, vk::Format::R16G16_SFLOAT),
        ("confidence", guidance_extent, vk::Format::R8_UNORM),
        ("disocclusion", guidance_extent, vk::Format::R8_UNORM),
        ("reactive", guidance_extent, vk::Format::R8_UNORM),
        ("composition", guidance_extent, vk::Format::R8_UNORM),
        ("relative_depth", guidance_extent, vk::Format::R32_SFLOAT),
        (
            "exposure",
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            vk::Format::R32_SFLOAT,
        ),
    ];
    let mut offset = 0_u64;
    let mut layouts = Vec::with_capacity(definitions.len());
    for (name, extent, format) in definitions {
        let size = image_size(extent, format)?;
        layouts.push(ResourceLayout {
            name,
            extent,
            format,
            offset,
            size,
        });
        offset = offset
            .checked_add(size as u64)
            .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)?;
        offset = align_up(offset, 4)?;
    }
    Ok(layouts)
}

fn image_size(extent: vk::Extent2D, format: vk::Format) -> Result<usize, vk::Result> {
    let bytes_per_pixel = match format {
        vk::Format::R8_UNORM => 1_u64,
        vk::Format::R16G16_SFLOAT => 4,
        vk::Format::R32_SFLOAT => 4,
        vk::Format::R8G8B8A8_UNORM
        | vk::Format::B8G8R8A8_UNORM
        | vk::Format::R8G8B8A8_SRGB
        | vk::Format::B8G8R8A8_SRGB => 4,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => return Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED),
    };
    u64::from(extent.width)
        .checked_mul(u64::from(extent.height))
        .and_then(|pixels| pixels.checked_mul(bytes_per_pixel))
        .and_then(|size| usize::try_from(size).ok())
        .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
}

fn align_up(value: u64, alignment: u64) -> Result<u64, vk::Result> {
    let mask = alignment.saturating_sub(1);
    value
        .checked_add(mask)
        .map(|value| value & !mask)
        .ok_or(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
}

fn format_name(format: vk::Format) -> String {
    match format {
        vk::Format::R8_UNORM => "R8_UNORM".into(),
        vk::Format::R16G16_SFLOAT => "R16G16_SFLOAT".into(),
        vk::Format::R32_SFLOAT => "R32_SFLOAT".into(),
        vk::Format::R8G8B8A8_UNORM => "R8G8B8A8_UNORM".into(),
        vk::Format::B8G8R8A8_UNORM => "B8G8R8A8_UNORM".into(),
        vk::Format::R8G8B8A8_SRGB => "R8G8B8A8_SRGB".into(),
        vk::Format::B8G8R8A8_SRGB => "B8G8R8A8_SRGB".into(),
        vk::Format::R16G16B16A16_SFLOAT => "R16G16B16A16_SFLOAT".into(),
        _ => format.as_raw().to_string(),
    }
}

fn json_string(value: &str) -> String {
    let mut result = String::with_capacity(value.len() + 2);
    result.push('"');
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            character if character.is_control() => {
                result.push_str(&format!("\\u{:04x}", character as u32))
            }
            character => result.push(character),
        }
    }
    result.push('"');
    result
}

fn metadata_json(pending: &PendingCapture, frame_prefix: &str) -> String {
    let metadata = &pending.metadata;
    let timings = metadata
        .gpu_timings_ms
        .iter()
        .map(|value| format!("{value:.6}"))
        .collect::<Vec<_>>()
        .join(",");
    let resources = pending
        .resources
        .iter()
        .map(|resource| {
            format!(
                "{{\"name\":{},\"file\":{},\"format\":{},\"extent\":[{},{}],\"offset\":{},\"bytes\":{}}}",
                json_string(resource.name),
                json_string(&format!("{frame_prefix}-{}.bin", resource.name)),
                json_string(&format_name(resource.format)),
                resource.extent.width,
                resource.extent.height,
                resource.offset,
                resource.size,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!(
        concat!(
            "{{\n",
            "  \"frame_id\":{},\n",
            "  \"generation_id\":{},\n",
            "  \"slot_index\":{},\n",
            "  \"slot_count\":{},\n",
            "  \"motion_slot_index\":{},\n",
            "  \"timestamp_ns\":{},\n",
            "  \"frame_delta_ns\":{{\"raw\":{},\"validated\":{},\"smoothed\":{}}},\n",
            "  \"guidance_scale\":{:.6},\n",
            "  \"motion_quality\":{},\n",
            "  \"numeric_encoding\":{},\n",
            "  \"game_extent\":[{},{}],\n",
            "  \"guidance_extent\":[{},{}],\n",
            "  \"output_extent\":[{},{}],\n",
            "  \"viewport\":[{:.6},{:.6},{:.6},{:.6}],\n",
            "  \"reset_reason\":{},\n",
            "  \"backend\":{},\n",
            "  \"guidance_mode\":{},\n",
            "  \"ablations\":{{\"motion\":{},\"relative_depth\":{},\"reactive\":{},\"composition\":{},\"exposure\":{},\"confidence_disocclusion\":{},\"post_capture_jitter\":{}}},\n",
            "  \"sharpening\":{{\"enabled\":{},\"sharpness\":{:.6}}},\n",
            "  \"fsr_inputs\":{{\"motion\":{},\"confidence\":{},\"depth\":{},\"exposure\":{},\"reactive\":{},\"composition\":{},\"jitter\":{}}},\n",
            "  \"history_age\":{},\n",
            "  \"gpu_timings_ms\":[{}],\n",
            "  \"resources\":[{}]\n",
            "}}\n"
        ),
        metadata.frame_id,
        metadata.generation_id,
        metadata.slot_index,
        metadata.slot_count,
        metadata.motion_slot_index,
        metadata.timestamp_ns,
        metadata.frame_delta_raw_ns,
        metadata.frame_delta_validated_ns,
        metadata.frame_delta_smoothed_ns,
        metadata.guidance_scale,
        json_string(&metadata.motion_quality),
        json_string(&metadata.numeric_encoding),
        metadata.game_extent[0],
        metadata.game_extent[1],
        metadata.guidance_extent[0],
        metadata.guidance_extent[1],
        metadata.output_extent[0],
        metadata.output_extent[1],
        metadata.viewport[0],
        metadata.viewport[1],
        metadata.viewport[2],
        metadata.viewport[3],
        json_string(&metadata.reset_reason),
        json_string(&metadata.backend),
        json_string(&metadata.guidance_mode),
        metadata.ablations.motion,
        metadata.ablations.relative_depth,
        metadata.ablations.reactive,
        metadata.ablations.composition,
        metadata.ablations.exposure,
        metadata.ablations.confidence_disocclusion,
        metadata.ablations.post_capture_jitter,
        metadata.sharpening_enabled,
        metadata.sharpness,
        json_string(&metadata.fsr_inputs.motion),
        json_string(&metadata.fsr_inputs.confidence),
        json_string(&metadata.fsr_inputs.depth),
        json_string(&metadata.fsr_inputs.exposure),
        json_string(&metadata.fsr_inputs.reactive),
        json_string(&metadata.fsr_inputs.composition),
        json_string(&metadata.fsr_inputs.jitter),
        metadata.history_age,
        timings,
        resources,
    )
}

fn write_atomic(root: &Path, name: &str, bytes: &[u8]) -> io::Result<()> {
    fs::create_dir_all(root)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |value| value.as_nanos());
    let temporary = root.join(format!(".{name}.{}.tmp", stamp));
    let final_path = root.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temporary, final_path)
}

#[cfg(test)]
mod tests {
    use super::{
        CaptureState, CaptureWriteTask, CapturedResource, DiagnosticCaptureConfig,
        DiagnosticFrameMetadata, DiagnosticFsrInputs, EnqueueCapture, PendingCapture, SlotState,
        capture_writer, image_size, metadata_json, resource_layouts, try_enqueue_capture,
        write_capture_task,
    };
    use ash::vk;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    use tuxscaling_temporal::GuidanceAblations;

    #[test]
    fn disabled_capture_requires_a_directory() {
        assert!(DiagnosticCaptureConfig::from_root(None).is_none());
        assert!(DiagnosticCaptureConfig::from_root(Some("".into())).is_none());
    }

    #[test]
    fn one_shot_capture_only_arms_one_frame() {
        let config = DiagnosticCaptureConfig::for_test(1);
        let mut state = CaptureState::new(config);
        assert!(state.arm(7));
        assert!(!state.arm(8));
        state.complete(7);
        assert_eq!(state.completed_frames(), 1);
    }

    #[test]
    fn frame_range_and_ring_slot_reuse_are_explicit() {
        let config = DiagnosticCaptureConfig::for_test_range(10, 12);
        let mut state = CaptureState::new(config);
        assert!(!state.arm(9));
        assert!(state.arm(10));
        assert_eq!(state.slot_for(10), 1);
        state.complete(10);
        assert!(state.arm(11));
        assert_eq!(state.slot_for(11), 2);
        state.complete(11);
        assert!(state.arm(12));
        assert_eq!(state.slot_for(12), 0);
        state.complete(12);
        assert!(!state.arm(13));
    }

    #[test]
    fn resize_and_backend_switch_keep_capture_state() {
        let config = DiagnosticCaptureConfig::for_test(4);
        let mut state = CaptureState::new(config);
        state.resize(3);
        state.set_backend("FSR 3.1.4");
        assert_eq!(state.slot_count(), 3);
        assert_eq!(state.backend(), "FSR 3.1.4");
        assert!(state.arm(4));
    }

    #[test]
    fn dropped_output_directory_disables_without_poisoning_frame_state() {
        let mut state = CaptureState::new(DiagnosticCaptureConfig::for_test(2));
        state.disable_for_io_error();
        assert!(!state.is_enabled());
        assert_eq!(state.slot_state(0), SlotState::Free);
    }

    #[test]
    fn device_loss_clears_pending_work_and_teardown_is_idempotent() {
        let mut state = CaptureState::new(DiagnosticCaptureConfig::for_test(2));
        assert!(state.arm(3));
        state.mark_device_lost();
        assert_eq!(state.pending(), 0);
        assert!(!state.is_enabled());
        state.shutdown();
        state.shutdown();
    }

    #[test]
    fn readback_waits_for_the_owning_fence_without_queue_idle() {
        let source = include_str!("present.rs");
        assert!(source.contains("service_completed"));
        assert!(source.contains("slot.fence"));
        assert!(!source.contains("queue_wait_idle"));
        assert!(!source.contains("QueueWaitIdle"));
        assert_eq!(PendingCapture::readback_phase(), "after-owning-fence");
    }

    #[test]
    fn capture_writer_queue_drops_completed_work_when_full() {
        use std::sync::mpsc::sync_channel;

        let (sender, receiver) = sync_channel(1);
        let make_task = |frame_id| CaptureWriteTask {
            root: PathBuf::from("target/diagnostic-capture-test"),
            frame_id,
            resources: Vec::new(),
            metadata: Vec::new(),
        };

        assert_eq!(
            try_enqueue_capture(&sender, make_task(1)),
            EnqueueCapture::Queued
        );
        assert_eq!(
            try_enqueue_capture(&sender, make_task(2)),
            EnqueueCapture::Full
        );
        drop(receiver);
        assert_eq!(
            try_enqueue_capture(&sender, make_task(3)),
            EnqueueCapture::Disconnected
        );
    }

    #[test]
    fn capture_writer_persists_raw_resources_and_metadata() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tuxscaling-capture-writer-{}-{nonce}",
            std::process::id()
        ));
        let task = CaptureWriteTask {
            root: root.clone(),
            frame_id: 9,
            resources: vec![("frame-00000009-source.bin".into(), vec![1, 2, 3, 4])],
            metadata: b"{\"frame_id\":9}".to_vec(),
        };

        write_capture_task(task).unwrap();

        assert_eq!(
            fs::read(root.join("frame-00000009-source.bin")).unwrap(),
            [1, 2, 3, 4]
        );
        assert_eq!(
            fs::read(root.join("frame-00000009.json")).unwrap(),
            b"{\"frame_id\":9}"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn capture_writer_shutdown_drains_queued_frames_before_returning() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "tuxscaling-capture-writer-drain-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let mut writer = capture_writer().unwrap();
        assert_eq!(
            writer.enqueue(CaptureWriteTask {
                root: root.clone(),
                frame_id: 11,
                resources: vec![("frame-00000011-source.bin".into(), vec![11, 12])],
                metadata: b"{\"frame_id\":11}".to_vec(),
            }),
            EnqueueCapture::Queued
        );

        writer.shutdown();

        assert_eq!(
            fs::read(root.join("frame-00000011-source.bin")).unwrap(),
            [11, 12]
        );
        assert_eq!(
            fs::read(root.join("frame-00000011.json")).unwrap(),
            b"{\"frame_id\":11}"
        );
        drop(writer);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staging_layout_contains_every_quality_signal_without_overlap() {
        let layouts = resource_layouts(
            vk::Extent2D {
                width: 128,
                height: 72,
            },
            vk::Extent2D {
                width: 128,
                height: 72,
            },
            vk::Extent2D {
                width: 192,
                height: 108,
            },
            vk::Format::R8G8B8A8_UNORM,
        )
        .unwrap();
        assert_eq!(layouts.len(), 11);
        for pair in layouts.windows(2) {
            assert!(pair[0].offset + pair[0].size as u64 <= pair[1].offset);
        }
        assert_eq!(
            image_size(
                vk::Extent2D {
                    width: 8,
                    height: 4
                },
                vk::Format::R8_UNORM
            ),
            Ok(32)
        );
        assert_eq!(
            image_size(
                vk::Extent2D {
                    width: 8,
                    height: 4
                },
                vk::Format::R16G16_SFLOAT
            ),
            Ok(128)
        );
    }

    #[test]
    fn metadata_json_describes_atomic_raw_resources_and_active_controls() {
        let pending = PendingCapture {
            frame_id: 7,
            metadata: DiagnosticFrameMetadata {
                frame_id: 7,
                generation_id: 2,
                slot_index: 1,
                slot_count: 3,
                motion_slot_index: 0,
                timestamp_ns: 123_456,
                frame_delta_raw_ns: 16_667,
                frame_delta_validated_ns: 16_667,
                frame_delta_smoothed_ns: 16_667,
                guidance_scale: 1.0,
                motion_quality: "balanced".into(),
                numeric_encoding: "srgb_nonlinear".into(),
                game_extent: [1280, 720],
                guidance_extent: [1280, 720],
                output_extent: [1920, 1080],
                viewport: [0.0, 0.125, 1.0, 0.75],
                reset_reason: "PresetChanged".into(),
                backend: "FSR 3.1.4".into(),
                guidance_mode: "Estimated".into(),
                ablations: GuidanceAblations {
                    motion: false,
                    relative_depth: true,
                    reactive: false,
                    composition: false,
                    exposure: false,
                    confidence_disocclusion: false,
                    post_capture_jitter: false,
                },
                sharpening_enabled: true,
                sharpness: 0.2,
                fsr_inputs: DiagnosticFsrInputs {
                    motion: "Estimated".into(),
                    confidence: "Estimated".into(),
                    depth: "SuppressedIncompatible".into(),
                    exposure: "SuppressedIncompatible".into(),
                    reactive: "Neutral".into(),
                    composition: "Neutral".into(),
                    jitter: "Neutral".into(),
                },
                history_age: 4,
                gpu_timings_ms: [0.0; 15],
            },
            resources: vec![CapturedResource {
                name: "source",
                extent: vk::Extent2D {
                    width: 1280,
                    height: 720,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                offset: 0,
                size: 1280 * 720 * 4,
            }],
        };
        let json = metadata_json(&pending, "frame-00000007");
        assert!(json.contains("\"backend\":\"FSR 3.1.4\""));
        assert!(json.contains("\"timestamp_ns\":123456"));
        assert!(json.contains("\"slot_index\":1"));
        assert!(json.contains("\"slot_count\":3"));
        assert!(json.contains("\"motion_slot_index\":0"));
        assert!(json.contains("\"frame_delta_ns\":{\"raw\":16667"));
        assert!(json.contains("\"guidance_scale\":1.000000"));
        assert!(json.contains("\"motion_quality\":\"balanced\""));
        assert!(json.contains("\"numeric_encoding\":\"srgb_nonlinear\""));
        assert!(json.contains("\"relative_depth\":true"));
        assert!(json.contains("\"sharpness\":0.200000"));
        assert!(json.contains("\"fsr_inputs\":{\"motion\":\"Estimated\""));
        assert!(json.contains("\"depth\":\"SuppressedIncompatible\""));
        assert!(json.contains("frame-00000007-source.bin"));
    }
}
