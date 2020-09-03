// Copyright 2020, ARM Limited.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use super::interrupt_controller::{Error, InterruptController};
use std::convert::TryInto;
use std::result;
use std::sync::Arc;
use vm_device::interrupt::{
    InterruptIndex, InterruptManager, InterruptSourceGroup, MsiIrqGroupConfig,
};
use vm_migration::{Migratable, MigratableError, Pausable, Snapshot, Snapshottable, Transportable};

type Result<T> = result::Result<T, Error>;

// Reserve 32 IRQs (GSI 32 ~ 64) for legacy device.
// GsiAllocator should allocate beyond this: from 64 on
pub const IRQ_LEGACY_COUNT: usize = 32;
pub const IRQ_SPI_OFFSET: usize = 32;

// This Gic struct implements InterruptController to provide interrupt delivery service.
// The Gic source files in arch/ folder maintain the Aarch64 specific Gic device.
// The 2 Gic instances could be merged together.
// Leave this refactoring to future. Two options may be considered:
//   1. Move Gic*.rs from arch/ folder here.
//   2. Move this file and ioapic.rs to arch/, as they are architecture specific.
pub struct Gic {
    device_entity: Option<Arc<dyn hypervisor::Device>>,
    // Vector holding values of GICR_TYPER for each vCPU
    gicr_typers: Vec<u64>,
    interrupt_source_group: Arc<Box<dyn InterruptSourceGroup>>,
}

impl Gic {
    pub fn new(
        vcpu_count: u8,
        interrupt_manager: Arc<dyn InterruptManager<GroupConfig = MsiIrqGroupConfig>>,
    ) -> Result<Gic> {
        let interrupt_source_group = interrupt_manager
            .create_group(MsiIrqGroupConfig {
                base: IRQ_SPI_OFFSET as InterruptIndex,
                count: IRQ_LEGACY_COUNT as InterruptIndex,
            })
            .map_err(Error::CreateInterruptSourceGroup)?;

        Ok(Gic {
            device_entity: None,
            gicr_typers: vec![0; vcpu_count.try_into().unwrap()],
            interrupt_source_group,
        })
    }

    pub fn set_device_entity(&mut self, device_entity: &Arc<dyn hypervisor::Device>) {
        self.device_entity = Some(Arc::clone(device_entity));
    }

    pub fn set_gicr_typers(&mut self, gicr_typers: Vec<u64>) {
        self.gicr_typers = gicr_typers;
    }
}

impl InterruptController for Gic {
    fn enable(&self) -> Result<()> {
        self.interrupt_source_group
            .enable()
            .map_err(Error::EnableInterrupt)?;
        Ok(())
    }

    // This should be called anytime an interrupt needs to be injected into the
    // running guest.
    fn service_irq(&mut self, irq: usize) -> Result<()> {
        self.interrupt_source_group
            .trigger(irq as InterruptIndex)
            .map_err(Error::TriggerInterrupt)?;

        Ok(())
    }
}

const GIC_SNAPSHOT_ID: &str = "gic";
impl Snapshottable for Gic {
    fn id(&self) -> String {
        GIC_SNAPSHOT_ID.to_string()
    }

    fn snapshot(&mut self) -> std::result::Result<Snapshot, MigratableError> {
        unimplemented!();
    }

    fn restore(&mut self, _snapshot: Snapshot) -> std::result::Result<(), MigratableError> {
        unimplemented!();
    }
}

impl Pausable for Gic {}
impl Transportable for Gic {}
impl Migratable for Gic {}
