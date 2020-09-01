// Copyright 2020, ARM Limited.
//
// SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause

use super::interrupt_controller::{Error, InterruptController};
use anyhow::anyhow;
use std::convert::TryInto;
use std::result;
use std::sync::Arc;
use vm_device::interrupt::{
    InterruptIndex, InterruptManager, InterruptSourceGroup, MsiIrqGroupConfig,
};
use vm_migration::{
    Migratable, MigratableError, Pausable, Snapshot, SnapshotDataSection, Snapshottable,
    Transportable,
};

use arch::aarch64::gic::dist_regs::{get_dist_regs, read_ctlr, set_dist_regs, write_ctlr};
use arch::aarch64::gic::icc_regs::{get_icc_regs, set_icc_regs};
use arch::aarch64::gic::kvm::save_pending_tables;
use arch::aarch64::gic::redist_regs::{get_redist_regs, set_redist_regs};

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
    id: String,
    device_entity: Option<Arc<dyn hypervisor::Device>>,
    // Vector holding values of GICR_TYPER for each vCPU
    gicr_typers: Vec<u64>,
    interrupt_source_group: Arc<Box<dyn InterruptSourceGroup>>,
}

#[derive(Serialize, Deserialize)]
pub struct GicState {
    dist: Vec<u32>,
    rdist: Vec<u32>,
    icc: Vec<u32>,
    // special register that enables interrupts and affinity routing
    gicd_ctlr: u32,
}

impl Gic {
    pub fn new(
        id: String,
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
            id,
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

    fn state(&self, gicr_typers: &Vec<u64>) -> Result<GicState> {
        // Flush redistributors pending tables to guest RAM.
        save_pending_tables(&self.device_entity.as_ref().unwrap())
            .map_err(|e| Error::SaveRegisters("RAM pending tables", e))?;

        let gicd_ctlr = read_ctlr(&self.device_entity.as_ref().unwrap())
            .map_err(|e| Error::SaveRegisters("distributor control register", e))?;

        let dist_state = get_dist_regs(&self.device_entity.as_ref().unwrap())
            .map_err(|e| Error::SaveRegisters("distributor registers", e))?;

        let rdist_state = get_redist_regs(&self.device_entity.as_ref().unwrap(), &gicr_typers)
            .map_err(|e| Error::SaveRegisters("redistributor registers", e))?;

        let icc_state = get_icc_regs(&self.device_entity.as_ref().unwrap(), &gicr_typers)
            .map_err(|e| Error::SaveRegisters("CPU interface registers", e))?;

        Ok(GicState {
            dist: dist_state,
            rdist: rdist_state,
            icc: icc_state,
            gicd_ctlr: gicd_ctlr,
        })
    }

    fn set_state(&mut self, gicr_typers: &Vec<u64>, state: &GicState) -> Result<()> {
        write_ctlr(&self.device_entity.as_ref().unwrap(), state.gicd_ctlr)
            .map_err(|e| Error::RestoreRegisters("distributor control register", e))?;

        set_dist_regs(&self.device_entity.as_ref().unwrap(), &state.dist)
            .map_err(|e| Error::RestoreRegisters("distributor registers", e))?;

        set_redist_regs(
            &self.device_entity.as_ref().unwrap(),
            gicr_typers,
            &state.rdist,
        )
        .map_err(|e| Error::RestoreRegisters("redistributor registers", e))?;

        set_icc_regs(
            &self.device_entity.as_ref().unwrap(),
            &gicr_typers,
            &state.icc,
        )
        .map_err(|e| Error::SaveRegisters("CPU interface registers", e))?;

        Ok(())
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
        let gicr_typers = self.gicr_typers.clone();
        let snapshot = serde_json::to_vec(&self.state(&gicr_typers).unwrap())
            .map_err(|e| MigratableError::Snapshot(e.into()))?;

        let mut gic_snapshot = Snapshot::new(self.id.as_str());
        gic_snapshot.add_data_section(SnapshotDataSection {
            id: format!("{}-section", self.id),
            snapshot,
        });

        Ok(gic_snapshot)
    }

    fn restore(&mut self, snapshot: Snapshot) -> std::result::Result<(), MigratableError> {
        if let Some(gic_section) = snapshot.snapshot_data.get(&format!("{}-section", self.id)) {
            let gic_state = match serde_json::from_slice(&gic_section.snapshot) {
                Ok(state) => state,
                Err(error) => {
                    return Err(MigratableError::Restore(anyhow!(
                        "Could not deserialize GIC {}",
                        error
                    )))
                }
            };

            let gicr_typers = self.gicr_typers.clone();
            return self.set_state(&gicr_typers, &gic_state).map_err(|e| {
                MigratableError::Restore(anyhow!("Could not restore GIC state {:?}", e))
            });
        }

        Err(MigratableError::Restore(anyhow!(
            "Could not find GIC snapshot section"
        )))
    }
}

impl Pausable for Gic {}
impl Transportable for Gic {}
impl Migratable for Gic {}
