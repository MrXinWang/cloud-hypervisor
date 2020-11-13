// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod kvm {
    use crate::aarch64::gic::dist_regs::{get_dist_regs, read_ctlr, set_dist_regs, write_ctlr};
    use crate::aarch64::gic::icc_regs::{get_icc_regs, set_icc_regs};
    use crate::aarch64::gic::kvm::{save_pending_tables, KvmGICDevice};
    use crate::aarch64::gic::redist_regs::{get_redist_regs, set_redist_regs};
    use crate::aarch64::gic::GICDevice;
    use crate::layout;
    use anyhow::anyhow;
    use hypervisor::kvm::kvm_bindings;
    use std::any::Any;
    use std::convert::TryInto;
    use std::sync::Arc;
    use std::{boxed::Box, result};
    use vm_migration::{
        Migratable, MigratableError, Pausable, Snapshot, SnapshotDataSection, Snapshottable,
        Transportable,
    };

    /// Errors thrown while saving/restoring the GICv3.
    #[derive(Debug)]
    pub enum Error {
        /// Error in saving RDIST pending tables into guest RAM.
        SavePendingTables(crate::aarch64::gic::Error),
        /// Error in saving GIC distributor registers.
        SaveDistributorRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GIC distributor registers.
        RestoreDistributorRegisters(crate::aarch64::gic::Error),
        /// Error in saving GIC distributor control registers.
        SaveDistributorCtrlRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GIC distributor control registers.
        RestoreDistributorCtrlRegisters(crate::aarch64::gic::Error),
        /// Error in saving GIC redistributor registers.
        SaveRedistributorRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GIC redistributor registers.
        RestoreRedistributorRegisters(crate::aarch64::gic::Error),
        /// Error in saving GIC CPU interface registers.
        SaveICCRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GIC CPU interface registers.
        RestoreICCRegisters(crate::aarch64::gic::Error),
    }

    type Result<T> = result::Result<T, Error>;

    pub struct KvmGICv3 {
        /// The hypervisor agnostic device
        device: Arc<dyn hypervisor::Device>,

        /// Vector holding values of GICR_TYPER for each vCPU
        gicr_typers: Vec<u64>,

        /// GIC device properties, to be used for setting up the fdt entry
        properties: [u64; 4],

        /// Number of CPUs handled by the device
        vcpu_count: u64,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Gicv3State {
        dist: Vec<u32>,
        rdist: Vec<u32>,
        icc: Vec<u32>,
        // special register that enables interrupts and affinity routing
        gicd_ctlr: u32,
    }

    impl KvmGICv3 {
        // Unfortunately bindgen omits defines that are based on other defines.
        // See arch/arm64/include/uapi/asm/kvm.h file from the linux kernel.
        pub const SZ_64K: u64 = 0x0001_0000;
        const KVM_VGIC_V3_DIST_SIZE: u64 = KvmGICv3::SZ_64K;
        const KVM_VGIC_V3_REDIST_SIZE: u64 = (2 * KvmGICv3::SZ_64K);

        // Device trees specific constants
        pub const ARCH_GIC_V3_MAINT_IRQ: u32 = 9;

        /// Get the address of the GIC distributor.
        pub fn get_dist_addr() -> u64 {
            layout::MAPPED_IO_START - KvmGICv3::KVM_VGIC_V3_DIST_SIZE
        }

        /// Get the size of the GIC distributor.
        pub fn get_dist_size() -> u64 {
            KvmGICv3::KVM_VGIC_V3_DIST_SIZE
        }

        /// Get the address of the GIC redistributors.
        pub fn get_redists_addr(vcpu_count: u64) -> u64 {
            KvmGICv3::get_dist_addr() - KvmGICv3::get_redists_size(vcpu_count)
        }

        /// Get the size of the GIC redistributors.
        pub fn get_redists_size(vcpu_count: u64) -> u64 {
            vcpu_count * KvmGICv3::KVM_VGIC_V3_REDIST_SIZE
        }

        /// Save the state of GIC.
        fn state(&self, gicr_typers: &[u64]) -> Result<Gicv3State> {
            // Flush redistributors pending tables to guest RAM.
            save_pending_tables(&self.device()).map_err(Error::SavePendingTables)?;

            let gicd_ctlr =
                read_ctlr(&self.device()).map_err(Error::SaveDistributorCtrlRegisters)?;

            let dist_state =
                get_dist_regs(&self.device()).map_err(Error::SaveDistributorRegisters)?;

            let rdist_state = get_redist_regs(&self.device(), &gicr_typers)
                .map_err(Error::SaveRedistributorRegisters)?;

            let icc_state =
                get_icc_regs(&self.device(), &gicr_typers).map_err(Error::SaveICCRegisters)?;

            Ok(Gicv3State {
                dist: dist_state,
                rdist: rdist_state,
                icc: icc_state,
                gicd_ctlr,
            })
        }

        /// Restore the state of GIC.
        fn set_state(&mut self, gicr_typers: &[u64], state: &Gicv3State) -> Result<()> {
            write_ctlr(&self.device(), state.gicd_ctlr)
                .map_err(Error::RestoreDistributorCtrlRegisters)?;

            set_dist_regs(&self.device(), &state.dist)
                .map_err(Error::RestoreDistributorRegisters)?;

            set_redist_regs(&self.device(), gicr_typers, &state.rdist)
                .map_err(Error::RestoreRedistributorRegisters)?;

            set_icc_regs(&self.device(), &gicr_typers, &state.icc)
                .map_err(Error::RestoreICCRegisters)?;

            Ok(())
        }
    }

    impl GICDevice for KvmGICv3 {
        fn device(&self) -> &Arc<dyn hypervisor::Device> {
            &self.device
        }

        fn its_device(&self) -> Option<&Arc<dyn hypervisor::Device>> {
            None
        }

        fn fdt_compatibility(&self) -> &str {
            "arm,gic-v3"
        }

        fn fdt_maint_irq(&self) -> u32 {
            KvmGICv3::ARCH_GIC_V3_MAINT_IRQ
        }

        fn device_properties(&self) -> &[u64] {
            &self.properties
        }

        fn vcpu_count(&self) -> u64 {
            self.vcpu_count
        }

        fn set_gicr_typers(&mut self, gicr_typers: Vec<u64>) {
            self.gicr_typers = gicr_typers;
        }

        fn as_any_concrete_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl KvmGICDevice for KvmGICv3 {
        fn version() -> u32 {
            kvm_bindings::kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_V3
        }

        fn create_device(
            gic_v3_device: Option<Arc<dyn hypervisor::Device>>,
            _gic_v3_its_device: Option<Arc<dyn hypervisor::Device>>,
            vcpu_count: u64,
        ) -> Box<dyn GICDevice> {
            Box::new(KvmGICv3 {
                device: gic_v3_device.unwrap(),
                gicr_typers: vec![0; vcpu_count.try_into().unwrap()],
                properties: [
                    KvmGICv3::get_dist_addr(),
                    KvmGICv3::get_dist_size(),
                    KvmGICv3::get_redists_addr(vcpu_count),
                    KvmGICv3::get_redists_size(vcpu_count),
                ],
                vcpu_count,
            })
        }

        fn init_device_attributes(gic_device: &dyn GICDevice) -> crate::aarch64::gic::Result<()> {
            /* Setting up the distributor attribute.
             We are placing the GIC below 1GB so we need to substract the size of the distributor.
            */
            Self::set_device_attribute(
                gic_device.device(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ADDR,
                u64::from(kvm_bindings::KVM_VGIC_V3_ADDR_TYPE_DIST),
                &KvmGICv3::get_dist_addr() as *const u64 as u64,
                0,
            )?;

            /* Setting up the redistributors' attribute.
            We are calculating here the start of the redistributors address. We have one per CPU.
            */
            Self::set_device_attribute(
                gic_device.device(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ADDR,
                u64::from(kvm_bindings::KVM_VGIC_V3_ADDR_TYPE_REDIST),
                &KvmGICv3::get_redists_addr(gic_device.vcpu_count()) as *const u64 as u64,
                0,
            )?;

            Ok(())
        }

        fn new(
            vm: &Arc<dyn hypervisor::Vm>,
            vcpu_count: u64,
        ) -> crate::aarch64::gic::Result<Box<dyn GICDevice>> {
            let vgic_v3_device = Self::init_device(vm)?;
            let vgic_v3_device_obj = Self::create_device(Some(vgic_v3_device), None, vcpu_count);

            Self::init_device_attributes(&*vgic_v3_device_obj)?;
            Self::finalize_device(&*vgic_v3_device_obj)?;

            Ok(vgic_v3_device_obj)
        }
    }

    pub const GIC_V3_SNAPSHOT_ID: &str = "gic-v3";
    impl Snapshottable for KvmGICv3 {
        fn id(&self) -> String {
            GIC_V3_SNAPSHOT_ID.to_string()
        }

        fn snapshot(&mut self) -> std::result::Result<Snapshot, MigratableError> {
            let gicr_typers = self.gicr_typers.clone();
            let snapshot = serde_json::to_vec(&self.state(&gicr_typers).unwrap())
                .map_err(|e| MigratableError::Snapshot(e.into()))?;

            let mut gic_v3_snapshot = Snapshot::new(self.id().as_str());
            gic_v3_snapshot.add_data_section(SnapshotDataSection {
                id: format!("{}-section", self.id()),
                snapshot,
            });

            Ok(gic_v3_snapshot)
        }

        fn restore(&mut self, snapshot: Snapshot) -> std::result::Result<(), MigratableError> {
            if let Some(gic_v3_section) = snapshot
                .snapshot_data
                .get(&format!("{}-section", self.id()))
            {
                let gic_v3_state = match serde_json::from_slice(&gic_v3_section.snapshot) {
                    Ok(state) => state,
                    Err(error) => {
                        return Err(MigratableError::Restore(anyhow!(
                            "Could not deserialize GICv3 {}",
                            error
                        )))
                    }
                };

                let gicr_typers = self.gicr_typers.clone();
                return self.set_state(&gicr_typers, &gic_v3_state).map_err(|e| {
                    MigratableError::Restore(anyhow!("Could not restore GICv3 state {:?}", e))
                });
            }

            Err(MigratableError::Restore(anyhow!(
                "Could not find GICv3 snapshot section"
            )))
        }
    }

    impl Pausable for KvmGICv3 {}
    impl Transportable for KvmGICv3 {}
    impl Migratable for KvmGICv3 {}
}
