// Copyright 2020 ARM Limited
// SPDX-License-Identifier: Apache-2.0

pub mod kvm {
    use crate::aarch64::gic::dist_regs::{get_dist_regs, read_ctlr, set_dist_regs, write_ctlr};
    use crate::aarch64::gic::gicv3::kvm::KvmGICv3;
    use crate::aarch64::gic::icc_regs::{get_icc_regs, set_icc_regs};
    use crate::aarch64::gic::its_regs::{gic_v3_its_attr_access_32, gic_v3_its_attr_access_64};
    use crate::aarch64::gic::kvm::{save_pending_tables, KvmGICDevice};
    use crate::aarch64::gic::redist_regs::{get_redist_regs, set_redist_regs};
    use crate::aarch64::gic::GICDevice;
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

    const GITS_CTLR: u32 = 0x0000;
    const GITS_IIDR: u32 = 0x0004;
    const GITS_CBASER: u32 = 0x0080;
    const GITS_CWRITER: u32 = 0x0088;
    const GITS_CREADR: u32 = 0x0090;
    const GITS_BASER: u32 = 0x0100;

    /// Errors thrown while saving/restoring the GICv3.
    #[derive(Debug)]
    pub enum Error {
        /// Error in saving RDIST pending tables into guest RAM.
        SavePendingTables(crate::aarch64::gic::Error),
        /// Error in saving GICv3 distributor registers.
        SaveDistributorRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GICv3 distributor registers.
        RestoreDistributorRegisters(crate::aarch64::gic::Error),
        /// Error in saving GICv3 distributor control registers.
        SaveDistributorCtrlRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GICv3 distributor control registers.
        RestoreDistributorCtrlRegisters(crate::aarch64::gic::Error),
        /// Error in saving GICv3 redistributor registers.
        SaveRedistributorRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GICv3 redistributor registers.
        RestoreRedistributorRegisters(crate::aarch64::gic::Error),
        /// Error in saving GICv3 CPU interface registers.
        SaveICCRegisters(crate::aarch64::gic::Error),
        /// Error in restoring GICv3 CPU interface registers.
        RestoreICCRegisters(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS IIDR register.
        SaveITSIIDR(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS IIDR register.
        RestoreITSIIDR(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS CBASER register.
        SaveITSCBASER(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS CBASER register.
        RestoreITSCBASER(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS CREADR register.
        SaveITSCREADR(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS CREADR register.
        RestoreITSCREADR(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS CWRITER register.
        SaveITSCWRITER(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS CWRITER register.
        RestoreITSCWRITER(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS BASER register.
        SaveITSBASER(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS BASER register.
        RestoreITSBASER(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS CTLR register.
        SaveITSCTLR(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS CTLR register.
        RestoreITSCTLR(crate::aarch64::gic::Error),
        /// Error in saving GICv3ITS restore tables.
        SaveITSRestoreTables(crate::aarch64::gic::Error),
        /// Error in restoring GICv3ITS restore tables.
        RestoreITSRestoreTables(crate::aarch64::gic::Error),
    }

    type Result<T> = result::Result<T, Error>;

    pub struct KvmGICv3ITS {
        /// The hypervisor agnostic device representing the GICv3
        gic_v3_device: Arc<dyn hypervisor::Device>,

        /// The hypervisor agnostic device representing the GICv3ITS
        gic_v3_its_device: Arc<dyn hypervisor::Device>,

        /// Vector holding values of GICR_TYPER for each vCPU
        gicr_typers: Vec<u64>,

        /// GIC device properties, to be used for setting up the fdt entry
        gic_properties: [u64; 4],

        /// MSI device properties, to be used for setting up the fdt entry
        msi_properties: [u64; 2],

        /// Number of CPUs handled by the device
        vcpu_count: u64,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Gicv3ITSState {
        /// GICv3 registers
        dist: Vec<u32>,
        rdist: Vec<u32>,
        icc: Vec<u32>,
        // special register that enables interrupts and affinity routing
        gicd_ctlr: u32,

        /// GICv3ITS registers
        its_ctlr: u32,
        its_iidr: u32,
        its_cbaser: u64,
        its_cwriter: u64,
        its_creadr: u64,
        its_baser: [u64; 8],
    }

    impl KvmGICv3ITS {
        const KVM_VGIC_V3_ITS_SIZE: u64 = (2 * KvmGICv3::SZ_64K);

        fn get_msi_size() -> u64 {
            KvmGICv3ITS::KVM_VGIC_V3_ITS_SIZE
        }

        fn get_msi_addr(vcpu_count: u64) -> u64 {
            KvmGICv3::get_redists_addr(vcpu_count) - KvmGICv3ITS::get_msi_size()
        }

        /// Save the state of GICv3ITS.
        fn state(&self, gicr_typers: &[u64]) -> Result<Gicv3ITSState> {
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

            // Save GICv3ITS registers
            let its_baser_state: [u64; 8] = [0; 8];
            debug!("=====SAVE: its_baser_state before get={:#?}=====", its_baser_state);
            for i in 0..8 {
                gic_v3_its_attr_access_64(
                    &self.its_device().unwrap(),
                    kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                    GITS_BASER + i * 8,
                    &its_baser_state[i as usize],
                    false,
                )
                .map_err(Error::SaveITSBASER)?;
            }
            debug!("=====SAVE: its_baser_state after get={:#?}=====", its_baser_state);

            let its_ctlr_state: u32 = 0;
            debug!("=====SAVE: its_ctlr_state before get={:#?}=====", its_ctlr_state);

            gic_v3_its_attr_access_32(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CTLR,
                &its_ctlr_state,
                false,
            )
            .map_err(Error::SaveITSCTLR)?;
            debug!("=====SAVE: its_ctlr_state after get={:#?}=====", its_ctlr_state);

            let its_cbaser_state: u64 = 0;
            debug!("=====SAVE: its_cbaser_state before get={:#?}=====", its_cbaser_state);
            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CBASER,
                &its_cbaser_state,
                false,
            )
            .map_err(Error::SaveITSCBASER)?;
            debug!("=====SAVE: its_cbaser_state after get={:#?}=====", its_cbaser_state);

            let its_creadr_state: u64 = 0;
            debug!("=====SAVE: its_creadr_state before get={:#?}=====", its_creadr_state);
            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CREADR,
                &its_creadr_state,
                false,
            )
            .map_err(Error::SaveITSCREADR)?;
            debug!("=====SAVE: its_creadr_state after get={:#?}=====", its_creadr_state);

            let its_cwriter_state: u64 = 0;
            debug!("=====SAVE: its_cwriter_state before get={:#?}=====", its_cwriter_state);
            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CWRITER,
                &its_cwriter_state,
                false,
            )
            .map_err(Error::SaveITSCWRITER)?;
            debug!("=====SAVE: its_cwriter_state after get={:#?}=====", its_cwriter_state);

            let its_iidr_state: u32 = 0;
            debug!("=====SAVE: its_iidr_state before get={:#?}=====", its_iidr_state);
            gic_v3_its_attr_access_32(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_IIDR,
                &its_iidr_state,
                false,
            )
            .map_err(Error::SaveITSIIDR)?;
            debug!("=====SAVE: its_iidr_state before get={:#?}=====", its_iidr_state);

            Ok(Gicv3ITSState {
                dist: dist_state,
                rdist: rdist_state,
                icc: icc_state,
                gicd_ctlr,
                its_ctlr: its_ctlr_state,
                its_iidr: its_iidr_state,
                its_cbaser: its_cbaser_state,
                its_cwriter: its_cwriter_state,
                its_creadr: its_creadr_state,
                its_baser: its_baser_state,
            })
        }

        /// Restore the state of GICv3ITS.
        fn set_state(&mut self, gicr_typers: &[u64], state: &Gicv3ITSState) -> Result<()> {
            write_ctlr(&self.device(), state.gicd_ctlr)
                .map_err(Error::RestoreDistributorCtrlRegisters)?;

            set_dist_regs(&self.device(), &state.dist)
                .map_err(Error::RestoreDistributorRegisters)?;

            set_redist_regs(&self.device(), gicr_typers, &state.rdist)
                .map_err(Error::RestoreRedistributorRegisters)?;

            set_icc_regs(&self.device(), &gicr_typers, &state.icc)
                .map_err(Error::RestoreICCRegisters)?;

            //Restore GICv3ITS registers
            debug!("=====RESTORE: its_iidr_state={:#?}=====", &state.its_iidr);
            gic_v3_its_attr_access_32(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_IIDR,
                &state.its_iidr,
                true,
            )
            .map_err(Error::RestoreITSIIDR)?;

            debug!("=====RESTORE: its_cbaser_state={:#?}=====", &state.its_cbaser);
            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CBASER,
                &state.its_cbaser,
                true,
            )
            .map_err(Error::RestoreITSCBASER)?;

            debug!("=====RESTORE: its_creadr_state={:#?}=====", &state.its_creadr);
            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CREADR,
                &state.its_creadr,
                true,
            )
            .map_err(Error::RestoreITSCREADR)?;

            debug!("=====RESTORE: its_cwriter_state={:#?}=====", &state.its_cwriter);
            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CWRITER,
                &state.its_cwriter,
                true,
            )
            .map_err(Error::RestoreITSCWRITER)?;

            debug!("=====RESTORE: its_baser_state={:#?}=====", &state.its_baser);
            for i in 0..8 {
                gic_v3_its_attr_access_64(
                    &self.its_device().unwrap(),
                    kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                    GITS_BASER + i * 8,
                    &state.its_baser[i as usize],
                    true,
                )
                .map_err(Error::RestoreITSBASER)?;
            }

            gic_v3_its_attr_access_64(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CTRL,
                kvm_bindings::KVM_DEV_ARM_ITS_RESTORE_TABLES,
                &0,
                true,
            )
            .map_err(Error::RestoreITSRestoreTables)?;

            debug!("=====RESTORE: its_ctlr_state={:#?}=====", &state.its_ctlr);
            gic_v3_its_attr_access_32(
                &self.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ITS_REGS,
                GITS_CTLR,
                &state.its_ctlr,
                true,
            )
            .map_err(Error::RestoreITSCTLR)?;

            Ok(())
        }
    }

    impl GICDevice for KvmGICv3ITS {
        fn device(&self) -> &Arc<dyn hypervisor::Device> {
            &self.gic_v3_device
        }

        fn its_device(&self) -> Option<&Arc<dyn hypervisor::Device>> {
            Some(&self.gic_v3_its_device)
        }

        fn fdt_compatibility(&self) -> &str {
            "arm,gic-v3"
        }

        fn msi_compatible(&self) -> bool {
            true
        }

        fn msi_compatibility(&self) -> &str {
            "arm,gic-v3-its"
        }

        fn fdt_maint_irq(&self) -> u32 {
            KvmGICv3::ARCH_GIC_V3_MAINT_IRQ
        }

        fn msi_properties(&self) -> &[u64] {
            &self.msi_properties
        }

        fn device_properties(&self) -> &[u64] {
            &self.gic_properties
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

    impl KvmGICDevice for KvmGICv3ITS {
        fn version() -> u32 {
            kvm_bindings::kvm_device_type_KVM_DEV_TYPE_ARM_VGIC_ITS
        }

        fn create_device(
            gic_v3_device: Option<Arc<dyn hypervisor::Device>>,
            gic_v3_its_device: Option<Arc<dyn hypervisor::Device>>,
            vcpu_count: u64,
        ) -> Box<dyn GICDevice> {
            Box::new(KvmGICv3ITS {
                gic_v3_device: gic_v3_device.unwrap(),
                gic_v3_its_device: gic_v3_its_device.unwrap(),
                gicr_typers: vec![0; vcpu_count.try_into().unwrap()],
                gic_properties: [
                    KvmGICv3::get_dist_addr(),
                    KvmGICv3::get_dist_size(),
                    KvmGICv3::get_redists_addr(vcpu_count),
                    KvmGICv3::get_redists_size(vcpu_count),
                ],
                msi_properties: [
                    KvmGICv3ITS::get_msi_addr(vcpu_count),
                    KvmGICv3ITS::get_msi_size(),
                ],
                vcpu_count,
            })
        }

        fn init_device_attributes(gic_device: &dyn GICDevice) -> crate::aarch64::gic::Result<()> {
            Self::set_device_attribute(
                gic_device.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_ADDR,
                u64::from(kvm_bindings::KVM_VGIC_ITS_ADDR_TYPE),
                &KvmGICv3ITS::get_msi_addr(gic_device.vcpu_count()) as *const u64 as u64,
                0,
            )?;

            Self::set_device_attribute(
                gic_device.its_device().unwrap(),
                kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CTRL,
                u64::from(kvm_bindings::KVM_DEV_ARM_VGIC_CTRL_INIT),
                0,
                0,
            )?;

            Ok(())
        }

        fn new(
            vm: &Arc<dyn hypervisor::Vm>,
            vcpu_count: u64,
        ) -> crate::aarch64::gic::Result<Box<dyn GICDevice>> {
            let vgic_v3_device = KvmGICv3::init_device(vm)?;
            let vgic_v3_its_device = Self::init_device(vm)?;
            let gicv3_its_device_obj =
                Self::create_device(Some(vgic_v3_device), Some(vgic_v3_its_device), vcpu_count);

            KvmGICv3::init_device_attributes(&*gicv3_its_device_obj)?;
            Self::init_device_attributes(&*gicv3_its_device_obj)?;
            Self::finalize_device(&*gicv3_its_device_obj)?;

            Ok(gicv3_its_device_obj)
        }
    }

    pub const GIC_V3_ITS_SNAPSHOT_ID: &str = "gic-v3-its";
    impl Snapshottable for KvmGICv3ITS {
        fn id(&self) -> String {
            GIC_V3_ITS_SNAPSHOT_ID.to_string()
        }

        fn snapshot(&mut self) -> std::result::Result<Snapshot, MigratableError> {
            let gicr_typers = self.gicr_typers.clone();
            let snapshot = serde_json::to_vec(&self.state(&gicr_typers).unwrap())
                .map_err(|e| MigratableError::Snapshot(e.into()))?;

            let mut gic_v3_its_snapshot = Snapshot::new(self.id().as_str());
            gic_v3_its_snapshot.add_data_section(SnapshotDataSection {
                id: format!("{}-section", self.id()),
                snapshot,
            });

            Ok(gic_v3_its_snapshot)
        }

        fn restore(&mut self, snapshot: Snapshot) -> std::result::Result<(), MigratableError> {
            if let Some(gic_v3_its_section) = snapshot
                .snapshot_data
                .get(&format!("{}-section", self.id()))
            {
                let gic_v3_its_state = match serde_json::from_slice(&gic_v3_its_section.snapshot) {
                    Ok(state) => state,
                    Err(error) => {
                        return Err(MigratableError::Restore(anyhow!(
                            "Could not deserialize GICv3ITS {}",
                            error
                        )))
                    }
                };

                let gicr_typers = self.gicr_typers.clone();
                return self
                    .set_state(&gicr_typers, &gic_v3_its_state)
                    .map_err(|e| {
                        MigratableError::Restore(anyhow!(
                            "Could not restore GICv3ITS state {:?}",
                            e
                        ))
                    });
            }

            Err(MigratableError::Restore(anyhow!(
                "Could not find GICv3ITS snapshot section"
            )))
        }
    }

    impl Pausable for KvmGICv3ITS {}
    impl Transportable for KvmGICv3ITS {}
    impl Migratable for KvmGICv3ITS {}
}
