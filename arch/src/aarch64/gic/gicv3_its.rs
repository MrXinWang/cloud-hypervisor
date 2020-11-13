// Copyright 2020 ARM Limited
// SPDX-License-Identifier: Apache-2.0

pub mod kvm {
    use std::any::Any;
    use std::convert::TryInto;
    use std::sync::Arc;
    use std::{boxed::Box, result};
    type Result<T> = result::Result<T, Error>;
    use crate::aarch64::gic::gicv3::kvm::KvmGICv3;
    use crate::aarch64::gic::kvm::KvmGICDevice;
    use crate::aarch64::gic::{Error, GICDevice};
    use hypervisor::kvm::kvm_bindings;

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

    impl KvmGICv3ITS {
        const KVM_VGIC_V3_ITS_SIZE: u64 = (2 * KvmGICv3::SZ_64K);

        fn get_msi_size() -> u64 {
            KvmGICv3ITS::KVM_VGIC_V3_ITS_SIZE
        }

        fn get_msi_addr(vcpu_count: u64) -> u64 {
            KvmGICv3::get_redists_addr(vcpu_count) - KvmGICv3ITS::get_msi_size()
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

        fn init_device_attributes(gic_device: &dyn GICDevice) -> Result<()> {
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
}
