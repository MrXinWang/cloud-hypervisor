// Copyright 2020 Arm Limited (or its affiliates). All rights reserved.
// Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod dist_regs;
pub mod gicv3;
pub mod gicv3_its;
pub mod icc_regs;
pub mod redist_regs;

pub use self::dist_regs::{get_dist_regs, read_ctlr, set_dist_regs, write_ctlr};
pub use self::icc_regs::{get_icc_regs, set_icc_regs};
pub use self::redist_regs::{get_redist_regs, set_redist_regs};
use std::any::Any;
use std::result;
use std::sync::Arc;

/// Errors thrown while setting up the GIC.
#[derive(Debug)]
pub enum Error {
    /// Error while calling KVM ioctl for setting up the global interrupt controller.
    CreateGIC(hypervisor::HypervisorVmError),
    /// Error while setting device attributes for the GIC.
    SetDeviceAttribute(hypervisor::HypervisorDeviceError),
    /// Error while getting device attributes for the GIC.
    GetDeviceAttribute(hypervisor::HypervisorDeviceError),
}
type Result<T> = result::Result<T, Error>;

pub trait GICDevice: Send {
    /// Returns the hypervisor agnostic Device of the GIC device
    fn device(&self) -> &Arc<dyn hypervisor::Device>;

    /// Returns the hypervisor agnostic Device of the GICITS device
    fn its_device(&self) -> Option<&Arc<dyn hypervisor::Device>>;

    /// Returns the fdt compatibility property of the device
    fn fdt_compatibility(&self) -> &str;

    /// Returns the maint_irq fdt property of the device
    fn fdt_maint_irq(&self) -> u32;

    /// Returns an array with GIC device properties
    fn device_properties(&self) -> &[u64];

    /// Returns the number of vCPUs this GIC handles
    fn vcpu_count(&self) -> u64;

    /// Returns whether the GIC device is MSI compatible or not
    fn msi_compatible(&self) -> bool {
        false
    }

    /// Returns the MSI compatibility property of the device
    fn msi_compatibility(&self) -> &str {
        ""
    }

    /// Returns the MSI reg property of the device
    fn msi_properties(&self) -> &[u64] {
        &[]
    }

    /// Get the values of GICR_TYPER for each vCPU.
    fn set_gicr_typers(&mut self, gicr_typers: Vec<u64>);

    /// Downcast the trait object to its concrete type.
    fn as_any_concrete_mut(&mut self) -> &mut dyn Any;
}

pub mod kvm {
    use super::GICDevice;
    use super::Result;
    use crate::aarch64::gic::gicv3::kvm::KvmGICv3;
    use crate::aarch64::gic::gicv3_its::kvm::KvmGICv3ITS;
    use hypervisor::kvm::kvm_bindings;
    use std::boxed::Box;
    use std::sync::Arc;

    /// Trait for GIC devices.
    pub trait KvmGICDevice: Send + Sync + GICDevice {
        /// Returns the GIC version of the device
        fn version() -> u32;

        /// Create the GIC device object
        fn create_device_object(
            device: Option<Arc<dyn hypervisor::Device>>,
            its_device: Option<Arc<dyn hypervisor::Device>>,
            vcpu_count: u64,
        ) -> Box<dyn GICDevice>;

        /// Setup the device-specific attributes
        fn init_device_attributes(gic_device_object: &dyn GICDevice) -> Result<()>;

        /// Initialize a GIC device
        fn init_device(vm: &Arc<dyn hypervisor::Vm>) -> Result<Arc<dyn hypervisor::Device>> {
            let mut gic_device = kvm_bindings::kvm_create_device {
                type_: Self::version(),
                fd: 0,
                flags: 0,
            };

            vm.create_device(&mut gic_device)
                .map_err(super::Error::CreateGIC)
        }

        /// Set a GIC device attribute
        fn set_device_attribute(
            device: &Arc<dyn hypervisor::Device>,
            group: u32,
            attr: u64,
            addr: u64,
            flags: u32,
        ) -> Result<()> {
            let attr = kvm_bindings::kvm_device_attr {
                group,
                attr,
                addr,
                flags,
            };
            device
                .set_device_attr(&attr)
                .map_err(super::Error::SetDeviceAttribute)?;

            Ok(())
        }

        /// Get a GIC device attribute
        fn get_device_attribute(
            device: &Arc<dyn hypervisor::Device>,
            group: u32,
            attr: u64,
            addr: u64,
            flags: u32,
        ) -> Result<()> {
            let mut attr = kvm_bindings::kvm_device_attr {
                group,
                attr,
                addr,
                flags,
            };
            device
                .get_device_attr(&mut attr)
                .map_err(super::Error::GetDeviceAttribute)?;

            Ok(())
        }

        /// Finalize the setup of a GIC device
        fn finalize_device(gic_device_object: &dyn GICDevice) -> Result<()>;
    }

    /// Create the GICv3 and the GICv3ITS device.
    pub fn create_gic(
        vm: &Arc<dyn hypervisor::Vm>,
        vcpu_count: u64,
    ) -> Result<(Box<dyn GICDevice>, Box<dyn GICDevice>)> {
        debug!("creating a GICv3");
        let gicv3_device = KvmGICv3::init_device(vm)?;
        debug!("creating a GICv3-ITS");
        let gicv3_its_device = KvmGICv3ITS::init_device(vm)?;

        let gicv3_device_obj =
            KvmGICv3::create_device_object(Some(gicv3_device.clone()), None, vcpu_count);
        let gicv3_its_device_obj = KvmGICv3ITS::create_device_object(
            Some(gicv3_device.clone()),
            Some(gicv3_its_device.clone()),
            vcpu_count,
        );

        /*
        KvmGICv3::init_device_attributes(&*gicv3_device_obj)?;
        KvmGICv3ITS::init_device_attributes(&*gicv3_its_device_obj)?;

        KvmGICv3ITS::finalize_device(&*gicv3_its_device_obj)?;
        KvmGICv3::finalize_device(&*gicv3_device_obj)?;
        */

        Ok((gicv3_device_obj, gicv3_its_device_obj))
    }

    /// Function that saves RDIST pending tables into guest RAM.
    ///
    /// The tables get flushed to guest RAM whenever the VM gets stopped.
    pub fn save_pending_tables(gic: &Arc<dyn hypervisor::Device>) -> Result<()> {
        let init_gic_attr = kvm_bindings::kvm_device_attr {
            group: kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CTRL,
            attr: u64::from(kvm_bindings::KVM_DEV_ARM_VGIC_SAVE_PENDING_TABLES),
            addr: 0,
            flags: 0,
        };
        gic.set_device_attr(&init_gic_attr)
            .map_err(super::Error::SetDeviceAttribute)
    }

    pub fn gicv3_its_attr_access(
        gicv3_its: &Arc<dyn hypervisor::Device>,
        group: u32,
        attr: u32,
        val: &u64,
        set: bool,
    ) -> Result<()> {
        let mut gicv3_its_attr = kvm_bindings::kvm_device_attr {
            group,
            attr: attr as u64,
            addr: val as *const u64 as u64,
            flags: 0,
        };
        if set {
            gicv3_its
                .set_device_attr(&gicv3_its_attr)
                .map_err(super::Error::SetDeviceAttribute)?;
        } else {
            gicv3_its
                .get_device_attr(&mut gicv3_its_attr)
                .map_err(super::Error::GetDeviceAttribute)?;
        }
        Ok(())
    }

    /// Function that saves ITS tables into guest RAM.
    ///
    /// The tables get flushed to guest RAM whenever the VM gets stopped.
    pub fn save_its_tables(gicv3_its: &Arc<dyn hypervisor::Device>) -> Result<()> {
        let init_gic_attr = kvm_bindings::kvm_device_attr {
            group: kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CTRL,
            attr: u64::from(kvm_bindings::KVM_DEV_ARM_ITS_SAVE_TABLES),
            addr: 0,
            flags: 0,
        };
        gicv3_its
            .set_device_attr(&init_gic_attr)
            .map_err(super::Error::SetDeviceAttribute)
    }

    /// Function that restores ITS tables into guest RAM.
    pub fn restore_its_tables(gicv3_its: &Arc<dyn hypervisor::Device>) -> Result<()> {
        let init_gic_attr = kvm_bindings::kvm_device_attr {
            group: kvm_bindings::KVM_DEV_ARM_VGIC_GRP_CTRL,
            attr: u64::from(kvm_bindings::KVM_DEV_ARM_ITS_RESTORE_TABLES),
            addr: 0,
            flags: 0,
        };
        gicv3_its
            .set_device_attr(&init_gic_attr)
            .map_err(super::Error::SetDeviceAttribute)
    }
}
