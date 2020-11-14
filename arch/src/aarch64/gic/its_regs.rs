use super::{Error, Result};
use hypervisor::kvm::kvm_bindings::kvm_device_attr;
use std::sync::Arc;

pub fn gic_v3_its_attr_access_32(
    gic_v3_its: &Arc<dyn hypervisor::Device>,
    group: u32,
    attr: u32,
    val: &u32,
    set: bool,
) -> Result<()> {
    let mut gic_v3_its_attr = kvm_device_attr {
        group,
        attr: attr as u64,
        addr: val as *const u32 as u64,
        flags: 0,
    };
    if set {
        gic_v3_its
            .set_device_attr(&gic_v3_its_attr)
            .map_err(Error::SetDeviceAttribute)?;
    } else {
        gic_v3_its
            .get_device_attr(&mut gic_v3_its_attr)
            .map_err(Error::GetDeviceAttribute)?;
    }
    Ok(())
}

pub fn gic_v3_its_attr_access_64(
    gic_v3_its: &Arc<dyn hypervisor::Device>,
    group: u32,
    attr: u32,
    val: &u64,
    set: bool,
) -> Result<()> {
    let mut gic_v3_its_attr = kvm_device_attr {
        group,
        attr: attr as u64,
        addr: val as *const u64 as u64,
        flags: 0,
    };
    if set {
        gic_v3_its
            .set_device_attr(&gic_v3_its_attr)
            .map_err(Error::SetDeviceAttribute)?;
    } else {
        gic_v3_its
            .get_device_attr(&mut gic_v3_its_attr)
            .map_err(Error::GetDeviceAttribute)?;
    }
    Ok(())
}
