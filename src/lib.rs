#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]

mod tests {
    use super::*;
    use std::mem;
    use std::ffi::{CString};
    use std::os::raw::c_char;

    #[test]
    fn inference_of_network_gives_correct_result() {
        let config_file = CString::new("").unwrap();
        let config_file_ptr: *const c_char = config_file.as_ptr();
   
        unsafe {
            let mut core: *mut ie_core_t = mem::zeroed();
            let status = ie_core_create(config_file_ptr, &mut core as *mut *mut ie_core_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }
           
            let input_model = CString::new("/home/artem/intel/openvino_2020.1.023/deployment_tools/model_optimizer/resnet-50.xml").unwrap();
            let input_model_ptr: *const c_char = input_model.as_ptr();
            let input_weights = CString::new("/home/artem/intel/openvino_2020.1.023/deployment_tools/model_optimizer/resnet-50.bin").unwrap();
            let input_weights_ptr: *const c_char = input_weights.as_ptr();
            let mut network: *mut ie_network_t = mem::zeroed();
            let status = ie_core_read_network(core, input_model_ptr,
                input_weights_ptr, &mut network as *mut *mut ie_network_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }

            let device_name = CString::new("CPU").unwrap();
            let device_name_ptr: *const c_char = device_name.as_ptr();
            let config: ie_config_t = ie_config{
                name: std::ptr::null_mut(),
                next: std::ptr::null_mut(),
                value: std::ptr::null_mut()
            };
            let mut executable_network: *mut ie_executable_network_t = mem::zeroed();
            let status = ie_core_load_network(core, network, device_name_ptr,
                &config as *const ie_config_t, &mut executable_network as *mut *mut ie_executable_network_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }

            let mut infer_request: *mut ie_infer_request_t = mem::zeroed();
            let status = ie_exec_network_create_infer_request(executable_network, &mut infer_request as *mut *mut ie_infer_request_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }

            let status = ie_infer_request_infer(infer_request);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }

            let mut output_blob: *mut ie_blob_t = mem::zeroed();
            let output_name = CString::new("prob").unwrap();
            let output_name_ptr: *const c_char = output_name.as_ptr();
 
            let status = ie_infer_request_get_blob(infer_request, output_name_ptr, &mut output_blob as *mut *mut ie_blob_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }



        }
    }
}