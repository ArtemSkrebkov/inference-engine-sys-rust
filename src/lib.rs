#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

extern crate image;
extern crate ndarray;

#[cfg(test)]

mod tests {
    use super::*;
    use std::mem;
    use std::ffi::{CString};
    use std::os::raw::c_char;
    use std::os::raw::c_void;
    use std::ptr;
    use rulinalg::utils;
    use ndarray_image;
    use ndarray::prelude::*;

    #[test]
    fn inference_of_network_gives_correct_result() {
        let config_file = CString::new("").unwrap();
        let config_file_ptr: *const c_char = config_file.as_ptr();
        let input_image = ndarray_image::open_image("test_data/cat.png", ndarray_image::Colors::Bgr)
                                .unwrap();//.into_shape((3, 224, 224)).unwrap();
        let dims = input_image.dim();
        let width = dims.0;
        let height = dims.1;
        let channels = dims.2;
        let len = width * height * channels;
                
        let mut input_image_f32 = Array3::<f32>::zeros((channels, height, width).f());
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    input_image_f32[[c, h, w]] = input_image[[h, w, c]] as f32;
                }
            }
        }

        let mut raw_pixels_f32: Vec<f32> = Vec::with_capacity(len);
        raw_pixels_f32.resize(len, 0.0);
        for c in 0..channels {
            for h in 0..height {
                for w in 0..width {
                    raw_pixels_f32[c * width * height + h * width + w] = input_image_f32[[c, h, w]];
                }
            }
        }

        unsafe {
            let mut core: *mut ie_core_t = mem::zeroed();
            let status = ie_core_create(config_file_ptr, &mut core as *mut *mut ie_core_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }
           
            let input_model = CString::new("test_data/resnet-50.xml").unwrap();
            let input_model_ptr: *const c_char = input_model.as_ptr();
            let input_weights = CString::new("test_data/resnet-50.bin").unwrap();
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

            let dims: dimensions = dimensions_t{
                ranks: 4,
                dims: [1, channels as u64, height as u64, width as u64,0,0,0,0] 
            };
            let tensor_desc: tensor_desc_t = tensor_desc_t{
                layout: layout_e_NHWC,
                dims: dims,
                precision: precision_e_FP32};
            let size:u64 = (channels * height * width * 4)as u64;
            let mut input: *mut ie_blob_t = mem::zeroed();
            let data = raw_pixels_f32.as_ptr() as *mut c_void;
            let status = ie_blob_make_memory_from_preallocated(&tensor_desc as *const tensor_desc_t, data, size, &mut input as *mut *mut ie_blob_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }

            let input_name = CString::new("data").unwrap();
            let input_name_ptr: *const c_char = input_name.as_ptr();
            let status = ie_infer_request_set_blob(infer_request, input_name_ptr, input);
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

            let mut output_buffer = ie_blob_buffer_t{
                __bindgen_anon_1: ie_blob_buffer__bindgen_ty_1 {
                    buffer: std::ptr::null_mut(),
                }
            };
            let status = ie_blob_get_cbuffer(output_blob, &mut output_buffer as *mut ie_blob_buffer_t);
            match status {
                s if s == (IEStatusCode_GENERAL_ERROR as _) => panic!("GENERAL_ERROR"),
                s if s == (IEStatusCode_UNEXPECTED as _) => panic!("UNEXPECTED"),
                s if s == (IEStatusCode_OK as _) => {},
                s => panic!("Unknown return value = {}", s),
            }
            let mut v: Vec<f32> = Vec::with_capacity(1000);
            v.set_len(1000);
            ptr::copy(output_buffer.__bindgen_anon_1.cbuffer as *const f32, v.as_mut_ptr(), 1000);
            let c = utils::argmax(&v);
            if c.0 != 283 {
                panic!("Wrong class = {}, {}: {}", c.0, c.1, v[283]);
            }
        }
    }
}