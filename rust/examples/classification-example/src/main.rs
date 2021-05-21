use image::{DynamicImage};
use image::io::Reader;
use std::convert::TryInto;
use std::fs;
use wasi_nn;
mod imagenet_classes;

pub fn main() {
    let xml = fs::read_to_string("fixture/mobilenet.xml").unwrap();
    println!("Read graph XML, first 50 characters: {}", &xml[..50]);

    let weights = fs::read("fixture/mobilenet.bin").unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let graph = unsafe {
        wasi_nn::load(
            &[&xml.into_bytes(), &weights],
            wasi_nn::GRAPH_ENCODING_OPENVINO,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
    println!("Created wasi-nn execution context with ID: {}", context);

    // Load a tensor that precisely matches the graph input tensor (see
    // `fixture/frozen_inference_graph.xml`).
    for i in 0..5 {
        let filename: String = format!("{}{}{}", "fixture/images/", i, ".jpg");
        let tensor_data = image_to_tensor(filename, 224, 224);
        println!("Read input tensor, size in bytes: {}", tensor_data.len());
        let tensor = wasi_nn::Tensor {
            dimensions: &[1, 3, 224, 224],
            r#type: wasi_nn::TENSOR_TYPE_F32,
            data: &tensor_data,
        };
        unsafe {
            wasi_nn::set_input(context, 0, tensor).unwrap();
        }

        // Execute the inference.
        unsafe {
            wasi_nn::compute(context).unwrap();
        }
        println!("Executed graph inference");

        // Retrieve the output.
        let mut output_buffer = vec![0f32; 1001];
        unsafe {
            wasi_nn::get_output(
                context,
                0,
                &mut output_buffer[..] as *mut [f32] as *mut u8,
                (output_buffer.len() * 4).try_into().unwrap(),
            )
            .unwrap();
        }

        let results = sort_results(&output_buffer);
        println!(
            "Found results, sorted top 5: {:?}",
            &results[..5]
        );

        for i in 0..5 {
            println!("{}.) {}", i + 1, imagenet_classes::IMAGENET_CLASSES[results[i].0]);
        }
    }

}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = Reader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img = dyn_img.to_bgr8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    // Create an array to hold the f32 value of those pixels
    let bytes_required = raw_u8_arr.len() * 4;
    let mut u8_f32_arr:Vec<u8> = vec![0; bytes_required];

    for i in 0..raw_u8_arr.len()  {
        // Read the number as a f32 and break it into u8 bytes
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let u8_bytes = u8_f32.to_ne_bytes();

        for j in 0..4 {
            u8_f32_arr[(i * 4) + j] = u8_bytes[j];
        }
    }
    return u8_f32_arr;
}
// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
