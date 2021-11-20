use image::{DynamicImage};
use image::io::Reader;
use std::convert::TryInto;
use std::fs;
use wasi_nn;
mod imagenet_classes;
use std::path::Path;
use std::str;

pub fn main() {
    // let xml = fs::read_to_string("fixture/mobilenet.xml").unwrap();
    // println!("Read graph XML, first 50 characters: {}", &xml[..50]);

    // let weights = fs::read("fixture/mobilenet.bin").unwrap();
    // println!("Read graph weights, size in bytes: {}", weights.len());

    let filename: String = "saved_model.pb".to_string();
    let modelPath: String = "fixture".to_string(); //Path::new("src").to_;
    let mypath = Path::new("fixture/saved_model.pb");

    println!("BJONES in example, PATH == {}", mypath.display());
    // let modelPath = fs::read("fixture/saved_model.pb").unwrap();
    let graph = unsafe {
        wasi_nn::load(
            // &[&xml.into_bytes(), &weights],
            &[&modelPath.into_bytes(), &filename.into_bytes()],
            wasi_nn::GRAPH_ENCODING_TENSORFLOW,
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
        println!("BJONES OPENING FILE {}", filename);
        let tensor_data = image_to_tensor(filename, 224, 224);
        println!("Read input tensor, size in bytes: {}", tensor_data.len());
        //BJONES TODO dimension order for Tensorflow / Openvino are different.
        let tensor = wasi_nn::Tensor {
            dimensions: &[1, 224, 224, 3],
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
        //BJONES THIS IS 1001 FOR OPENVINO AND 1000 FOR TENSORFLOW
        // let mut output_buffer = vec![0f32; 1001];
        let mut output_buffer = vec![0f32; 1000];
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
        // .skip(1) //BJONES TODO I think this is only needed by openvino?
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

    //BJONES TODO Tensorflow uses rgb8 vs bgr8
    // let bgr_img = dyn_img.to_bgr8();
    let bgr_img = dyn_img.to_rgb8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];

    //BJONES TODO  i'm using an empty vector. Fix this to fill it in like I do for the
    // Tensorflow demo
    let mut u8_f32_arr:Vec<u8> = vec![0; raw_u8_arr.len()];
    let mut f32_arr:Vec<f32> = vec![0.0; raw_u8_arr.len()];
    for i in 0..raw_u8_arr.len() {
        u8_f32_arr[i] = raw_u8_arr[i];
        f32_arr[i] = raw_u8_arr[i] as f32;
    }

    let res: bool = u8_f32_arr[0] as f32 == f32_arr[0];
    println!("RES == {}", res);

    for x in 0..15 {
        println!("u8_f32_arr[{}] = {} vs f32_arr[{}] = {} ", x, u8_f32_arr[x], x, f32_arr[x]);
    }
    // Create an array to hold the f32 value of those pixels

    // BJONES TODO CHANGE THIS FOR OPENVINO
    // let bytes_required = raw_u8_arr.len() * 4;
    // let mut u8_f32_arr:Vec<u8> = vec![0; bytes_required];

    //BJONES TODO AFTER LUNCH
    // The array has to be u8 to be passed. On the tf.rs side I need to convert
    // back to f32 before using the Tensor.
    // for i in 0..raw_u8_arr.len()  {
    //     // Read the number as a f32 and break it into u8 bytes
    //     let u8_f32: f32 = raw_u8_arr[i] as f32;
    //     let u8_bytes = u8_f32.to_ne_bytes();

    //     for j in 0..4 {
    //         u8_f32_arr[(i * 4) + j] = u8_bytes[j];
    //     }
    // }
    return u8_f32_arr;
    // return raw_u8_arr;
}
// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
