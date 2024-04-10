use image2tensor::*;
use std::{convert::TryInto, fs};
use wasi_nn;
mod imagenet_classes;

wit_bindgen::generate!({
    path: "../wit/example.wit",
    world: "classification-world",
});

struct Export;

impl Guest for Export {
    fn run_classification() {
        let xml = fs::read_to_string("build/mobilenet.xml").unwrap();
        println!("Read graph XML, first 50 characters: {}", &xml[..50]);

        let weights = fs::read("build/mobilenet.bin").unwrap();
        println!("Read graph weights, size in bytes: {}", weights.len());

        // Default is `Openvino` + `CPU`.
        let builders = vec![xml.into_bytes(), weights];

        let graph = wasi_nn::graph::load(
            &builders,
            wasi_nn::graph::GraphEncoding::Openvino,
            wasi_nn::graph::ExecutionTarget::Cpu,
        )
        .unwrap();
        println!("Loaded graph into wasi-nn with ID: {:?}", graph);

        let context = graph.init_execution_context().unwrap();
        println!("Created wasi-nn execution context with ID: {:?}", context);

        // Load a tensor that precisely matches the graph input tensor (see
        // `fixture/frozen_inference_graph.xml`).
        for i in 0..5 {
            let filename: String = format!("{}{}{}", "build/images/", i, ".jpg");
            // Convert the image. If it fails just exit
            let tensor_data = convert_image_to_tensor_bytes(
                &filename,
                224,
                224,
                TensorType::F32,
                ColorOrder::BGR,
            )
            .or_else(|e| Err(e))
            .unwrap();

            println!("Read input tensor, size in bytes: {}", tensor_data.len());

            // Set inference input.
            let tensor = wasi_nn::tensor::Tensor::new(
                &vec![1, 3, 224, 224],
                wasi_nn::tensor::TensorType::Fp32,
                &tensor_data,
            );

            context.set_input("0", tensor).unwrap();

            // Execute the inference.
            context.compute().unwrap();
            println!("Executed graph inference");

            // Retrieve the output.
            let output = context.get_output("0").unwrap();
            let results = bytes_to_f32_vec(output.data());
            let results = sort_results(&results);
            println!("Found results, sorted top 5: {:?}", &results[..5]);

            for i in 0..5 {
                println!(
                    "{}.) {}",
                    i + 1,
                    imagenet_classes::IMAGENET_CLASSES[results[i].0]
                );
            }
        }
    }
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        // Skip the background class
        .skip(1)
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks
        .into_iter()
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    v.into_iter().collect()
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);

export!(Export);
