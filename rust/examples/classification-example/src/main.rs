use image2tensor::*;
use std::fs;
use wasi_nn;
mod imagenet_classes;

pub fn main() {
    let xml = fs::read_to_string("fixture/mobilenet.xml").unwrap();
    println!("Read graph XML, first 50 characters: {}", &xml[..50]);

    let weights = fs::read("fixture/mobilenet.bin").unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    // Default is `Openvino` + `CPU`.
    let graph = wasi_nn::GraphBuilder::default()
        .build_from_bytes([xml.into_bytes(), weights])
        .unwrap();
    println!("Loaded graph into wasi-nn with ID: {:?}", graph);

    let mut context = graph.init_execution_context().unwrap();
    println!("Created wasi-nn execution context with ID: {:?}", context);

    // Load a tensor that precisely matches the graph input tensor (see
    // `fixture/frozen_inference_graph.xml`).
    for i in 0..5 {
        let filename: String = format!("{}{}{}", "fixture/images/", i, ".jpg");
        // Convert the image. If it fails just exit
        let tensor_data =
            convert_image_to_tensor_bytes(&filename, 224, 224, TensorType::F32, ColorOrder::BGR)
                .or_else(|e| Err(e))
                .unwrap();

        println!("Read input tensor, size in bytes: {}", tensor_data.len());

        // Set inference input.
        let dimensions = [1, 3, 224, 224];
        context
            .set_input(0, wasi_nn::TensorType::F32, &dimensions, &tensor_data)
            .unwrap();

        // Execute the inference.
        context.compute().unwrap();
        println!("Executed graph inference");

        // Retrieve the output.
        let mut output_buffer = vec![0f32; 1001];
        context.get_output(0, &mut output_buffer).unwrap();

        let results = sort_results(&output_buffer);
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

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
