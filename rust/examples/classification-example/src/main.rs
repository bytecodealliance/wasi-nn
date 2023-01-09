use image2tensor::*;
use std::convert::TryInto;
use std::fs;
use wasi_nn;
mod imagenet_classes;

pub fn main() {
    let xml = fs::read_to_string("fixture/mobilenet.xml").unwrap();
    println!("Read graph XML, first 50 characters: {}", &xml[..50]);

    let weights = fs::read("fixture/mobilenet.bin").unwrap();
    println!("Read graph weights, size in bytes: {}", weights.len());

    let xmlbytes = xml.as_bytes();
    let weightbytes = weights.as_slice();
    let mut builders: Vec<&[u8]> = vec![xmlbytes, weightbytes];

    let my_graph = wasi_nn::WasiNnGraph::load(
        builders.into_iter(),
        wasi_nn::GRAPH_ENCODING_OPENVINO,
        wasi_nn::EXECUTION_TARGET_CPU,
    );
    let graph = my_graph.unwrap();
    let mut context = graph.get_execution_context();

    // TODO: Need to add code to the Wasmtime side to get the input / output tensor shapes
    let intypes = graph.get_input_types();
    let outtypes = graph.get_output_types();

    // Load a tensor that precisely matches the graph input tensor (see
    // `fixture/frozen_inference_graph.xml`).
    for i in 0..5 {
        let filename: String = format!("{}{}{}", "fixture/images/", i, ".jpg");
        // Convert the image. If it fails just exit
        let tensor_data =
            convert_image_to_bytes(&filename, 224, 224, TensorType::F32, ColorOrder::BGR)
                .or_else(|e| Err(e))
                .unwrap();

        println!("Read input tensor, size in bytes: {}", tensor_data.len());

        let tensor = wasi_nn::Tensor {
            dimensions: &[1, 3, 224, 224],
            type_: wasi_nn::TENSOR_TYPE_F32,
            data: &tensor_data,
        };

        context.set_input(0, tensor);

        // Execute the inference and get the output.
        let mut output_buffer = vec![0f32; 1001];
        let _res = context.compute();
        let _wrote = context.get_output(
            0,
            &mut output_buffer[..] as *mut [f32] as *mut u8,
            (output_buffer.len() * 4).try_into().unwrap(),
        );

        println!("Executed graph inference");
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
