use image::{DynamicImage};
use image::io::Reader;
use std::{convert::TryInto, str::FromStr};
use std::fs;
use wasi_nn;
mod imagenet_classes;
use std::env;
// #[cfg(feature = "performance")]
use std::time::{Duration, Instant};

pub fn main() {
    let loop_size: u32 = env!("LOOP_SIZE").parse().unwrap();
    let tensor_desc_data = fs::read_to_string("fixture/tensor.desc").unwrap();
    let tensor_desc = TensorDescription::from_str(&tensor_desc_data).unwrap();
    match env!("BACKEND") {
        "openvino" => {
            println!("##################################################################\n");
            println!("Running benchmark using OpenVINO, {} model, and looping for {} times...\n", env!("MODEL"), loop_size);
            println!("##################################################################");
            execute(wasi_nn::GRAPH_ENCODING_OPENVINO, &tensor_desc.dimensions(), vec![0f32; 1001], loop_size);
        },
        "tensorflow" => {
            let tf_dim = [tensor_desc.dimensions()[0], tensor_desc.dimensions()[2], tensor_desc.dimensions()[3], tensor_desc.dimensions()[1]];
            println!("#####################################################################\n");
            println!("Running benchmark using Tensorflow, {} model, and looping for {} times...\n", env!("MODEL"), loop_size);
            println!("#####################################################################");
            execute(wasi_nn::GRAPH_ENCODING_TENSORFLOW, &tf_dim, vec![0f32; 1000], loop_size);
        },
        _ => {
            println!("Unknown backend, exiting...");
            return();
        }
    }
}

fn execute(backend: wasi_nn::GraphEncoding, dimensions: &[u32], mut output_buffer: Vec<f32>, loop_size: u32) {
    println!("** Using the {} backend **", backend);

    // #[cfg(feature = "performance")]
    let init_time = Instant::now();
    // #[cfg(feature = "performance")]
    let mut id_secs: Duration;

    let gba: Vec<Vec<u8>> = create_gba(backend);

    let graph = unsafe {
        wasi_nn::load(
            &[&gba[0], &gba[1]],
            backend,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    println!("Loaded graph into wasi-nn with ID: {}", graph);
    let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };

    // #[cfg(feature = "performance")]
    let init_secs = init_time.elapsed();

    println!("Created wasi-nn execution context with ID: {}", context);
    #[cfg(feature = "performance")]
    printtime("Initiating the backend", init_secs.as_micros());

    // Load a tensor that precisely matches the graph input tensor (see
    for i in 0..5 {
        let filename: String = format!("{}{}{}", "fixture/images/", i, ".jpg");

        // let tensor_data: Vec<u8> = image_to_tensor(filename, 224, 224, backend);
        let tensor_data: Vec<u8> = image_to_tensor(filename, dimensions[2], dimensions[3], backend);


        let tensor = wasi_nn::Tensor {
                        dimensions: dimensions,
                        r#type: wasi_nn::TENSOR_TYPE_F32,
                        data: &tensor_data,
                    };

        unsafe {
            wasi_nn::set_input(context, 0, tensor).unwrap();
        }

        // #[cfg(feature = "performance")]
        let mut totaltime = 0;
        for j in 0..loop_size {
            // #[cfg(feature = "performance")]
            let id_time = Instant::now();
            // Execute the inference.
            unsafe {
                wasi_nn::compute(context).unwrap();
            }
            id_secs = id_time.elapsed();
            totaltime += id_secs.as_micros();

            #[cfg(feature = "performance")]
            if j == 1 {
                printtime("First run took", totaltime);
            }

            if j == loop_size - 1 {
                #[cfg(feature = "performance")]
                let msg = format!("{}{}", loop_size, " runs took");
                #[cfg(feature = "performance")]
                printtime(&msg,  totaltime);

                unsafe {
                    wasi_nn::get_output(
                        context,
                        0,
                        &mut output_buffer[..] as *mut [f32] as *mut u8,
                        (output_buffer.len() * 4).try_into().unwrap(),
                    )
                    .unwrap();
                }

                let results = sort_results(&output_buffer, backend);
                println!(
                    "Found results, sorted top 5: {:?}",
                    &results[..5]
                );


                for i in 0..5 {
                    println!("{}.) {}", i + 1, imagenet_classes::IMAGENET_CLASSES[results[i].0]);
                }
            }
        }
    }
}

fn create_gba (backend: u8) -> Vec<Vec<u8>> {
    let result: Vec<Vec<u8>> = match backend {
        wasi_nn::GRAPH_ENCODING_OPENVINO => {
            let xml = fs::read_to_string("fixture/model.xml").unwrap();
            println!("Read graph XML, first 50 characters: {}", &xml[..50]);
            let weights = fs::read("fixture/model.bin").unwrap();
            println!("Read graph weights, size in bytes: {}", weights.len());

            Vec::from([xml.into_bytes(), weights])
        },
        wasi_nn::GRAPH_ENCODING_TENSORFLOW => {
            let filename: String = "saved_model.pb".to_string();
            let model_path: String = env!("MAPDIR").to_string();
            Vec::from([model_path.into_bytes(), filename.into_bytes()])
        },
        _ => {
            println!("Unknown backend {}", backend);
            vec![]
        }

    };
    return result;
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32], backend: u8) -> Vec<InferenceResult> {
    let skipval = match backend {
        wasi_nn::GRAPH_ENCODING_OPENVINO => { 1 },
        _ => { 0 }
    };

    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .skip(skipval)
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}



fn image_to_tensor(path: String, height: u32, width: u32, backend: u8) -> Vec<u8> {
    println!("BJONES h/w = {}/{}", height, width);
    let result: Vec<u8> = match backend {
        wasi_nn::GRAPH_ENCODING_OPENVINO => {
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

            u8_f32_arr
        },
        wasi_nn::GRAPH_ENCODING_TENSORFLOW => {
            let pixels = Reader::open(path).unwrap().decode().unwrap();
            let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
            let bgr_img = dyn_img.to_rgb8();
            // Get an array of the pixel values
            let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
            // Create an array to hold the f32 value of those pixels
            let mut u8_f32_arr:Vec<u8> = vec![0; raw_u8_arr.len()];

            for i in 0..raw_u8_arr.len() {
                u8_f32_arr[i] = raw_u8_arr[i];
            }

            u8_f32_arr
        },
        _ => {
            println!("Unknown backend {}", backend);
            vec![]
        }
    };
    return result;
}

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);

#[cfg(feature = "performance")]
fn printtime (msg: &str, dur: u128) {
    let bdrlen = (msg.len() + 20) as u16;

    for _i in 0..bdrlen {
        print!("-");
    }

    let durstr = dur.to_string();
    let charnum = durstr.chars().count();

    println!("");
    println!("** {} took {}.{} ms **", msg, &durstr[0..charnum-3], &durstr[(charnum-3)..charnum]);

    for _i in 0..bdrlen {
        print!("-");
    }
    println!("");
}

/// This structure makes it possible to use runtime-defined tensor dimensions by reading the
/// tensor description from a string, e.g. `TensorDescription::from_str("u8x1x3x300x300")`.
struct TensorDescription(wasi_nn::TensorType, Vec<u32>);

impl TensorDescription {
    pub fn precision(&self) -> wasi_nn::TensorType {
        self.0
    }
    pub fn dimensions<'a>(&'a self) -> &'a [u32] {
        &self.1
    }
}

impl FromStr for TensorDescription {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.trim().split('x').collect();
        if parts.len() < 2 {
            return Err(
                "Not enough parts in description string; should be [precision]x[dimension]*",
            );
        }
        let precision = match parts[0] {
            "u8" => wasi_nn::TENSOR_TYPE_U8,
            "i32" => wasi_nn::TENSOR_TYPE_I32,
            "f16" => wasi_nn::TENSOR_TYPE_F16,
            "f32" => wasi_nn::TENSOR_TYPE_F32,
            _ => return Err("Unknown precision string; should be one of [u8|i32|f16|f32]"),
        };
        let mut dimensions = Vec::new();
        for part in parts.into_iter().skip(1) {
            let dimension = u32::from_str(part).map_err(|_| "Failed to parse dimension as i32")?;
            dimensions.push(dimension);
        }
        Ok(Self(precision, dimensions))
    }
}