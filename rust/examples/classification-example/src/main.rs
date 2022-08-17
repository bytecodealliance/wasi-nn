use image::{DynamicImage};
use image::io::Reader;
use std::io::Write;
use std::{convert::TryInto, str::FromStr};
use std::fs;
use wasi_nn;
mod imagenet_classes;
use std::env;
use std::time::{Duration, Instant};
use floating_duration::{TimeAsFloat};
use statistical::{mean, standard_deviation};

pub fn main() {
    let runs: u32 = env!("RUNS").parse().unwrap();
    let tensor_desc_data = fs::read_to_string("fixture/tensor.desc").unwrap();
    let tensor_desc = TensorDescription::from_str(&tensor_desc_data).unwrap();
    let batch_sz:usize = env!("BATCH_SZ").parse().unwrap();
    match env!("BACKEND") {
        "openvino" => {
            let ov_dim = [batch_sz as u32, tensor_desc.dimensions()[1], tensor_desc.dimensions()[2], tensor_desc.dimensions()[3]];
            println!("##################################################################\n");
            println!("Running {} benchmark using:\nOpenVINO\n{} model\nlooping for {} times...\n",env!("BUILD_TYPE"), env!("MODEL"), runs);
            println!("##################################################################");
            let out_sz:usize = 1001 * batch_sz;

            execute(wasi_nn::GRAPH_ENCODING_OPENVINO, &ov_dim, vec![0f32; out_sz], runs);

        },
        "tensorflow" => {
            // TensorFlow orders the shape slightly different than OpenVINO
            let tf_dim = [batch_sz as u32, tensor_desc.dimensions()[2], tensor_desc.dimensions()[3], tensor_desc.dimensions()[1]];
            println!("#####################################################################\n");
            println!("Running {} benchmark using:\nTensorflow\n{} model\nlooping for {} times...\n",env!("BUILD_TYPE"), env!("MODEL"), runs);
            println!("#####################################################################");
            let out_sz:usize = 1001 * batch_sz;
            execute(wasi_nn::GRAPH_ENCODING_TENSORFLOW, &tf_dim, vec![0f32; out_sz], runs);
        },
        _ => {
            println!("Unknown backend, exiting...");
            return();
        }
    }
}

fn execute(backend: wasi_nn::GraphEncoding, dimensions: &[u32], mut output_buffer: Vec<f32>, runs: u32) {
    println!("** Using the {} backend **", backend);
    println!("Tensor shape / size is {:?}", dimensions);
    let init_time = Instant::now();
    let mut id_secs: Duration;
    let mut all_results: Vec<f64> = vec![];
    let mut final_results: Vec<f64> = vec![];
    let mut finaltime: f64 = 0.0;
    let batch_sz:usize = env!("BATCH_SZ").parse().unwrap();
    let max_file_num:usize = env!("MAX_FILE_NUM").parse().unwrap();
    let mut curr_img_index = 1;
    let mut finished_runs = 0;
    let mut gba_r: Vec<&[u8]> = vec![];
    let gba = create_gba(backend);

    for i in 0..gba.len() {
        gba_r.push(gba[i].as_slice());
    }

    let graph = unsafe {
        wasi_nn::load(
            &gba_r,
            backend,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    println!("Loaded graph into wasi-nn with ID: {}", graph);
    let context = unsafe { wasi_nn::init_execution_context(graph).unwrap() };
    let init_secs = init_time.elapsed();
    println!("Created wasi-nn execution context with ID: {}", context);
    println!("Initiating the backend took {}ms", init_secs.as_fractional_millis());

    let filen: String = format!("{}{}", "fixture/images/", "1.jpg");
    let td: Vec<u8> = image_to_tensor(filen, dimensions, backend);
    let mut tensor_data_batch: Vec<u8> = vec![0;td.len() * batch_sz];
        let mut totaltime: f64 = 0.0;

        while finished_runs < runs {

            for i in 0..batch_sz {
                let filename: String = format!("{}{}{}", "fixture/images/", curr_img_index, ".jpg");
                let tensor_data: Vec<u8> = image_to_tensor(filename, dimensions, backend);
                let jump = i * tensor_data.len();
                    for k in 0..tensor_data.len() {
                        let testd = tensor_data[k];
                        tensor_data_batch[k + jump] = tensor_data[k];
                    }

                if curr_img_index < max_file_num {
                    curr_img_index += 1;
                } else {
                    curr_img_index = 1;
                }
            }
                let tensor = wasi_nn::Tensor {
                                dimensions: dimensions,
                                r#type: wasi_nn::TENSOR_TYPE_F32,
                                data: &tensor_data_batch,
                            };
                unsafe {
                    wasi_nn::set_input(context, 0, tensor).unwrap();
                }

            let id_time = Instant::now();
            // Execute the inference.
            unsafe {
                wasi_nn::compute(context).unwrap();
            }

            id_secs = id_time.elapsed();
            totaltime += id_secs.as_fractional_millis();
            all_results.push(id_secs.as_fractional_millis());
            finished_runs += 1;

            if finished_runs >= runs {
                println!("############################################################");
                println!("Image results");
                println!("------------------------------------------------------------");
                unsafe {
                    wasi_nn::get_output(
                        context,
                        0,
                        &mut output_buffer[..] as *mut [f32] as *mut u8,
                        (output_buffer.len() * 4).try_into().unwrap(),
                    )
                    .unwrap();
                }

                let results = sort_results(&output_buffer, backend, batch_sz);

                for x in 0..batch_sz {
                    println!("---------------- Image # {} ---------------------------------", x);
                    for i in 0..5 {
                        println!("{}.) {} = ({:?})", i + 1, imagenet_classes::IMAGENET_CLASSES[results[x][i].0], results[0][i]);
                    }
                }

                println!("------------------------------------------------------------");
                println!("############################################################");
                final_results.append(&mut all_results);
                all_results.clear();
                finaltime += totaltime;

            }
        }
    print_csv(&final_results, "testout".to_string(), finaltime);
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
            let model_path: String = env!("MAPDIR").to_string();
            Vec::from([model_path.into_bytes(),
                        "signature,serving_default".to_owned().into_bytes(),
                        "tag,serve".to_owned().into_bytes(),
                        ])
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

fn sort_results(buffer: &[f32], backend: u8, batch_size: usize) -> Vec<Vec<InferenceResult>> {
    let skipval = match backend {
        wasi_nn::GRAPH_ENCODING_OPENVINO => { 1 },
        _ => { 0 }
    };

    let chunks: Vec<&[f32]> =  buffer.chunks(1000).collect();
    let mut ret_vec: Vec<Vec<InferenceResult>> = vec![];

    for i in 0..batch_size {

        let mut results: Vec<InferenceResult> = chunks[i as usize]
            .iter()
            .skip(skipval)
            .enumerate()
            .map(|(c, p)| InferenceResult(c, *p))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ret_vec.push(results);
    }
    ret_vec
}

fn image_to_tensor(path: String, dimensions: &[u32], backend: u8) -> Vec<u8> {
    let result: Vec<u8> = match backend {
        wasi_nn::GRAPH_ENCODING_OPENVINO => {
            let pixels = Reader::open(path).unwrap().decode().unwrap();
            let dyn_img: DynamicImage = pixels.resize_exact(dimensions[2], dimensions[3], image::imageops::Triangle);
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
            let dyn_img: DynamicImage = pixels.resize_exact(dimensions[1], dimensions[2], image::imageops::Triangle);
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

fn print_csv(all_results: &Vec<f64>, filename: String, totaltime: f64) {
    let runs: u32 = env!("RUNS").parse().unwrap();

    if all_results.len() > 5 {
        println!(
                "** First 5 inference times **\n {:?}",
                &all_results[..5]
            );
    } else {
        println!("** Inference times **\n{:?}", all_results);
    }

    println!("\n** Performance results **");
    println!("{} runs took {}ms total",runs, totaltime);
    let res_mean = mean(&all_results);
    let mut res_dev: f64 = 0.0;

    if all_results.len() >1 {
        res_dev = standard_deviation(&all_results, Some(res_mean));
        println!("AVG = {:?}", res_mean);
        println!("STD_DEV = {:?}", res_dev);
    }

    let filename_sum = filename.clone() + ".csv";
    let filename_all = filename.clone() + "_all.csv";
    let mut outfile_sum = fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename_sum.clone());

    let mut outfile_all =fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename_all.clone());

    if outfile_all.is_ok() {
        let mut outfile_all = outfile_all.unwrap();
            let cvs_all_str = String::from("run,time\n");
            outfile_all.write_all(cvs_all_str.as_bytes());
        for i in 0..all_results.len() {
            outfile_all.write_all(format!("{},{}\n", i, all_results[i]).as_bytes());
        }
    } else {
        println!("Couldn't save CSV data to {}", filename_all);
    }

    if outfile_sum.is_ok() {
        let mut outfile_sum = outfile_sum.unwrap();
            let cvs_sum_str = String::from("runs,total_time,avg_time,std_dev\n");
            outfile_sum.write_all(cvs_sum_str.as_bytes());

        outfile_sum.write_all(format!("{},{},{},{}\n", runs, totaltime, res_mean, res_dev).as_bytes());
    } else {
        println!("Couldn't save CSV data to {}", filename_sum);
    }

}
// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);

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
