use image2tensor::*;
use std::io::Write;
use std::{convert::TryInto, str::FromStr};
use std::fs;
use wasi_nn;
mod imagenet_classes;
use std::env;
use std::time::{Duration, Instant};
use floating_duration::{TimeAsFloat};
use statistical::{mean, standard_deviation};
use std::path::Path;

pub fn main() {
    let runs: u32 = env!("RUNS").parse().unwrap();
    let tensor_desc_data = fs::read_to_string("fixture/tensor.desc").unwrap();
    let tensor_desc = TensorDescription::from_str(&tensor_desc_data).unwrap();
    let batch_sz:usize = env!("BATCH_SZ").parse().unwrap();
    let mut curr_img:CurrImg = CurrImg{index: 0, max_index: 0, out_img_size: 0, curr_path: None};

    match env!("BACKEND") {
        "openvino" => {
            curr_img.out_img_size = 1001;
            let ov_dim = [batch_sz as u32, tensor_desc.dimensions()[1], tensor_desc.dimensions()[2], tensor_desc.dimensions()[3]];
            println!("##################################################################\n");
            println!("Running {} benchmark using:\nOpenVINO\n{} model\nlooping for {} times...\n",env!("BUILD_TYPE"), env!("MODEL"), runs);
            println!("##################################################################");
            let out_sz:usize = curr_img.out_img_size * batch_sz;

            execute(wasi_nn::GRAPH_ENCODING_OPENVINO, &ov_dim, vec![0f32; out_sz], TensorType::F32, ColorOrder::BGR, runs, curr_img);

        },
        "tensorflow" => {
            // TensorFlow orders the shape slightly different than OpenVINO
            curr_img.out_img_size = 1000;
            let tf_dim = [batch_sz as u32, tensor_desc.dimensions()[2], tensor_desc.dimensions()[3], tensor_desc.dimensions()[1]];
            println!("#####################################################################\n");
            println!("Running {} benchmark using:\nTensorflow\n{} model\nlooping for {} times...\n",env!("BUILD_TYPE"), env!("MODEL"), runs);
            println!("#####################################################################");
            let out_sz:usize = curr_img.out_img_size * batch_sz;
            execute(wasi_nn::GRAPH_ENCODING_TENSORFLOW, &tf_dim, vec![0f32; out_sz], TensorType::F32, ColorOrder::RGB, runs, curr_img);
        },
        _ => {
            println!("Unknown backend, exiting...");
            return();
        }
    }
}

fn execute(backend: wasi_nn::GraphEncoding, dimensions: &[u32], mut output_buffer: Vec<f32>, precision: TensorType, color_order: ColorOrder, runs: u32, mut curr_img: CurrImg) {
    println!("** Using the {:?} backend **", backend);
    println!("Tensor shape / size is {:?}", dimensions);
    let init_time = Instant::now();
    let mut id_secs: Duration;
    let mut all_results: Vec<f64> = vec![];
    let mut final_results: Vec<f64> = vec![];
    let mut finaltime: f64 = 0.0;
    let batch_sz:usize = env!("BATCH_SZ").parse().unwrap();
    let max: u32 = env!("MAX_FILE_NUM").parse().unwrap();
    curr_img.set_max(max);
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

    // Calculate the size of the buffer needed for a single image.
    let mut filename = curr_img.get_next_img_path();
    let td = convert_image_to_bytes(&filename, dimensions[2], dimensions[2], precision, color_order).or_else(|e| {
        Err(e)
    }).unwrap();
    let mut tensor_data_batch: Vec<u8> = vec![0;td.len() * batch_sz];
    let mut totaltime: f64 = 0.0;

        while finished_runs < runs {
            for i in 0..batch_sz {
                let tensor_data = convert_image_to_bytes(&filename, dimensions[2], dimensions[2], precision, color_order).or_else(|e| {
                    Err(e)
                }).unwrap();
                let jump = i * tensor_data.len();
                    for k in 0..tensor_data.len() {
                        tensor_data_batch[k + jump] = tensor_data[k];
                    }
                if batch_sz - i > 1 {
                    filename = curr_img.get_next_img_path();
                }
            }
                let tensor = wasi_nn::Tensor {
                                dimensions: dimensions,
                                type_: wasi_nn::TENSOR_TYPE_F32,
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
            if id_secs.as_fractional_millis() < 100.0 {
                totaltime += id_secs.as_fractional_millis();
                all_results.push(id_secs.as_fractional_millis());
            }



            println!("############################################################");
            println!("Image results run {}", finished_runs);
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

            let results = sort_results(&output_buffer, backend, batch_sz, curr_img.out_img_size);

            for x in 0..batch_sz {
                println!("---------------- Image # {} ---------------------------------", x);
                for i in 0..5 {
                    println!("{}.) {} = ({:?})", i + 1, imagenet_classes::IMAGENET_CLASSES[results[x][i].0], results[0][i]);
                }
            }

            println!("------------------------------------------------------------");
            println!("############################################################");

            finished_runs += 1;

            if finished_runs >= runs {
                final_results.append(&mut all_results);
                all_results.clear();
                finaltime += totaltime;

            }

            filename = curr_img.get_next_img_path();
        }
    print_csv(&final_results, "testout".to_string(), finaltime);
}

fn create_gba (backend: wasi_nn::GraphEncoding) -> Vec<Vec<u8>> {
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
                        "serving_default".to_owned().into_bytes(),
                        "serve".to_owned().into_bytes(),
                        ])
        },
        _ => {
            println!("Unknown backend {:?}", backend);
            vec![]
        }

    };
    return result;
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.

fn sort_results(buffer: &[f32], backend: wasi_nn::GraphEncoding, batch_size: usize, out_img_size: usize) -> Vec<Vec<InferenceResult>> {
    let skipval = match backend {
        wasi_nn::GRAPH_ENCODING_OPENVINO => { 1 },
        _ => { 0 }
    };

    let chunks: Vec<&[f32]> =  buffer.chunks(out_img_size).collect();
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

fn print_csv(all_results: &Vec<f64>, filename: String, totaltime: f64) {
    let runs: u32 = env!("RUNS").parse().unwrap();
    let mut _wr;

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
    let outfile_sum = fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename_sum.clone());

    let outfile_all =fs::OpenOptions::new()
        .write(true)
        .append(true)
        .open(filename_all.clone());

    if outfile_all.is_ok() {
        let mut outfile_all = outfile_all.unwrap();
            let cvs_all_str = String::from("run,time\n");
            _wr = outfile_all.write_all(cvs_all_str.as_bytes());
        for i in 0..all_results.len() {
            _wr = outfile_all.write_all(format!("{},{}\n", i, all_results[i]).as_bytes());
        }
    } else {
        println!("Couldn't save CSV data to {}", filename_all);
    }

    if outfile_sum.is_ok() {
        let mut outfile_sum = outfile_sum.unwrap();
        let cvs_sum_str = String::from("runs,total_time,avg_time,std_dev\n");
        _wr = outfile_sum.write_all(cvs_sum_str.as_bytes());

        _wr = outfile_sum.write_all(format!("{},{},{},{}\n", runs, totaltime, res_mean, res_dev).as_bytes());
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


struct CurrImg {
    index: u32,
    max_index: u32,
    out_img_size: usize,
    curr_path: Option<String>,
}

impl CurrImg {
    fn get_next_index(&mut self) -> u32 {
        if self.index < self.max_index {
            self.index += 1;
        } else {
            self.index = 1;
        }
        return self.index;
    }

    fn get_next_img_path (&mut self) -> String {
        let mut img_path: String = format!("{}{}{}", "fixture/images/", self.index, ".jpg");
        while !Path::new(&img_path).exists() {
            img_path = format!("{}{}{}", "fixture/images/", self.get_next_index(), ".jpg");
        }
        self.curr_path = Some(img_path.clone());
        self.get_next_index();
        return img_path;
    }

    fn set_max(&mut self, new_max: u32) {
        self.max_index = new_max;
    }
}
