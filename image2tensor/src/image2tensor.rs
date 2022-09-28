use image::{io::Reader, Bgr, DynamicImage, ImageBuffer, Rgb};

#[derive(Debug, Copy, Clone)]
pub enum TensorType {
    F16,
    F32,
    U8,
    I32,
}

#[derive(Debug, Copy, Clone)]
pub enum ColorOrder {
    RGB,
    BGR,
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to the type requested.
pub fn convert_image(
    path: &String,
    width: u32,
    height: u32,
    precision: TensorType,
    order: ColorOrder,
) -> Vec<u8> {
    let raw_file = Reader::open(path);
    let raw_file = match raw_file {
        Ok(file) => file,
        Err(e) => {
            println!("Failed to open the file: {:?}", e);
            return vec![];
        }
    };

    let decoded = raw_file.decode();
    let decoded = match decoded {
        Ok(pixels) => pixels,
        Err(e) => {
            println!("Failed to decode the image: {:?}", e);
            return vec![];
        }
    };

    let dyn_img: DynamicImage = decoded.resize_exact(width, height, image::imageops::Triangle);
    let rgb_bind: ImageBuffer<Rgb<u8>, Vec<u8>>;
    let bgr_bind: ImageBuffer<Bgr<u8>, Vec<u8>>;

    // Get an array of the pixel values
    let raw_u8_arr = match order {
        ColorOrder::RGB => {
            rgb_bind = dyn_img.to_rgb8();
            rgb_bind.as_raw()
        }
        ColorOrder::BGR => {
            bgr_bind = dyn_img.to_bgr8();
            bgr_bind.as_raw()
        }
    };

    return save_bytes(raw_u8_arr, precision);
}

pub fn calculate_buffer_size(width: u32, height: u32, precision: TensorType) -> usize {
    let bytes_per_pixel = get_bytes_per_pixel(precision);
    let pixels: u32 = width * height * 3;
    return pixels as usize * bytes_per_pixel;
}

fn save_bytes(arr: &[u8], tt: TensorType) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];

    for i in 0..arr.len() {
        // Split out the bytes based on the TensorType.
        let bytes: Vec<u8>;
        bytes = match tt {
            TensorType::F16 => (arr[i] as f32).to_ne_bytes().to_vec(),
            TensorType::F32 => (arr[i] as f32).to_ne_bytes().to_vec(),
            TensorType::U8 => (arr[i] as u8).to_ne_bytes().to_vec(),
            TensorType::I32 => (arr[i] as i32).to_ne_bytes().to_vec(),
        };

        for j in 0..bytes.len() {
            out.push(bytes[j]);
        }
    }

    return out;
}

fn get_bytes_per_pixel(precision: TensorType) -> usize {
    let bytes_per_pixel = match precision {
        TensorType::F32 | TensorType::I32 => 4,
        TensorType::F16 => 2,
        _ => 1,
    };

    return bytes_per_pixel;
}
