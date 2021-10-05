use image::DynamicImage;
use image::io::Reader;

#[derive(Debug, Copy, Clone)]
pub enum TensorType {
    F16,
    F32,
    U8,
    I32
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to the type requested.
pub fn convert_image (
    path: String,
    width: u32,
    height: u32,
    precision: TensorType,
    out_buff: &mut [u8]
    ) -> i32 {
    let raw_file = Reader::open(path);
    let raw_file = match raw_file {
        Ok(file) => file,
        Err(e) => {
            println!("Failed to open the file: {:?}", e);
            return -1;
        },
    };

    let decoded = raw_file.decode(); //.unwrap();
    let decoded = match decoded {
        Ok(pixels) => pixels,
        Err(e) => {
            println!("Failed to decode the image: {:?}", e);
            return -1;
        },
    };

    let dyn_img: DynamicImage = decoded.resize_exact(width, height, image::imageops::Triangle);
    // Switch from RGB to BGR
    let bgr_img = dyn_img.to_bgr8();

    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    let bytes_per_pixel: usize = get_bytes_per_pixel(precision);
    let total_bytes: i32 = (raw_u8_arr.len() * bytes_per_pixel) as i32;

    // Create an array to hold the f32 value of those pixels
    for i in 0..raw_u8_arr.len()  {

        // Read the number as a f32 and break it into u8 bytes
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let u8_bytes = u8_f32.to_ne_bytes();

        for j in 0..bytes_per_pixel {
            out_buff[(i * bytes_per_pixel) + j] = u8_bytes[j];
        }

    }

    return total_bytes;
}

pub fn calculate_buffer_size (
    width: u32,
    height: u32,
    precision: TensorType) -> usize {

    let bytes_per_pixel = get_bytes_per_pixel(precision);
    let pixels: u32 = width * height * 3;
    return pixels as usize * bytes_per_pixel;
}

fn get_bytes_per_pixel (precision: TensorType) -> usize {
    let mut bytes_per_pixel: usize = 1;
    match precision {
        TensorType::F32 | TensorType::I32 => bytes_per_pixel = 4,
        TensorType::F16 => bytes_per_pixel = 2,
        _ => bytes_per_pixel = 1
    }
    return bytes_per_pixel;
}