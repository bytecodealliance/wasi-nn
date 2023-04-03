use image::{io::Reader, DynamicImage};

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
// the pixel precision to the type requested. NOTE: this function assumes the image is in
// standard 8bit RGB format. It should work with standard image formats such as .jpg, .png, etc
pub fn convert_image_to_bytes(
    path: &str,
    width: u32,
    height: u32,
    precision: TensorType,
    order: ColorOrder,
) -> Result<Vec<u8>, String> {
    // Open the file and create the reader.
    let raw_file = Reader::open(path)
        .or_else(|_| Err(format!("Failed to open the file: {:?}", path)))
        .unwrap();

    // Create the DynamicImage by decoding the image.
    let decoded = raw_file
        .decode()
        .or_else(|_| Err(format!("Failed to decode the file: {:?}", path)))
        .unwrap();

    // Resize the image to the specified W/H and get an array of u8 RGB values.
    let dyn_img: DynamicImage = decoded.resize_exact(width, height, image::imageops::Triangle);
    let mut img_bytes = dyn_img.into_bytes();

    // Get an array of the pixel values and return it.
    match order {
        ColorOrder::RGB => Ok(save_bytes(&img_bytes, precision)),
        ColorOrder::BGR => Ok(save_bytes(rgb_to_bgr(&mut img_bytes), precision)),
    }
}

// standard 8bit RGB format. It should work with standard image formats such as .jpg, .png, etc
pub fn convert_image_bytes_to_tensor_bytes(
    bytes: &[u8],
    width: u32,
    height: u32,
    precision: TensorType,
    order: ColorOrder,
) -> Result<Vec<u8>, String> {
    // Create the DynamicImage by decoding the image.
    let decoded = image::load_from_memory(bytes)
        .expect("Unable to load image from bytes.");

    // Resize the image to the specified W/H and get an array of u8 RGB values.
    let dyn_img: DynamicImage = decoded.resize_exact(width, height, image::imageops::Triangle);
    let mut img_bytes = dyn_img.into_bytes();

    // Get an array of the pixel values and return it.
    match order {
        ColorOrder::RGB => Ok(save_bytes(&img_bytes, precision)),
        ColorOrder::BGR => Ok(save_bytes(rgb_to_bgr(&mut img_bytes), precision)),
    }
}

pub fn calculate_buffer_size(width: u32, height: u32, precision: TensorType) -> usize {
    let bytes_per_pixel = get_bytes_per_pixel(precision);
    let pixels: u32 = width * height * 3;
    pixels as usize * bytes_per_pixel
}

// Save the bytes into the specified TensorType format.
fn save_bytes(arr: &[u8], tt: TensorType) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];

    for i in 0..arr.len() {
        // Split out the bytes based on the TensorType.
        let bytes: Vec<u8>;
        bytes = match tt {
            TensorType::F16 => (arr[i] as f32).to_ne_bytes().to_vec(),
            TensorType::F32 => (arr[i] as f32).to_ne_bytes().to_vec(),
            TensorType::U8 => (arr[i]).to_ne_bytes().to_vec(),
            TensorType::I32 => (arr[i] as i32).to_ne_bytes().to_vec(),
        };

        for j in 0..bytes.len() {
            out.push(bytes[j]);
        }
    }
    out
}

fn get_bytes_per_pixel(precision: TensorType) -> usize {
    match precision {
        TensorType::F32 | TensorType::I32 => 4,
        TensorType::F16 => 4,   // Currently Rust doesn't support F16 natively, so we use f32.
        TensorType::U8 => 1,
    }
}

// Converts an RGB array to BGR
fn rgb_to_bgr(arr: &mut [u8]) -> &[u8] {
    for i in (0..arr.len()).step_by(3) {
        let b_bak = arr[i + 2];
        // swap R and B
        arr[i + 2] = arr[i];
        arr[i] = b_bak;
    }
    arr
}
