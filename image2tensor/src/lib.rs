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

// Take the image located at 'path', open it, resize it to `height` x `width`, and then convert the
// pixel precision to the type requested. NOTE: this function assumes the image is in standard 8-bit
// RGB format. It should work with standard image formats such as `.jpg`, `.png`, etc.
pub fn convert_image_to_tensor_bytes(
    path: &str,
    width: u32,
    height: u32,
    precision: TensorType,
    order: ColorOrder,
) -> Result<Vec<u8>, String> {
    // Open the file and create the reader.
    let raw_file = Reader::open(path)
        .map_err(|_| format!("Failed to open the file: {:?}", path))
        .unwrap();

    // Create the DynamicImage by decoding the image.
    let decoded = raw_file
        .decode()
        .map_err(|_| format!("Failed to decode the file: {:?}", path))
        .unwrap();

    convert_dynamic_image_to_tensor_bytes(decoded, width, height, precision, order)
}

/// Same as [convert_image_to_bytes] but accepts a `bytes` slice instead.
pub fn convert_image_bytes_to_tensor_bytes(
    bytes: &[u8],
    width: u32,
    height: u32,
    precision: TensorType,
    order: ColorOrder,
) -> Result<Vec<u8>, String> {
    // Create the DynamicImage by decoding the image.
    let decoded = image::load_from_memory(bytes).expect("Unable to load image from bytes.");

    convert_dynamic_image_to_tensor_bytes(decoded, width, height, precision, order)
}

fn convert_dynamic_image_to_tensor_bytes(
    image: DynamicImage,
    width: u32,
    height: u32,
    precision: TensorType,
    order: ColorOrder,
) -> Result<Vec<u8>, String> {
    // Resize the image to the specified W/H and get an array of u8 RGB values.
    let dyn_img: DynamicImage = image.resize_exact(width, height, image::imageops::Triangle);
    let mut img_bytes = dyn_img.into_bytes();

    // Get an array of the pixel values and return it.
    match order {
        ColorOrder::RGB => Ok(save_bytes(&img_bytes, precision)),
        ColorOrder::BGR => Ok(save_bytes(rgb_to_bgr(&mut img_bytes), precision)),
    }
}

/// Calculate the expected tensor data size of an image of `width` x `height` with the given
/// `precision`.
pub fn calculate_buffer_size(width: u32, height: u32, precision: TensorType) -> usize {
    let bytes_per_pixel = get_bytes_per_pixel(precision);
    let pixels: u32 = width * height * 3;
    pixels as usize * bytes_per_pixel
}

/// Save the bytes into the specified TensorType format.
fn save_bytes(buffer: &[u8], tt: TensorType) -> Vec<u8> {
    let mut out: Vec<u8> = vec![];

    for &byte in buffer {
        // Split out the bytes based on the TensorType.
        let ne_bytes = match tt {
            TensorType::F16 => todo!("unable to convert to f16 yet"),
            TensorType::F32 => (byte as f32).to_ne_bytes().to_vec(),
            TensorType::U8 => (byte).to_ne_bytes().to_vec(),
            TensorType::I32 => (byte as i32).to_ne_bytes().to_vec(),
        };

        for byte in ne_bytes {
            out.push(byte);
        }
    }
    out
}

fn get_bytes_per_pixel(precision: TensorType) -> usize {
    match precision {
        TensorType::F32 | TensorType::I32 => 4,
        TensorType::F16 => 4, // Currently Rust doesn't support F16 natively, so we use f32.
        TensorType::U8 => 1,
    }
}

/// Converts an RGB array to BGR.
fn rgb_to_bgr(buffer: &mut [u8]) -> &[u8] {
    for i in (0..buffer.len()).step_by(3) {
        buffer.swap(i + 2, i);
    }
    buffer
}
