### Image2Tensor

This package provides several helper functions for converting images to tensors.

### Use

- `calculate_buffer_size(width, height, precision)` : Given the width, height, and precision of your desired tensor, it will return the number of bytes you need to allocate. This is used to determine the size of the array you need to allocate for the output buffer you pass to convert_image.
- `convert_image(path, width, height, precision, out_buff)` : Will convert the image located at the path into a tensor of the requested dimensions and precision. The tensor data will be stored in out_buff. If the conversion fails for any reason, convert_image will return -1. Otherwise it will return the number of bytes written to out_buff.

### Examples
```
use image2tensor;
let width: u32 = 224;
let height: u32 = 224;
let precision = image2tensor::TensorType::F32;
let bufsize = image2tensor::calculate_buffer_size(width, height, precision);
let mut out_buffer = vec![0; bufsize];
let bytes_written: i32 = image2tensor::convert_image(path_std_str, width, height, precision, out_arr);
```

### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
