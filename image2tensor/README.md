### `image2tensor`

This package provides several helper functions for converting images to tensors. It is designed
primarily for the `wasm32-wasi` target.

### Use

- `calculate_buffer_size(width, height, precision)`: given the width, height, and precision of your
  desired tensor, it will return the number of bytes you need to allocate. This is useful to
  determine the size of the array you need to allocate for the output buffer you pass to
  convert_image.
- `convert_image_to_bytes(path, width, height, precision, order)`: convert the image located at the
  path into a byte array with the requested dimensions and precision. NOTE: This currently only
  works with images that are in standard 8bit RGB color format.

### Build

```console
$ cargo build --target wasm32-wasi
```

### Examples

```rust
use image2tensor;
let width: u32 = 224;
let height: u32 = 224;
let bytes = image2tensor::convert_image_to_bytes("path/to/file", width, height, TensorType::F32, ColorOrder::BGR);
```

### License

This project is licensed under the Apache 2.0 license. See [LICENSE] for more details.

[LICENSE]: LICENSE


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
this project by you, as defined in the Apache-2.0 license, shall be licensed as above, without any
additional terms or conditions.
