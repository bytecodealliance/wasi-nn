# **Utilizing wasi-nn from AssemblyScript**
JavaScript is one of the most popular programming languages today, and is used in a wide range of places outside the browser. A lot of what makes it popular is it's low entry barrier and abundance of ready to use libraries, making it easy to get started and make something great. However, due to the nature of being an interpreted language, its not always the most performant option, and often it can be difficult (or impossible) to access powerful low level capabilities. AssemblyScript and The [wasi-nn proposal](https://github.com/WebAssembly/wasi-nn) are geared towards alleviating these shortcomings. AssemblyScipt allows your JavaScript code to be compiled down to WebAssembly, which performs much closer to native speeds. And WASI-NN gives WebAssembly programs access to host-provided machine learning (ML) functions. Effectively this can give us the best of both worlds. The speed and security of native code, along with the simplicity and quick development provided by AssemblyScript.
## **Benefits of AssemblyScript**
---
AssemblyScript is a variant of Typescript, meaning it should be very familiar to anyone who's accustomed to writing software with TypeScript or even plain JavaScript. This means its easy for JavaScript developers to jump in and get started, it also means porting other libraries is often possible as well. This makes AssemblyScript an attractive choice for developers working in the webspace, who want to get into machine learning but don't want to learn an entirely new language like Rust.

## **AssemblyScript vs JavaScript**
---
While AssemblyScript is very similar to JavaScript, it does have much stricter rules in order to compile down to WebAssembly. AssemblyScript intentionally avoids the dynamic nature of JavaScript where it cannot be compiled ahead of time efficiently. This means variables need to have a clearly defined type, and can't change dynamically. Here are a few things to keep in mind when writing your AssemblyScript program.

### **Static Typing**
WebAssembly uses more specific integer and floating point types. Instead of simply using 'number' we have 'u8, u32, f32, u16, etc'. In fact, the standard JavaScript number is just an alias for f64 in AssemblyScript.

### **No any or undefined allowed**
```
// Invalid
function foo(a?) {
var b = a + 1
return b
}

// Valid
function foo(a: i32 = 0): i32 {
var b = a + 1
return b
}
```

### **No union types**
```
// Invalid
function foo(a: i32 | string): void {
}

// Valid
function foo<T>(a: T): void {
}
```
### **Strictly typed objects**
```
// Invalid
var a = {}
a.prop = "hello world"

// Valid
class A {
constructor(public prop: string) {}
}
var a = new A("hello world")
```

## **WASI-NN bindings for AssemblyScript ([as-wasi-nn](https://www.npmjs.com/package/as-wasi-nn))**
---
These bindings provide access to the wasi-nn system calls, as well as some useful objects to help in the creation of machine learning code. These bindings are exposing the same functionality as the Rust bindings written about [here](https://bytecodealliance.org/articles/using-wasi-nn-in-wasmtime). At the heart of it, its the exact same code being run by both the AssemblyScript and Rust bindings. Each simply gives you an option of what language you want to write your program in. And since they both compile down to WebAssembly code, the AssemblyScript version is almost as fast as Rust!

## **Currently available functions and objects**
---
- Graph - Graph data object.
    -  load - Load a model using one or more opaque byte arrays.
    - initExecutionContext - Initialize the execution context.
- ExecutionContext - The context object. Provides the setInput and compute functions as well.
    - setInput - Bind tensors to the context.
    - compute - Computes the machine learning inference using the bound context.
    - getOutput - Retrieve the inference result tensors.
- Tensor - Tensor data object.
    - asPointer - Returns a pointer to the tensor object.
    - toArrayBuffer - Returns an ArrayBuffer with a copy of the bytes in this Tensor.
- GraphEncoding - Enum value for what graph encoder to use. Currently only supports OpenVino.
- ExecutionTarget - Enum value for what processor to use for computation. (cpu, gpu, or tpu).

## **An example**
---
You can find a code example on how to use these bindings in the [git repo](https://github.com/bytecodealliance/wasi-nn/tree/main/assemblyscript/examples). This example uses wasi-nn to identify items in pictures and uses a mobilenet test fixture, found [here.](https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet).

The example essentially consists of 7 steps. Note that readBytes is a helper function used to load files into a array of bytes.
- 1.) Load the graph using Graph.load. In this step we pass in mobilenet.xml and mobilenet.bin as the graph builder array, specify GraphEncoding.openvino as our encoder, and specify we want to use the CPU with ExecutionTarget.cpu.
- 2.) Create the context using Graph.initExecutionContext.
- 3.) Create a Tensor of the desired dimensions, of type TensorType.f32, from the file 0.bgr. Our demo loops five times for images 0.bgr through 4.bgr.
- 4.) Set the tensor as the input of the ExecutionContext using ExecutionContext.setInput.
- 5.) Run ExecutionContext.compute to generate the results.
- 6.) Save the results to a new u8 array using ExecutionContext.getOutput.
- 7.) Sort results to find the most likely objects in the picture, and print the top 5.

### **To run the example**
---
The easiest way to build the demo is to simply run `./build.sh as`, which will handle the steps below for you.

To build it manually, you first need to compile your wasm file using the AssemblyScript compiler, asc.
```
asc examples/object-classification.ts --runtime stub --use abort=examples/object-classification/wasi_abort --target release --enable simd
```

Then run the generated .wasm file with Wasmtime. Note that when we run the example we need to include our build directory so Wasmtime has access to the needed files, and use the experimental-wasi-nn flag to enable use of wasi-nn.
```
wasmtime run build/optimized.wasm --dir build --wasi-modules=experimental-wasi-nn
```
