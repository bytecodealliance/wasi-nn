/**
 * Export the wasi-nn APIs with an ergonomic, AssemblyScript-compatible binding. We expect users to
 * use these bindings, not the raw `wasi_ephemeral_nn.ts` bindings.
 *
 * The equivalent Rust bindings (i.e. high-level, ergonomic) for wasi-nn currently live as an
 * example in the Wasmtime repository--they should be moved to their own repository at some point.
 * For comparison, those Rust bindings look like in part (see
 * https://github.com/bytecodealliance/wasmtime/blob/main/crates/wasi-nn/examples/wasi-nn-rust-bindings/src/generated.rs#L57-L61):
 *
 *   pub unsafe fn load(builder: GraphBuilderArray, encoding: GraphEncoding, target:
 *     ExecutionTarget) -> Result<Graph>
 *
 * Our AssemblyScript bindings should expose this level of functionality until interface types or
 * WITX tooling make this project irrelevant.
 */
import * as wasi_ephemeral_nn from './wasi_ephemeral_nn';

/**
 * A machine learning model.
 */
export class Graph {
    private constructor(private pointer: i32) { }

    /**
     * Create a `Graph` from one or more binary blobs.
     * @param builder the binary blobs that make up the graph
     * @param encoding the framework required for
     * @param target the device on which to run the graph
     * @returns an initialized `Graph`
     */
    static load(builder: u8[][], encoding: GraphEncoding, target: ExecutionTarget): Graph {
        let graphBuilder: u32[] = [];

        for (let i = 0; i < builder.length; i++) {
            graphBuilder.push(getArrayPtr(builder[i]));
            graphBuilder.push(builder[i].length);
        }

        let graphPointer: u32 = changetype<u32>(memory.data(4));
        let resultCode = wasi_ephemeral_nn.load(getArrayPtr(graphBuilder), builder.length, encoding, target, graphPointer);
        if (resultCode != 0) {
            throw new WasiNnError("Unable to load graph", resultCode);
        }
        return new Graph(graphPointer);
    }

    /**
     * Create an execution context for performing inference requests. This indirection separates the
     * "graph loading" phase (potentially expensive) from the "graph execution" phase.
     * @returns an `ExecutionContext`
     */
    initExecutionContext(): ExecutionContext {
        let executionContextPointer: u32 = changetype<u32>(memory.data(4));
        let resultCode = wasi_ephemeral_nn.init_execution_context(load<u32>(this.pointer), executionContextPointer);
        if (resultCode != 0) {
            throw new WasiNnError("Unable to initialize an execution context", resultCode);
        }
        return new ExecutionContext(executionContextPointer);
    }
}

/**
 * The allowed encodings for `Graph`s. This must match what is defined in the specification.
 */
export const enum GraphEncoding {
    openvino = 0,
    onnx = 1,
    tensorflow = 2,
    pytorch = 3,
    tensorflowlite = 4
}

/**
 * The allowed execution targets. These values must match the values defined in the specification.
 */
export const enum ExecutionTarget {
    cpu = 0,
    gpu = 1,
    tpu = 2,
}

/**
 * The context for computing an inference request.
 */
export class ExecutionContext {
    // TODO this should be module-private.
    constructor(private pointer: i32) { }

    /**
     * Set an input parameter prior to calling `compute`. This will fail if the `index` is not a
     * valid graph input.
     * @param index the index of the input parameter to set
     * @param tensor the input tensor data
     */
    setInput(index: u32, tensor: Tensor): void {
        let resultCode = wasi_ephemeral_nn.set_input(load<u32>(this.pointer), index, tensor.asPointer());
        if (resultCode != 0) {
            throw new WasiNnError("Unable to set input tensor", resultCode);
        }
    }

    /**
     * Compute an inference request.
     */
    compute(): void {
        let resultCode = wasi_ephemeral_nn.compute(load<u32>(this.pointer));
        if (resultCode != 0) {
            throw new WasiNnError("Unable to compute inference", resultCode);
        }
    }

    /**
     * Retrieve the result of an inference after calling `compute`. NOTE: If you get a
     * `NotEnoughMemory` error, try upping the size of `outputBuffer`.
     * @param index the index of the output parameter to retrieve
     * @param outputBuffer the output buffer to use as the destination for the output bytes; once we
     * have functions to get the output buffer size, this parameter will become unnecessary and
     * should be removed (TODO).
     * @returns the output tensor
     */
    getOutput(index: u32, outputBuffer: Array<u8>): Tensor {
        let maxBufferLength = outputBuffer.length;
        let bytesWritten: u32 = changetype<u32>(memory.data(4));
        let resultCode = wasi_ephemeral_nn.get_output(load<u32>(this.pointer), index,
            changetype<u32>(getArrayPtr(outputBuffer)),
            maxBufferLength,
            bytesWritten);
        if (resultCode != 0) {
            throw new WasiNnError("Unable to get output tensor", resultCode);
        }
        return new Tensor([maxBufferLength], TensorType.u8, outputBuffer);
    }
}

/**
 * Contains the data passed to inference.
 */
export class Tensor {
    constructor(public dimensions: u32[], public type: TensorType, public data: u8[]) { }

    /**
     * @returns a pointer to a structure describing the tensor; this conforms to the WITX `$tensor`
     * description as well as the ABI expected by Wasmtime.
     * @see https://github.com/WebAssembly/wasi-nn/blob/master/phases/ephemeral/witx/wasi_ephemeral_nn.witx#L56
     * @see https://github.com/bytecodealliance/wasmtime/blob/main/crates/wasi-nn/examples/wasi-nn-rust-bindings/src/generated.rs#L19-L35
    */
    asPointer(): u32 {
        let struct: u32[] = [getArrayPtr(this.dimensions), this.dimensions.length, this.type,
        getArrayPtr(this.data), this.data.length];
        return getArrayPtr(struct);
        // May need to pin `struct` so it is not garbage-collected? (See
        // https://www.assemblyscript.org/garbage-collection.html#garbage-collection). Perhaps not,
        // as long as we use the stub runtime (see
        // https://www.assemblyscript.org/garbage-collection.html#stub-runtime).
    }

    /**
     * Convert data to an `ArrayBuffer` for using data views.
     * @returns an ArrayBuffer with a copy of the bytes in `this.data`
     */
    toArrayBuffer(): ArrayBuffer {
        const buffer = new ArrayBuffer(this.data.length);
        // TODO figure out why we cannot use `forEach` here.
        for (let i = 0; i < this.data.length; i++) {
            store<u8>(changetype<u32>(buffer) + i, this.data[i])
        }
        return buffer;
    }
}

/**
 * The type of data contained within a `Tensor`.
 */
export const enum TensorType {
    f16 = 0,
    f32 = 1,
    u8 = 2,
    i32 = 3
}

/**
 * A wasi-nn failure.
 */
export class WasiNnError extends Error {
    constructor(message: string = "", code: i32 = -1) {
        super(message + "; error code = " + code);
        this.name = "WasiNnError";
    }
}

/**
 * Helper function to capture the pointer to the beginning of an array.
 * @param data an array
 * @returns a pointer to the array data
 */
// @ts-ignore: decorator
@inline
function getArrayPtr<T>(data: T[]): u32 {
    // Use the documented `dataStart` field; see
    // https://www.assemblyscript.org/memory.html#arraybufferview-layout. We cast to a `u32` to
    // match the wasi-nn expected types.
    return changetype<u32>(data.dataStart);
}
