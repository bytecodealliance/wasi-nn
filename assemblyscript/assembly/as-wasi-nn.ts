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

function getArrayPtr<T>(data: T): u32 {
    // Only typed arrays have byteOffset. Cast to typed.
    let u8Data = Uint8Array.wrap(data.buffer);
    return (changetype<u32>(u8Data.buffer) + u8Data.byteOffset);
}

export class Graph {
    private constructor(private pointer: i32) { }

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

    initExecutionContext(): ExecutionContext {
        let executionContextPointer: u32 = changetype<u32>(memory.data(4));
        let resultCode = wasi_ephemeral_nn.init_execution_context(load<u32>(this.pointer), executionContextPointer);
        if (resultCode != 0) {
            throw new WasiNnError("Unable to initialize an execution context", resultCode);
        }
        return new ExecutionContext(executionContextPointer);
    }
}

export const enum GraphEncoding {
    openvino = 0,
}

export const enum ExecutionTarget {
    cpu = 0,
    gpu = 1,
    tpu = 2,
}

export class ExecutionContext {
    // TODO this should be module-private.
    constructor(private pointer: i32) { }

    setInput(index: u32, tensor: Tensor): void {
        let resultCode = wasi_ephemeral_nn.set_input(load<u32>(this.pointer), index, tensor.asPointer());
        if (resultCode != 0) {
            throw new WasiNnError("Unable to set input tensor", resultCode);
        }
    }

    compute(): void {
        let resultCode = wasi_ephemeral_nn.compute(load<u32>(this.pointer));
        if (resultCode != 0) {
            throw new WasiNnError("Unable to compute inference", resultCode);
        }
    }

    // TODO once we have functions to get the output buffer size, we shouldn't need the user to pass in the buffer.
    // NOTE: If you get a NotEnoughMemory error, try upping the size of outputBuffer
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

export const enum TensorType {
    f16 = 0,
    f32 = 1,
    u8 = 2,
    i32 = 3
}

export class WasiNnError extends Error {
    constructor(message: string = "", code: i32 = -1) {
        super(message + "; error code = " + code);
        this.name = "WasiNnError";
    }
}
