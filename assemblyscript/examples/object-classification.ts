import { FileSystem, Console, Process } from "as-wasi";
import { Graph, Tensor, TensorType, GraphEncoding, ExecutionTarget } from "../assembly/as-wasi-nn";
import { IMAGENET_CLASSES } from "../assembly/imagenet_classes";

/**
 * Demonstrate running a ML classification using the wasi-nn API.
 * @returns an exit code; 0 if successful
 */
export function main(): i32 {
    Console.log("Loading graph...");
   
    const graph = Graph.load([readBytes("mobilenet.xml"), readBytes("mobilenet.bin")], GraphEncoding.openvino, ExecutionTarget.cpu);

    Console.log("Setting up execution context...");
    const context = graph.initExecutionContext();
    const input = new Tensor([1, 3, 224, 224], TensorType.f32, readBytes("tensor-1x224x224x3-f32.bgr"));
    context.setInput(0, input);

    Console.log("Running classification...");
    context.compute();
    let maxBufferLength = 4004; // Size of our output buffer
    const output = context.getOutput(0, new Array<u8>(4004).fill(0));

    const results = sortResults(output, 5);
    Console.log("Top 5 results: ");
    // TODO figure out why we cannot use `forEach` here.
    for (let i = 0; i < results.length; i++) {
        Console.log((i + 1).toString() + ".) " + IMAGENET_CLASSES[results[i].id] + " : (" + results[i].id.toString() + ", " + results[i].probability.toString() + ")");
    }
    return 0;
}

/**
 * Read the bytes from a file.
 * @param filePath a path to a file
 * @returns all of the bytes read from a file
 */
function readBytes(filePath: string): u8[] {
    const openDescriptor = FileSystem.open(filePath, "r");
    if (openDescriptor === null) {
        throw new Error("Failed to open file: " + filePath);
    }
    const readBytes = openDescriptor.readAll();
    if (readBytes === null) {
        throw new Error("Failed to read bytes from file: " + filePath);
    }
    return readBytes;
}

/**
 * Extract the sorted classification results from an output tensor.
 * @param output the output tensor
 * @param topK the number of results to include (e.g. "top 5 results")
 * @returns an array of results
 */
function sortResults(output: Tensor, topK: u32): Result[] {
    const probabilities = Float32Array.wrap(output.toArrayBuffer()).slice(1);
    const results = new Array<Result>(probabilities.length);
    // TODO figure out why we cannot use `map` here.
    for (let i = 0; i < probabilities.length; i++) {
        results[i] = new Result(i, probabilities[i]);
    }
    results.sort((a: Result, b: Result) => a.probability > b.probability ? -1 : 1);
    return results.slice(0, topK);
}

/**
 * A helper structure for recording the classification ID and probability.
 */
class Result {
    constructor(public id: i32, public probability: f32) { }
}

/**
 * This is a duplicate of wasi_abort from as-wasi (see
 * https://github.com/jedisct1/as-wasi/blob/master/assembly/as-wasi.ts#L1100); that function should
 * be exported in as-wasi's `index` (TODO) to make it accessible using `--use
 * abort=as-wasi/wasi_abort` (see https://www.assemblyscript.org/debugging.html#overriding-abort).
 * @param message 
 * @param fileName 
 * @param lineNumber 
 * @param columnNumber 
 */
export function wasi_abort(
    message: string = "",
    fileName: string = "",
    lineNumber: u32 = 0,
    columnNumber: u32 = 0
): void {
    Console.error(
        fileName + ":" + lineNumber.toString() + ":" + columnNumber.toString() + ": error: " + message
    );
    Process.exit(1);
}

main();
