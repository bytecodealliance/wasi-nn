import { FileSystem, Console, Process } from "as-wasi";
import * as wasi_nn from "../assembly/as-wasi-nn";
import { IMAGENET_CLASSES } from "../assembly/imagenet_classes";

/**
 * Demonstrate running a ML classification using the wasi-nn API.
 * @returns an exit code; 0 if successful
 */
export function main(): i32 {
    Console.log("Loading graph...");
    const graph = wasi_nn.Graph.load([readBytes("mobilenet.xml"), readBytes("mobilenet.bin")], wasi_nn.GraphEncoding.openvino, wasi_nn.ExecutionTarget.cpu);

    Console.log("Setting up execution context...");
    const context = graph.initExecutionContext();

    for (let i = 0; i < 5; i++) {
        let imgData = wasi_nn.convert_image("build/images/" + i.toString() + ".jpg", 224, 224, wasi_nn.TensorType.f32);
        if (imgData.length > 0) {
            const input = new wasi_nn.Tensor([1, 3, 224, 224], wasi_nn.TensorType.f32, imgData);
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
        } else {
            console.log("Failed to convert the image " + i.toString() + ".jpg, skipping it.");
        }
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
function sortResults(output: wasi_nn.Tensor, topK: u32): Result[] {
    const probabilities = Float32Array.wrap(output.toArrayBuffer()).slice(1);
    const results = new Array<Result>(probabilities.length);
    // TODO figure out why we cannot use `map` here.
    for (let i = 0; i < probabilities.length; i++) {
        results[i] = new Result(i, probabilities[i]);
    }
    results.sort((a, b) => a.probability > b.probability ? -1 : 1);
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
