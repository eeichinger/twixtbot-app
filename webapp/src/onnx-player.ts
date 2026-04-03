/**
 * onnx-player.ts — onnxruntime-web wrapper
 * Loads the exported TwixNet ONNX model and provides async eval().
 */

import * as ort from 'onnxruntime-web';
import type { Game } from './twixt.js';
import { toInputArrays, threeToOne } from './naf.js';
import { SIZE } from './twixt.js';

// Point onnxruntime-web at its own .wasm files (hashed into assets/ by Vite).
ort.env.wasm.wasmPaths = import.meta.env.BASE_URL;

export class OnnxPlayer {
  private session: ort.InferenceSession | null = null;

  /**
   * Load the ONNX model.  Call once before the first eval().
   * Passes both backends to a single create() call so onnxruntime handles the
   * fallback internally.  Using two separate create() calls is wrong: the first
   * call internally initialises the WASM runtime; if that fails (e.g. no
   * SharedArrayBuffer yet), the second call reports "previous initWasm() failed"
   * even though SharedArrayBuffer may now be available.
   */
  async load(modelUrl: string): Promise<void> {
    console.log('[OnnxPlayer] crossOriginIsolated:', typeof crossOriginIsolated !== 'undefined' ? crossOriginIsolated : 'N/A');
    console.log('[OnnxPlayer] SharedArrayBuffer:', typeof SharedArrayBuffer !== 'undefined');
    console.log('[OnnxPlayer] wasmPaths:', ort.env.wasm.wasmPaths);
    console.log('[OnnxPlayer] numThreads:', ort.env.wasm.numThreads);

    // Only include webgpu if the browser actually exposes the WebGPU API.
    // Attempting webgpu on browsers that don't support it (e.g. iOS Safari in a
    // module worker) causes ort to call initWasm() internally and fail, which
    // permanently poisons the WASM init state so the fallback also fails.
    const providers: ort.InferenceSession.ExecutionProviderConfig[] =
      (typeof navigator !== 'undefined' && 'gpu' in navigator)
        ? ['webgpu', 'wasm']
        : ['wasm'];
    console.log('[OnnxPlayer] executionProviders:', providers);

    this.session = await ort.InferenceSession.create(modelUrl, {
      // ort tries backends in order: WebGPU (GPU-accelerated) → WASM (CPU fallback).
      executionProviders: providers,
    }).catch((e) => {
      throw new Error(`Failed to load ONNX model from ${modelUrl}: ${e}`);
    });
    const backend = (this.session as unknown as { handler: { constructor: { name: string } } })
      .handler.constructor.name.toLowerCase().includes('webgpu') ? 'WebGPU' : 'WASM';
    console.log(`[OnnxPlayer] Using ${backend} backend`);
  }

  /**
   * Run one forward pass for the given game position.
   * Returns [score, policyLogits] where score ∈ (-1, 1) and
   * policyLogits is Float32Array[528] of raw un-normalised logits.
   */
  async eval(game: Game): Promise<[number, Float32Array]> {
    if (!this.session) throw new Error('Model not loaded — call load() first');

    const { pegs, links, locs } = toInputArrays(game);

    const feeds: Record<string, ort.Tensor> = {
      pegs:  new ort.Tensor('float32', pegs,  [1, 2, SIZE, SIZE]),
      links: new ort.Tensor('float32', links, [1, 8, SIZE, SIZE]),
      locs:  new ort.Tensor('float32', locs,  [1, 2, SIZE, SIZE]),
    };

    const results = await this.session.run(feeds);

    const policyData = results['policy'].data as Float32Array;  // [1, 528]
    const valueData  = results['value'].data as Float32Array;   // [1, 3]

    // Slice batch dim (always batch=1)
    const policy = policyData.slice(0, 528);
    const value  = valueData.slice(0, 3);

    const score = threeToOne(value as unknown as [number, number, number]);
    return [score, policy];
  }
}
