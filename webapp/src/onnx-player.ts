/**
 * onnx-player.ts — onnxruntime-web wrapper
 * Loads the exported TwixNet ONNX model and provides async eval().
 */

import * as ort from 'onnxruntime-web';
import type { Game } from './twixt.js';
import { toInputArrays, threeToOne } from './naf.js';
import { SIZE } from './twixt.js';

// Point onnxruntime-web at its own .wasm files.
// Vite copies these from node_modules into dist/ via the optimizeDeps exclude.
ort.env.wasm.wasmPaths = '/';

export class OnnxPlayer {
  private session: ort.InferenceSession | null = null;

  /**
   * Load the ONNX model.  Call once before the first eval().
   * Tries WebGPU first (fast, GPU-accelerated), falls back to WASM.
   */
  async load(modelUrl: string): Promise<void> {
    // Try WebGPU (Chrome 113+, Android Chrome, newer Safari)
    try {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu'],
      });
      console.log('[OnnxPlayer] Using WebGPU backend');
      return;
    } catch (_e) {
      // WebGPU not available
    }

    // Fall back to multi-threaded WASM
    try {
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm'],
      });
      console.log('[OnnxPlayer] Using WASM backend');
    } catch (e) {
      throw new Error(`Failed to load ONNX model from ${modelUrl}: ${e}`);
    }
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
