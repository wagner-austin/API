/**
 * Vitest setup file for jsdom environment.
 *
 * Provides minimal stubs for Web APIs not included in jsdom.
 */

// Polyfill Blob.arrayBuffer() if not available in jsdom
if (typeof Blob.prototype.arrayBuffer !== "function") {
  Blob.prototype.arrayBuffer = function (): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (): void => {
        const result = reader.result;
        if (result instanceof ArrayBuffer) {
          resolve(result);
        } else {
          reject(new Error("Failed to read blob as ArrayBuffer"));
        }
      };
      reader.onerror = (): void => {
        reject(reader.error);
      };
      reader.readAsArrayBuffer(this);
    });
  };
}

// Minimal AudioContext stub for audio visualization tests
class FakeAnalyserNode {
  fftSize = 256;
  frequencyBinCount = 128;

  getByteFrequencyData(array: Uint8Array): void {
    // Fill with fake data (50% level)
    for (let i = 0; i < array.length; i++) {
      array[i] = 64;
    }
  }
}

class FakeAudioContext {
  createMediaStreamSource(_stream: MediaStream): { connect: (node: FakeAnalyserNode) => void } {
    return {
      connect: (_node: FakeAnalyserNode): void => {
        // No-op
      },
    };
  }

  createAnalyser(): FakeAnalyserNode {
    return new FakeAnalyserNode();
  }

  close(): Promise<void> {
    return Promise.resolve();
  }
}

// Add to global scope
globalThis.AudioContext = FakeAudioContext as unknown as typeof AudioContext;
