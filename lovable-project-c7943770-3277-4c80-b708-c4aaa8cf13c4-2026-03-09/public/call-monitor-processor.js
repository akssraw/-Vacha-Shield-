class CallMonitorProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 1024;
    this.buffer = new Float32Array(this.chunkSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel || channel.length === 0) {
      return true;
    }

    let readOffset = 0;
    while (readOffset < channel.length) {
      const remaining = this.chunkSize - this.offset;
      const copyCount = Math.min(remaining, channel.length - readOffset);
      this.buffer.set(channel.subarray(readOffset, readOffset + copyCount), this.offset);
      this.offset += copyCount;
      readOffset += copyCount;

      if (this.offset === this.chunkSize) {
        this.port.postMessage(this.buffer.slice(0));
        this.offset = 0;
      }
    }

    return true;
  }
}

registerProcessor("call-monitor-processor", CallMonitorProcessor);
