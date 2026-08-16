/**
 * Demucs audio processor - Core separation logic
 */

import { CONSTANTS } from './constants.js';
import { stft, istft, reflectPad } from './fft.js';

const { SAMPLE_RATE, FFT_SIZE, HOP_SIZE, TRAINING_SAMPLES, MODEL_SPEC_BINS, MODEL_SPEC_FRAMES, SEGMENT_OVERLAP, TRACKS, MODEL_CACHE_NAME } = CONSTANTS;

/**
 * Convert model frequency output to complex spectrogram per track
 */
export function standaloneMask(freqOutput) {
  const numTracks = 4;
  const numChannels = 4;
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const result = [];

  for (let t = 0; t < numTracks; t++) {
    const trackSpec = {
      leftReal: new Float32Array(numBins * numFrames),
      leftImag: new Float32Array(numBins * numFrames),
      rightReal: new Float32Array(numBins * numFrames),
      rightImag: new Float32Array(numBins * numFrames)
    };

    for (let f = 0; f < numFrames; f++) {
      for (let b = 0; b < numBins; b++) {
        const baseIdx = t * numChannels * numBins * numFrames;
        const outIdx = b * numFrames + f;
        trackSpec.leftReal[outIdx] = freqOutput[baseIdx + 0 * numBins * numFrames + b * numFrames + f];
        trackSpec.leftImag[outIdx] = freqOutput[baseIdx + 1 * numBins * numFrames + b * numFrames + f];
        trackSpec.rightReal[outIdx] = freqOutput[baseIdx + 2 * numBins * numFrames + b * numFrames + f];
        trackSpec.rightImag[outIdx] = freqOutput[baseIdx + 3 * numBins * numFrames + b * numFrames + f];
      }
    }
    result.push(trackSpec);
  }

  return result;
}

/**
 * Convert complex spectrogram back to time domain (iSTFT with proper offsets)
 */
export function standaloneIspec(trackSpec, targetLength) {
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const hopLength = HOP_SIZE;
  const paddedBins = numBins + 1;
  const paddedFrames = numFrames + 4;

  const padChannel = (real, imag) => {
    const paddedReal = new Float32Array(paddedFrames * paddedBins);
    const paddedImag = new Float32Array(paddedFrames * paddedBins);

    for (let f = 0; f < numFrames; f++) {
      for (let b = 0; b < numBins; b++) {
        const srcIdx = b * numFrames + f;
        const dstFrame = f + 2;
        const dstIdx = dstFrame * paddedBins + b;
        paddedReal[dstIdx] = real[srcIdx];
        paddedImag[dstIdx] = imag[srcIdx];
      }
    }
    return { real: paddedReal, imag: paddedImag };
  };

  const leftPadded = padChannel(trackSpec.leftReal, trackSpec.leftImag);
  const rightPadded = padChannel(trackSpec.rightReal, trackSpec.rightImag);

  const centerPad = FFT_SIZE / 2;
  const pad = Math.floor(hopLength / 2) * 3;
  const istftLength = (paddedFrames - 1) * hopLength + FFT_SIZE;

  const leftOut = istft(leftPadded.real, leftPadded.imag, paddedFrames, paddedBins, FFT_SIZE, hopLength, istftLength);
  const rightOut = istft(rightPadded.real, rightPadded.imag, paddedFrames, paddedBins, FFT_SIZE, hopLength, istftLength);

  const totalOffset = centerPad + pad;
  const left = leftOut.subarray(totalOffset, totalOffset + targetLength);
  const right = rightOut.subarray(totalOffset, totalOffset + targetLength);

  return { left: new Float32Array(left), right: new Float32Array(right) };
}

/**
 * Prepare model input from stereo audio
 */
export function prepareModelInput(leftChannel, rightChannel) {
  const inputLength = TRAINING_SAMPLES;

  const paddedLeft = new Float32Array(inputLength);
  const paddedRight = new Float32Array(inputLength);
  const copyLen = Math.min(leftChannel.length, inputLength);
  paddedLeft.set(leftChannel.subarray(0, copyLen));
  paddedRight.set(rightChannel.subarray(0, copyLen));

  const le = Math.ceil(inputLength / HOP_SIZE);
  const pad = Math.floor(HOP_SIZE / 2) * 3;
  const padRight = pad + le * HOP_SIZE - inputLength;

  const stftInputLeft = reflectPad(paddedLeft, pad, padRight);
  const stftInputRight = reflectPad(paddedRight, pad, padRight);

  const centerPad = FFT_SIZE / 2;
  const centeredLeft = reflectPad(stftInputLeft, centerPad, centerPad);
  const centeredRight = reflectPad(stftInputRight, centerPad, centerPad);

  const stftLeft = stft(centeredLeft, FFT_SIZE, HOP_SIZE);
  const stftRight = stft(centeredRight, FFT_SIZE, HOP_SIZE);

  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const frameOffset = 2;

  const magSpec = new Float32Array(4 * numBins * numFrames);

  for (let f = 0; f < numFrames; f++) {
    const srcFrame = f + frameOffset;
    for (let b = 0; b < numBins; b++) {
      const srcIdx = srcFrame * stftLeft.numBins + b;
      magSpec[0 * numBins * numFrames + b * numFrames + f] = stftLeft.real[srcIdx];
      magSpec[1 * numBins * numFrames + b * numFrames + f] = stftLeft.imag[srcIdx];
      magSpec[2 * numBins * numFrames + b * numFrames + f] = stftRight.real[srcIdx];
      magSpec[3 * numBins * numFrames + b * numFrames + f] = stftRight.imag[srcIdx];
    }
  }

  const waveform = new Float32Array(2 * inputLength);
  waveform.set(paddedLeft, 0);
  waveform.set(paddedRight, inputLength);

  return { waveform, magSpec, numBins, numFrames, originalLength: leftChannel.length };
}

/**
 * Main Demucs processor class
 */
export class DemucsProcessor {
  constructor(options = {}) {
    this.ort = options.ort || null;
    this.session = null;
    this.modelPath = options.modelPath || './htdemucs_embedded.onnx';
    this.sessionOptions = options.sessionOptions || {};
    this.onProgress = options.onProgress || (() => {});
    this.onLog = options.onLog || (() => {});
    this.onDownloadProgress = options.onDownloadProgress || (() => {});
    this.onCacheHit = options.onCacheHit || (() => {});
  }

  async loadModel(modelPathOrBuffer) {
    if (!this.ort) {
      throw new Error('ONNX Runtime not provided. Pass ort in constructor options.');
    }

    this.onLog('model', 'Loading model...');

    let modelBuffer;
    if (modelPathOrBuffer instanceof ArrayBuffer) {
      modelBuffer = modelPathOrBuffer;
    } else {
      modelBuffer = await this._fetchModel(modelPathOrBuffer || this.modelPath);
    }

    const defaultSessionOptions = {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'basic'
    };

    this.session = await this.ort.InferenceSession.create(modelBuffer, {
      ...defaultSessionOptions,
      ...this.sessionOptions
    });

    this.onLog('model', 'Model loaded successfully');
    return this.session;
  }

  /**
   * 【追加 2026-08-17】モデル取得。Cache API に保存済みならダウンロードを省く。
   * 初回だけ約172MBを取得し、以後は端末内のキャッシュから即座に読む。
   */
  async _fetchModel(url) {
    let cache = null;
    try {
      // file:// や 非セキュアコンテキストでは caches が無い/使えない
      if (typeof caches !== 'undefined') cache = await caches.open(MODEL_CACHE_NAME);
    } catch (e) {
      cache = null;
    }

    if (cache) {
      try {
        const hit = await cache.match(url);
        if (hit) {
          this.onLog('model', 'Model loaded from cache');
          this.onCacheHit();
          return await hit.arrayBuffer();
        }
      } catch (e) {
        this.onLog('model', 'Cache lookup failed (ignored): ' + e.message);
      }
    }

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`分離モデルの取得に失敗しました (HTTP ${response.status})`);
    }

    // 保存用にクローンしてから本体をストリーミングで読む
    // ※ 206 Partial Content は Cache API に保存できない仕様なので 200 のときだけ
    let toCache = null;
    if (cache && response.status === 200) {
      try { toCache = response.clone(); } catch (e) { toCache = null; }
    }

    const buffer = await this._readWithProgress(response);

    if (toCache) {
      try {
        await cache.put(url, toCache);
        this.onLog('model', 'Model cached for next time');
      } catch (e) {
        // 容量不足などで保存できなくても処理自体は続行する
        this.onLog('model', 'Model cache failed (ignored): ' + e.message);
      }
    }

    return buffer;
  }

  /**
   * 【改変 2026-08-17】受信バッファの二重確保をなくす。
   * 以前は chunk を配列に貯めてから別配列へコピーしていたため、172MB のモデルで
   * 一時的に約344MB を占有していた。Content-Length ぶんを先に確保して直接書き込む。
   */
  async _readWithProgress(response) {
    const contentLength = response.headers.get('Content-Length');
    const totalSize = contentLength ? parseInt(contentLength, 10) : 0;
    if (!totalSize || !response.body) {
      // 長さが分からない場合は進捗表示を諦めて一括取得
      return await response.arrayBuffer();
    }

    const reader = response.body.getReader();
    let combined = new Uint8Array(totalSize);
    let offset = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.length > combined.length) {
        // Content-Length と実サイズが食い違う場合（圧縮転送など）の保険
        const grown = new Uint8Array(Math.max(combined.length * 2, offset + value.length));
        grown.set(combined.subarray(0, offset));
        combined = grown;
      }
      combined.set(value, offset);
      offset += value.length;
      this.onDownloadProgress(offset, totalSize);
    }

    return offset === combined.length ? combined.buffer : combined.buffer.slice(0, offset);
  }

  /**
   * ストリーミング分離。
   *
   * 【改変 2026-08-17 / メモリ対策】
   * 以前は曲の全長ぶんの出力バッファ（4トラック×2ch＋重み配列＝4分の曲で約380MB）を
   * 最初に確保していた。これが長い曲や低メモリ端末でブラウザごとクラッシュする主因だった。
   *
   * 重ね合わせの構造上、サンプル位置 p に寄与するのは
   *   start <= p < start + TRAINING_SAMPLES
   * を満たすセグメントだけなので、セグメント start の処理が終わった時点で
   * 「p < start + stride」の範囲は値が確定している。
   * そこで確定ぶんを onChunk() で順に吐き出し、バッファは TRAINING_SAMPLES 分だけを
   * 使い回す（＝曲の長さに関係なくメモリ消費が一定）。
   * 重ね合わせの計算自体は変えていないため、出力波形は従来と完全に同一。
   *
   * @param {Float32Array} leftChannel
   * @param {Float32Array} rightChannel
   * @param {(chunk:{offset:number,length:number,tracks:Object}) => void} [onChunk]
   *        省略時は全部を集めて従来と同じ形（{drums,bass,other,vocals}）で返す互換モード。
   */
  async separate(leftChannel, rightChannel, onChunk) {
    if (!this.session) {
      throw new Error('Model not loaded. Call loadModel() first.');
    }

    // 互換モード: onChunk 無しなら全部集めて返す（メモリは従来同様に必要）
    if (typeof onChunk !== 'function') {
      const whole = {};
      for (const name of TRACKS) {
        whole[name] = {
          left: new Float32Array(leftChannel.length),
          right: new Float32Array(leftChannel.length)
        };
      }
      await this.separate(leftChannel, rightChannel, (c) => {
        for (const name of TRACKS) {
          whole[name].left.set(c.tracks[name].left, c.offset);
          whole[name].right.set(c.tracks[name].right, c.offset);
        }
      });
      return whole;
    }

    const totalSamples = leftChannel.length;
    const stride = Math.floor(TRAINING_SAMPLES * (1 - SEGMENT_OVERLAP));
    const numSegments = Math.ceil((totalSamples - TRAINING_SAMPLES) / stride) + 1;

    // 未確定領域だけを保持するスライディングバッファ（曲長に依存しない固定サイズ）
    const BUF_LEN = TRAINING_SAMPLES;
    const acc = TRACKS.map(() => ({
      left: new Float32Array(BUF_LEN),
      right: new Float32Array(BUF_LEN)
    }));
    const accWeights = new Float32Array(BUF_LEN);
    let bufStart = 0;   // acc[] の先頭が対応する、曲全体でのサンプル位置

    // bufStart 〜 absEnd を確定ぶんとして吐き出し、残りを先頭へ詰め直す
    const flushTo = (absEnd) => {
      const len = absEnd - bufStart;
      if (len <= 0) return;
      const tracks = {};
      for (let t = 0; t < TRACKS.length; t++) {
        const l = new Float32Array(len);
        const r = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          const w = accWeights[i];
          if (w > 0) {
            l[i] = acc[t].left[i] / w;
            r[i] = acc[t].right[i] / w;
          }
        }
        tracks[TRACKS[t]] = { left: l, right: r };
      }
      onChunk({ offset: bufStart, length: len, tracks });
      // 未確定ぶんを先頭へ寄せ、空いた末尾をゼロで埋める
      for (let t = 0; t < TRACKS.length; t++) {
        acc[t].left.copyWithin(0, len);
        acc[t].left.fill(0, BUF_LEN - len);
        acc[t].right.copyWithin(0, len);
        acc[t].right.fill(0, BUF_LEN - len);
      }
      accWeights.copyWithin(0, len);
      accWeights.fill(0, BUF_LEN - len);
      bufStart = absEnd;
    };

    let segmentIdx = 0;

    for (let start = 0; start < totalSamples; start += stride) {
      const end = Math.min(start + TRAINING_SAMPLES, totalSamples);
      const segmentLength = end - start;

      const segLeft = new Float32Array(TRAINING_SAMPLES);
      const segRight = new Float32Array(TRAINING_SAMPLES);

      for (let i = 0; i < segmentLength; i++) {
        segLeft[i] = leftChannel[start + i];
        segRight[i] = rightChannel[start + i];
      }

      const input = prepareModelInput(segLeft, segRight);

      const waveformTensor = new this.ort.Tensor('float32', input.waveform, [1, 2, TRAINING_SAMPLES]);
      const magSpecTensor = new this.ort.Tensor('float32', input.magSpec, [1, 4, MODEL_SPEC_BINS, MODEL_SPEC_FRAMES]);

      const feeds = {};
      feeds[this.session.inputNames[0]] = waveformTensor;
      if (this.session.inputNames.length > 1) {
        feeds[this.session.inputNames[1]] = magSpecTensor;
      }

      const inferResults = await this.session.run(feeds);

      let timeData = null, timeShape = null;
      let freqData = null;

      for (const name of this.session.outputNames) {
        const tensor = inferResults[name];
        if (tensor.dims.length === 4 && tensor.dims[2] === 2) {
          timeData = tensor.data;
          timeShape = tensor.dims;
        } else if (tensor.dims.length === 5 && tensor.dims[2] === 4) {
          freqData = tensor.data;
        }
      }

      if (!timeData) {
        throw new Error('Could not find time-domain output tensor');
      }

      let combinedOutputs = null;
      if (freqData) {
        const trackSpecs = standaloneMask(freqData);
        combinedOutputs = [];

        for (let t = 0; t < 4; t++) {
          const freqOutput = standaloneIspec(trackSpecs[t], TRAINING_SAMPLES);
          const numChannels = timeShape[2];
          const samples = timeShape[3];
          const timeLeft = new Float32Array(samples);
          const timeRight = new Float32Array(samples);

          for (let i = 0; i < samples; i++) {
            timeLeft[i] = timeData[t * numChannels * samples + 0 * samples + i];
            timeRight[i] = timeData[t * numChannels * samples + 1 * samples + i];
          }

          const combined = {
            left: new Float32Array(samples),
            right: new Float32Array(samples)
          };
          for (let i = 0; i < samples; i++) {
            combined.left[i] = timeLeft[i] + (freqOutput.left[i] || 0);
            combined.right[i] = timeRight[i] + (freqOutput.right[i] || 0);
          }
          combinedOutputs.push(combined);
        }
      }

      const numTracks = timeShape[1];
      const numChannels = timeShape[2];
      const samples = timeShape[3];

      const overlapWindow = new Float32Array(segmentLength);
      for (let i = 0; i < segmentLength; i++) {
        const fadeIn = Math.min(i / (stride * 0.5), 1);
        const fadeOut = Math.min((segmentLength - i) / (stride * 0.5), 1);
        overlapWindow[i] = Math.min(fadeIn, fadeOut);
      }

      // スライディングバッファ内での書き込み位置（flush 済みなので通常は 0）
      const base = start - bufStart;
      if (base < 0 || base + segmentLength > BUF_LEN) {
        throw new Error(`内部エラー: 出力バッファ範囲外 (base=${base}, segmentLength=${segmentLength})`);
      }

      for (let t = 0; t < numTracks; t++) {
        for (let i = 0; i < segmentLength; i++) {
          let leftVal, rightVal;
          if (combinedOutputs) {
            leftVal = combinedOutputs[t].left[i];
            rightVal = combinedOutputs[t].right[i];
          } else {
            const leftIdx = t * numChannels * samples + 0 * samples + i;
            const rightIdx = t * numChannels * samples + 1 * samples + i;
            leftVal = timeData[leftIdx];
            rightVal = timeData[rightIdx];
          }
          acc[t].left[base + i] += leftVal * overlapWindow[i];
          acc[t].right[base + i] += rightVal * overlapWindow[i];
        }
      }

      for (let i = 0; i < segmentLength; i++) {
        accWeights[base + i] += overlapWindow[i];
      }

      // 推論結果のテンソルを明示的に解放（特に WebGPU の GPU バッファ）。
      // timeData / freqData を使い終えたこの位置で行う。
      timeData = null; freqData = null; combinedOutputs = null;
      for (const name of this.session.outputNames) {
        const t2 = inferResults[name];
        if (t2 && typeof t2.dispose === 'function') { try { t2.dispose(); } catch (e) { /* 解放失敗は無視 */ } }
      }
      if (typeof waveformTensor.dispose === 'function') { try { waveformTensor.dispose(); } catch (e) {} }
      if (typeof magSpecTensor.dispose === 'function') { try { magSpecTensor.dispose(); } catch (e) {} }

      // 確定した部分を吐き出してメモリを解放する
      flushTo(Math.min(start + stride, totalSamples));

      segmentIdx++;
      this.onProgress({
        progress: segmentIdx / numSegments,
        currentSegment: segmentIdx,
        totalSegments: numSegments
      });
    }

    // 通常は最終セグメントで totalSamples まで flush 済みだが、念のため残りを出す
    flushTo(totalSamples);

    return { totalSamples };
  }
}
