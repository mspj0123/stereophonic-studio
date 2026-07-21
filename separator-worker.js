// -*- 音源分離ワーカー -*-
// 重い処理（モデル読込・STFT/iSTFT・ONNX推論）をこの Web Worker 上で実行し、
// メインスレッド（UI）を塞がないようにする。→「ページが応答しません」を防ぐ。
// コードは MIT: onnxruntime-web / vendor/demucs-web
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.all.mjs';
import { DemucsProcessor, CONSTANTS } from './vendor/demucs-web/index.js';

const { SAMPLE_RATE } = CONSTANTS;

let processor = null;
let loadedEP = null;

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

// 線形補間リサンプル（44100Hz に揃える）— メインではなくここで実行
function resample(ch, fromRate, toRate, targetLen) {
    const ratio = toRate / fromRate;
    const out = new Float32Array(targetLen);
    for (let i = 0; i < targetLen; i++) {
        const src = i / ratio, i0 = Math.floor(src), i1 = Math.min(i0 + 1, ch.length - 1);
        out[i] = ch[i0] * (1 - (src - i0)) + ch[i1] * (src - i0);
    }
    return out;
}

async function ensureProcessor(useGpu, threads) {
    const eps = useGpu ? ['webgpu', 'wasm'] : ['wasm'];
    const key = eps.join(',');
    ort.env.wasm.numThreads = threads;
    if (useGpu) { ort.env.webgpu = ort.env.webgpu || {}; ort.env.webgpu.powerPreference = 'high-performance'; }
    if (processor && loadedEP === key) return;
    processor = new DemucsProcessor({
        ort,
        sessionOptions: {
            executionProviders: eps,
            graphOptimizationLevel: 'basic',
            enableCpuMemArena: false,
            enableMemPattern: false
        },
        onProgress: ({ progress, currentSegment, totalSegments }) =>
            post({ type: 'progress', progress, currentSegment, totalSegments }),
        onDownloadProgress: (loaded, total) => post({ type: 'download', loaded, total })
    });
    post({ type: 'phase', phase: 'loadmodel' });
    await processor.loadModel(CONSTANTS.DEFAULT_MODEL_URL);
    loadedEP = key;
}

self.onmessage = async (e) => {
    const msg = e.data || {};
    if (msg.type !== 'separate') return;
    try {
        let { left, right, sampleRate, useGpu, threads } = msg;

        if (sampleRate !== SAMPLE_RATE) {
            post({ type: 'phase', phase: 'resample' });
            const newLen = Math.floor(left.length * SAMPLE_RATE / sampleRate);
            left = resample(left, sampleRate, SAMPLE_RATE, newLen);
            right = resample(right, sampleRate, SAMPLE_RATE, newLen);
        }

        await ensureProcessor(!!useGpu, threads || 1);

        post({ type: 'phase', phase: 'separate' });
        const tracks = await processor.separate(left, right);

        const out = {};
        const transfers = [];
        for (const name of ['vocals', 'drums', 'bass', 'other']) {
            const tr = tracks[name];
            if (!tr) continue;
            out[name] = { left: tr.left, right: tr.right };
            transfers.push(tr.left.buffer, tr.right.buffer);
        }
        post({ type: 'done', tracks: out }, transfers);
    } catch (err) {
        // WebGPUを使っていて失敗したら、呼び出し側にフォールバック指示を返す
        post({ type: 'error', message: (err && err.message) ? err.message : String(err), usedGpu: !!msg.useGpu });
    }
};
