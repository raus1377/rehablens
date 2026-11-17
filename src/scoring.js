/**
 * Scoring for the live coach.
 *  - preprocess(): ring buffer of joint frames -> normalised [300,75] window at 30 Hz.
 *    MUST match ml/rehablens_ml/data.py::normalise (wrist-centred, divided by mean wrist->middle-knuckle distance).
 *  - OnnxScorer: runs the distilled student model (public/models/student.onnx) with onnxruntime-web.
 *  - ClassicScorer: signal-processing fallback (ROM / tremor / rhythm) used until a model is available.
 */
export const T = 300;
export const FS = 30;
const J = 25;
export const EXERCISES = ['fist', 'finger_tap', 'wrist_rot'];
const TIPS = [4, 9, 14, 19, 24];

const winBuf = new Float32Array(T * J * 3);

/**
 * ring: { t: Float32Array(cap), p: Float32Array(cap*75), n: count, head: next write index, cap }
 * Resamples the last T/FS seconds to a uniform 30 Hz grid ending at `now`. If fewer than T/FS seconds have been
 * recorded, the observed span is tiled (periodic extension) so early scores are not biased by a frozen hand.
 * Returns {x, coverage} or null.
 */
export function preprocess(ring, now) {
    const { t, p, cap } = ring;
    const n = Math.min(ring.n, cap);
    if (n < 10) return null;
    const idx = (k) => (ring.head - n + k + cap * 2) % cap; // k=0 oldest
    const first = t[idx(0)];
    const span = now - first;
    if (span < 1) return null;
    const t0 = now - T / FS;
    for (let g = 0; g < T; g++) {
        let tg = t0 + g / FS;
        if (tg < first) tg += Math.ceil((first - tg) / span) * span;
        tg = Math.min(tg, now);
        let lo = 0, hi = n - 1; // last k with t[k] <= tg
        while (lo < hi) {
            const mid = (lo + hi + 1) >> 1;
            if (t[idx(mid)] <= tg) lo = mid; else hi = mid - 1;
        }
        const a = idx(lo), b = idx(Math.min(lo + 1, n - 1));
        const dt = t[b] - t[a];
        const w = dt > 1e-6 ? Math.min(1, Math.max(0, (tg - t[a]) / dt)) : 0;
        for (let c = 0; c < J * 3; c++) winBuf[g * 75 + c] = p[a * 75 + c] * (1 - w) + p[b * 75 + c] * w;
    }
    return { x: winBuf, coverage: Math.min(1, span / (T / FS)) };
}

/** In-place wrist-centre + hand-size normalise a [T,75] window. */
export function normalise(x, out) {
    out = out || new Float32Array(x.length);
    let scale = 0;
    for (let f = 0; f < T; f++) {
        const o = f * 75;
        const wx = x[o], wy = x[o + 1], wz = x[o + 2];
        for (let j = 0; j < J; j++) {
            out[o + 3 * j] = x[o + 3 * j] - wx;
            out[o + 3 * j + 1] = x[o + 3 * j + 1] - wy;
            out[o + 3 * j + 2] = x[o + 3 * j + 2] - wz;
        }
        scale += Math.hypot(out[o + 33], out[o + 34], out[o + 35]); // joint 11 = middle-finger proximal (knuckle)
    }
    scale = scale / T + 1e-6;
    for (let i = 0; i < out.length; i++) out[i] /= scale;
    return out;
}

const softmax = (v) => {
    const m = Math.max(...v);
    const e = Array.from(v, (a) => Math.exp(a - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((a) => a / s);
};

export class OnnxScorer {
    constructor() {
        this.session = null;
        this.ready = false;
        this.busy = false;
        this.name = 'AI model';
        this.norm = new Float32Array(T * 75);
        this.ort = null;
    }
    async load(url) {
        const ort = await import('onnxruntime-web/wasm');
        const [{ default: mjs }, { default: wasm }] = await Promise.all([
            import('../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs?url'),
            import('../node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm?url'),
        ]);
        ort.env.wasm.wasmPaths = { mjs, wasm };
        ort.env.wasm.numThreads = 1; // avoids SharedArrayBuffer / cross-origin-isolation requirements
        this.session = await ort.InferenceSession.create(url, { executionProviders: ['wasm'] });
        this.ort = ort;
        this.ready = true;
    }
    /** x: raw [T,75] window (metres). Returns { score 0-100, sigma, exercise, probs } */
    async run(x) {
        if (!this.ready || this.busy) return null;
        this.busy = true;
        try {
            normalise(x, this.norm);
            const input = new this.ort.Tensor('float32', this.norm, [1, T, 75]);
            const out = await this.session.run({ seq: input });
            const probs = softmax(out.ex.data);
            const score = Math.min(100, Math.max(0, out.score.data[0] * 100));
            const sigma = Math.sqrt(Math.exp(out.logvar.data[0])) * 100; // score std-dev in points
            return { score, sigma, exercise: probs.indexOf(Math.max(...probs)), probs };
        } finally {
            this.busy = false;
        }
    }
}

/** Signal-processing fallback. Returns the same shape as OnnxScorer.run (plus metrics). */
export class ClassicScorer {
    constructor() { this.name = 'signal analysis'; this.ready = true; }
    run(x) {
        const n = normalise(x);
        const sig = new Float32Array(T);
        for (let f = 0; f < T; f++) {
            let s = 0;
            for (const j of TIPS) s += Math.hypot(n[f * 75 + 3 * j], n[f * 75 + 3 * j + 1], n[f * 75 + 3 * j + 2]);
            sig[f] = s / TIPS.length;
        }
        return metrics(sig);
    }
}

/** Metrics of a 1-D motion signal sampled at FS: range of motion, tremor share (4-12 Hz), rhythm regularity. */
export function metrics(sig) {
    const N = sig.length;
    let mean = 0;
    for (let i = 0; i < N; i++) mean += sig[i];
    mean /= N;
    const s = Array.from(sig, (v) => v - mean);
    const sorted = [...sig].sort((a, b) => a - b);
    const rom = sorted[Math.floor(N * 0.95)] - sorted[Math.floor(N * 0.05)];
    // power spectrum via direct DFT up to 12 Hz (N=300 -> cheap enough once per second)
    let tot = 0, trem = 0;
    for (let k = 1; k <= Math.floor((12 * N) / FS); k++) {
        let re = 0, im = 0;
        const w = (2 * Math.PI * k) / N;
        for (let i = 0; i < N; i++) { re += s[i] * Math.cos(w * i); im -= s[i] * Math.sin(w * i); }
        const pw = re * re + im * im;
        tot += pw;
        if ((k * FS) / N >= 4) trem += pw;
    }
    const tremor = tot > 0 ? trem / tot : 0;
    // rhythm: peak-to-peak interval variation (5-frame smoothing so sensor noise does not create fake peaks)
    const sm = s.map((_, i) => { let a = 0, c = 0; for (let k = -2; k <= 2; k++) if (s[i + k] !== undefined) { a += s[i + k]; c++; } return a / c; });
    const sd = Math.sqrt(sm.reduce((a, b) => a + b * b, 0) / N) || 1e-9;
    const peaks = [];
    for (let i = 2; i < N - 2; i++) {
        if (sm[i] > sm[i - 1] && sm[i] >= sm[i + 1] && sm[i] > 0.2 * sd && (!peaks.length || i - peaks[peaks.length - 1] > FS * 0.3)) peaks.push(i);
    }
    let cv = 1;
    if (peaks.length >= 3) {
        const iv = peaks.slice(1).map((p, i) => p - peaks[i]);
        const m = iv.reduce((a, b) => a + b, 0) / iv.length;
        cv = Math.sqrt(iv.reduce((a, b) => a + (b - m) ** 2, 0) / iv.length) / m;
    }
    const romScore = Math.min(1, Math.max(0, (rom - 0.25) / 0.45)); // rom in hand-size units
    const q = 0.5 * romScore + 0.25 * (1 - Math.min(1, tremor * 3)) + 0.25 * (1 - Math.min(1, cv * 2.5));
    return { score: 100 * q, sigma: 12, exercise: -1, probs: null, rom, tremor, cv, reps: peaks.length };
}

/** 1-D motion signal for an exercise from a raw [T,75] window: openness (fist/finger_tap) or palm-normal angle (wrist_rot). */
export function signalFor(exercise, x) {
    const n = normalise(x);
    const sig = new Float32Array(T);
    if (exercise !== 2) {
        for (let f = 0; f < T; f++) {
            let s = 0;
            for (const j of TIPS) s += Math.hypot(n[f * 75 + 3 * j], n[f * 75 + 3 * j + 1], n[f * 75 + 3 * j + 2]);
            sig[f] = s / TIPS.length;
        }
        return sig;
    }
    const nrm = new Float32Array(T * 3);
    const m = [0, 0, 0];
    for (let f = 0; f < T; f++) {
        const o = f * 75;
        const ax = n[o + 15], ay = n[o + 16], az = n[o + 17]; // index metacarpal (joint 5)
        const bx = n[o + 60], by = n[o + 61], bz = n[o + 62]; // little metacarpal (joint 20)
        let cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
        const l = Math.hypot(cx, cy, cz) || 1;
        cx /= l; cy /= l; cz /= l;
        nrm[3 * f] = cx; nrm[3 * f + 1] = cy; nrm[3 * f + 2] = cz;
        m[0] += cx; m[1] += cy; m[2] += cz;
    }
    const ml = Math.hypot(m[0], m[1], m[2]) || 1;
    for (let f = 0; f < T; f++) {
        const d = (nrm[3 * f] * m[0] + nrm[3 * f + 1] * m[1] + nrm[3 * f + 2] * m[2]) / ml;
        sig[f] = Math.acos(Math.min(1, Math.max(-1, d)));
    }
    return sig;
}

/** Fallback calibration: per-exercise least-squares fit of the form score on (range of motion, tremor share, rhythm CV),
 *  fitted on synthetic data (ml/ synth generator; leave-one-out MAE ~6 points). Replaced by the AI model when it loads. */
const COEF = {"0": [73.008, 19.409, -143.753, -5.661], "1": [70.136, 192.139, -26.377, -4.529], "2": [91.859, 1.387, -103.424, -26.259]};
export function calibratedScore(m, exercise) {
    const c = COEF[exercise] || COEF[0];
    const v = c[0] + c[1] * m.rom + c[2] * Math.min(m.tremor, 1) + c[3] * Math.min(m.cv, 1);
    return Math.min(100, Math.max(0, v));
}
