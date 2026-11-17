/**
 * Hands-only rehab coach (default mode). No controllers, no menus:
 *   pinch & hold -> 3 guided exercises (ghost-hand demo, then 12 s of live scoring each) -> summary.
 * Live score comes from the distilled AI model (public/models/student.onnx) when present,
 * otherwise from the signal-processing fallback. Rep count, range of motion, tremor and rhythm are always
 * computed from the hand-joint signal. Fingertip markers show which finger is lagging.
 */
import { createSystem, CanvasTexture, Mesh, MeshBasicMaterial, PlaneGeometry, SphereGeometry } from '@iwsdk/core';
import { computeJoints, GhostHand } from './ghost.js';
import { calibratedScore, ClassicScorer, metrics, normalise, OnnxScorer, preprocess, signalFor } from './scoring.js';

const NAMES = ['Fist open / close', 'Finger tapping', 'Wrist rotation'];
const ORDER = [0, 1, 2];
const TIPS = [4, 9, 14, 19, 24];
const FINGER = ['thumb', 'index', 'middle', 'ring', 'little'];
const PINCH_M = 0.025;
const HOLD_S = 0.8;
const DEMO_S = 4;
const PRACTICE_S = 12;
const CAP = 1200;

const S = { IDLE: 0, DEMO: 1, PRACTICE: 2, SUMMARY: 3 };
const modelUrl = `${import.meta.env.BASE_URL}models/student.onnx`;

export class CoachSystem extends createSystem({}) {
    init() {
        this.state = S.IDLE;
        this.step = 0; // index into ORDER
        this.timer = 0;
        this.hold = 0;
        this.clock = 0;
        this.nextScore = 0;
        this.results = [];
        this.cur = { score: null, sigma: 0, reps: 0, cue: '', m: null };
        this.ema = null;
        this.sumScore = 0; this.nScore = 0;
        this.sig = { min: 0, max: 0, low: true, init: false };
        this.nRef = null;
        this.last = '';
        // hand input
        this.transforms = new Float32Array(16 * 25);
        this.pos = { left: new Float32Array(75), right: new Float32Array(75) };
        this.tracked = { left: false, right: false };
        this.side = 'right';
        // ring buffer of the active hand
        this.ring = { t: new Float32Array(CAP), p: new Float32Array(CAP * 75), n: 0, head: 0, cap: CAP };
        // per-finger excursion over the last ~3 s (wrist->tip distance)
        this.fing = new Float32Array(90 * 5);
        this.fingN = 0;
        this.fingROM = new Float32Array(5).fill(1);

        // scorers
        this.classic = new ClassicScorer();
        this.onnx = new OnnxScorer();
        this.modelState = 'loading';
        this.onnx.load(modelUrl).then(
            () => { this.modelState = 'ai'; this.last = ''; },
            (e) => { this.modelState = 'fallback'; this.last = ''; console.warn('[coach] model load failed:', e && (e.message || e)); },
        );

        // visuals
        this.canvas = document.createElement('canvas');
        this.canvas.width = 1024;
        this.canvas.height = 512;
        this.ctx = this.canvas.getContext('2d');
        this.tex = new CanvasTexture(this.canvas);
        const panel = new Mesh(new PlaneGeometry(0.8, 0.4), new MeshBasicMaterial({ map: this.tex, transparent: true }));
        panel.position.set(0.1, 1.5, -0.85);
        this.world.createTransformEntity(panel);
        this.ghost = new GhostHand(this.world);
        // Desktop demo: with no WebXR headset available, a simulated hand drives the coach so the link still shows the full experience.
        this.userHand = new GhostHand(this.world, { pos: [0.55, 1.2, -0.75], color: 0x16a34a, line: 0x15803d, fk: { weak: 3, weakScale: 0.1, tremor: 0, t: 0 } });
        const q = new URLSearchParams(location.search).get('demo');
        this.simOn = q === '1';
        this.simIdle = 0;
        if (q !== '0' && q !== '1') {
            if (navigator.xr && navigator.xr.isSessionSupported) navigator.xr.isSessionSupported('immersive-vr').then((ok) => { this.simOn = !ok; }, () => { this.simOn = true; });
            else this.simOn = true;
        }
        const sg = new SphereGeometry(0.011, 12, 10);
        this.tipMats = TIPS.map(() => new MeshBasicMaterial({ color: 0x38bdf8 }));
        this.tipMeshes = this.tipMats.map((m) => {
            const mesh = new Mesh(sg, m);
            mesh.visible = false;
            this.world.createTransformEntity(mesh);
            return mesh;
        });
        // tracked-hand dots (the emulator draws no hands; on a headset they sit on top of the system hand)
        const lg = new SphereGeometry(0.006, 8, 6), lm = new MeshBasicMaterial({ color: 0x22c55e });
        this.liveDots = Array.from({ length: 25 }, () => { const m = new Mesh(lg, lm); m.visible = false; this.world.createTransformEntity(m); return m; });
        this.welcome = this.world.getSceneObject?.('welcome-panel');
        this.draw();

        // test hook (used by automated checks): feed a [n,75] sequence and read back the score
        window.__rehab = {
            state: () => ({ tracked: this.tracked[this.side], state: this.state, step: this.step, model: this.modelState, cur: this.cur, results: this.results }),
            feed: (frames, hz) => {
                this.resetRing();
                frames.forEach((f, i) => this.pushFrame(i / hz, f));
                return (this.testNow = frames.length / hz);
            },
            score: async (exercise) => this.scoreNow(this.testNow, exercise, true),
            norm: () => Array.from(normalise(preprocess(this.ring, this.testNow).x)),
        };
    }

    resetRing() { this.ring.n = 0; this.ring.head = 0; }
    pushFrame(t, p) {
        const r = this.ring;
        r.t[r.head] = t;
        r.p.set(p, r.head * 75);
        r.head = (r.head + 1) % r.cap;
        r.n = Math.min(r.n + 1, r.cap);
    }

    readHands() {
        const frame = this.world.xrFrame, ref = this.world.xrReferenceSpace, session = this.world.session;
        this.tracked.left = this.tracked.right = false;
        if (!frame || !ref || !session) return;
        for (const src of session.inputSources) {
            if (!src.hand) continue;
            const side = src.handedness;
            if (side !== 'left' && side !== 'right') continue;
            const spaces = Array.from(src.hand.values());
            if (spaces.length !== 25 || !frame.fillPoses(spaces, ref, this.transforms)) continue;
            const out = this.pos[side];
            for (let j = 0; j < 25; j++) {
                out[3 * j] = this.transforms[16 * j + 12];
                out[3 * j + 1] = this.transforms[16 * j + 13];
                out[3 * j + 2] = this.transforms[16 * j + 14];
            }
            this.tracked[side] = true;
        }
        this.side = this.tracked.right || !this.tracked.left ? 'right' : 'left';
    }

    /** Desktop demo: synthetic user hand (stiff ring finger on the fist exercise, mild tremor on wrist rotation). */
    simHands(time) {
        this.tracked.left = false;
        this.tracked.right = true;
        this.side = 'right';
        const ex = ORDER[Math.min(this.step, ORDER.length - 1)];
        computeJoints(ex, time * 2 * Math.PI * 0.8 - 0.3, this.pos.right, { weak: ex === 0 ? 3 : -1, weakScale: 0.1, tremor: ex === 2 ? 0.05 : 0, t: time });
    }

    pinching() {
        for (const side of ['right', 'left']) {
            if (!this.tracked[side]) continue;
            const p = this.pos[side];
            if (Math.hypot(p[12] - p[27], p[13] - p[28], p[14] - p[29]) < PINCH_M) return true; // thumb tip (4) vs index tip (9)
        }
        return false;
    }

    start() {
        this.step = 0;
        this.results = [];
        this.simIdle = 0;
        this.beginDemo();
    }
    beginDemo() {
        this.state = S.DEMO;
        this.timer = DEMO_S;
        this.ghost.visible = true;
    }
    beginPractice() {
        this.state = S.PRACTICE;
        this.timer = PRACTICE_S;
        this.resetRing();
        this.fingN = 0;
        this.cur = { score: null, sigma: 0, reps: 0, cue: 'Copy the ghost hand', m: null };
        this.ema = null; this.sumScore = 0; this.nScore = 0; this.nextScore = 3;
        this.sig = { min: 0, max: 0, low: true, init: false };
        this.nRef = null;
        this.stat = { rom: [], tremor: [], cv: [] };
    }
    endPractice() {
        const ex = ORDER[this.step];
        this.results.push({
            ex,
            score: this.nScore ? this.sumScore / this.nScore : null,
            reps: this.cur.reps,
            rom: avg(this.stat.rom), tremor: avg(this.stat.tremor), cv: avg(this.stat.cv),
        });
        this.step++;
        if (this.step >= ORDER.length) { this.state = S.SUMMARY; this.ghost.visible = false; this.hold = 0; this.saveHistory(); } else this.beginDemo();
    }
    /** Keep a small per-browser session history (real sessions only, not the looping desktop demo). */
    saveHistory() {
        this.prevOverall = null; this.sessionCount = 0;
        try {
            const key = 'rehablens.history';
            const h = JSON.parse(localStorage.getItem(key) || '[]');
            const real = this.results.filter((r) => r.score != null);
            const overall = real.length ? Math.round(real.reduce((a, r) => a + r.score, 0) / real.length) : null;
            if (h.length) this.prevOverall = h[h.length - 1].overall;
            if (overall != null && !(this.simOn && !this.world.session)) { h.push({ t: Date.now(), overall }); localStorage.setItem(key, JSON.stringify(h.slice(-30))); }
            this.sessionCount = h.length;
        } catch (e) { /* storage unavailable: skip history */ }
    }

    /** Run scoring on the current window. */
    async scoreNow(now, exercise, force = false) {
        const w = preprocess(this.ring, now);
        if (!w) return null;
        const sig = signalFor(exercise, w.x);
        const m = metrics(sig);
        // split-conformal 90% half-width (points) from held-out synthetic subjects (seed 777, never used to fit the calibration)
        let res = { ...m, score: calibratedScore(m, exercise), sigma: 12.8 };
        if (this.onnx.ready) {
            const r = await this.onnx.run(w.x);
            if (r) res = { ...m, score: r.score, sigma: r.sigma, exercise: r.exercise, probs: r.probs };
        }
        return { ...res, coverage: w.coverage, m };
    }

    cueFor(m) {
        if (!m) return 'Copy the ghost hand';
        if (this.fingROM && ORDER[this.step] === 0) {
            let worst = 0, best = 0;
            for (let i = 0; i < 5; i++) { if (this.fingROM[i] < this.fingROM[worst]) worst = i; if (this.fingROM[i] > this.fingROM[best]) best = i; }
            if (this.fingROM[best] > 0 && this.fingROM[worst] < 0.55 * this.fingROM[best]) return `Bend your ${FINGER[worst]} finger more`;
        }
        if (m.rom < 0.45) return 'Open and close wider';
        if (m.tremor > 0.3) return 'Move smoothly - slow down a little';
        if (m.cv > 0.4) return 'Keep a steady rhythm';
        return 'Great form - keep going';
    }

    updateReps(sigNow) {
        const s = this.sig;
        if (!s.init) { s.min = s.max = sigNow; s.init = true; return; }
        s.min = Math.min(sigNow, s.min + 0.02 * (s.max - s.min) * 0.01);
        s.max = Math.max(sigNow, s.max - 0.02 * (s.max - s.min) * 0.01);
        const range = s.max - s.min;
        if (range < 1e-4) return;
        const u = (sigNow - s.min) / range;
        if (s.low && u > 0.65) { s.low = false; this.cur.reps++; }
        else if (!s.low && u < 0.35) s.low = true;
    }

    update(delta, time) {
        const inXR = !!this.world.session;
        const sim = !inXR && this.simOn;
        if (this.welcome) this.welcome.visible = !inXR && !sim;
        if (!inXR && !sim) {
            this.ghost.visible = false;
            this.userHand.visible = false;
            for (const m of this.tipMeshes) m.visible = false;
            this.setText('Enter VR to begin', '');
            return;
        }
        this.clock = time;
        if (sim) this.simHands(time);
        else this.readHands();
        const tracked = this.tracked[this.side];
        const pinch = sim ? this.simIdle > (this.state === S.IDLE ? 1.5 : 8) : this.pinching();
        if (sim && (this.state === S.IDLE || this.state === S.SUMMARY)) this.simIdle += delta;

        for (let i = 0; i < 25; i++) {
            const d = this.liveDots[i];
            d.visible = inXR && tracked;
            if (d.visible) { const p = this.pos[this.side]; d.position.set(p[3 * i], p[3 * i + 1], p[3 * i + 2]); }
        }
        // user's fingertip markers (colour = how well each finger moves)
        for (let i = 0; i < 5; i++) {
            const m = this.tipMeshes[i];
            m.visible = !sim && tracked && this.state === S.PRACTICE;
            if (!m.visible) continue;
            const p = this.pos[this.side], j = TIPS[i];
            m.position.set(p[3 * j], p[3 * j + 1], p[3 * j + 2]);
            const r = this.fingROM[i] / (Math.max(...this.fingROM) || 1);
            this.tipMats[i].color.setHex(ORDER[this.step] !== 0 ? 0x38bdf8 : r > 0.7 ? 0x4ade80 : r > 0.5 ? 0xfbbf24 : 0xf87171);
        }

        if (sim) {
            this.userHand.visible = this.state === S.PRACTICE;
            if (this.state === S.PRACTICE) { this.userHand.opts.t = time; this.userHand.opts.tremor = ORDER[this.step] === 2 ? 0.05 : 0; this.userHand.opts.weak = ORDER[this.step] === 0 ? 3 : -1; this.userHand.setTip(19, ORDER[this.step] === 0 ? 0xef4444 : 0x16a34a); this.userHand.update(ORDER[this.step], this.clock * 2 * Math.PI * 0.8 - 0.3); }
        }
        switch (this.state) {
            case S.IDLE:
            case S.SUMMARY: {
                this.hold = pinch ? this.hold + delta : 0;
                if (this.hold >= HOLD_S) { this.hold = 0; this.start(); }
                break;
            }
            case S.DEMO: {
                this.timer -= delta;
                this.ghost.update(ORDER[this.step], this.clock * 2 * Math.PI * 0.8);
                if (this.timer <= 0) this.beginPractice();
                break;
            }
            case S.PRACTICE: {
                this.timer -= delta;
                const ex = ORDER[this.step];
                this.ghost.update(ex, this.clock * 2 * Math.PI * 0.8);
                if (tracked) {
                    const p = this.pos[this.side];
                    this.pushFrame(time, p);
                    // per-finger openness (wrist->tip) ring
                    const fi = this.fingN % 90;
                    let mean = 0;
                    for (let i = 0; i < 5; i++) {
                        const j = TIPS[i];
                        const d = Math.hypot(p[3 * j] - p[0], p[3 * j + 1] - p[1], p[3 * j + 2] - p[2]);
                        this.fing[fi * 5 + i] = d; mean += d;
                    }
                    this.fingN++;
                    this.updateReps(this.exSignal(ex, p, mean / 5));
                }
                this.nextScore -= delta;
                if (this.nextScore <= 0 && tracked) {
                    this.nextScore = 1;
                    this.updateFingerROM();
                    this.scoreNow(time, ex).then((r) => {
                        if (!r || this.state !== S.PRACTICE) return;
                        this.ema = this.ema == null ? r.score : 0.6 * this.ema + 0.4 * r.score;
                        this.cur.score = this.ema; this.cur.sigma = r.sigma; this.cur.m = r.m;
                        this.cur.cue = this.cueFor(r.m);
                        if (r.coverage > 0.35) { this.sumScore += r.score; this.nScore++; this.stat.rom.push(r.m.rom); this.stat.tremor.push(r.m.tremor); this.stat.cv.push(r.m.cv); }
                    });
                }
                if (this.timer <= 0) this.endPractice();
                break;
            }
        }
        this.draw(tracked);
    }

    exSignal(ex, p, openness) {
        if (ex !== 2) return openness;
        const ax = p[15] - p[0], ay = p[16] - p[1], az = p[17] - p[2];
        const bx = p[60] - p[0], by = p[61] - p[1], bz = p[62] - p[2];
        const cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
        const l = Math.hypot(cx, cy, cz) || 1;
        if (!this.nRef) this.nRef = [cx / l, cy / l, cz / l];
        return 1 - (cx * this.nRef[0] + cy * this.nRef[1] + cz * this.nRef[2]) / l;
    }

    updateFingerROM() {
        const n = Math.min(this.fingN, 90);
        if (n < 20) return;
        for (let i = 0; i < 5; i++) {
            const v = [];
            for (let k = 0; k < n; k++) v.push(this.fing[k * 5 + i]);
            v.sort((a, b) => a - b);
            this.fingROM[i] = v[Math.floor(n * 0.95)] - v[Math.floor(n * 0.05)];
        }
    }

    // ---------- drawing ----------
    setText(a, b) { this.draw(false, a, b); }

    draw(tracked = true, forceA, forceB) {
        const ex = ORDER[Math.min(this.step, ORDER.length - 1)];
        const sc = this.cur.score == null ? -1 : Math.round(this.cur.score);
        const key = [this.state, this.step, Math.ceil(this.timer), sc, this.cur.reps, this.cur.cue, this.modelState, tracked, forceA, this.results.length, Math.round(this.hold * 5), this.simOn].join('|');
        if (key === this.last) return;
        this.last = key;
        const c = this.ctx;
        c.clearRect(0, 0, 1024, 512);
        c.fillStyle = 'rgba(8,12,24,0.86)';
        c.beginPath(); c.roundRect(0, 0, 1024, 512, 40); c.fill();
        c.fillStyle = '#7dd3fc'; c.font = '600 30px sans-serif';
        c.fillText('REHABLENS', 40, 62);
        c.textAlign = 'right'; c.fillStyle = this.modelState === 'ai' ? '#4ade80' : '#fbbf24'; c.font = '26px sans-serif';
        c.fillText(this.simOn && !this.world.session ? 'DESKTOP DEMO - simulated hand' : this.modelState === 'ai' ? 'AI model active' : this.modelState === 'loading' ? 'loading model...' : 'signal analysis mode', 984, 62);
        c.textAlign = 'left';
        c.fillStyle = '#fff';
        if (forceA) {
            c.font = '700 56px sans-serif'; c.fillText(forceA, 40, 230);
        } else if (this.state === S.IDLE) {
            c.font = '700 64px sans-serif'; c.fillText(this.simOn && !this.world.session ? 'Desktop demo' : tracked ? 'Pinch and hold to start' : 'Show your hands', 40, 220);
            c.font = '32px sans-serif'; c.fillStyle = '#cbd5e1';
            c.fillText(this.simOn && !this.world.session ? 'Best on Meta Quest with hand tracking. Starting demo...' : '3 short exercises, scored live by motion analysis. No controllers.', 40, 290);
            this.bar(c, this.hold / HOLD_S, 360);
        } else if (this.state === S.DEMO) {
            c.font = `600 34px sans-serif`; c.fillStyle = '#94a3b8'; c.fillText(`Exercise ${this.step + 1} of 3`, 40, 130);
            c.font = '700 66px sans-serif'; c.fillStyle = '#fff'; c.fillText(NAMES[ex], 40, 215);
            c.font = '36px sans-serif'; c.fillStyle = '#cbd5e1'; c.fillText('Watch the ghost hand, then copy it', 40, 280);
            c.font = '700 150px sans-serif'; c.fillStyle = '#7dd3fc'; c.fillText(String(Math.max(1, Math.ceil(this.timer))), 40, 440);
        } else if (this.state === S.PRACTICE) {
            c.font = '600 32px sans-serif'; c.fillStyle = '#94a3b8'; c.fillText(`${NAMES[ex]}  ·  ${this.step + 1}/3`, 40, 125);
            const col = sc < 0 ? '#94a3b8' : sc >= 75 ? '#4ade80' : sc >= 50 ? '#fbbf24' : '#f87171';
            c.fillStyle = col; c.font = '700 190px sans-serif'; c.fillText(sc < 0 ? '--' : String(sc), 40, 330);
            c.font = '34px sans-serif'; c.fillStyle = '#cbd5e1';
            if (sc >= 0 && this.cur.sigma) c.fillText(`form score  ±${Math.round(this.cur.sigma)} (90% interval)`, 40, 380);
            else c.fillText('measuring form...', 40, 380);
            c.textAlign = 'right'; c.fillStyle = '#fff'; c.font = '700 100px sans-serif'; c.fillText(String(this.cur.reps), 984, 250);
            c.font = '30px sans-serif'; c.fillStyle = '#94a3b8'; c.fillText('reps', 984, 292); c.textAlign = 'left';
            c.font = '600 38px sans-serif'; c.fillStyle = '#fde68a'; c.fillText(this.cur.cue, 40, 440);
            this.bar(c, 1 - this.timer / PRACTICE_S, 470);
            if (!tracked) { c.fillStyle = '#f87171'; c.font = '32px sans-serif'; c.fillText('Hand not visible', 560, 380); }
        } else {
            c.font = '700 54px sans-serif'; c.fillText('Session complete', 40, 125);
            let y = 195, tot = 0, n = 0;
            for (const r of this.results) {
                const s = r.score == null ? 0 : r.score;
                tot += s; n++;
                c.font = '30px sans-serif'; c.fillStyle = '#cbd5e1'; c.fillText(NAMES[r.ex], 40, y);
                c.fillStyle = '#1e293b'; c.fillRect(380, y - 28, 420, 34);
                c.fillStyle = s >= 75 ? '#4ade80' : s >= 50 ? '#fbbf24' : '#f87171'; c.fillRect(380, y - 28, 4.2 * s, 34);
                c.fillStyle = '#fff'; c.fillText(`${Math.round(s)}`, 820, y);
                c.font = '24px sans-serif'; c.fillStyle = '#94a3b8';
                c.fillText(r.ex === 1 ? `${r.reps} taps · tremor ${Math.round(r.tremor * 100)}%` : `${r.reps} reps · range ${Math.round(r.rom * 100)}% · tremor ${Math.round(r.tremor * 100)}%`, 380, y + 28);
                y += 85;
            }
            c.font = '700 40px sans-serif'; c.fillStyle = '#7dd3fc';
            const ov = Math.round(n ? tot / n : 0);
            c.fillText(`Overall ${ov} / 100` + (this.simOn && !this.world.session ? '  ·  demo restarts soon' : '  ·  pinch and hold to go again'), 40, 465);
            if (this.prevOverall != null && !(this.simOn && !this.world.session)) {
                const d = ov - this.prevOverall;
                c.font = '26px sans-serif'; c.fillStyle = '#94a3b8';
                c.fillText(`Last session ${this.prevOverall}  ·  ${d >= 0 ? '+' : ''}${d} today  ·  session ${this.sessionCount}`, 40, 500);
            }
        }
        this.tex.needsUpdate = true;
    }

    bar(c, f, y) {
        c.fillStyle = '#1e293b'; c.fillRect(40, y, 944, 14);
        c.fillStyle = '#38bdf8'; c.fillRect(40, y, 944 * Math.min(1, Math.max(0, f)), 14);
    }
}

function avg(a) { return a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0; }
