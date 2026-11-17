/**
 * RehabLens — hands-only joint recorder.
 *
 * Reads the 25 WebXR hand joints per hand every XR frame, runs a tiny
 * pinch-to-start state machine (no controllers, no menus), and saves each
 * take as JSON through the dev server (/__save) or a browser download.
 *
 * Label a take with URL params, e.g.
 *   ?subject=s01&exercise=finger_tap&form=good&dur=10
 */
import {
    createSystem,
    CanvasTexture,
    Mesh,
    MeshBasicMaterial,
    PlaneGeometry,
} from '@iwsdk/core';

const N_JOINTS = 25; // WebXR hand-tracking joint count
const THUMB_TIP = 4;
const INDEX_TIP = 9;
const PINCH_M = 0.025; // thumb-index distance (m) that counts as a pinch
const HOLD_S = 0.6; // pinch-and-hold time to start a take
const COUNTDOWN_S = 3;

const params = new URLSearchParams(location.search);
const META = {
    subject: params.get('subject') || 's00',
    exercise: params.get('exercise') || 'finger_tap',
    form: params.get('form') || 'good',
    duration: Number(params.get('dur')) || 10,
};

const State = { IDLE: 0, COUNTDOWN: 1, RECORD: 2, SAVING: 3 };

export class HandRecorderSystem extends createSystem({}) {
    init() {
        this.state = State.IDLE;
        this.timer = 0;
        this.holdTimer = 0;
        this.flash = 0;
        this.take = 1;
        this.frames = [];
        this.jointNames = null;
        this.last = ''; // last panel text, to avoid redrawing every frame
        this.msg = 'Enter VR, then pinch and hold to start';
        this.sub = '';

        // Reused buffers: nothing is allocated per frame except the saved row.
        this.transforms = new Float32Array(16 * N_JOINTS);
        this.pos = { left: new Float32Array(3 * N_JOINTS), right: new Float32Array(3 * N_JOINTS) };
        this.tracked = { left: false, right: false };

        // Status panel (canvas texture on a plane, 1 m in front of the user)
        this.canvas = document.createElement('canvas');
        this.canvas.width = 1024;
        this.canvas.height = 384;
        this.ctx = this.canvas.getContext('2d');
        this.tex = new CanvasTexture(this.canvas);
        const mesh = new Mesh(
            new PlaneGeometry(0.9, 0.3375),
            new MeshBasicMaterial({ map: this.tex, transparent: true }),
        );
        mesh.position.set(0, 1.45, -0.9);
        this.world.createTransformEntity(mesh);
        this.draw();
    }

    draw() {
        const key = this.msg + '|' + this.sub;
        if (key === this.last) return;
        this.last = key;
        const c = this.ctx;
        c.clearRect(0, 0, 1024, 384);
        c.fillStyle = 'rgba(10,14,25,0.82)';
        c.beginPath();
        c.roundRect(0, 0, 1024, 384, 36);
        c.fill();
        c.fillStyle = '#7dd3fc';
        c.font = '600 34px sans-serif';
        c.fillText('RehabLens recorder', 40, 70);
        c.fillStyle = '#ffffff';
        c.font = '700 54px sans-serif';
        c.fillText(this.msg, 40, 170);
        c.fillStyle = '#cbd5e1';
        c.font = '30px sans-serif';
        c.fillText(this.sub, 40, 230);
        c.fillStyle = '#94a3b8';
        c.font = '28px sans-serif';
        c.fillText(
            `subject ${META.subject} · ${META.exercise} · form: ${META.form} · take ${this.take}`,
            40,
            330,
        );
        this.tex.needsUpdate = true;
    }

    /** Read both hands for this frame. Returns a bitmask: 1=left, 2=right. */
    readHands() {
        const frame = this.world.xrFrame;
        const ref = this.world.xrReferenceSpace;
        const session = this.world.session;
        this.tracked.left = this.tracked.right = false;
        if (!frame || !ref || !session) return 0;
        let mask = 0;
        for (const src of session.inputSources) {
            if (!src.hand) continue;
            const side = src.handedness; // 'left' | 'right'
            if (side !== 'left' && side !== 'right') continue;
            if (!this.jointNames) this.jointNames = Array.from(src.hand.keys());
            const spaces = Array.from(src.hand.values());
            if (spaces.length !== N_JOINTS) continue;
            if (!frame.fillPoses(spaces, ref, this.transforms)) continue;
            const out = this.pos[side];
            for (let j = 0; j < N_JOINTS; j++) {
                out[3 * j] = this.transforms[16 * j + 12];
                out[3 * j + 1] = this.transforms[16 * j + 13];
                out[3 * j + 2] = this.transforms[16 * j + 14];
            }
            this.tracked[side] = true;
            mask |= side === 'left' ? 1 : 2;
        }
        return mask;
    }

    isPinching(side) {
        if (!this.tracked[side]) return false;
        const p = this.pos[side];
        const dx = p[3 * THUMB_TIP] - p[3 * INDEX_TIP];
        const dy = p[3 * THUMB_TIP + 1] - p[3 * INDEX_TIP + 1];
        const dz = p[3 * THUMB_TIP + 2] - p[3 * INDEX_TIP + 2];
        return Math.hypot(dx, dy, dz) < PINCH_M;
    }

    update(delta, time) {
        const inXR = !!this.world.session;
        if (!inXR) {
            this.msg = 'Enter VR, then pinch and hold to start';
            this.sub = '';
            this.state = State.IDLE;
            this.draw();
            return;
        }
        const mask = this.readHands();

        switch (this.state) {
            case State.IDLE: {
                if (this.flash > 0) {
                    this.flash -= delta; // keep the "Saved" message visible for ~2.5 s
                    this.holdTimer = 0;
                    break;
                }
                this.msg = mask ? 'Pinch and hold to start' : 'Show your hands to the headset';
                this.sub = `Records ${META.duration}s of ${META.exercise.replace('_', ' ')}`;
                const pinching = this.isPinching('right') || this.isPinching('left');
                this.holdTimer = pinching ? this.holdTimer + delta : 0;
                if (this.holdTimer >= HOLD_S) {
                    this.state = State.COUNTDOWN;
                    this.timer = COUNTDOWN_S;
                    this.holdTimer = 0;
                }
                break;
            }
            case State.COUNTDOWN: {
                this.timer -= delta;
                this.msg = `Get ready… ${Math.max(1, Math.ceil(this.timer))}`;
                this.sub = 'Start the exercise when recording begins';
                if (this.timer <= 0) {
                    this.state = State.RECORD;
                    this.timer = META.duration;
                    this.frames = [];
                    this.t0 = time;
                }
                break;
            }
            case State.RECORD: {
                this.timer -= delta;
                // layout: [t, mask, left xyz*25, right xyz*25]
                const frame = new Float32Array(2 + 3 * N_JOINTS * 2);
                frame[0] = time - this.t0;
                frame[1] = mask;
                frame.set(this.pos.left, 2);
                frame.set(this.pos.right, 2 + 3 * N_JOINTS);
                this.frames.push(frame);
                this.msg = `Recording  ${Math.max(0, this.timer).toFixed(1)}s`;
                this.sub = `${this.frames.length} frames · keep both hands in view`;
                if (this.timer <= 0) {
                    this.state = State.SAVING;
                    this.save();
                }
                break;
            }
            case State.SAVING:
                this.msg = 'Saving…';
                this.sub = '';
                break;
        }
        this.draw();
    }

    async save() {
        const n = this.frames.length;
        const dur = n > 1 ? this.frames[n - 1][0] - this.frames[0][0] : 0;
        const payload = {
            schema: 'rehablens.handseq.v1',
            meta: { ...META, take: this.take, hz: n / Math.max(dur, 1e-6), nFrames: n },
            joints: this.jointNames,
            // row = [t, trackedMask(1=L,2=R), L(25*xyz), R(25*xyz)] in metres, reference-space coords
            frames: this.frames.map((f) => Array.from(f, (v) => Math.round(v * 1e4) / 1e4)),
        };
        const name = `${META.subject}_${META.exercise}_${META.form}_t${this.take}.json`;
        const body = JSON.stringify(payload);
        let ok = false;
        try {
            const res = await fetch('/__save', {
                method: 'POST',
                headers: { 'content-type': 'application/json', 'x-filename': name },
                body,
            });
            ok = res.ok;
        } catch {
            ok = false;
        }
        if (!ok) {
            // Fallback: browser download (works in the desktop emulator).
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([body], { type: 'application/json' }));
            a.download = name;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 5000);
        }
        this.msg = ok ? 'Saved ✓' : 'Downloaded ✓';
        this.sub = `${name} · pinch to record the next take`;
        this.take += 1;
        this.frames = [];
        this.flash = 2.5;
        this.state = State.IDLE;
    }
}
