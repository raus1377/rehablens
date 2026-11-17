/** Ghost demonstration hand: forward kinematics ported from ml/rehablens_ml/synth.py (same hand model the AI was pretrained on). */
import { BufferGeometry, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh, MeshBasicMaterial, SphereGeometry } from '@iwsdk/core';

const DIGITS = [
    [4, -0.03, [0.04, 0.035, 0.03, 0.025]],
    [5, -0.025, [0.06, 0.04, 0.025, 0.02, 0.018]],
    [5, -0.005, [0.062, 0.045, 0.028, 0.022, 0.018]],
    [5, 0.015, [0.058, 0.042, 0.026, 0.02, 0.017]],
    [5, 0.033, [0.05, 0.03, 0.02, 0.016, 0.014]],
];
const MAXF = [0.9, 1.0, 0.8];
const BONES = []; // [parent, child] joint pairs
{
    let k = 1;
    for (const [nj] of DIGITS) {
        BONES.push([0, k]);
        for (let s = 0; s < nj - 1; s++) BONES.push([k + s, k + s + 1]);
        k += nj;
    }
}


/** Forward kinematics of the synthetic hand. Writes 25 joints (xyz, wrist at origin, metres) into `j`.
 *  opts.weak = digit index with reduced flexion (simulated stiff finger), opts.weakScale, opts.tremor = roll jitter (rad), opts.t = time (s). */
export function computeJoints(exercise, phase, j, opts = {}) {
    const base = 0.5 - 0.5 * Math.cos(phase);
    const flex = new Float32Array(15);
    let roll = 0;
    if (exercise === 0) {
        for (let d = 0; d < 5; d++) for (let a = 0; a < 3; a++) flex[d * 3 + a] = base * MAXF[a];
    } else if (exercise === 1) {
        for (let d = 1; d < 5; d++) {
            const b = Math.max(0, Math.sin(phase - ((d - 1) * Math.PI) / 2)) ** 2;
            for (let a = 0; a < 3; a++) flex[d * 3 + a] = b * MAXF[a];
        }
        for (let a = 0; a < 3; a++) flex[a] = 0.4 * base * MAXF[a];
    } else roll = 0.9 * Math.sin(phase);
    if (opts.weak != null && opts.weak >= 0) for (let a = 0; a < 3; a++) flex[opts.weak * 3 + a] *= opts.weakScale ?? 0.35;
    if (opts.tremor) roll += opts.tremor * Math.sin(2 * Math.PI * 8 * (opts.t || 0));
    let k = 1;
    for (let d = 0; d < 5; d++) {
        const [nj, bx, seg] = DIGITS[d];
        let x = bx, y = 0, z = -0.01, ang = 0;
        j[3 * k] = x; j[3 * k + 1] = y; j[3 * k + 2] = z; k++;
        for (let s = 0; s < nj - 1; s++) {
            if (s >= 1) ang += flex[d * 3 + Math.min(s - 1, 2)];
            y += -Math.sin(ang) * seg[s];
            z += -Math.cos(ang) * seg[s];
            j[3 * k] = x; j[3 * k + 1] = y; j[3 * k + 2] = z; k++;
        }
    }
    j[0] = j[1] = j[2] = 0;
    const c = Math.cos(roll), s = Math.sin(roll);
    for (let i = 0; i < 25; i++) {
        const px = j[3 * i], py = j[3 * i + 1];
        j[3 * i] = c * px - s * py; j[3 * i + 1] = s * px + c * py;
    }
    return j;
}

export class GhostHand {
    constructor(world, o = {}) {
        this.opts = o.fk || {};
        this.group = new Group();
        this.joints = new Float32Array(75);
        const mat = new MeshBasicMaterial({ color: o.color ?? 0x0284c7, transparent: true, opacity: 0.9 });
        const geo = new SphereGeometry(0.009, 10, 8);
        this.dots = [];
        for (let i = 0; i < 25; i++) {
            const m = new Mesh(geo, mat);
            this.group.add(m);
            this.dots.push(m);
        }
        this.lineGeo = new BufferGeometry();
        this.linePos = new Float32Array(BONES.length * 6);
        this.lineGeo.setAttribute('position', new Float32BufferAttribute(this.linePos, 3));
        const lines = new LineSegments(this.lineGeo, new LineBasicMaterial({ color: o.line ?? 0x0369a1 }));
        lines.frustumCulled = false; // positions change every frame
        this.group.add(lines);
        this.group.rotation.order = 'YXZ';
        this.group.rotation.set(Math.PI / 2, Math.PI, 0); // fingers up, palm toward the viewer
        this.group.scale.setScalar(1.7);
        this.group.position.set(...(o.pos || [-0.45, 1.2, -0.75]));
        world.createTransformEntity(this.group);
        this.group.visible = false;
    }
    set visible(v) { this.group.visible = v; }
    /** Recolour one joint (own material so the others keep the shared one). */
    setTip(i, hex) {
        const d = this.dots[i];
        if (!d.userData.own) { d.material = d.material.clone(); d.userData.own = true; }
        d.material.color.setHex(hex);
    }

    /** exercise 0=fist 1=finger_tap 2=wrist_rot ; phase in radians (2*pi per repetition) */
    update(exercise, phase) {
        computeJoints(exercise, phase, this.joints, this.opts);
        const j = this.joints;
        for (let i = 0; i < 25; i++) this.dots[i].position.set(j[3 * i], j[3 * i + 1], j[3 * i + 2]);
        BONES.forEach(([a, b], i) => {
            const pa = this.dots[a].position, pb = this.dots[b].position;
            this.linePos.set([pa.x, pa.y, pa.z, pb.x, pb.y, pb.z], i * 6);
        });
        this.lineGeo.attributes.position.needsUpdate = true;
    }
}
