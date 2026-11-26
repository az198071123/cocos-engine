/*
 Rasterizes TTF label text in a Web Worker on iOS web.

 iOS WebKit (Safari and in-app web views) leaks GPU-process memory when a page that draws
 WebGL every frame draws text with a web font (@font-face) into a 2D canvas that has been
 idle for a while: every such draw keeps another copy of the font file in the GPU process,
 so a NONE label that changes once a second grows it by the font size per second until iOS
 kills it and the WebGL context is lost. Measured on iOS 27.2.1, it does not happen when the
 same canvas draws every frame, with a system font, on a page without WebGL, or when the
 drawing happens on an OffscreenCanvas inside a worker, which is what this module does.

 The main thread still measures and lays out the text with its own canvas (measureText does
 not leak). TextProcessing records the drawing calls, this module replays them in the worker,
 reads the pixels back with getImageData (straight alpha, the same as uploading a canvas with
 UNPACK_PREMULTIPLY_ALPHA off) and hands them back to be uploaded into the label texture.
*/

import { JSB } from 'internal:constants';
import { systemInfo } from 'pal/system-info';
import { OS } from '../../../../pal/system-info/enum-type';

/** A recorded 2D-context call: `[0, property, value]` or `[1, method, ...args]`. */
export type RasterOp = [0, string, unknown] | [1, string, ...unknown[]];

const RECORDED_PROPERTIES = [
    'font', 'fillStyle', 'strokeStyle', 'lineWidth', 'lineJoin', 'textAlign', 'textBaseline',
    'shadowColor', 'shadowBlur', 'shadowOffsetX', 'shadowOffsetY',
];
const RECORDED_METHODS = ['clearRect', 'fillRect', 'fillText', 'strokeText'];

/**
 * Stands in for the label canvas context while a label is drawn: property writes and drawing
 * calls are recorded (property writes also reach the real context so measureText stays right),
 * measureText goes to the real context. Covers exactly what TextProcessing draws with.
 */
export class RecordingContext2D {
    public readonly ops: RasterOp[] = [];
    constructor (public readonly real: CanvasRenderingContext2D) {
        // The worker canvas starts from defaults: carry over what _updateLabelDimensions set.
        this.ops.push([0, 'font', real.font], [0, 'textAlign', real.textAlign], [0, 'textBaseline', real.textBaseline]);
    }

    public measureText (text: string): TextMetrics {
        return this.real.measureText(text);
    }
}

for (const name of RECORDED_PROPERTIES) {
    Object.defineProperty(RecordingContext2D.prototype, name, {
        get (this: RecordingContext2D): unknown { return (this.real as unknown as Record<string, unknown>)[name]; },
        set (this: RecordingContext2D, value: unknown) {
            (this.real as unknown as Record<string, unknown>)[name] = value;
            this.ops.push([0, name, value]);
        },
    });
}
for (const name of RECORDED_METHODS) {
    const proto = RecordingContext2D.prototype as unknown as Record<string, unknown>;
    proto[name] = function record (this: RecordingContext2D, ...args: unknown[]): void {
        this.ops.push([1, name, ...args]);
    };
}

// Draws run one at a time on a single OffscreenCanvas; each waits for the fonts it names.
const WORKER_SOURCE = `
let canvas = null, ctx = null, queue = Promise.resolve();
const fonts = new Map();
function loadFont (family, url) {
    if (fonts.has(family)) return;
    try {
        const face = new FontFace(family, 'url("' + url + '")');
        self.fonts.add(face);
        fonts.set(family, face.load().then(
            () => { self.postMessage({ t: 'font', family, ok: true }); },
            () => { self.postMessage({ t: 'font', family, ok: false }); },
        ));
    } catch (e) { fonts.set(family, Promise.resolve(null)); }
}
self.onmessage = (e) => {
    const m = e.data;
    if (m.t === 'probe') {
        let ok = false;
        try { ok = typeof FontFace === 'function' && !!self.fonts && !!new OffscreenCanvas(1, 1).getContext('2d'); } catch (err) { ok = false; }
        self.postMessage({ t: 'probe', ok });
    } else if (m.t === 'font') {
        loadFont(m.family, m.url);
    } else if (m.t === 'draw') {
        queue = queue.then(() => Promise.all(m.families.map((f) => fonts.get(f)))).then(() => {
            if (!canvas) { canvas = new OffscreenCanvas(m.w, m.h); ctx = canvas.getContext('2d'); } else { canvas.width = m.w; canvas.height = m.h; }
            for (const op of m.ops) { if (op[0] === 0) ctx[op[1]] = op[2]; else ctx[op[1]].apply(ctx, op.slice(2)); }
            const pixels = ctx.getImageData(0, 0, m.w, m.h).data;
            self.postMessage({ t: 'draw', id: m.id, pixels: pixels.buffer }, [pixels.buffer]);
        }).catch((err) => { self.postMessage({ t: 'draw', id: m.id, error: String(err) }); });
    }
};
`;

type DrawCallback = (pixels: Uint8Array | null) => void;
interface WorkerReply { t: string, ok?: boolean, id?: number, family?: string, pixels?: ArrayBuffer }

/** One label redraw: replay `ops` on a width×height canvas; `done` gets straight-alpha RGBA, or null if the worker failed. */
export interface RasterJob {
    width: number;
    height: number;
    family: string;
    ops: RasterOp[];
    done: DrawCallback;
}

// Per texture: at most one draw in flight, and only the newest waiting one is kept, so a label that changes every
// frame still shows its latest text and cannot pile up work in the worker.
interface RasterSlot { busy: boolean, next: RasterJob | null }

function unquote (family: string): string {
    return family.replace(/^["']|["']$/g, '');
}

class TextRasterWorker {
    private _worker: Worker | null = null;
    private _state: 'idle' | 'probing' | 'ready' | 'off' = 'idle';
    private _fonts = new Map<string, string>(); // family -> absolute url
    private _loaded = new Set<string>(); // families the worker has finished loading
    private _nextId = 1;
    private _callbacks = new Map<number, DrawCallback>();
    private _slots = new WeakMap<object, RasterSlot>();

    /**
     * Whether a label in `family` can be drawn in the worker: only web fonts loaded through loadFont (system fonts do
     * not leak), and only once the worker has the font, so a slow font download never holds labels back.
     * Starts the worker on first call.
     */
    public canDraw (family: string): boolean {
        if (this._state === 'idle') this._start();
        return this._state === 'ready' && this._loaded.has(unquote(family));
    }

    public registerFont (family: string, url: string): void {
        const name = unquote(family);
        if (this._fonts.has(name)) return;
        this._fonts.set(name, url);
        if (this._state === 'ready') this._worker!.postMessage({ t: 'font', family: name, url });
    }

    /** Draws `job` for `key` (the label texture), or queues it behind the draw already in flight for that key. */
    public request (key: object, job: RasterJob): void {
        let slot = this._slots.get(key);
        if (!slot) {
            slot = { busy: false, next: null };
            this._slots.set(key, slot);
        }
        if (slot.busy) {
            slot.next = job; // an older waiting job is simply dropped
            return;
        }
        this._send(key, slot, job);
    }

    /** Forget `key`'s pending work: a reply still in flight is ignored (the label went back to the main thread). */
    public cancel (key: object): void {
        this._slots.delete(key);
    }

    private _send (key: object, slot: RasterSlot, job: RasterJob): void {
        const id = this._nextId++;
        slot.busy = true;
        this._callbacks.set(id, (pixels): void => {
            if (this._slots.get(key) !== slot) return; // cancelled meanwhile
            slot.busy = false;
            job.done(pixels);
            const next = slot.next;
            slot.next = null;
            if (next && pixels) this._send(key, slot, next);
        });
        try {
            this._worker!.postMessage({ t: 'draw', id, w: job.width, h: job.height, families: [unquote(job.family)], ops: job.ops });
        } catch (e) {
            this._fail();
        }
    }

    private _start (): void {
        this._state = 'off';
        if (JSB || !systemInfo.isBrowser || systemInfo.os !== OS.IOS
            || typeof Worker !== 'function' || typeof OffscreenCanvas !== 'function' || typeof Blob !== 'function') {
            return;
        }
        try {
            const url = URL.createObjectURL(new Blob([WORKER_SOURCE], { type: 'text/javascript' }));
            this._worker = new Worker(url);
            URL.revokeObjectURL(url);
        } catch (e) {
            this._worker = null;
            return;
        }
        this._state = 'probing';
        this._worker.onmessage = (e: MessageEvent<WorkerReply>): void => this._onMessage(e.data);
        this._worker.onerror = (): void => this._fail();
        this._worker.postMessage({ t: 'probe' });
    }

    private _onMessage (m: WorkerReply): void {
        if (m.t === 'probe') {
            if (!m.ok) { this._fail(); return; }
            this._state = 'ready';
            this._fonts.forEach((url, family) => this._worker!.postMessage({ t: 'font', family, url }));
        } else if (m.t === 'font') {
            if (m.ok && m.family) this._loaded.add(m.family);
        } else if (m.t === 'draw') {
            // Any failed draw turns the worker off for good: _fail answers every pending draw with
            // null, and the label then redraws on the main thread instead of retrying every frame.
            if (!m.pixels) { this._fail(); return; }
            const done = this._callbacks.get(m.id!);
            this._callbacks.delete(m.id!);
            if (done) done(new Uint8Array(m.pixels));
        }
    }

    private _fail (): void {
        this._state = 'off';
        if (this._worker) this._worker.terminate();
        this._worker = null;
        const callbacks = Array.from(this._callbacks.values());
        this._callbacks.clear();
        callbacks.forEach((done) => done(null));
    }
}

export const textRasterWorker = new TextRasterWorker();
