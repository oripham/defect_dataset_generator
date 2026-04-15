/**
 * MaskEditor
 * Provides a canvas-based mask drawing tool.
 * - bgCanvas   : shows the background image (product photo)
 * - maskCanvas : transparent overlay; user paints semi-transparent red
 * - Internally maintains a separate B/W mask canvas (white = defect region)
 */
class MaskEditor {
  constructor(maskCanvasId, bgCanvasId, width, height) {
    this.w = width;
    this.h = height;

    this.maskCv = document.getElementById(maskCanvasId);
    this.maskCv.width = width;
    this.maskCv.height = height;
    this.mctx = this.maskCv.getContext('2d');

    this.bgCv = document.getElementById(bgCanvasId);
    this.bgCv.width = width;
    this.bgCv.height = height;
    this.bctx = this.bgCv.getContext('2d');

    // Actual mask (white/black) stored in a hidden canvas
    this.hiddenCv = document.createElement('canvas');
    this.hiddenCv.width = width;
    this.hiddenCv.height = height;
    this.hctx = this.hiddenCv.getContext('2d');
    this.hctx.fillStyle = 'black';
    this.hctx.fillRect(0, 0, width, height);

    this.tool = 'brush';
    this.brushSize = 20;
    this.isDrawing = false;
    this.startX = 0;
    this.startY = 0;
    this.bgImage = null;
    this.undoStack = [];   // stores ImageData snapshots of hiddenCv

    this._setupEvents();
    this._render();
  }

  // ---- Event setup ----

  _setupEvents() {
    const cv = this.maskCv;
    console.log('[MaskEditor] canvas:', cv.width, 'x', cv.height, 'element:', cv);
    cv.addEventListener('mousedown',  e => { console.log('[MaskEditor] mousedown'); this._onDown(e); });
    cv.addEventListener('mousemove',  e => this._onMove(e));
    cv.addEventListener('mouseup',    e => this._onUp(e));
    cv.addEventListener('mouseleave', e => this._onUp(e));
    // Touch support
    cv.addEventListener('touchstart', e => { e.preventDefault(); this._onDown(this._touch(e)); }, {passive:false});
    cv.addEventListener('touchmove',  e => { e.preventDefault(); this._onMove(this._touch(e)); }, {passive:false});
    cv.addEventListener('touchend',   e => { e.preventDefault(); this._onUp(e); },                {passive:false});
  }

  _touch(e) {
    const t = e.touches[0];
    return { clientX: t.clientX, clientY: t.clientY };
  }

  _pos(e) {
    const r = this.maskCv.getBoundingClientRect();
    const scaleX = this.w / r.width;
    const scaleY = this.h / r.height;
    return {
      x: (e.clientX - r.left) * scaleX,
      y: (e.clientY - r.top) * scaleY,
    };
  }

  _onDown(e) {
    const {x, y} = this._pos(e);
    this._saveUndo();
    this.isDrawing = true;
    this.startX = x;
    this.startY = y;
    if (this.tool === 'brush' || this.tool === 'eraser') {
      this._paint(x, y, x, y);
      this._render();
    }
    this._previewSnap = this._getHiddenSnap();
  }

  _onMove(e) {
    if (!this.isDrawing) return;
    const {x, y} = this._pos(e);
    if (this.tool === 'brush' || this.tool === 'eraser') {
      this._paint(this.startX, this.startY, x, y);
      this.startX = x;
      this.startY = y;
      this._render();
    } else {
      // Preview rect/ellipse
      this._restoreHiddenSnap(this._previewSnap);
      this._drawShape(this.startX, this.startY, x, y);
      this._render();
    }
  }

  _onUp(e) {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    if (this.tool === 'rect' || this.tool === 'ellipse') {
      const {x, y} = e.type.startsWith('touch') ? {x:0,y:0} : this._pos(e);
      this._restoreHiddenSnap(this._previewSnap);
      this._drawShape(this.startX, this.startY, x, y);
    }
    this._render();
  }

  // ---- Drawing on hidden canvas ----

  _paint(x0, y0, x1, y1) {
    const ctx = this.hctx;
    ctx.globalCompositeOperation = this.tool === 'eraser' ? 'destination-out' : 'source-over';
    ctx.strokeStyle = 'white';
    ctx.lineWidth = this.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.globalCompositeOperation = 'source-over';
  }

  _drawShape(x0, y0, x1, y1) {
    const ctx = this.hctx;
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'white';
    if (this.tool === 'rect') {
      ctx.fillRect(
        Math.min(x0,x1), Math.min(y0,y1),
        Math.abs(x1-x0), Math.abs(y1-y0)
      );
    } else if (this.tool === 'ellipse') {
      ctx.beginPath();
      ctx.ellipse(
        (x0+x1)/2, (y0+y1)/2,
        Math.abs(x1-x0)/2, Math.abs(y1-y0)/2,
        0, 0, Math.PI*2
      );
      ctx.fill();
    }
  }

  // ---- Render display canvas ----

  _render() {
    const ctx = this.mctx;
    ctx.clearRect(0, 0, this.w, this.h);

    // Draw red overlay where mask is white
    const imgData = this.hctx.getImageData(0, 0, this.w, this.h);
    const d = imgData.data;
    const out = ctx.createImageData(this.w, this.h);
    const od = out.data;
    for (let i = 0; i < d.length; i += 4) {
      const bright = d[i]; // red channel of white pixel
      if (bright > 128) {
        od[i]   = 220;  // R
        od[i+1] = 50;   // G
        od[i+2] = 50;   // B
        od[i+3] = 180;  // A
      }
      // else transparent (show background)
    }
    ctx.putImageData(out, 0, 0);
  }

  // ---- Undo ----

  _saveUndo() {
    const snap = this._getHiddenSnap();
    this.undoStack.push(snap);
    if (this.undoStack.length > 50) this.undoStack.shift();
  }

  _getHiddenSnap() {
    return this.hctx.getImageData(0, 0, this.w, this.h);
  }

  _restoreHiddenSnap(snap) {
    this.hctx.putImageData(snap, 0, 0);
  }

  undo() {
    if (!this.undoStack.length) return;
    this.hctx.putImageData(this.undoStack.pop(), 0, 0);
    this._render();
  }

  // ---- Public actions ----

  clear() {
    this._saveUndo();
    this.hctx.fillStyle = 'black';
    this.hctx.fillRect(0, 0, this.w, this.h);
    this._render();
  }

  invert() {
    this._saveUndo();
    const img = this.hctx.getImageData(0, 0, this.w, this.h);
    const d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      d[i]   = 255 - d[i];
      d[i+1] = 255 - d[i+1];
      d[i+2] = 255 - d[i+2];
      // keep alpha
    }
    this.hctx.putImageData(img, 0, 0);
    this._render();
  }

  loadBackground(url) {
    const img = new Image();
    img.onload = () => {
      this.bgImage = img;
      this.bctx.clearRect(0, 0, this.w, this.h);
      // Fit image to canvas
      const scale = Math.min(this.w / img.width, this.h / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      const dx = (this.w - dw) / 2;
      const dy = (this.h - dh) / 2;
      this.bctx.fillStyle = '#111';
      this.bctx.fillRect(0, 0, this.w, this.h);
      this.bctx.drawImage(img, dx, dy, dw, dh);
      this._render();
    };
    img.onerror = () => alert('背景画像の読み込みに失敗しました');
    img.crossOrigin = 'anonymous';
    img.src = url;
  }

  removeBg() {
    this.bgImage = null;
    this.bctx.clearRect(0, 0, this.w, this.h);
    this.bctx.fillStyle = '#111';
    this.bctx.fillRect(0, 0, this.w, this.h);
    this._render();
  }

  /** Returns the mask as a PNG data URL (white=defect, black=keep). */
  getMaskDataURL() {
    // The hidden canvas is already white/black.
    // Make sure background is filled black (alpha fix).
    const tmp = document.createElement('canvas');
    tmp.width = this.w;
    tmp.height = this.h;
    const tc = tmp.getContext('2d');
    tc.fillStyle = 'black';
    tc.fillRect(0, 0, this.w, this.h);
    tc.drawImage(this.hiddenCv, 0, 0);
    // Convert to grayscale (remove color, keep luminance)
    const id = tc.getImageData(0, 0, this.w, this.h);
    const d = id.data;
    for (let i = 0; i < d.length; i += 4) {
      const lum = (d[i] + d[i+1] + d[i+2]) / 3;
      const v = lum > 128 ? 255 : 0;
      d[i] = d[i+1] = d[i+2] = v;
      d[i+3] = 255;
    }
    tc.putImageData(id, 0, 0);
    return tmp.toDataURL('image/png');
  }

  loadMaskFromDataURL(dataURL) {
    const img = new Image();
    img.onload = () => {
      this._saveUndo();
      this.hctx.fillStyle = 'black';
      this.hctx.fillRect(0, 0, this.w, this.h);
      this.hctx.drawImage(img, 0, 0, this.w, this.h);
      // Threshold to B/W
      const id = this.hctx.getImageData(0, 0, this.w, this.h);
      const d = id.data;
      for (let i = 0; i < d.length; i += 4) {
        const lum = (d[i]+d[i+1]+d[i+2])/3;
        const v = lum > 128 ? 255 : 0;
        d[i] = d[i+1] = d[i+2] = v; d[i+3] = 255;
      }
      this.hctx.putImageData(id, 0, 0);
      this._render();
    };
    img.src = dataURL;
  }
}
