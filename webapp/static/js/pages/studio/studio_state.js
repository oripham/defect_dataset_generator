// static/js/pages/studio/studio_state.js

let currentGroup = "cap";
let okImageB64 = null;
let ngRefB64 = null;
let maskB64 = null;
let resultB64 = null;
let batchImagePool = [];
let batchJobId = null;
let batchTimer = null;

// NG Ref Crop State
let _ngFullImg = null;   // HTMLImageElement full NG
let _ngCropRect = null;  // {x,y,w,h} in canvas px
let _ngCropDraw = false, _ngCropSX = 0, _ngCropSY = 0;
let _ngCropScale = 1;
const _NG_CVW = 500;     // canvas display width

// Metal Cap Draw Mask State
let _mcBaseImg = null;   // HTMLImageElement for the OK image
let _mcMaskOff = null;   // offscreen canvas tracking painted pixels
let _mcDrawing = false;
let _mcBrushSz = 20;
