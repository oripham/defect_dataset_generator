// static/js/pages/studio/studio_state.js

let currentGroup = "cap";
var okImageB64 = null;
var ngRefB64 = null;
var maskB64 = null;
var resultB64 = null;
let batchImagePool = [];
let batchJobId = null;
let batchTimer = null;

// NG Ref State
let _ngFullImg = null;   // HTMLImageElement full NG

// Metal Cap Draw Mask State
let _mcBaseImg = null;   // HTMLImageElement for the OK image
let _mcMaskOff = null;   // offscreen canvas tracking painted pixels
let _mcDrawing = false;
let _mcBrushSz = 20;
