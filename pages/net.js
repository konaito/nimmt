/* NimmtNet の素のJS forward。PyTorch の nn.GELU()（erf版）/ LayerNorm(eps=1e-5) と数値一致させる。
   ブラウザ(<script>)と node(mjs import なし・global 登録) の両方から使う。 */

"use strict";

// Abramowitz–Stegun 7.1.26（|誤差| < 1.5e-7）
function erf(x) {
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return sign * y;
}

function gelu(x) {
  return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

function linear(W, b, x, outDim, inDim) {
  const out = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let s = b[o];
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) s += W[row + i] * x[i];
    out[o] = s;
  }
  return out;
}

function layerNorm(gamma, beta, x, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let v = 0;
  for (let i = 0; i < n; i++) { const d = x[i] - mean; v += d * d; }
  v /= n;
  const inv = 1 / Math.sqrt(v + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
  return out;
}

function b64ToF32(b64) {
  let bin;
  if (typeof atob === "function") bin = atob(b64);
  else bin = Buffer.from(b64, "base64").toString("binary");
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

class NimmtNetJS {
  constructor(blob) {
    this.cfg = blob.config;                     // {g_dim, c_dim, r_dim, hidden, depth}
    this.t = {};
    for (const [k, v] of Object.entries(blob.tensors)) {
      this.t[k] = { shape: v.shape, data: b64ToF32(v.b64) };
    }
  }

  encode(g) {
    const { hidden, depth } = this.cfg;
    // enc の Sequential index: [Linear, GELU, LN] × depth → Linear:3i, LN:3i+2
    let x = g;
    let inDim = this.cfg.g_dim;
    for (let d = 0; d < depth; d++) {
      const li = 3 * d, ni = 3 * d + 2;
      x = linear(this.t[`enc.${li}.weight`].data, this.t[`enc.${li}.bias`].data, x, hidden, inDim);
      for (let i = 0; i < hidden; i++) x[i] = gelu(x[i]);
      x = layerNorm(this.t[`enc.${ni}.weight`].data, this.t[`enc.${ni}.bias`].data, x);
      inDim = hidden;
    }
    return x;
  }

  _head(prefix, h, feat, featDim) {
    const { hidden } = this.cfg;
    const mid = hidden / 2;
    const inDim = hidden + featDim;
    const x = new Float32Array(inDim);
    x.set(h, 0);
    x.set(feat, hidden);
    let y = linear(this.t[`${prefix}.0.weight`].data, this.t[`${prefix}.0.bias`].data, x, mid, inDim);
    for (let i = 0; i < mid; i++) y[i] = gelu(y[i]);
    y = linear(this.t[`${prefix}.2.weight`].data, this.t[`${prefix}.2.bias`].data, y, 1, mid);
    return y[0];
  }

  cardLogits(h, cardFeats, mask) {
    const out = new Array(cardFeats.length);
    for (let i = 0; i < cardFeats.length; i++) {
      out[i] = mask[i] ? this._head("card_head", h, cardFeats[i], this.cfg.c_dim) : -Infinity;
    }
    return out;
  }

  rowLogits(h, rowFeats) {
    const out = new Array(rowFeats.length);
    for (let i = 0; i < rowFeats.length; i++) {
      out[i] = this._head("row_head", h, rowFeats[i], this.cfg.r_dim);
    }
    return out;
  }
}

if (typeof module !== "undefined") module.exports = { NimmtNetJS, erf, gelu };
if (typeof globalThis !== "undefined") globalThis.NimmtNetJS = NimmtNetJS;
