//! BCPack: lossless packing of BC1-BC7 block streams, as the console's
//! DirectStorage hardware unpacks them. Decode only; BC2 is left out (no
//! title data uses it). Ported from VIctory's victory-formats/src/bcpack.rs on 2026-11-23
//! (https://github.com/Marvisak/VIctory, GPL-3.0, see THIRD_PARTY.md).
//!
//! A packed stream has no header. It is 12 or 16 interleaved rANS *lanes*;
//! each lane is a run of fragments (header byte, length, payload) coded with
//! one of 128 static models, and a fixed schedule decides which lane takes
//! the next input byte. The lanes' symbols form one record per block: two
//! back references (to one of the last 256 blocks, or to a per-format
//! dictionary that stands in for blocks -1 to -255) and the residuals of the
//! endpoints and indices against them. The models, the index rank tables,
//! the BC7 control codes and the dictionaries are constants of the format,
//! kept in `bcpack-tables.bin` beside this file ([`Tables::builtin`]).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error(pub &'static str);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BCPack: {}", self.0)
    }
}

impl std::error::Error for Error {}

const HISTORY: usize = 256;
/// rANS: decode while the state is above this, read while below.
const LOW: u32 = 1 << 24;
/// Input bytes per schedule round.
const ROUND_BYTES: usize = 8;

use std::sync::OnceLock;

use super::tables::{self, Bc6Ep, Bc6Ix, Bc7Ep, Bc7Ix, Plain};

/// The format's constant tables, initialised once from `tables.rs`.
pub struct Tables {
    pub freq: &'static [[u32; tables::BUCKETS]; tables::MODELS],
    pub start: [[u32; tables::BUCKETS]; tables::MODELS],
    pub recip: [[u64; tables::BUCKETS]; tables::MODELS],
    pub rank1: &'static [u8; 256],
    pub rank3a: &'static [u8; 64],
    pub rank3c: &'static [u8; 256],
    pub rank4: &'static [u8; 64],
    pub rank5: &'static [u8; 64],
    pub ctrl7: &'static [u16; 512],
    pub bc1: &'static [Plain],
    pub bc3: &'static [Plain],
    pub bc4: &'static [Plain],
    pub bc5: &'static [Plain],
    pub bc6: &'static [(Bc6Ep, Bc6Ix)],
    pub bc7: &'static [(Bc7Ep, Bc7Ix)],
}

impl Tables {
    pub fn builtin() -> &'static Tables {
        static T: OnceLock<Tables> = OnceLock::new();
        T.get_or_init(|| {
            let mut start = [[0u32; tables::BUCKETS]; tables::MODELS];
            let mut recip = [[0u64; tables::BUCKETS]; tables::MODELS];
            for m in 0..tables::MODELS {
                let used = m / 16 + 2;
                let mut at = 0u32;
                for b in 0..used {
                    start[m][b] = at;
                    recip[m][b] = (1u64 << 32).div_ceil(u64::from(tables::FREQ[m][b]));
                    at += tables::FREQ[m][b] << b.saturating_sub(1);
                }
                start[m][used..].fill(1 << 16);
            }
            Tables {
                freq: &tables::FREQ,
                start,
                recip,
                rank1: &tables::RANK1,
                rank3a: &tables::RANK3A,
                rank3c: &tables::RANK3C,
                rank4: &tables::RANK4,
                rank5: &tables::RANK5,
                ctrl7: &tables::CTRL7,
                bc1: &tables::BC1,
                bc3: &tables::BC3,
                bc4: &tables::BC4,
                bc5: &tables::BC5,
                bc6: &tables::BC6,
                bc7: &tables::BC7,
            }
        })
    }
}


/// Unpacks `src` (BC number `bc`, 1 and 3-7) into `dst`, whose length gives
/// the block count. The stream carries no size: bytes after the last one the
/// lanes need are ignored, a stream that runs out is an error.
pub fn decode(bc: u8, src: &[u8], dst: &mut [u8], tables: &Tables) -> Result<(), Error> {
    let block = match bc {
        1 | 4 => 8,
        3 | 5 | 6 | 7 => 16,
        _ => return Err(Error("unsupported BC format")),
    };
    if dst.is_empty() || !dst.len().is_multiple_of(block) {
        return Err(Error("destination is not a whole number of blocks"));
    }
    let blocks = dst.len() / block;
    let (widths, per) = lane_layout(bc);
    let stride = widths.len() * per;
    let mut records = vec![0u8; blocks * stride];
    let mut next = src.iter().copied();
    lanes(tables, widths, per, blocks, |_| next.next(), &mut records)?;
    let records = records.chunks_exact(stride);
    let out = dst.chunks_exact_mut(block);
    let t = tables;
    match bc {
        1 => plain(records, out, &t.bc1, &BC1, |s| bc1_residual(t, s), bc1_pack),
        3 => plain(records, out, &t.bc3, &BC3, |s| bc3_residual(t, s), bc3_pack),
        4 => plain(records, out, &t.bc4, &BC4, |s| bc4_residual(t, s), bc4_pack),
        5 => plain(records, out, &t.bc5, &BC5, |s| bc5_residual(t, s), bc5_pack),
        6 => bc6(records, out, &t.bc6),
        _ => bc7(records, out, t),
    }
}

// Lanes.

/// Format `bc`'s lanes (symbol bits - 1 of each) and the symbols each lane gives per block.
fn lane_layout(bc: u8) -> (&'static [u8], usize) {
    match bc {
        1 => (&[7, 7, 4, 5, 4, 4, 5, 4, 7, 7, 7, 7], 1),
        3 => (&[3, 3, 7, 5, 5, 5, 5, 4, 5, 4, 7, 7], 2),
        4 => (&[7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 1),
        5 => (&[3, 3, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5], 2),
        6 => (&[3, 3, 7, 7, 7, 4, 4, 4, 5, 5, 5, 7, 7, 7, 7, 4], 2),
        _ => (&[3, 3, 7, 7, 7, 7, 7, 6, 6, 4, 5, 7, 7, 7, 7, 4], 2),
    }
}

/// The lane stage alone (tests): the records of `blocks` blocks of format `bc` in `src`.
#[cfg(test)]
pub(super) fn records(bc: u8, src: &[u8], blocks: usize) -> Result<(Vec<u8>, usize, usize), Error> {
    let (widths, per) = lane_layout(bc);
    let mut records = vec![0u8; blocks * widths.len() * per];
    let mut next = src.iter().copied();
    lanes(Tables::builtin(), widths, per, blocks, |_| next.next(), &mut records)?;
    Ok((records, widths.len(), per))
}

#[derive(Clone, Copy, Default)]
struct Lane {
    /// Symbol bits - 1 at the full alphabet.
    width: u8,
    /// 0 header, 1 length (low byte), 2 length high byte, 3 payload.
    phase: u8,
    long: bool,
    reset: bool,
    model: u8,
    need: bool,
    latch: Option<u8>,
    out: Option<u8>,
    stalled: bool,
    x: u32,
    left: u32,
}

impl Lane {
    /// One step: take the latched byte (header, length or payload) and
    /// decode at most one symbol.
    fn step(&mut self, t: &Tables) -> Result<(), Error> {
        if self.phase != 3 {
            let Some(b) = self.latch.take() else {
                return Ok(());
            };
            self.need = true;
            match self.phase {
                0 => {
                    // The top two bits shrink the alphabet; width >= 3.
                    let group = self.width - (b >> 6);
                    self.model = group * 16 + (b & 15);
                    self.reset = b & 0x10 != 0;
                    self.long = b & 0x20 != 0;
                    self.left = 0;
                    self.phase = 1;
                }
                1 if self.long => {
                    self.left = b.into();
                    self.phase = 2;
                }
                1 => {
                    self.left = u32::from(b) + 1;
                    self.phase = 3;
                }
                _ => {
                    self.left = (u32::from(b) << 8 | self.left) + 257;
                    self.phase = 3;
                }
            }
            return Ok(());
        }
        let mut x = self.x;
        if x < LOW {
            if let Some(b) = self.latch.take() {
                self.left = self.left.wrapping_sub(1);
                x = x << 8 | u32::from(b);
            }
        }
        self.need = self.left != 0 && self.latch.is_none();
        if x > LOW {
            let freq = &t.freq[usize::from(self.model)];
            let start = &t.start[usize::from(self.model)];
            let slot = x & 0xffff;
            // The bucket: starts are ascending, and 65536 past the model's last.
            let b = start[1..].iter().filter(|&&s| s <= slot).count();
            // Buckets 0 and 1 hold one symbol. Elsewhere offset and frequency
            // are below 2^16, where the reciprocal divides exactly.
            let sym = match b {
                0 => 0,
                1 => 1,
                _ => {
                    let rec = t.recip[usize::from(self.model)][b];
                    (1 << (b - 1)) + ((u64::from(slot - start[b]) * rec) >> 32) as u8
                }
            };
            // Slot 0 is never used: symbol 0 owns 1 .. f[0] - 1.
            let (f, c) = if sym == 0 {
                (freq[0] - 1, 1)
            } else {
                let k = (8 - sym.leading_zeros()) as usize;
                let f = freq[k];
                (
                    f,
                    (u32::from(sym) - (1 << (k - 1)))
                        .wrapping_mul(f)
                        .wrapping_add(start[k])
                        & 0xffff,
                )
            };
            self.x = (x >> 16).wrapping_mul(f).wrapping_sub(c).wrapping_add(slot);
            self.out = Some(sym);
        } else if self.left != 0 {
            // At exactly 2^24 the lane neither reads nor decodes again; only a
            // lane that has delivered its symbols may end up there.
            self.x = x;
        } else {
            // The fragment ends; a byte read in this step does not reach the carried state.
            self.phase = 0;
            self.need = self.latch.is_none();
            if self.reset {
                self.x = 0;
            }
        }
        Ok(())
    }
}

/// Decodes `per` symbols per block on every lane into `records` (one record
/// of `widths.len() * per` symbols per block), taking input from `fetch`
/// (given the lane that takes the byte) in the order of the schedule: each
/// round the lanes step in order and take at most 8 bytes between them.
// shortcut: one lane step at a time (BC1 roughly 30-60 MB/s, BC7 15-20 MB/s
// on one core); if texture streaming needs more, run the schedule only to
// split the input per lane and decode the lanes apart.
fn lanes(
    t: &Tables,
    widths: &[u8],
    per: usize,
    blocks: usize,
    mut fetch: impl FnMut(usize) -> Option<u8>,
    records: &mut [u8],
) -> Result<(), Error> {
    let mut lanes = [Lane::default(); 16];
    let lanes = &mut lanes[..widths.len()];
    for (lane, &width) in lanes.iter_mut().zip(widths) {
        lane.width = width;
        lane.need = true;
    }
    let stride = widths.len() * per;
    let full = (1u32 << widths.len()) - 1;
    let (mut pos, mut done, mut idle) = (0, 0u32, 0);
    while pos < blocks * per {
        let (mut taken, mut moved) = (0, false);
        for (j, lane) in lanes.iter_mut().enumerate() {
            let bit = 1 << j;
            let before = taken;
            if !lane.stalled {
                lane.step(t)?;
            }
            let waiting = lane.out.is_some() && done & bit != 0;
            if lane.need && !waiting && taken < ROUND_BYTES {
                // Past the end nothing arrives; in a valid stream only finished lanes ask there.
                if let Some(b) = fetch(j) {
                    lane.latch = Some(b);
                    taken += 1;
                }
            }
            lane.stalled = (lane.need && before >= ROUND_BYTES) || waiting;
            if !lane.stalled && lane.out.is_some() {
                let sym = lane.out.take().unwrap();
                records[pos / per * stride + j * per + pos % per] = sym;
                done |= bit;
                moved = true;
            }
        }
        if done == full {
            pos += 1;
            done = 0;
            moved = true;
        }
        // Nothing read and nothing delivered for a few rounds: no lane can go on.
        idle = if taken > 0 || moved { 0 } else { idle + 1 };
        if idle > 3 {
            return Err(Error("the stream ends early or is corrupt"));
        }
    }
    Ok(())
}

// Blocks.

/// The record's endpoint and index references.
fn references(s: &[u8], paired: bool) -> (usize, usize) {
    let (d, ix) = if paired {
        (interleave(s[0], s[1]), interleave(s[2], s[3]))
    } else {
        (s[0], s[1])
    };
    let delta = (d >> 1) ^ (d & 1).wrapping_neg();
    (usize::from(ix.wrapping_add(delta)), usize::from(ix))
}

/// Spreads a 4-bit symbol onto even bit positions `0, 2, 4, 6`.
const SPREAD4: [u8; 16] = [
    0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
    0x40, 0x41, 0x44, 0x45, 0x50, 0x51, 0x54, 0x55,
];

/// Spreads an 8-bit symbol onto even bit positions `0, 2, ..., 14`.
const SPREAD8: [u16; 256] = {
    let mut t = [0u16; 256];
    let mut v = 0usize;
    while v < 256 {
        let mut out = 0u16;
        let mut k = 0;
        while k < 8 {
            out |= (((v >> k) & 1) as u16) << (2 * k);
            k += 1;
        }
        t[v] = out;
        v += 1;
    }
    t
};

/// Two 4-bit symbols, the first on the even bits.
#[inline(always)]
fn interleave(a: u8, b: u8) -> u8 {
    SPREAD4[usize::from(a & 15)] | (SPREAD4[usize::from(b & 15)] << 1)
}

/// Two 8-bit symbols, the first on the even bits.
#[inline(always)]
fn wide_interleave(a: u8, b: u8) -> u16 {
    SPREAD8[usize::from(a)] | (SPREAD8[usize::from(b)] << 1)
}

/// Block `i`'s reference `r`: 0 none, else `r` blocks back, before the first
/// block into the dictionary.
fn pick<'a, T>(i: usize, r: usize, history: &'a [T], dictionary: &'a [T], zero: &'a T) -> &'a T {
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) % HISTORY],
        r => &dictionary[r - i - 1],
    }
}

/// Zigzag residual (0, -1, 1, -2, ...) modulo 2^bits.
#[inline(always)]
fn zz(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// Element masks and active counts of the plain formats' endpoints and indices.
struct Widths {
    ep_mask: [u8; 8],
    ep_len: usize,
    ix_mask: [u8; 32],
    ix_len: usize,
    paired: bool,
}

const fn widths(ep: [u8; 8], ix3: usize, ix2: usize, paired: bool) -> Widths {
    let mut ep_mask = [0u8; 8];
    let mut ep_len = 0;
    let mut k = 0;
    while k < 8 {
        if ep[k] != 0 {
            ep_mask[k] = ((1u16 << ep[k]) - 1) as u8;
            ep_len = k + 1;
        }
        k += 1;
    }
    let mut ix_mask = [0u8; 32];
    let ix_len = ix3 + ix2;
    let mut k = 0;
    while k < ix_len {
        ix_mask[k] = if k < ix3 { 7 } else { 3 };
        k += 1;
    }
    Widths { ep_mask, ep_len, ix_mask, ix_len, paired }
}

const BC1: Widths = widths([5, 6, 5, 5, 6, 5, 0, 0], 0, 16, false);
const BC3: Widths = widths([5, 6, 5, 5, 6, 5, 8, 8], 16, 16, true);
const BC4: Widths = widths([8, 8, 0, 0, 0, 0, 0, 0], 16, 0, false);
const BC5: Widths = widths([8, 8, 8, 8, 0, 0, 0, 0], 32, 0, true);

fn plain<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[Plain],
    w: &Widths,
    residual: impl Fn(&[u8]) -> Plain,
    pack: fn(&Plain, &mut [u8]),
) -> Result<(), Error> {
    let mut history = vec![Plain::default(); HISTORY];
    let zero = Plain::default();
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, w.paired);
        let ep_ref = &pick(i, ep_ref, &history, dictionary, &zero).ep;
        let ix_ref = &pick(i, ix_ref, &history, dictionary, &zero).ix;
        let mut v = residual(s);
        for ((v, &r), &mask) in v.ep[..w.ep_len].iter_mut().zip(ep_ref).zip(&w.ep_mask) {
            *v = v.wrapping_add(r) & mask;
        }
        if w.ix_len == 16 {
            for ((v, &r), &mask) in v.ix[..16].iter_mut().zip(&ix_ref[..16]).zip(&w.ix_mask[..16]) {
                *v = v.wrapping_add(r) & mask;
            }
        } else {
            for ((v, &r), &mask) in v.ix[..w.ix_len].iter_mut().zip(ix_ref).zip(&w.ix_mask) {
                *v = v.wrapping_add(r) & mask;
            }
        }
        history[i % HISTORY] = v;
        pack(&v, block);
    }
    Ok(())
}

/// 2-bit index residuals: one symbol per 2x2 quad.
#[inline(always)]
fn quads(rank: &[u8; 256], s: &[u8; 4], ix: &mut [u8; 16]) {
    for (q, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym)];
        let base = (q >> 1) * 8 + (q & 1) * 2;
        ix[base] = v & 3;
        ix[base + 1] = (v >> 2) & 3;
        ix[base + 4] = (v >> 4) & 3;
        ix[base + 5] = (v >> 6) & 3;
    }
}

/// 3-bit index residuals: one symbol per two pixels in raster order.
#[inline(always)]
fn pairs(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    let (chunks, _) = ix.as_chunks_mut::<2>();
    for (&sym, dst) in s.iter().zip(chunks) {
        let v = rank[usize::from(sym & 63)];
        dst[0] = v & 7;
        dst[1] = (v >> 4) & 7;
    }
}

fn bc1_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for (k, bits) in [5, 6, 5, 5, 6, 5].into_iter().enumerate() {
        v.ep[k] = zz(s[2 + k], bits);
    }
    quads(t.rank1, s[8..12].try_into().unwrap(), (&mut v.ix[..16]).try_into().unwrap());
    v
}

fn bc3_residual(t: &Tables, s: &[u8]) -> Plain {
    // Colour lanes carry each channel's two endpoints together.
    let mut v = Plain {
        ep: [
            zz(s[14], 5),
            zz(s[16], 6),
            zz(s[18], 5),
            zz(s[15], 5),
            zz(s[17], 6),
            zz(s[19], 5),
            zz(s[4], 8),
            zz(s[5], 8),
        ],
        ..Default::default()
    };
    pairs(&t.rank3a, &s[6..14], &mut v.ix[..16]);
    quads(t.rank3c, s[20..24].try_into().unwrap(), (&mut v.ix[16..]).try_into().unwrap());
    v
}

fn bc4_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    v.ep[..2].copy_from_slice(&[zz(s[2], 8), zz(s[3], 8)]);
    pairs(&t.rank4, &s[4..12], &mut v.ix[..16]);
    v
}

fn bc5_residual(t: &Tables, s: &[u8]) -> Plain {
    let mut v = Plain::default();
    for k in 0..4 {
        v.ep[k] = zz(s[4 + k], 8);
    }
    pairs(&t.rank5, &s[8..24], &mut v.ix);
    v
}

/// Indices along the endpoint line -> BC1 and BC4 codes.
const BC1_CODE: [u32; 4] = [0, 2, 3, 1];
const BC4_CODE: [u64; 8] = [0, 2, 3, 4, 5, 6, 7, 1];

#[inline(always)]
fn bc1_color(ep: &[u8; 6], ix: &[u8; 16], out: &mut [u8; 8]) {
    let c0 = u16::from(ep[0]) << 11 | u16::from(ep[1]) << 5 | u16::from(ep[2]);
    let c1 = u16::from(ep[3]) << 11 | u16::from(ep[4]) << 5 | u16::from(ep[5]);
    let mut bits = 0u32;
    for (p, &val) in ix.iter().enumerate() {
        bits |= BC1_CODE[usize::from(val & 3)] << (2 * p);
    }
    let word = u64::from(c0) | (u64::from(c1) << 16) | (u64::from(bits) << 32);
    *out = word.to_le_bytes();
}

/// BC4 half: endpoints are kept 128 below their stored value.
#[inline(always)]
fn bc4_alpha(e0: u8, e1: u8, ix: &[u8; 16], out: &mut [u8; 8]) {
    let mut bits = 0u64;
    for (p, &val) in ix.iter().enumerate() {
        bits |= BC4_CODE[usize::from(val & 7)] << (16 + 3 * p);
    }
    let v = u64::from(e0 ^ 0x80) | u64::from(e1 ^ 0x80) << 8 | bits;
    *out = v.to_le_bytes();
}

fn bc1_pack(v: &Plain, out: &mut [u8]) {
    bc1_color(
        v.ep[..6].try_into().unwrap(),
        v.ix[..16].try_into().unwrap(),
        (&mut out[..8]).try_into().unwrap(),
    );
}

fn bc3_pack(v: &Plain, out: &mut [u8]) {
    let (a, c) = out.split_at_mut(8);
    bc4_alpha(v.ep[6], v.ep[7], v.ix[..16].try_into().unwrap(), a.try_into().unwrap());
    bc1_color(
        v.ep[..6].try_into().unwrap(),
        v.ix[16..].try_into().unwrap(),
        (&mut c[..8]).try_into().unwrap(),
    );
}

fn bc4_pack(v: &Plain, out: &mut [u8]) {
    bc4_alpha(v.ep[0], v.ep[1], v.ix[..16].try_into().unwrap(), (&mut out[..8]).try_into().unwrap());
}

fn bc5_pack(v: &Plain, out: &mut [u8]) {
    let (a, b) = out.split_at_mut(8);
    bc4_alpha(v.ep[0], v.ep[1], v.ix[..16].try_into().unwrap(), a.try_into().unwrap());
    bc4_alpha(v.ep[2], v.ep[3], v.ix[16..].try_into().unwrap(), (&mut b[..8]).try_into().unwrap());
}

/// Even and odd bit nibbles of every byte, zigzag-decoded to signed 4-bit residuals.
const NIBBLES: [[i16; 2]; 256] = {
    let mut t = [[0i16; 2]; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut even = 0u8;
        let mut odd = 0u8;
        let mut k = 0;
        while k < 4 {
            even |= (((b as u8) >> (2 * k)) & 1) << k;
            odd |= (((b as u8) >> (2 * k + 1)) & 1) << k;
            k += 1;
        }
        let ze = (even >> 1) as i16 ^ -((even & 1) as i16);
        let zo = (odd >> 1) as i16 ^ -((odd & 1) as i16);
        t[b] = [ze, zo];
        b += 1;
    }
    t
};

/// A byte's even and odd bits as two zigzag-coded 4-bit residuals.
#[inline(always)]
fn nibbles(b: u8, odd_first: bool) -> [i16; 2] {
    let [even, odd] = NIBBLES[usize::from(b)];
    if odd_first { [odd, even] } else { [even, odd] }
}

/// Index residual plus reference, both left-aligned to 16 bits at their own
/// widths, back to `bits` bits (signed).
#[inline(always)]
fn add_index(cur: i16, bits: u8, refv: i16, ref_bits: u8) -> i16 {
    let up = |v: i16, b: u8| {
        if b == 0 || b > 16 {
            0
        } else {
            (v as u16) << (16 - b)
        }
    };
    let sum = up(cur, bits).wrapping_add(up(refv, ref_bits)) as i16;
    if bits == 0 { 0 } else { sum >> (16 - bits) }
}

/// Little-endian bit writer for one 128-bit block.
#[derive(Default)]
struct Bits {
    v: u128,
    at: u32,
}

impl Bits {
    #[inline(always)]
    fn put(&mut self, value: u64, n: u32) {
        if n > 0 && self.at < 128 {
            let masked = if n < 64 { value & ((1u64 << n) - 1) } else { value };
            self.v |= u128::from(masked) << self.at;
        }
        self.at += n;
    }
}

// BC7.

/// Public BC7 anchor indices: second subset of two; second and third of three.
const ANCHOR2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8,
    2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2,
    2, 15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHOR3A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5,
    15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5,
    10, 8, 13, 15, 12, 3, 3,
];
const ANCHOR3B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6,
    10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

/// Endpoint bits of R G B A per mode (public), before p-bits and rotation.
const BC7_BITS: [[u8; 4]; 8] = [
    [4, 4, 4, 0],
    [6, 6, 6, 0],
    [5, 5, 5, 0],
    [7, 7, 7, 0],
    [5, 5, 5, 6],
    [7, 7, 7, 8],
    [7, 7, 7, 7],
    [5, 5, 5, 5],
];
/// Index bits of the primary and secondary sets per mode.
const BC7_INDEX: [[u8; 2]; 8] = [
    [3, 0],
    [3, 0],
    [2, 0],
    [2, 0],
    [3, 2],
    [2, 2],
    [4, 0],
    [2, 0],
];

fn pbits(mode: u16) -> bool {
    matches!(mode, 0 | 1 | 3 | 6 | 7)
}

/// Channel precisions: the p-bit is red's lowest bit; rotation swaps alpha
/// with the rotated channel.
fn bc7_precision(ctrl: u16) -> [u8; 4] {
    let (mode, rot) = (ctrl & 7, usize::from(ctrl >> 3 & 3));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

fn bc7<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    t: &Tables,
) -> Result<(), Error> {
    let mut history = vec![(Bc7Ep::default(), Bc7Ix::default()); HISTORY];
    let zero = (Bc7Ep::default(), Bc7Ix::default());
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, true);
        let ctrl = t.ctrl7[usize::from(s[31] & 15) << 5 | usize::from(s[30] & 31)];
        if ctrl == 0 {
            return Err(Error("unused BC7 control code"));
        }
        let mode = usize::from(ctrl & 7);
        let subsets = usize::from((ctrl & 0xff) >> 6);
        let prec = bc7_precision(ctrl);
        let three = subsets == 3;
        let or0 = |v: u8| if three { 0 } else { v };
        let if3 = |v: u8| if three { v } else { 0 };
        let stored = [
            [s[4], s[5], s[6], s[7], s[8], s[9], or0(s[10]), or0(s[11])],
            [
                s[12],
                s[13],
                s[14],
                s[15],
                s[16],
                s[17],
                or0(s[18]),
                or0(s[19]),
            ],
            [
                s[20],
                s[21],
                if3(s[10]),
                if3(s[11]),
                if3(s[18]),
                if3(s[19]),
                0,
                0,
            ],
        ];
        let r = &pick(i, ep_ref, &history, &t.bc7, &zero).0;
        let mut ep = Bc7Ep {
            ctrl,
            ..Default::default()
        };
        // Rows the reference lacks build on its first row.
        let common = subsets.min(usize::from((r.ctrl & 0xff) >> 6));
        let columns = if prec[3] > 0 { 8 } else { 6 };
        for c in 0..columns {
            let p = prec[c / 2];
            if p == 0 {
                continue;
            }
            let mask = 0xffffu16 << (16 - p);
            for (row, st) in stored.iter().enumerate().take(subsets) {
                let v = st[c];
                let residual =
                    (u16::from(v >> 1) ^ (v & 1).wrapping_neg() as i8 as u16) << (16 - p);
                let base = r.rows[if row < common { row } else { 0 }][c] & mask;
                ep.rows[row][c] = base.wrapping_add(residual);
            }
        }
        let bits = BC7_INDEX[mode];
        let rx = &pick(i, ix_ref, &history, &t.bc7, &zero).1;
        let mut ix = Bc7Ix {
            bits,
            ..Default::default()
        };
        for (set, offsets) in [[22, 24, 26, 28], [12, 14, 16, 20]].into_iter().enumerate() {
            if bits[set] == 0 {
                continue;
            }
            for (k, o) in offsets.into_iter().enumerate() {
                for (b, &byte) in s[o..o + 2].iter().enumerate() {
                    let pair = nibbles(byte, k < 3);
                    for (h, &v) in pair.iter().enumerate() {
                        let p = 4 * k + 2 * b + h;
                        ix.ix[set][p] = add_index(v, bits[set], rx.ix[set][p], rx.bits[set]);
                    }
                }
            }
        }
        history[i % HISTORY] = (ep, ix);
        bc7_pack(&ep, &ix, prec, block);
    }
    Ok(())
}

fn bc7_pack(ep: &Bc7Ep, ix: &Bc7Ix, prec: [u8; 4], out: &mut [u8]) {
    let ctrl = ep.ctrl;
    let mode = ctrl & 7;
    let subsets = usize::from((ctrl & 0xff) >> 6);
    let part = usize::from(ctrl >> 8 & 63);
    let rot = usize::from(ctrl >> 3 & 3);
    let mut w = Bits::default();
    w.put(1 << mode, u32::from(mode) + 1);
    match mode {
        0 => w.put(part as u64, 4),
        1 | 2 | 3 | 7 => w.put(part as u64, 6),
        4 => {
            w.put(rot as u64, 2);
            w.put(u64::from(ctrl >> 5 & 1), 1);
        }
        5 => w.put(rot as u64, 2),
        _ => {}
    }
    let ends = 2 * subsets;
    let field = |row: usize, col: usize, p: u8| u64::from(ep.rows[row][col] >> (16 - p));
    let mut order = [0, 1, 2, 3];
    if rot != 0 {
        order.swap(3, rot - 1);
    }
    for ch in order {
        let p = prec[ch];
        if p == 0 {
            continue;
        }
        let drop = u32::from(ch == 0 && pbits(mode));
        for e in 0..ends {
            w.put(field(e / 2, 2 * ch + e % 2, p) >> drop, u32::from(p) - drop);
        }
    }
    match mode {
        1 => (0..subsets).for_each(|row| w.put(field(row, 0, prec[0]) & 1, 1)),
        m if pbits(m) => (0..ends).for_each(|e| w.put(field(e / 2, e % 2, prec[0]) & 1, 1)),
        _ => {}
    }
    let anchor_mask: u16 = match subsets {
        2 => 1 | (1 << ANCHOR2[part]),
        3 => 1 | (1 << ANCHOR3A[part]) | (1 << ANCHOR3B[part]),
        _ => 1,
    };
    let mut indices = |set: usize| {
        let bits = u32::from(ix.bits[set]);
        for (p, &v) in ix.ix[set].iter().enumerate() {
            let n = bits - u32::from((anchor_mask >> p) & 1);
            w.put(u64::from(v as u16), n);
        }
    };
    match mode {
        4 => {
            indices(1);
            indices(0);
        }
        5 => {
            indices(0);
            indices(1);
        }
        _ => indices(0),
    }
    out.copy_from_slice(&w.v.to_le_bytes());
}

// BC6H.

/// Bits of endpoint 0, then of the other endpoints' R, G, B, per mode 1-14.
const BC6_BITS: [[u8; 4]; 15] = [
    [0; 4],
    [10, 5, 5, 5],
    [7, 6, 6, 6],
    [11, 5, 4, 4],
    [11, 4, 5, 4],
    [11, 4, 4, 5],
    [9, 5, 5, 5],
    [8, 6, 5, 5],
    [8, 5, 6, 5],
    [8, 5, 5, 6],
    [6, 6, 6, 6],
    [10, 10, 10, 10],
    [11, 9, 9, 9],
    [12, 8, 8, 8],
    [16, 4, 4, 4],
];
/// Mode code and its width, modes 1-14.
const BC6_CODE: [(u64, u32); 15] = [
    (0, 0),
    (0, 2),
    (1, 2),
    (2, 5),
    (6, 5),
    (10, 5),
    (14, 5),
    (18, 5),
    (22, 5),
    (26, 5),
    (30, 5),
    (3, 5),
    (7, 5),
    (11, 5),
    (15, 5),
];

/// A run of the public BC6H layout: `len` bits of channel `ch` (0-2; 3 is
/// the partition) of endpoint `ep`, from bit `first` up.
#[derive(Clone, Copy)]
struct Run {
    ch: u8,
    ep: u8,
    first: u8,
    len: u8,
}

const fn r(ch: u8, ep: u8, first: u8, len: u8) -> Run {
    Run { ch, ep, first, len }
}

const R: u8 = 0;
const G: u8 = 1;
const B: u8 = 2;
const D: Run = r(3, 0, 0, 5);

/// The header after the mode bits, per mode 1-14. Modes 13 and 14 write the
/// high base bits in ascending order (what the console's packer does).
#[rustfmt::skip]
const BC6_LAYOUT: [&[Run]; 15] = [
    &[],
    &[
        r(G, 2, 4, 1), r(B, 2, 4, 1), r(B, 3, 4, 1), r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(G, 2, 5, 1), r(G, 3, 4, 1), r(G, 3, 5, 1), r(R, 0, 0, 7), r(B, 3, 0, 1), r(B, 3, 1, 1),
        r(B, 2, 4, 1), r(G, 0, 0, 7), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 7),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 5), r(R, 0, 10, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(G, 3, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 5), r(G, 0, 10, 1), r(G, 3, 0, 4), r(B, 1, 0, 4), r(B, 0, 10, 1),
        r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 0, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(G, 2, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 1), r(B, 2, 4, 1),
        r(G, 2, 0, 4), r(G, 1, 0, 4), r(G, 0, 10, 1), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5),
        r(B, 0, 10, 1), r(B, 2, 0, 4), r(R, 2, 0, 4), r(B, 3, 1, 1), r(B, 3, 2, 1), r(R, 3, 0, 4),
        r(B, 3, 4, 1), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 9), r(B, 2, 4, 1), r(G, 0, 0, 9), r(G, 2, 4, 1), r(B, 0, 0, 9), r(B, 3, 4, 1),
        r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4), r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4),
        r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5), r(B, 3, 2, 1), r(R, 3, 0, 5),
        r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(G, 3, 4, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 3, 2, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 3, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 5),
        r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 6),
        r(R, 3, 0, 6), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 0, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(G, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(G, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 6), r(G, 3, 0, 4), r(B, 1, 0, 5), r(B, 3, 1, 1), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 8), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 8), r(B, 2, 5, 1), r(G, 2, 4, 1),
        r(B, 0, 0, 8), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 5), r(G, 3, 4, 1), r(G, 2, 0, 4),
        r(G, 1, 0, 5), r(B, 3, 0, 1), r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 5),
        r(B, 3, 2, 1), r(R, 3, 0, 5), r(B, 3, 3, 1), D,
    ],
    &[
        r(R, 0, 0, 6), r(G, 3, 4, 1), r(B, 3, 0, 1), r(B, 3, 1, 1), r(B, 2, 4, 1), r(G, 0, 0, 6),
        r(G, 2, 5, 1), r(B, 2, 5, 1), r(B, 3, 2, 1), r(G, 2, 4, 1), r(B, 0, 0, 6), r(G, 3, 5, 1),
        r(B, 3, 3, 1), r(B, 3, 5, 1), r(B, 3, 4, 1), r(R, 1, 0, 6), r(G, 2, 0, 4), r(G, 1, 0, 6),
        r(G, 3, 0, 4), r(B, 1, 0, 6), r(B, 2, 0, 4), r(R, 2, 0, 6), r(R, 3, 0, 6), D,
    ],
    &[r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 10), r(G, 1, 0, 10), r(B, 1, 0, 10)],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 9), r(R, 0, 10, 1), r(G, 1, 0, 9),
        r(G, 0, 10, 1), r(B, 1, 0, 9), r(B, 0, 10, 1),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 8), r(R, 0, 10, 2), r(G, 1, 0, 8),
        r(G, 0, 10, 2), r(B, 1, 0, 8), r(B, 0, 10, 2),
    ],
    &[
        r(R, 0, 0, 10), r(G, 0, 0, 10), r(B, 0, 0, 10), r(R, 1, 0, 4), r(R, 0, 10, 6), r(G, 1, 0, 4),
        r(G, 0, 10, 6), r(B, 1, 0, 4), r(B, 0, 10, 6),
    ],
];

fn bc6_index_bits(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

fn bc6<'a>(
    records: impl Iterator<Item = &'a [u8]>,
    out: impl Iterator<Item = &'a mut [u8]>,
    dictionary: &[(Bc6Ep, Bc6Ix)],
) -> Result<(), Error> {
    let mut history = vec![(Bc6Ep::default(), Bc6Ix::default()); HISTORY];
    let zero = (
        Bc6Ep::default(),
        Bc6Ix {
            mode: 11,
            ..Default::default()
        },
    );
    let z = |v: u16| (v >> 1) ^ (v & 1).wrapping_neg();
    let wide = wide_interleave;
    for (i, (s, block)) in records.zip(out).enumerate() {
        let (ep_ref, ix_ref) = references(s, true);
        let mode = s[30].wrapping_add(1);
        if !(1..=14).contains(&mode) {
            return Err(Error("BC6H mode out of range"));
        }
        let two = mode < 11;
        let shift = BC6_BITS[usize::from(mode)].map(|b| 16 - u32::from(b));
        let e0 = [wide(s[4], s[5]), wide(s[6], s[7]), wide(s[8], s[9])].map(z);
        let e1 = [wide(s[10], s[11]), wide(s[12], s[13]), wide(s[14], s[15])].map(z);
        let (e2, e3) = if two {
            (
                [s[16], s[18], s[20]].map(|v| z(v.into())),
                [s[17], s[19], s[21]].map(|v| z(v.into())),
            )
        } else {
            (e0, e1)
        };
        let r = &pick(i, ep_ref, &history, dictionary, &zero).0;
        let mut ep = Bc6Ep::default();
        // Endpoint 0 at the base precision, the others per channel; the
        // reference is cut by the mask after the sum.
        let mask = |sh: u32| 0xffffu16 << sh;
        for ((e, &r), res) in ep.e[0].iter_mut().zip(&r.e[0]).zip(e0) {
            *e = r.wrapping_add(res << shift[0]) & mask(shift[0]);
        }
        for (k, res) in [e1, e2, e3]
            .iter()
            .enumerate()
            .take(if two { 3 } else { 1 })
        {
            for c in 0..3 {
                ep.e[k + 1][c] =
                    r.e[k + 1][c].wrapping_add(res[c] << shift[1 + c]) & mask(shift[1 + c]);
            }
        }
        if !two {
            for c in 0..3 {
                ep.e[2][c] = (ep.e[0][c] >> shift[0]) << shift[1];
            }
            ep.e[3] = ep.e[1];
        }
        let rx = &pick(i, ix_ref, &history, dictionary, &zero).1;
        let bits = bc6_index_bits(mode);
        let mut ix = Bc6Ix {
            mode,
            ..Default::default()
        };
        for k in 0..8 {
            for (h, v) in nibbles(s[22 + k], k < 6).into_iter().enumerate() {
                let p = 2 * k + h;
                ix.ix[p] = add_index(v, bits, rx.ix[p], bc6_index_bits(rx.mode));
            }
        }
        history[i % HISTORY] = (ep, ix);
        // Fields: the working value has its top bit flipped.
        let field = |k: usize, c: usize| {
            let sh = if k == 0 { shift[0] } else { shift[1 + c] };
            u64::from((ep.e[k][c] ^ 0x8000) >> sh)
        };
        let mut w = Bits::default();
        let (code, n) = BC6_CODE[usize::from(mode)];
        w.put(code, n);
        let part = s[31] & 31;
        for run in BC6_LAYOUT[usize::from(mode)] {
            if run.ch == 3 {
                w.put(part.into(), 5);
            } else {
                w.put(
                    field(run.ep.into(), run.ch.into()) >> run.first,
                    run.len.into(),
                );
            }
        }
        let anchor_mask: u16 = if two { 1 | (1 << ANCHOR2[usize::from(part)]) } else { 1 };
        let bias = 1u16 << (bits - 1);
        for (p, &v) in ix.ix.iter().enumerate() {
            let v = (v as u16).wrapping_add(bias);
            w.put(v.into(), u32::from(bits) - u32::from((anchor_mask >> p) & 1));
        }
        block.copy_from_slice(&w.v.to_le_bytes());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    #[test]
    fn lut_helpers_match_scalar_reference_exhaustively() {
        let spread4 = |v: u8| (0..4).fold(0u8, |acc, k| acc | ((v >> k) & 1) << (2 * k));
        let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | ((v >> (2 * k)) & 1) << k);
        let z4 = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
        for a in 0..=255u8 {
            let (even, odd) = (z4(half(a)), z4(half(a >> 1)));
            assert_eq!(nibbles(a, false), [even, odd]);
            assert_eq!(nibbles(a, true), [odd, even]);
            for b in 0..=255u8 {
                assert_eq!(interleave(a, b), spread4(a) | (spread4(b) << 1));
                let wide_ref = (0..8).fold(0u16, |acc, k| {
                    acc | (u16::from((a >> k) & 1) << (2 * k)) | (u16::from((b >> k) & 1) << (2 * k + 1))
                });
                assert_eq!(wide_interleave(a, b), wide_ref);
            }
        }

        // Precomputed Widths masks and active lengths for BC1/BC3/BC4/BC5.
        for (w, ep_bits, ix3, ix2) in [
            (&BC1, [5u8, 6, 5, 5, 6, 5, 0, 0], 0usize, 16usize),
            (&BC3, [5, 6, 5, 5, 6, 5, 8, 8], 16, 16),
            (&BC4, [8, 8, 0, 0, 0, 0, 0, 0], 16, 0),
            (&BC5, [8, 8, 8, 8, 0, 0, 0, 0], 32, 0),
        ] {
            let want_ep_len = ep_bits.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            assert_eq!(w.ep_len, want_ep_len);
            assert_eq!(w.ix_len, ix3 + ix2);
            for (k, &bits) in ep_bits.iter().enumerate() {
                assert_eq!(w.ep_mask[k], ((1u16 << bits) - 1) as u8);
            }
            for (k, &mask) in w.ix_mask.iter().enumerate() {
                let bits = if k < ix3 { 3 } else if k < ix3 + ix2 { 2 } else { 0 };
                assert_eq!(mask, (1u8 << bits) - 1);
            }
        }

        // BC7 and BC6H u16 anchor bitmasks match slice/branch anchor predicates.
        for part in 0..64usize {
            for subsets in [1usize, 2, 3] {
                let anchors: &[u8] = match subsets {
                    2 => &[0, ANCHOR2[part]],
                    3 => &[0, ANCHOR3A[part], ANCHOR3B[part]],
                    _ => &[0],
                };
                let mask: u16 = match subsets {
                    2 => 1 | (1 << ANCHOR2[part]),
                    3 => 1 | (1 << ANCHOR3A[part]) | (1 << ANCHOR3B[part]),
                    _ => 1,
                };
                for p in 0..16usize {
                    assert_eq!(((mask >> p) & 1) != 0, anchors.contains(&(p as u8)));
                }
            }
            if part < 32 {
                for two in [false, true] {
                    let anchor = if two { ANCHOR2[part] } else { 0 };
                    let mask: u16 = if two { 1 | (1 << ANCHOR2[part]) } else { 1 };
                    for p in 0..16usize {
                        assert_eq!(((mask >> p) & 1) != 0, p == 0 || p == usize::from(anchor));
                    }
                }
            }
        }

        // Bits::put matches u128 reference across n = 0, small widths, 64, and overflow past 128.
        let mut got = Bits::default();
        let mut ref_v = 0u128;
        let mut ref_at = 0u32;
        for (val, n) in [(0xFFFFu64, 0u32), (0x55, 5), (0xDEAD_BEEF_CAFE_BABE, 64), (0x3FF, 10), (0xFFFF, 60)] {
            got.put(val, n);
            if n > 0 && ref_at < 128 {
                ref_v |= (u128::from(val) & ((1u128 << n) - 1)) << ref_at;
            }
            ref_at += n;
            assert_eq!((got.v, got.at), (ref_v, ref_at));
        }
    }

    #[test]
    fn all_bc_fixtures_and_packers_match_baseline() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/data/bcpack");
        let tables = Tables::builtin();
        let cases: [(&str, u8, usize, u64); 12] = [
            ("bc1-2-0", 1, 2 * 8, 0x00241ed045f935b7),
            ("bc1-16-1", 1, 16 * 8, 0xaa27d5c23c617a11),
            ("bc1-16-9", 1, 16 * 8, 0x5af81e16cde0f88b),
            ("bc1-700-2", 1, 700 * 8, 0x6c1294776a5e0f7f),
            ("bc1-8192-3", 1, 8192 * 8, 0xa0d8a27ea25cb660),
            ("bc3-16-1", 3, 16 * 8, 0xc9b128a6e1beda7b),
            ("bc4-16-1", 4, 16 * 8, 0x80173d43d3e12784),
            ("bc5-16-1", 5, 16 * 8, 0x4338d58b289e83fc),
            ("bc6-16-1", 6, 16 * 8, 0x4312c31241f68f5f),
            ("bc7-2-0", 7, 2 * 8, 0xb3e3f4cd147b0999),
            ("bc7-16-1", 7, 16 * 8, 0x013a7ec1ceec8b71),
            ("bc7-8192-3", 7, 8192 * 8, 0x0b54870998ec2325),
        ];
        for (stem, mode, out_len, want_hash) in cases {
            let src = std::fs::read(dir.join(format!("{stem}.bin"))).unwrap();
            let mut dst = vec![0u8; out_len];
            decode(mode, &src, &mut dst, tables).unwrap_or_else(|e| panic!("{stem}: {e}"));
            assert_eq!(fnv1a(&dst), want_hash, "{stem} output diverged from baseline");
        }
        let bad_src = std::fs::read(dir.join("bc7-700-2.bin")).unwrap();
        let mut bad_dst = vec![0u8; 700 * 8];
        assert_eq!(decode(7, &bad_src, &mut bad_dst, tables), Err(Error("unused BC7 control code")));

        // Verify fixed-size bc1_color/bc4_alpha and every BC6H mode 1..=14 / BC7 mode 0..=7.
        let ep6 = [19u8, 45, 7, 29, 12, 31];
        let ix16: [u8; 16] = std::array::from_fn(|p| ((p * 5 + 3) & 7) as u8);
        let (mut got1, mut got4) = ([0u8; 8], [0u8; 8]);
        bc1_color(&ep6, &ix16, &mut got1);
        let c0 = u16::from(ep6[0]) << 11 | u16::from(ep6[1]) << 5 | u16::from(ep6[2]);
        let c1 = u16::from(ep6[3]) << 11 | u16::from(ep6[4]) << 5 | u16::from(ep6[5]);
        let ref_bits1 = (0..16).fold(0u32, |acc, p| acc | (BC1_CODE[usize::from(ix16[p] & 3)] << (2 * p)));
        assert_eq!(&got1[..2], &c0.to_le_bytes());
        assert_eq!(&got1[2..4], &c1.to_le_bytes());
        assert_eq!(&got1[4..8], &ref_bits1.to_le_bytes());
        bc4_alpha(0x2A, 0xD5, &ix16, &mut got4);
        let ref_v4 = u64::from(0x2Au8 ^ 0x80)
            | (u64::from(0xD5u8 ^ 0x80) << 8)
            | (0..16).fold(0u64, |acc, p| acc | (BC4_CODE[usize::from(ix16[p] & 7)] << (16 + 3 * p)));
        assert_eq!(got4, ref_v4.to_le_bytes());

        // Exercise all 14 BC6H modes (1..=14) and all 8 BC7 modes (0..=7) end-to-end.
        let bc6_recs: Vec<[u8; 32]> = (1u8..=14)
            .map(|m| {
                let mut r = [0u8; 32];
                for (k, slot) in r.iter_mut().enumerate().take(30) {
                    *slot = (m.wrapping_mul(17).wrapping_add((k as u8).wrapping_mul(11))) & 0x7F;
                }
                r[30] = m - 1;
                r[31] = (m * 3) & 31;
                r
            })
            .collect();
        let mut bc6_out = [[0u8; 16]; 14];
        bc6(bc6_recs.iter().map(|r| &r[..]), bc6_out.iter_mut().map(|b| &mut b[..]), tables.bc6).unwrap();
        assert!(bc6_out.iter().all(|b| b.iter().any(|&x| x != 0)));

        let mut bc7_recs = Vec::new();
        for mode in 0u16..8 {
            let idx = tables.ctrl7.iter().position(|&c| c != 0 && (c & 7) == mode).unwrap();
            let mut r = [0u8; 32];
            for (k, slot) in r.iter_mut().enumerate().take(30) {
                *slot = ((mode as u8 + 1).wrapping_mul(13).wrapping_add((k as u8).wrapping_mul(7))) & 0x3F;
            }
            r[30] = (idx & 31) as u8;
            r[31] = ((idx >> 5) & 15) as u8;
            bc7_recs.push(r);
        }
        let mut bc7_out = [[0u8; 16]; 8];
        bc7(bc7_recs.iter().map(|r| &r[..]), bc7_out.iter_mut().map(|b| &mut b[..]), tables).unwrap();
        for (mode, block) in bc7_out.iter().enumerate() {
            assert_eq!(block[0].trailing_zeros() as usize, mode, "BC7 mode {mode} header bit");
        }
    }
}

