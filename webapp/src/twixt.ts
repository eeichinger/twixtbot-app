/**
 * twixt.ts — TwixT game engine
 * Direct port of src/twixt.py (Python 3) to TypeScript.
 *
 * Board coordinates: Point(x, y)
 *   x = column (0..SIZE-1),  y = row (0..SIZE-1)
 *   WHITE connects y=0 to y=SIZE-1 (top/bottom borders)
 *   BLACK connects x=0 to x=SIZE-1 (left/right borders)
 *
 * Border restrictions:
 *   WHITE may not play at x=0 or x=SIZE-1
 *   BLACK may not play at y=0 or y=SIZE-1
 */

export const SIZE = 24;
export const BLACK = 0;
export const WHITE = 1;

// Link-array index bits (match Python constants)
const LINK_LONGY    = 4;   // set when (a.x+b.x) is odd
const LINK_DIFFSIGN = 2;   // set when the slope is negative

// All 8 knight-move directions
export const DLINKS: [number, number][] = [
  [-2,-1], [-1,-2], [1,-2], [2,-1],
  [ 2, 1], [ 1, 2], [-1, 2], [-2, 1],
];

// Cross-link lookup table (see any_crossing_links in Python)
// Each entry: [dlong0, dshort0,  dlong1, dshort1]  →  candidate endpoints c = a + dlong*[0] + dshort*[1], d = a + dlong*[2] + dshort*[3]
const CROSS_LINKS: [number,number,number,number][] = [
  [-1, 1,  1, 0],
  [ 0, 1,  2, 0],
  [ 1, 1,  3, 0],
  [ 0, 1,  1,-1],
  [ 0, 2,  1, 0],
  [ 1, 1,  2,-1],
  [ 1, 2,  2, 0],
  [ 0,-1,  1, 1],
  [ 1, 0,  2, 2],
];

// -------------------------------------------------------------------------
// Point
// -------------------------------------------------------------------------

export interface Point { readonly x: number; readonly y: number; }

export function pt(x: number, y: number): Point { return { x, y }; }

export function ptAdd(a: Point, b: [number, number]): Point {
  return { x: a.x + b[0], y: a.y + b[1] };
}

export function ptKey(p: Point): string {
  return `${p.x},${p.y}`;
}

export function ptFromKey(k: string): Point {
  const [x, y] = k.split(',').map(Number);
  return { x, y };
}

export function ptEqual(a: Point, b: Point): boolean {
  return a.x === b.x && a.y === b.y;
}

/** Parse a board-notation string like "a1" or "B12" into a Point. */
export function ptFromString(s: string): Point {
  const c = s[0];
  const x = c >= 'A' && c <= 'Z' ? c.charCodeAt(0) - 65 : c.charCodeAt(0) - 97;
  const y = parseInt(s.slice(1), 10) - 1;
  return { x, y };
}

export function ptToString(p: Point): string {
  return String.fromCharCode(97 + p.x) + (p.y + 1);
}

// -------------------------------------------------------------------------
// SelectSet — O(1) add / remove / contains  (port of twixt.SelectSet)
// -------------------------------------------------------------------------

export class SelectSet {
  private itemByIndex: Point[] = [];
  private indexByKey: Map<string, number> = new Map();

  add(p: Point): void {
    const k = ptKey(p);
    if (this.indexByKey.has(k)) throw new Error(`Duplicate element ${k}`);
    this.indexByKey.set(k, this.itemByIndex.length);
    this.itemByIndex.push(p);
  }

  remove(p: Point): void {
    const k = ptKey(p);
    const idx = this.indexByKey.get(k);
    if (idx === undefined) return;
    const last = this.itemByIndex.length - 1;
    if (idx === last) {
      this.itemByIndex.pop();
    } else {
      const moved = this.itemByIndex.pop()!;
      this.itemByIndex[idx] = moved;
      this.indexByKey.set(ptKey(moved), idx);
    }
    this.indexByKey.delete(k);
  }

  contains(p: Point): boolean {
    return this.indexByKey.has(ptKey(p));
  }

  get length(): number { return this.itemByIndex.length; }

  at(i: number): Point { return this.itemByIndex[i]; }

  [Symbol.iterator](): Iterator<Point> {
    let i = 0;
    const items = this.itemByIndex;
    return {
      next(): IteratorResult<Point> {
        if (i < items.length) return { value: items[i++], done: false };
        return { value: undefined as unknown as Point, done: true };
      }
    };
  }

  clone(): SelectSet {
    const copy = new SelectSet();
    copy.itemByIndex = [...this.itemByIndex];
    copy.indexByKey  = new Map(this.indexByKey);
    return copy;
  }
}

// -------------------------------------------------------------------------
// Link helpers
// -------------------------------------------------------------------------

/** Returns [arrayIndex, flatOffset] for the link between a and b owned by color. */
function getLinkIndex(a: Point, b: Point, color: number): [number, number] {
  let ix1 = color;
  if ((a.x + b.x) % 2 !== 0) ix1 += LINK_LONGY;
  const cx = (a.x + b.x) >> 1;
  const cy = (a.y + b.y) >> 1;
  if ((b.y - a.y) * (b.x - a.x) < 0) ix1 += LINK_DIFFSIGN;
  return [ix1, cx * SIZE + cy];
}

function getLink(links: Int8Array[], a: Point, b: Point, color: number): number {
  const [ix1, ix2] = getLinkIndex(a, b, color);
  return links[ix1][ix2];
}

function setLink(links: Int8Array[], a: Point, b: Point, color: number, value: number): void {
  const [ix1, ix2] = getLinkIndex(a, b, color);
  links[ix1][ix2] = value;
}

/** True if any link of `color` would cross the potential link from a→b. */
function anyCrossingLinks(links: Int8Array[], a: Point, b: Point, color: number): boolean {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dsx = (dx & 1) * dx;
  const dsy = (dy & 1) * dy;
  const dlx = (dx - dsx) >> 1;
  const dly = (dy - dsy) >> 1;

  for (const [cl0, cl1, cl2, cl3] of CROSS_LINKS) {
    const cx = a.x + dlx * cl0 + dsx * cl1;
    const cy = a.y + dly * cl0 + dsy * cl1;
    const dx2 = a.x + dlx * cl2 + dsx * cl3;
    const dy2 = a.y + dly * cl2 + dsy * cl3;
    if (cx < 0 || cx >= SIZE || cy < 0 || cy >= SIZE) continue;
    if (dx2 < 0 || dx2 >= SIZE || dy2 < 0 || dy2 >= SIZE) continue;
    if (getLink(links, { x: cx, y: cy }, { x: dx2, y: dy2 }, color)) return true;
  }
  return false;
}

// -------------------------------------------------------------------------
// Game
// -------------------------------------------------------------------------

/** A single item in reachableHistory: "x,y" or the sentinel "win". */
type ReachableKey = string;  // "x,y" | "win"

export type MoveRecord = Point | 'swap';

export class Game {
  pegs:  [Int8Array, Int8Array];   // [color][x*SIZE+y]
  links: Int8Array[];               // [8 arrays][x*SIZE+y]
  turn:  number;                    // BLACK=0, WHITE=1
  history: MoveRecord[];
  openPegs: [SelectSet, SelectSet]; // [color] → available points
  reachable: [Set<ReachableKey>, Set<ReachableKey>];
  reachableHistory: ReachableKey[][];
  private redoStack: MoveRecord[] = [];
  private _inRedo = false;

  constructor() {
    this.pegs  = [new Int8Array(SIZE*SIZE), new Int8Array(SIZE*SIZE)];
    this.links = Array.from({length: 8}, () => new Int8Array(SIZE*SIZE));
    this.turn  = WHITE;
    this.history = [];
    this.openPegs = [new SelectSet(), new SelectSet()];
    this.reachable = [new Set(), new Set()];
    this.reachableHistory = [];

    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        const p = pt(x, y);
        if (x !== 0 && x !== SIZE - 1) this.openPegs[WHITE].add(p);
        if (y !== 0 && y !== SIZE - 1) this.openPegs[BLACK].add(p);
      }
    }
  }

  static inbounds(p: Point): boolean {
    return p.x >= 0 && p.x < SIZE && p.y >= 0 && p.y < SIZE;
  }

  getPeg(p: Point, color: number): number {
    return this.pegs[color][p.x * SIZE + p.y];
  }

  getLink(a: Point, b: Point, color: number): number {
    return getLink(this.links, a, b, color);
  }

  legalPlays(): SelectSet {
    return this.openPegs[this.turn];
  }

  justWon(): boolean {
    return this.isWinning(1 - this.turn);
  }

  isWinning(color: number): boolean {
    return this.reachable[color].has('win');
  }

  play(move: MoveRecord): void {
    if (!this._inRedo) this.redoStack = [];
    if (move === 'swap') { this._playSwap(); return; }

    const { x, y } = move;

    // Place links to all existing friendly pegs that can be reached
    for (const [dx, dy] of DLINKS) {
      const q = pt(x + dx, y + dy);
      if (!Game.inbounds(q)) continue;
      if (!this.getPeg(q, this.turn)) continue;
      if (anyCrossingLinks(this.links, move, q, 1 - this.turn)) continue;
      setLink(this.links, move, q, this.turn, 1);
    }

    this.pegs[this.turn][x * SIZE + y] = 1;
    this.history.push(move);
    this.reachableHistory.push(this._updateAddReachable(move));
    this.turn = 1 - this.turn;

    this.openPegs[0].remove(move);
    this.openPegs[1].remove(move);
  }

  undo(): void {
    const umove = this.history[this.history.length - 1];
    if (umove === 'swap') {
      this._undoSwap();
      this.redoStack.push('swap');
      return;
    }

    const uturn = 1 - this.turn;
    const { x, y } = umove;

    this.pegs[uturn][x * SIZE + y] = 0;

    for (const [dx, dy] of DLINKS) {
      const lend = pt(x + dx, y + dy);
      if (Game.inbounds(lend)) setLink(this.links, umove, lend, uturn, 0);
    }

    this.history.pop();
    this.turn = uturn;

    const rh = this.reachableHistory.pop()!;
    for (const k of rh) {
      this.reachable[uturn].delete(k);
    }

    if (x !== 0 && x !== SIZE - 1) this.openPegs[WHITE].add(umove);
    if (y !== 0 && y !== SIZE - 1) this.openPegs[BLACK].add(umove);
    this.redoStack.push(umove);
  }

  /** Re-apply the most recently undone move. No-op if no undo to redo. */
  redo(): void {
    const move = this.redoStack[this.redoStack.length - 1];
    if (move === undefined) return;
    this.redoStack.pop();
    this._inRedo = true;
    this.play(move);
    this._inRedo = false;
  }

  get canRedo(): boolean { return this.redoStack.length > 0; }

  clone(): Game {
    const g = new Game();
    g.pegs  = [new Int8Array(this.pegs[0]), new Int8Array(this.pegs[1])];
    g.links = this.links.map(a => new Int8Array(a));
    g.turn  = this.turn;
    g.history = [...this.history];
    g.openPegs = [this.openPegs[0].clone(), this.openPegs[1].clone()];
    g.reachable = [new Set(this.reachable[0]), new Set(this.reachable[1])];
    g.reachableHistory = this.reachableHistory.map(r => [...r]);
    // redoStack intentionally not cloned — redo history is a UI concern
    return g;
  }

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  private _playSwap(): void {
    const a = this.history[0] as Point;
    const b = pt(a.y, a.x);
    this.pegs[WHITE][a.x * SIZE + a.y] = 0;
    this.pegs[BLACK][b.x * SIZE + b.y] = 1;
    this.history.push('swap');
    this.turn = WHITE;

    // Transpose reachable[WHITE] → reachable[BLACK]
    const newBlack: Set<ReachableKey> = new Set();
    for (const k of this.reachable[WHITE]) {
      if (k === 'win') { newBlack.add('win'); continue; }
      const p = ptFromKey(k);
      newBlack.add(ptKey(pt(p.y, p.x)));
    }
    this.reachable[WHITE] = new Set();
    this.reachable[BLACK] = newBlack;
    this.reachableHistory.push([]);
  }

  private _undoSwap(): void {
    const a = this.history[0] as Point;
    const b = pt(a.y, a.x);
    this.pegs[WHITE][a.x * SIZE + a.y] = 1;
    this.pegs[BLACK][b.x * SIZE + b.y] = 0;
    this.history.pop();
    this.turn = BLACK;

    const newWhite: Set<ReachableKey> = new Set();
    for (const k of this.reachable[BLACK]) {
      if (k === 'win') { newWhite.add('win'); continue; }
      const p = ptFromKey(k);
      newWhite.add(ptKey(pt(p.y, p.x)));
    }
    this.reachable[BLACK] = new Set();
    this.reachable[WHITE] = newWhite;
    this.reachableHistory.pop();
  }

  /**
   * Incremental reachability update after placing `move` for `this.turn`.
   * Called BEFORE flipping this.turn (so this.turn == the player who just moved).
   * Returns the list of keys added to reachable[this.turn] in this step.
   */
  private _updateAddReachable(move: Point): ReachableKey[] {
    const color = this.turn;
    const myReachable = this.reachable[color];
    const added: ReachableKey[] = [];

    // Is this peg at the far border?
    if (move.x === SIZE - 1 || move.y === SIZE - 1) {
      added.push(ptKey(move));
    } else {
      // Does it link to an already-reachable peg?
      for (const [dx, dy] of DLINKS) {
        const other = pt(move.x + dx, move.y + dy);
        if (!myReachable.has(ptKey(other))) continue;
        if (!getLink(this.links, move, other, color)) continue;
        added.push(ptKey(move));
        if (move.x === 0 || move.y === 0) {
          added.push('win');
          myReachable.add(ptKey(move));
          myReachable.add('win');
          return added;
        }
        break;
      }
    }

    if (added.length === 0) return added;

    myReachable.add(ptKey(move));
    const unvisited: Point[] = [move];

    while (unvisited.length > 0) {
      const chk = unvisited.pop()!;
      for (const [dx, dy] of DLINKS) {
        const other = pt(chk.x + dx, chk.y + dy);
        if (!Game.inbounds(other)) continue;
        const ok = ptKey(other);
        if (myReachable.has(ok)) continue;
        if (!this.getPeg(other, color)) continue;
        if (!getLink(this.links, chk, other, color)) continue;
        added.push(ok);
        unvisited.push(other);
        myReachable.add(ok);
        if (other.x === 0 || other.y === 0) {
          added.push('win');
          myReachable.add('win');
          return added;
        }
      }
    }
    return added;
  }
}

// -------------------------------------------------------------------------
// Replay a move history onto a fresh Game (used in worker.ts)
// -------------------------------------------------------------------------

export function replayHistory(history: MoveRecord[]): Game {
  const g = new Game();
  for (const m of history) g.play(m);
  return g;
}

/** Return all (p1, p2, color) links currently on the board (for rendering). */
export interface LinkDesc { p1: Point; p2: Point; color: number; }

export function allLinks(game: Game): LinkDesc[] {
  const out: LinkDesc[] = [];
  for (let ix1 = 0; ix1 < 8; ix1++) {
    const color    = ix1 & 1;
    const vertical = (ix1 & LINK_LONGY)    !== 0;
    const diffsign = (ix1 & LINK_DIFFSIGN) !== 0;

    for (let flat = 0; flat < SIZE * SIZE; flat++) {
      if (!game.links[ix1][flat]) continue;
      const cx = flat >> 5;        // Math.floor(flat / SIZE) but SIZE=32 is not 24... use division
      // flat = cx*SIZE + cy
      const _cx = Math.floor(flat / SIZE);
      const _cy = flat % SIZE;

      let p1: Point, p2: Point;
      if (vertical) {
        // xlo = cx, xhi = cx+1; ylo = cy-1, yhi = cy+1
        if (diffsign) {
          p1 = pt(_cx,   _cy + 1);
          p2 = pt(_cx + 1, _cy - 1);
        } else {
          p1 = pt(_cx,   _cy - 1);
          p2 = pt(_cx + 1, _cy + 1);
        }
      } else {
        // xlo = cx-1, xhi = cx+1; ylo = cy, yhi = cy+1
        if (diffsign) {
          p1 = pt(_cx - 1, _cy + 1);
          p2 = pt(_cx + 1, _cy);
        } else {
          p1 = pt(_cx - 1, _cy);
          p2 = pt(_cx + 1, _cy + 1);
        }
      }
      out.push({ p1, p2, color });
    }
  }
  return out;
}
