/**
 * ui.ts — TwixT board renderer and touch/click handler.
 *
 * Renders onto an HTML Canvas element.
 * Board layout: 24×24 grid of equally-spaced nodes.
 *   - WHITE plays left↔right borders (excluded from play: x=0, x=23)
 *     actually WHITE connects y=0↔y=23 borders; restricted from x=0,x=23.
 *     Wait: WHITE is restricted from x=0 and x=23 border *columns*,
 *     and wins by connecting y=0 row to y=23 row.
 *   - BLACK is restricted from y=0 and y=23 border *rows*,
 *     and wins by connecting x=0 col to x=23 col.
 *
 * Visual convention (matching common TwixT board diagrams):
 *   - The board is drawn with x increasing right, y increasing down.
 *   - Border zones are tinted to show they are off-limits.
 *
 * Touch UX: drag-to-place with floating callout.
 *   - touchstart  → snap to nearest legal cell, show preview callout
 *   - touchmove   → callout tracks the finger
 *   - touchend    → commit the move
 *   - touchcancel → discard
 * Desktop: single click / pointerdown as before.
 */

import { Game, Point, SIZE, BLACK, WHITE, allLinks, pt, ptToString } from './twixt.js';

// -------------------------------------------------------------------------
// Colours
// -------------------------------------------------------------------------

// Color scheme: light sage-green playing field with dark outer ring.
// Blue (#0a3c96) for human (BLACK), orange (#b04800) for AI (WHITE).
// High contrast on light board; works for deuteranopia, protanopia, tritanopia.
const COLORS = {
  // Dark outer ring where labels live
  bg:              '#2c3428',
  // Light sage-green playing field (drawn as filled rect over bg)
  boardField:      '#d8ecc4',
  // Subtle grid lines on the light field
  grid:            'rgba(0,30,0,0.10)',
  // Anbindungslinien (strategic guiding lines)
  guideLine:       'rgba(0,50,0,0.30)',
  // Left/right strips = BLACK's goal edges → tinted blue
  borderZoneBlack: 'rgba(0,80,200,0.12)',
  // Top/bottom strips = WHITE's goal edges → tinted orange
  borderZoneWhite: 'rgba(200,100,0,0.12)',
  // Accent lines along goal edges
  borderLineBlack: '#005ab5',
  borderLineWhite: '#b06000',
  // Empty intersection dots on the light board
  node:            'rgba(0,40,0,0.45)',
  nodeHover:       '#1a50b0',
  // Human player = blue
  pegBlack:        '#0a3c96',
  pegBlackRim:     '#6090e0',
  // AI player = orange
  pegWhite:        '#b04800',
  pegWhiteRim:     '#e0a040',
  linkBlack:       '#0a3c96',
  linkWhite:       '#b04800',
  lastMove:        '#cc2040',   // red ring for most-recent move
  winLine:         '#8820a8',   // purple
};

// -------------------------------------------------------------------------
// BoardUI
// -------------------------------------------------------------------------

export interface UICallbacks {
  onMove: (p: Point) => void;
}

export class BoardUI {
  private canvas:  HTMLCanvasElement;
  private ctx:     CanvasRenderingContext2D;
  private cb:      UICallbacks;

  private cellSize = 0;
  private padLeft  = 0;
  private padTop   = 0;

  private hoveredCell: Point | null = null;
  /** Non-null while a touch drag is in progress; holds the snapped target cell. */
  private dragCell:    Point | null = null;
  private game: Game | null = null;
  private enabled = false;   // whether human can tap

  constructor(canvas: HTMLCanvasElement, cb: UICallbacks) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d')!;
    this.cb     = cb;

    this._setupEvents();
    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  setGame(game: Game, enabled: boolean): void {
    this.game    = game;
    this.enabled = enabled;
    this._resize(); // re-measure in case the canvas was hidden (size=0) during init
  }

  setEnabled(enabled: boolean): void {
    this.enabled = enabled;
  }

  // -------------------------------------------------------------------------
  // Layout helpers
  // -------------------------------------------------------------------------

  private _resize(): void {
    const container = this.canvas.parentElement!;
    const size = Math.min(container.clientWidth, container.clientHeight, window.innerWidth, window.innerHeight - 56);
    this.canvas.width  = size;
    this.canvas.height = size;

    const margin = Math.max(size * 0.055, 14);
    this.padLeft  = margin;
    this.padTop   = margin;
    this.cellSize = (size - 2 * margin) / (SIZE - 1);

    this.render();
  }

  private _toCanvas(p: Point): [number, number] {
    return [
      this.padLeft  + p.x * this.cellSize,
      this.padTop   + p.y * this.cellSize,
    ];
  }

  private _fromCanvas(cx: number, cy: number): Point | null {
    const x = Math.round((cx - this.padLeft)  / this.cellSize);
    const y = Math.round((cy - this.padTop)   / this.cellSize);
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return null;
    return pt(x, y);
  }

  /**
   * Convert a client-coordinate touch/pointer position to the nearest legal
   * board cell, or null if out of bounds or illegal.
   */
  private _snapToLegal(clientX: number, clientY: number, canvasYOffset = 0): Point | null {
    const r = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width  / r.width;
    const scaleY = this.canvas.height / r.height;
    const p = this._fromCanvas(
      (clientX - r.left) * scaleX,
      (clientY - r.top)  * scaleY + canvasYOffset,
    );
    return p && this._isLegalForHuman(p) ? p : null;
  }

  // -------------------------------------------------------------------------
  // Input events
  // -------------------------------------------------------------------------

  private _setupEvents(): void {
    // ---- Desktop: hover highlight ----------------------------------------
    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.enabled) return;
      const r = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width  / r.width;
      const scaleY = this.canvas.height / r.height;
      const p = this._fromCanvas(
        (e.clientX - r.left) * scaleX,
        (e.clientY - r.top)  * scaleY,
      );
      this.hoveredCell = p;
      this.render();
    });

    this.canvas.addEventListener('pointerleave', () => {
      this.hoveredCell = null;
      this.render();
    });

    // ---- Desktop: single click / stylus ----------------------------------
    // Keep pointerdown for mouse/stylus. Skip touch (handled below).
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'touch') return; // handled by touchstart/touchend
      e.preventDefault();
      if (!this.enabled || !this.game) return;
      const p = this._snapToLegal(e.clientX, e.clientY);
      if (p) this.cb.onMove(p);
    });

    // ---- Mobile: drag-to-place with callout indicator --------------------
    //
    // On touchstart: snap to nearest legal cell, show preview callout.
    // On touchmove:  update snapped cell as finger slides (only re-render
    //                when the snapped cell actually changes, for performance).
    // On touchend:   commit the move at the last snapped cell.
    // On touchcancel: discard (e.g. system gesture / incoming call).
    //
    // All handlers use { passive: false } so e.preventDefault() works,
    // preventing browser scroll/zoom on the canvas.

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      if (!this.enabled || !this.game) return;
      const t = e.changedTouches[0];
      this.dragCell = this._snapToLegal(t.clientX, t.clientY, -Math.max(this.cellSize * 2, 49));
      this.render();
    }, { passive: false });

    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (!this.enabled || !this.game) return;
      const t = e.changedTouches[0];
      const next = this._snapToLegal(t.clientX, t.clientY, -Math.max(this.cellSize * 2, 65));
      // Only re-render when the snapped cell changes (avoids redundant canvas redraws).
      if (next?.x !== this.dragCell?.x || next?.y !== this.dragCell?.y) {
        this.dragCell = next;
        this.render();
      }
    }, { passive: false });

    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      const cell = this.dragCell;
      this.dragCell = null;
      this.render();          // clear callout before the board updates
      if (cell) this.cb.onMove(cell);
    }, { passive: false });

    this.canvas.addEventListener('touchcancel', () => {
      this.dragCell = null;
      this.render();
    });
  }

  private _isLegalForHuman(p: Point): boolean {
    if (!this.game) return false;
    return this.game.legalPlays().contains(p);
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  render(): void {
    if (!this.game) return;
    const { ctx, canvas, cellSize } = this;
    const cs = cellSize;
    const pegR = Math.max(cs * 0.30, 4);

    // 1. Dark outer ring (label area)
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 2. Light playing field, then overlays in order
    this._drawBoardField();
    this._drawBorderZones();
    this._drawGuideLines();
    this._drawGrid();
    this._drawLabels();
    this._drawLinks();
    this._drawNodes(pegR);
    // Coordinate tooltip for hover (desktop) — drawn above nodes, below drag callout.
    if (this.hoveredCell && this.enabled && !this.dragCell && this._isLegalForHuman(this.hoveredCell)) {
      const [cx, cy] = this._toCanvas(this.hoveredCell);
      this._drawCoordLabel(cx, cy - pegR * 2.4, ptToString(this.hoveredCell));
    }
    // Offset peg preview drawn last so it appears on top of everything.
    if (this.dragCell) this._drawDragCallout(pegR);
  }

  /** Fill the inner playing area with the light board-field colour. */
  private _drawBoardField(): void {
    const { ctx, cellSize } = this;
    const pad = cellSize * 0.5;
    const [x0, y0] = this._toCanvas(pt(0, 0));
    const [xN, yN] = this._toCanvas(pt(SIZE - 1, SIZE - 1));
    ctx.fillStyle = COLORS.boardField;
    ctx.fillRect(x0 - pad, y0 - pad, (xN - x0) + 2 * pad, (yN - y0) + 2 * pad);
  }

  /**
   * Draw the Anbindungslinien (strategic guiding lines).
   *
   * Each corner of the inner playable area (B2, W2, B23, W23 in board notation)
   * spawns two lines at 1:2 and 2:1 slopes — the two extreme knight-move
   * trajectories that can still reach the far baseline.  That gives 8 lines
   * total (2 per corner × 4 corners), related by 90° rotation symmetry.
   *
   * In 0-indexed board coordinates the 4 anchor points and their targets are:
   *   TL (1, 1)  → (21,11) slope½   and → (11,21) slope2
   *   TR (22, 1) → ( 2,11) slope½   and → (12,21) slope2
   *   BR (22,22) → ( 2,12) slope½   and → (12, 2) slope2
   *   BL ( 1,22) → (21,12) slope½   and → (11, 2) slope2
   */
  private _drawGuideLines(): void {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = COLORS.guideLine;
    ctx.lineWidth   = 1.5;
    ctx.lineCap     = 'round';
    ctx.setLineDash([5, 7]);

    const lines: [Point, Point][] = [
      // Top-left corner (B2)
      [pt( 1,  1), pt(21, 11)],
      [pt( 1,  1), pt(11, 21)],
      // Top-right corner (W2)
      [pt(22,  1), pt( 2, 11)],
      [pt(22,  1), pt(12, 21)],
      // Bottom-right corner (W23)
      [pt(22, 22), pt( 2, 12)],
      [pt(22, 22), pt(12,  2)],
      // Bottom-left corner (B23)
      [pt( 1, 22), pt(21, 12)],
      [pt( 1, 22), pt(11,  2)],
    ];

    for (const [a, b] of lines) {
      const [ax, ay] = this._toCanvas(a);
      const [bx, by] = this._toCanvas(b);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
    ctx.restore();
  }

  private _drawLabels(): void {
    const { ctx, canvas, cellSize } = this;
    const fontSize = Math.max(Math.round(cellSize * 0.5), 7);
    ctx.font         = `${fontSize}px 'Courier Prime', monospace`;
    ctx.fillStyle    = 'rgba(255,255,255,1.0)';
    ctx.textBaseline = 'middle';

    const cols    = 'ABCDEFGHIJKLMNOPQRSTUVWX';
    const halfPad = this.padLeft / 2;

    // Column labels (A–X) above and below the board
    ctx.textAlign = 'center';
    for (let x = 0; x < SIZE; x++) {
      const [cx] = this._toCanvas(pt(x, 0));
      ctx.fillText(cols[x], cx, halfPad);                  // top
      ctx.fillText(cols[x], cx, canvas.height - halfPad);  // bottom
    }

    // Row labels (1–24) left and right of the board
    for (let y = 0; y < SIZE; y++) {
      const [, cy] = this._toCanvas(pt(0, y));
      ctx.textAlign = 'center';
      ctx.fillText(String(y + 1), halfPad, cy);                  // left
      ctx.fillText(String(y + 1), canvas.width - halfPad, cy);   // right
    }
  }

  private _drawBorderZones(): void {
    const { ctx } = this;
    const cs = this.cellSize;
    const hw = Math.max(cs * 0.15, 2);  // half-width of edge accent line

    const [x0, y0] = this._toCanvas(pt(0, 0));
    const [xN, yN] = this._toCanvas(pt(SIZE - 1, SIZE - 1));

    // Top & bottom strips = WHITE's goal edges (orange)
    ctx.fillStyle = COLORS.borderZoneWhite;
    ctx.fillRect(x0, y0, xN - x0, cs * 0.5);
    ctx.fillRect(x0, yN - cs * 0.5, xN - x0, cs * 0.5);

    // Left & right strips = BLACK's goal edges (blue)
    ctx.fillStyle = COLORS.borderZoneBlack;
    ctx.fillRect(x0, y0, cs * 0.5, yN - y0);
    ctx.fillRect(xN - cs * 0.5, y0, cs * 0.5, yN - y0);

    // Accent lines along each goal edge
    ctx.lineWidth = hw;
    ctx.strokeStyle = COLORS.borderLineWhite;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(xN, y0); ctx.stroke();  // top
    ctx.beginPath(); ctx.moveTo(x0, yN); ctx.lineTo(xN, yN); ctx.stroke();  // bottom

    ctx.strokeStyle = COLORS.borderLineBlack;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, yN); ctx.stroke();  // left
    ctx.beginPath(); ctx.moveTo(xN, y0); ctx.lineTo(xN, yN); ctx.stroke();  // right
  }

  private _drawGrid(): void {
    const { ctx } = this;
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth   = 0.5;

    for (let i = 0; i < SIZE; i++) {
      const [x0, y0] = this._toCanvas(pt(i, 0));
      const [, yN]   = this._toCanvas(pt(i, SIZE - 1));
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, yN); ctx.stroke();

      const [xS, yS] = this._toCanvas(pt(0, i));
      const [xE]     = this._toCanvas(pt(SIZE - 1, i));
      ctx.beginPath(); ctx.moveTo(xS, yS); ctx.lineTo(xE, yS); ctx.stroke();
    }
  }

  private _drawLinks(): void {
    if (!this.game) return;
    const { ctx } = this;
    const cs = this.cellSize;

    for (const { p1, p2, color } of allLinks(this.game)) {
      const [x1, y1] = this._toCanvas(p1);
      const [x2, y2] = this._toCanvas(p2);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = color === BLACK ? COLORS.linkBlack : COLORS.linkWhite;
      ctx.lineWidth   = Math.max(cs * 0.14, 2);
      ctx.lineCap     = 'round';
      ctx.stroke();
    }
  }

  private _drawNodes(pegR: number): void {
    if (!this.game) return;
    const { ctx, game } = this;
    const lastMove = game.history.length > 0
      ? game.history[game.history.length - 1]
      : null;

    for (let x = 0; x < SIZE; x++) {
      for (let y = 0; y < SIZE; y++) {
        const p = pt(x, y);
        const [cx, cy] = this._toCanvas(p);
        const isBlack = game.pegs[BLACK][x * SIZE + y];
        const isWhite = game.pegs[WHITE][x * SIZE + y];
        const isLast  = lastMove && lastMove !== 'swap'
          && (lastMove as Point).x === x && (lastMove as Point).y === y;
        const isHover = this.hoveredCell && this.hoveredCell.x === x && this.hoveredCell.y === y;

        if (isBlack || isWhite) {
          const fill = isBlack ? COLORS.pegBlack  : COLORS.pegWhite;
          const rim  = isBlack ? COLORS.pegBlackRim : COLORS.pegWhiteRim;
          ctx.beginPath();
          ctx.arc(cx, cy, pegR, 0, 2 * Math.PI);
          ctx.fillStyle = fill;
          ctx.fill();
          ctx.strokeStyle = isLast ? COLORS.lastMove : rim;
          ctx.lineWidth   = isLast ? Math.max(pegR * 0.35, 2) : 1;
          ctx.stroke();
        } else {
          // Empty node — slightly larger for good visibility on the light field
          const r = pegR * 0.38;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, 2 * Math.PI);
          ctx.fillStyle = isHover && this._isLegalForHuman(p)
            ? COLORS.nodeHover : COLORS.node;
          ctx.fill();
        }
      }
    }
  }

  /**
   * Draw the offset peg preview while a touch drag is active.
   *
   * The peg is drawn 2 cell-widths above the snapped board position so it is
   * never hidden under the fingertip.  This is the same convention used by Go
   * apps on iOS.  The peg drawn here IS the peg that will be placed — no stem
   * or lollipop.
   */
  private _drawDragCallout(pegR: number): void {
    if (!this.dragCell || !this.game) return;
    const { ctx, game } = this;
    const turn = game.turn;
    const fill = turn === BLACK ? COLORS.pegBlack : COLORS.pegWhite;
    const rim  = turn === BLACK ? COLORS.pegBlackRim : COLORS.pegWhiteRim;

    const [cx, cy] = this._toCanvas(this.dragCell);

    ctx.beginPath();
    ctx.arc(cx, cy, pegR, 0, 2 * Math.PI);
    ctx.fillStyle   = fill;
    ctx.fill();
    ctx.strokeStyle = rim;
    ctx.lineWidth   = 1;
    ctx.stroke();

    // Coordinate label below the floating peg (between peg and finger).
    this._drawCoordLabel(cx, cy + pegR * 2.2, ptToString(this.dragCell));
  }

  /** Draw a small dark pill with a coordinate label centred at (cx, cy). */
  private _drawCoordLabel(cx: number, cy: number, label: string): void {
    const { ctx } = this;
    const fs = Math.max(Math.round(this.cellSize * 0.52), 9);
    ctx.save();
    ctx.font         = `bold ${fs}px 'Courier Prime', monospace`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';

    const tw  = ctx.measureText(label).width;
    const pad = Math.max(fs * 0.3, 3);
    const bw  = tw + pad * 2;
    const bh  = fs + pad * 1.5;

    ctx.fillStyle = 'rgba(10,20,40,0.85)';
    ctx.beginPath();
    ctx.rect(cx - bw / 2, cy - bh / 2, bw, bh);
    ctx.fill();

    ctx.fillStyle = '#e8f4fd';
    ctx.fillText(label, cx, cy);
    ctx.restore();
  }
}
