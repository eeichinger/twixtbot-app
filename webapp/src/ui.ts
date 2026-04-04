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
 */

import { Game, Point, SIZE, BLACK, WHITE, allLinks, pt } from './twixt.js';

// -------------------------------------------------------------------------
// Colours
// -------------------------------------------------------------------------

// Wong color-blind-safe palette: blue (#0072B2) for human (BLACK),
// orange (#E69F00) for AI (WHITE).  Works for deuteranopia, protanopia,
// tritanopia.  Background is bright white/cream.
const COLORS = {
  bg:              '#f0f4f8',
  grid:            '#b0bec5',
  // Left/right strips = BLACK's goal edges → tinted blue
  borderZoneBlack: 'rgba(0, 114, 178, 0.15)',
  // Top/bottom strips = WHITE's goal edges → tinted orange
  borderZoneWhite: 'rgba(230, 159, 0, 0.15)',
  // Accent lines along goal edges
  borderLineBlack: '#0072b2',
  borderLineWhite: '#e69f00',
  node:            '#90a4ae',
  nodeHover:       '#0072b2',
  // Human player = blue
  pegBlack:        '#0057b8',
  pegBlackRim:     '#56b4e9',
  // AI player = orange
  pegWhite:        '#c87800',
  pegWhiteRim:     '#f0c040',
  linkBlack:       '#0072b2',
  linkWhite:       '#e69f00',
  lastMove:        '#d55e00',   // vermillion — distinct from both blue & orange
  winLine:         '#cc79a7',   // reddish-purple
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

    const margin = Math.max(size * 0.04, 12);
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

  // -------------------------------------------------------------------------
  // Input events
  // -------------------------------------------------------------------------

  private _setupEvents(): void {
    this.canvas.addEventListener('pointermove', (e) => {
      if (!this.enabled) return;
      const r    = this.canvas.getBoundingClientRect();
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

    // Use touchstart (not pointerdown) as the primary tap handler on mobile.
    // iOS Safari does not reliably dispatch pointerdown on <canvas> elements
    // even with touch-action:none; touchstart is always delivered.
    const handleTap = (clientX: number, clientY: number) => {
      if (!this.enabled || !this.game) return;
      const r = this.canvas.getBoundingClientRect();
      const scaleX = this.canvas.width  / r.width;
      const scaleY = this.canvas.height / r.height;
      const p = this._fromCanvas(
        (clientX - r.left) * scaleX,
        (clientY - r.top)  * scaleY,
      );
      if (!p || !this._isLegalForHuman(p)) return;
      this.cb.onMove(p);
    };

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const t = e.changedTouches[0];
      handleTap(t.clientX, t.clientY);
    }, { passive: false });

    // Keep pointerdown for mouse/stylus on desktop.
    this.canvas.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'touch') return; // already handled by touchstart
      e.preventDefault();
      handleTap(e.clientX, e.clientY);
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
    const pegR = Math.max(cs * 0.28, 4);

    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    this._drawBorderZones();
    this._drawGrid();
    this._drawLinks();
    this._drawNodes(pegR);
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
      ctx.lineWidth   = Math.max(cs * 0.12, 2);
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
          // Empty node
          const r = pegR * 0.35;
          ctx.beginPath();
          ctx.arc(cx, cy, r, 0, 2 * Math.PI);
          ctx.fillStyle = isHover && this._isLegalForHuman(p)
            ? COLORS.nodeHover : COLORS.node;
          ctx.fill();
        }
      }
    }
  }
}
