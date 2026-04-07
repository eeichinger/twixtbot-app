/**
 * lg-sgf.ts — Parser for Little Golem TwixT SGF files (.tsgf).
 *
 * LG uses Go-style two-letter coordinate encoding internally:
 *   First letter  = column (a=0 … x=23)
 *   Second letter = row    (a=0 … x=23)
 * So "hd" → column h (x=7), row d (y=3) → the app's Point {x:7, y:3}.
 *
 * This matches the existing ptFromString / ptToString convention:
 *   ptFromString("h4") = {x:7, y:3}  and  lgCoordToPoint("hd") = {x:7, y:3}
 */

import { pt } from './twixt.js';
import type { Point, MoveRecord } from './twixt.js';

export interface ParsedGame {
  id: string;
  blackPlayer: string;
  whitePlayer: string;
  boardSize: number;
  /** Raw result string from SGF: "B+", "W+", "0" (draw), "?" etc. */
  result: string;
  moves: MoveRecord[];
}

/** Convert a two-letter LG coordinate like "hd" to a board Point. */
function lgCoordToPoint(coord: string): Point {
  return pt(
    coord.charCodeAt(0) - 97,  // 'a' = 0
    coord.charCodeAt(1) - 97,
  );
}

/**
 * Parse the raw text of a LG .tsgf file and return a ParsedGame.
 *
 * @param text  Full text content of the .tsgf file.
 * @param id    LG game ID to embed in the result (optional).
 */
export function parseTSGF(text: string, id = ''): ParsedGame {
  const game: ParsedGame = {
    id,
    blackPlayer: '?',
    whitePlayer: '?',
    boardSize: 24,
    result: '?',
    moves: [],
  };

  const pbMatch = text.match(/PB\[([^\]]+)\]/);
  if (pbMatch) game.blackPlayer = pbMatch[1].trim();

  const pwMatch = text.match(/PW\[([^\]]+)\]/);
  if (pwMatch) game.whitePlayer = pwMatch[1].trim();

  const szMatch = text.match(/SZ\[(\d+)\]/);
  if (szMatch) game.boardSize = parseInt(szMatch[1], 10);

  const reMatch = text.match(/RE\[([^\]]*)\]/);
  if (reMatch) game.result = reMatch[1].trim();

  // Parse moves: ;B[hd] / ;W[hd] (standard SGF) or ;b[hd] / ;r[hd] (LG uses r for Red/White)
  const moveRegex = /;[BWR]\[([^\]]*)\]/gi;
  let match;
  while ((match = moveRegex.exec(text)) !== null) {
    const raw = match[1].toLowerCase();
    if (raw === 'swap') {
      game.moves.push('swap');
    } else if (raw === 'resign' || raw === 'tt' || raw === '') {
      // Resign or pass — marks end of play; not a real move
      break;
    } else if (/^[a-x]{2}$/.test(raw)) {
      game.moves.push(lgCoordToPoint(raw));
    }
    // Any other value is silently ignored
  }

  return game;
}

/** Convert a board Point to a two-letter LG coordinate like "hd". */
function pointToLgCoord(p: Point): string {
  return String.fromCharCode(97 + p.x) + String.fromCharCode(97 + p.y);
}

export interface SerializeTsgfOptions {
  /** Name of the TSGF Black player (first mover). */
  blackPlayer?: string;
  /** Name of the TSGF White player (second mover). */
  whitePlayer?: string;
  /**
   * SGF result string: "B+" (Black/first-mover wins), "W+" (White/second-mover wins),
   * "0" (draw), or "?" (unknown / game in progress).
   */
  result?: string;
}

/**
 * Serialize a move history to a Little Golem–compatible .tsgf string.
 *
 * IMPORTANT — color mapping:
 *   In the webapp WHITE is the first mover; in SGF/TSGF the first mover is "B" (Black).
 *   So the caller must swap the semantic meaning when building the options:
 *     webapp WHITE (first mover)  → TSGF PB / "B[…]" moves
 *     webapp BLACK (second mover) → TSGF PW / "W[…]" moves
 *   The result string must follow the same convention ("B+" = first mover won).
 */
export function serializeTSGF(moves: MoveRecord[], options: SerializeTsgfOptions = {}): string {
  const pb = options.blackPlayer ?? '?';
  const pw = options.whitePlayer ?? '?';
  const re = options.result     ?? '?';

  let sgf = `(;GM[21]FF[4]SZ[24]RU[PP]PB[${pb}]PW[${pw}]RE[${re}]`;

  for (let i = 0; i < moves.length; i++) {
    const color = i % 2 === 0 ? 'B' : 'W';
    const move = moves[i];
    if (move === 'swap') {
      sgf += `;${color}[swap]`;
    } else {
      sgf += `;${color}[${pointToLgCoord(move as Point)}]`;
    }
  }

  sgf += ')';
  return sgf;
}

/**
 * Return a human-readable result string.
 * Handles both SGF format ("B+", "W+", "0") and game-list format
 * ("win", "lost", "draw" — relative to the player whose list was fetched).
 */
export function formatResult(result: string): string {
  if (result === 'B+' || result.startsWith('B+')) return 'Black wins';
  if (result === 'W+' || result.startsWith('W+')) return 'White wins';
  if (result === '0' || result === 'draw') return 'Draw';
  if (result === 'win') return 'Win';
  if (result === 'lost') return 'Lost';
  return result || '—';
}
