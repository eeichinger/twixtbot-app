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

  // Parse moves: ;B[hd] or ;W[swap] or ;B[resign] or ;W[tt] (resign/pass sentinel)
  const moveRegex = /;[BW]\[([^\]]*)\]/gi;
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

/** Return a human-readable result string, e.g. "Black wins", "White wins", "Draw". */
export function formatResult(result: string): string {
  if (result === 'B+' || result.startsWith('B+')) return 'Black wins';
  if (result === 'W+' || result.startsWith('W+')) return 'White wins';
  if (result === '0') return 'Draw';
  return result || '—';
}
