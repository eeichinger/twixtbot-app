/**
 * lg-api.ts — Little Golem read-only API client.
 *
 * CORS: LG does not send CORS headers, so browser fetch() is blocked.
 * PROXY_PREFIX below defaults to corsproxy.io (zero infrastructure, public
 * read-only data). Replace with a Cloudflare Worker or other proxy if needed.
 *
 * TODO: Confirm LG player-game-list URL format against live site and adjust
 *       parseGameListHtml() column indices as needed.
 */

import { parseTSGF, type ParsedGame } from './lg-sgf.js';

// ---------------------------------------------------------------------------
// CORS proxy — change this constant to switch proxy implementations
// ---------------------------------------------------------------------------

// corsproxy.io: free, zero-config, public data only
const PROXY_PREFIX = 'https://corsproxy.io/?url=';

const LG_BASE = 'https://www.littlegolem.net';

function proxied(url: string): string {
  return PROXY_PREFIX + encodeURIComponent(url);
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface GameSummary {
  id: string;
  blackPlayer: string;
  whitePlayer: string;
  result: string;   // Raw LG result: "B+", "W+", "?" etc.
  boardSize: number;
  moveCount: number;
}

// ---------------------------------------------------------------------------
// Game by ID
// ---------------------------------------------------------------------------

/**
 * Fetch and parse a single game's SGF from Little Golem.
 * Throws on network error or non-2xx response.
 */
export async function fetchGame(id: string): Promise<ParsedGame> {
  const url = `${LG_BASE}/servlet/sgf/${id}/game${id}.tsgf`;
  let res: Response;
  try {
    res = await fetch(proxied(url));
  } catch (err) {
    throw new Error(`Network error: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  const text = await res.text();
  if (!text.includes('GM[21]')) {
    throw new Error('Not a TwixT game (GM[21] not found)');
  }
  return parseTSGF(text, id);
}

// ---------------------------------------------------------------------------
// Player game list
// ---------------------------------------------------------------------------

/**
 * Fetch a player's finished TwixT PP games from Little Golem.
 *
 * LG serves a plain-text tab-separated game list at:
 *   /jsp/info/player_game_list_txt.jsp?gtid=twixt.PP&plid=<name>
 *
 * If that endpoint returns nothing useful, falls back to HTML scraping of
 * the player's game page to extract at least the game IDs.
 */
export async function fetchPlayerGames(playerName: string): Promise<GameSummary[]> {
  const txtUrl = `${LG_BASE}/jsp/info/player_game_list_txt.jsp?gtid=twixt.PP&plid=${encodeURIComponent(playerName)}`;
  let res: Response;
  try {
    res = await fetch(proxied(txtUrl));
  } catch (err) {
    throw new Error(`Network error: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!res.ok) throw new Error(`Server returned ${res.status}`);

  const body = await res.text();

  // Try plain-text format first (tab-separated)
  const txtGames = parseGameListTxt(body);
  if (txtGames.length > 0) return txtGames;

  // Fall back to HTML parsing (extract game IDs from href links)
  const htmlGames = parseGameListHtml(body);
  if (htmlGames.length > 0) return htmlGames;

  // The page may have returned HTML for an unknown player; check for a "no games" signal
  if (body.includes('No games') || body.includes('no games') || body.trim().length < 50) {
    return [];
  }

  throw new Error('Could not parse game list — player not found or format changed');
}

// ---------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------

/**
 * Parse LG's tab-separated game list format.
 *
 * Best-guess column layout (adjust indices if LG changes the format):
 *   0: game ID
 *   1: game type string
 *   2: black player name
 *   3: white player name
 *   4: result ("B+", "W+", "0", "*" = ongoing)
 *   5+: other fields
 */
function parseGameListTxt(text: string): GameSummary[] {
  const games: GameSummary[] = [];
  for (const line of text.split('\n')) {
    const cols = line.trim().split('\t');
    if (cols.length < 5) continue;
    const id = cols[0]?.trim();
    if (!id || !/^\d+$/.test(id)) continue;  // must be a numeric ID
    const result = cols[4]?.trim() ?? '?';
    if (result === '*') continue;  // skip ongoing games (no SGF to replay)
    games.push({
      id,
      blackPlayer: cols[2]?.trim() || '?',
      whitePlayer: cols[3]?.trim() || '?',
      result,
      boardSize: 24,
      moveCount: 0,
    });
  }
  return games;
}

/**
 * Fall-back HTML parser: extract game IDs from href links.
 * Produces minimal GameSummary objects (names/result will be "?").
 */
function parseGameListHtml(html: string): GameSummary[] {
  const games: GameSummary[] = [];
  const seen = new Set<string>();
  // Links like /jsp/game/game.jsp?gid=12345
  const re = /\/jsp\/game\/game\.jsp\?gid=(\d+)/g;
  let m;
  while ((m = re.exec(html)) !== null) {
    const id = m[1];
    if (!seen.has(id)) {
      seen.add(id);
      games.push({ id, blackPlayer: '?', whitePlayer: '?', result: '?', boardSize: 24, moveCount: 0 });
    }
  }
  return games;
}
