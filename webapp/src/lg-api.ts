/**
 * lg-api.ts — Little Golem read-only API client.
 *
 * CORS: LG does not send CORS headers, so browser fetch() is blocked.
 * PROXY_PREFIX defaults to corsproxy.io (zero infrastructure, public data).
 * Replace with a Cloudflare Worker or other proxy if needed.
 *
 * Player search is a two-step flow:
 *   1. searchPlayers(name)          → PlayerResult[] (name + numeric plid)
 *   2. fetchPlayerGamesByPlid(plid) → GameSummary[]
 *
 * Game-by-ID is a single step:
 *   fetchGame(id) → ParsedGame  (fetches and parses the .tsgf SGF file)
 */

import { parseTSGF, type ParsedGame } from './lg-sgf.js';

// ---------------------------------------------------------------------------
// CORS proxy — change this constant to switch proxy implementations
// ---------------------------------------------------------------------------

const PROXY_PREFIX = 'https://corsproxy.io/?url=';

const LG_BASE = 'https://www.littlegolem.net';

function proxied(url: string): string {
  return PROXY_PREFIX + encodeURIComponent(url);
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface PlayerResult {
  plid: string;    // numeric LG player ID, e.g. "2674"
  name: string;
  rating: string;  // display string, e.g. "1. kyu" — empty if unavailable
}

export interface GameSummary {
  id: string;
  blackPlayer: string;
  whitePlayer: string;
  result: string;   // Raw LG result: "B+", "W+", "?" etc.
  boardSize: number;
  moveCount: number;
}

// ---------------------------------------------------------------------------
// Step 1 — player name search
// ---------------------------------------------------------------------------

/**
 * Search Little Golem for TwixT players whose name contains `name`.
 * Uses the public player list page (no login required).
 *
 * URL: /jsp/info/player_list.jsp?gtvar=twixt_DEFAULT&filter=NAME
 *
 * HTML row format:
 *   <td><a href="player.jsp?plid=2674">Alan Hensel</a></td>
 *   <td>...</td>
 *   <td><span title="2235.3">1. kyu</span></td>
 */
export async function searchPlayers(name: string): Promise<PlayerResult[]> {
  const url = `${LG_BASE}/jsp/info/player_list.jsp?gtvar=twixt_DEFAULT&filter=${encodeURIComponent(name)}`;
  const res = await fetchProxied(url);
  const html = await res.text();
  return parsePlayerListHtml(html);
}

function parsePlayerListHtml(html: string): PlayerResult[] {
  const players: PlayerResult[] = [];
  // Each row: <a href="player.jsp?plid=NNNN">Name</a>
  const rowRe = /href="player\.jsp\?plid=(\d+)">([^<]+)<\/a>/g;
  let m;
  while ((m = rowRe.exec(html)) !== null) {
    const plid = m[1];
    const name = m[2].trim();
    // Try to find the rating span in the next ~200 chars after this match
    const snippet = html.slice(m.index, m.index + 300);
    const ratingM = snippet.match(/<span[^>]*>([^<]+)<\/span>/);
    const rating = ratingM ? ratingM[1].trim() : '';
    players.push({ plid, name, rating });
  }
  return players;
}

// ---------------------------------------------------------------------------
// Step 2 — game list for a player
// ---------------------------------------------------------------------------

/**
 * Fetch TwixT PP games for a player identified by their numeric LG player ID.
 *
 * Tries the plain-text export endpoint first (fast, structured); falls back
 * to HTML parsing if it returns a non-text response.
 *
 * Text URL: /jsp/info/player_game_list_txt.jsp?gtid=twixt.PP&plid=PLID
 * HTML URL: /jsp/info/player_game_list.jsp?gtid=twixt.PP&plid=PLID
 */
export async function fetchPlayerGamesByPlid(plid: string): Promise<GameSummary[]> {
  // Try plain-text endpoint first
  const txtUrl = `${LG_BASE}/jsp/info/player_game_list_txt.jsp?gtid=twixt.PP&plid=${encodeURIComponent(plid)}`;
  const res = await fetchProxied(txtUrl);
  const body = await res.text();

  const txtGames = parseGameListTxt(body);
  if (txtGames.length > 0) return txtGames;

  // Fall back to HTML
  const htmlGames = parseGameListHtml(body);
  if (htmlGames.length > 0) return htmlGames;

  // Empty response or player has no finished TwixT PP games
  return [];
}

// ---------------------------------------------------------------------------
// Game by ID
// ---------------------------------------------------------------------------

/**
 * Fetch and parse a single game's SGF from Little Golem.
 * URL: /servlet/sgf/{id}/game{id}.tsgf
 */
export async function fetchGame(id: string): Promise<ParsedGame> {
  const url = `${LG_BASE}/servlet/sgf/${id}/game${id}.tsgf`;
  const res = await fetchProxied(url);
  const text = await res.text();
  if (!text.includes('GM[21]')) {
    // Might be a login redirect or error page
    if (text.includes('<html') || text.includes('login')) {
      throw new Error('Game not accessible — LG may require login for this content');
    }
    throw new Error('Not a TwixT game (GM[21] not found)');
  }
  return parseTSGF(text, id);
}

// ---------------------------------------------------------------------------
// Shared fetch helper
// ---------------------------------------------------------------------------

async function fetchProxied(url: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(proxied(url));
  } catch (err) {
    throw new Error(`Network error: ${err instanceof Error ? err.message : String(err)}`);
  }
  if (!res.ok) throw new Error(`Server returned ${res.status} for ${url}`);
  return res;
}

// ---------------------------------------------------------------------------
// Game list parsers
// ---------------------------------------------------------------------------

/**
 * Parse tab-separated player_game_list_txt.jsp response.
 *
 * Best-guess column layout (adjust if LG changes format):
 *   0: game ID   1: game type   2: black player   3: white player
 *   4: result ("B+", "W+", "0", "*"=ongoing)   5+: other
 */
function parseGameListTxt(text: string): GameSummary[] {
  const games: GameSummary[] = [];
  for (const line of text.split('\n')) {
    const cols = line.trim().split('\t');
    if (cols.length < 5) continue;
    const id = cols[0]?.trim();
    if (!id || !/^\d+$/.test(id)) continue;
    const result = cols[4]?.trim() ?? '?';
    if (result === '*') continue;  // skip ongoing games
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
 * Produces minimal GameSummary objects.
 */
function parseGameListHtml(html: string): GameSummary[] {
  const games: GameSummary[] = [];
  const seen = new Set<string>();
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
