/**
 * lg-api.ts — Little Golem read-only API client.
 *
 * CORS: LG does not send CORS headers, so browser fetch() is blocked.
 * Requests are routed through a Cloudflare Worker at eeichinger.workers.dev.
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
// Mock mode — set to false to enable real LG requests
// ---------------------------------------------------------------------------

const MOCK_MODE = false;

// ---------------------------------------------------------------------------
// CORS proxy — change this constant to switch proxy implementations
// ---------------------------------------------------------------------------

const PROXY_PREFIX = 'https://littlegolem-proxy.eeichinger.workers.dev/?url=';

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
  /**
   * From the game list page: "win" | "lost" | "draw" (player's perspective).
   * From a parsed SGF:        "B+" | "W+" | "0" | "?".
   */
  result: string;
  boardSize: number;
  moveCount: number;
  /** Opponent display name, available when parsed from the game list page. */
  opponent?: string;
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const MOCK_PLAYERS: PlayerResult[] = [
  { plid: '2674', name: 'Alan Hensel',  rating: '1. kyu'  },
  { plid: '1890', name: 'Richard Malaschitz', rating: '2. dan' },
  { plid: '3101', name: 'twixtbot',    rating: '3. kyu'  },
  { plid: '4455', name: 'Peyrol',      rating: '5. kyu'  },
  { plid: '5012', name: 'oakinger',    rating: '4. kyu'  },
];

// Mock game lists match the shape of real parseGameListHtml output:
// blackPlayer/whitePlayer are '?' (not known from the listing page),
// opponent holds the display name, result is player-perspective win/lost/draw.
const MOCK_GAMES_BY_PLID: Record<string, GameSummary[]> = {
  '2674': [
    { id: '2546140', blackPlayer: '?', whitePlayer: '?', opponent: 'TwixtBot',          result: 'win',  boardSize: 24, moveCount: 31 },
    { id: '2060663', blackPlayer: '?', whitePlayer: '?', opponent: 'TwixtBot',          result: 'lost', boardSize: 24, moveCount: 31 },
    { id: '2545876', blackPlayer: '?', whitePlayer: '?', opponent: 'Richard Malaschitz', result: 'lost', boardSize: 24, moveCount: 47 },
    { id: '2501234', blackPlayer: '?', whitePlayer: '?', opponent: 'Peyrol',             result: 'win',  boardSize: 24, moveCount: 38 },
    { id: '2498765', blackPlayer: '?', whitePlayer: '?', opponent: 'oakinger',           result: 'lost', boardSize: 24, moveCount: 52 },
    { id: '2477000', blackPlayer: '?', whitePlayer: '?', opponent: 'Richard Malaschitz', result: 'win',  boardSize: 24, moveCount: 29 },
  ],
  '3101': [
    { id: '2546140', blackPlayer: '?', whitePlayer: '?', opponent: 'Alan Hensel', result: 'lost', boardSize: 24, moveCount: 31 },
    { id: '2512000', blackPlayer: '?', whitePlayer: '?', opponent: 'Peyrol',      result: 'win',  boardSize: 24, moveCount: 44 },
  ],
  '5012': [
    { id: '2498765', blackPlayer: '?', whitePlayer: '?', opponent: 'Alan Hensel',        result: 'win',  boardSize: 24, moveCount: 52 },
    { id: '2531000', blackPlayer: '?', whitePlayer: '?', opponent: 'Richard Malaschitz', result: 'lost', boardSize: 24, moveCount: 37 },
  ],
};

// Real game from Little Golem: Alan Hensel (B) vs TwixtBot (W), game 2060663.
// Uses LG's native b[..]/r[..] notation (r = Red = White).
const MOCK_SGF_2060663 = `(;FF[4]EV[twixt]PB[Alan Hensel]PW[TwixtBot]SZ[24]SO[https://www.littlegolem.net];b[kg];r[ki];b[ii];r[jg];b[gh];r[ke];b[ei];r[fd];b[fc];r[in];b[po];r[km];b[mk];r[lk];b[qh];r[qk];b[tk];r[sl];b[ol];r[pm];b[nn];r[qf];b[og];r[oe];b[tf];r[sf];b[sd];r[rh];b[ui];r[vg];b[sh];r[resign])`;

// A realistic ~30-move TwixT PP game in LG SGF format.
const MOCK_SGF_2546140 = `(;GM[21]FF[4]SZ[24]RU[PP]PB[Alan Hensel]PW[twixtbot]RE[B+]
;B[hd];W[qd];B[swap];W[hd];B[me];W[lh];B[il];W[ji];B[kk]
;W[ni];B[mg];W[oi];B[og];W[qg];B[pe];W[qj];B[nk];W[pk]
;B[ml];W[pl];B[pm];W[ql];B[qm];W[rm];B[rn];W[sn];B[so]
;W[ro];B[rp];W[tt])`;

// Generic SGF for any game ID not explicitly mocked.
function makeMockSgf(bp: string, wp: string, result: string): string {
  return `(;GM[21]FF[4]SZ[24]RU[PP]PB[${bp}]PW[${wp}]RE[${result}]` +
    `;B[hk];W[rk];B[swap];W[hk];B[md];W[lh];B[jl];W[ji];B[kk]` +
    `;W[ni];B[mf];W[oh];B[og];W[qg];B[pe];W[qj];B[nk];W[pk]` +
    `;B[ml];W[pl];B[tt])`;
}

async function mockSearchPlayers(name: string): Promise<PlayerResult[]> {
  await delay(300);
  const q = name.toLowerCase();
  return MOCK_PLAYERS.filter(p => p.name.toLowerCase().includes(q));
}

async function mockFetchPlayerGamesByPlid(plid: string): Promise<GameSummary[]> {
  await delay(400);
  return MOCK_GAMES_BY_PLID[plid] ?? [];
}

async function mockFetchGame(id: string): Promise<ParsedGame> {
  await delay(500);
  if (id === '2546140') return parseTSGF(MOCK_SGF_2546140, id);
  if (id === '2060663') return parseTSGF(MOCK_SGF_2060663, id);
  // Find game in any player list to get player names / result
  for (const games of Object.values(MOCK_GAMES_BY_PLID)) {
    const g = games.find(g => g.id === id);
    if (g) return parseTSGF(makeMockSgf(g.blackPlayer, g.whitePlayer, g.result), id);
  }
  return parseTSGF(makeMockSgf('Player1', 'Player2', 'B+'), id);
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
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
  if (MOCK_MODE) return mockSearchPlayers(name);
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
 * Fetch TwixT games for a player identified by their numeric LG player ID.
 * URL: /jsp/info/player_game_list.jsp?gtid=twixt&plid=PLID
 */
export async function fetchPlayerGamesByPlid(plid: string): Promise<GameSummary[]> {
  if (MOCK_MODE) return mockFetchPlayerGamesByPlid(plid);
  const url = `${LG_BASE}/jsp/info/player_game_list.jsp?gtid=twixt&plid=${encodeURIComponent(plid)}`;
  const res = await fetchProxied(url);
  const html = await res.text();
  return parseGameListHtml(html);
}

// ---------------------------------------------------------------------------
// Game by ID
// ---------------------------------------------------------------------------

/**
 * Fetch and parse a single game's SGF from Little Golem.
 * URL: /servlet/sgf/{id}/game{id}.tsgf
 */
export async function fetchGame(id: string): Promise<ParsedGame> {
  if (MOCK_MODE) return mockFetchGame(id);
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
 * Parse the game-list HTML table from player_game_list.jsp.
 *
 * Table columns per row (verified against live HTML, April 2026):
 *   0: game link "#NNNNNN"
 *   1: opponent display name
 *   2: opponent rating span
 *   3: tournament / game type ("Twixt PP  Size 24")
 *   4: move count (number)
 *   5: result from queried player's perspective ("win" | "lost" | "draw")
 *
 * blackPlayer/whitePlayer remain "?" — which color each player had is only
 * available from the SGF, not from this listing page.
 */
function parseGameListHtml(html: string): GameSummary[] {
  const games: GameSummary[] = [];
  const seen = new Set<string>();

  for (const row of html.split('</tr>')) {
    const gidM = row.match(/href="\/jsp\/game\/game\.jsp\?gid=(\d+)"/);
    if (!gidM) continue;
    const id = gidM[1];
    if (seen.has(id)) continue;
    seen.add(id);

    // Extract text content of each <td>...</td> in this row
    const cells = [...row.matchAll(/<td[^>]*>(.*?)<\/td>/gi)]
      .map(m => m[1].replace(/<[^>]+>/g, '').replace(/&nbsp;/g, ' ').trim());

    // cells[1] = opponent name, cells[4] = move count, cells[5] = win/lost/draw
    const opponent = cells[1] || undefined;
    const moveCount = parseInt(cells[4] ?? '0', 10) || 0;
    const result = (cells[5] ?? '?').toLowerCase();

    games.push({
      id,
      blackPlayer: '?',
      whitePlayer: '?',
      result,   // "win" | "lost" | "draw"
      boardSize: 24,
      moveCount,
      opponent,
    });
  }
  return games;
}
