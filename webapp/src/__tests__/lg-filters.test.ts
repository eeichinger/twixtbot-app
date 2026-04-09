/**
 * lg-filters.test.ts — unit tests for filterGameSummaries (L10 + L11).
 *
 * All logic is pure client-side filtering on an already-fetched GameSummary[].
 */

import { describe, it, expect } from 'vitest';
import { filterGameSummaries, type GameSummary } from '../lg-api.js';

const GAMES: GameSummary[] = [
  { id: '1', blackPlayer: 'A', whitePlayer: 'B', result: 'win',  boardSize: 24, moveCount: 30, opponent: 'Alpha' },
  { id: '2', blackPlayer: 'A', whitePlayer: 'B', result: 'lost', boardSize: 24, moveCount: 40, opponent: 'Beta'  },
  { id: '3', blackPlayer: 'A', whitePlayer: 'B', result: 'win',  boardSize: 24, moveCount: 25, opponent: 'Beta'  },
  { id: '4', blackPlayer: 'A', whitePlayer: 'B', result: 'draw', boardSize: 24, moveCount: 50, opponent: 'Alpha' },
  { id: '5', blackPlayer: 'A', whitePlayer: 'B', result: 'lost', boardSize: 24, moveCount: 35, opponent: 'Gamma' },
];

describe('filterGameSummaries — result filter', () => {
  it('"all" returns the full list', () => {
    expect(filterGameSummaries(GAMES, 'all', null)).toHaveLength(5);
  });

  it('"win" returns only wins', () => {
    const result = filterGameSummaries(GAMES, 'win', null);
    expect(result).toHaveLength(2);
    expect(result.every(g => g.result === 'win')).toBe(true);
  });

  it('"lost" returns only losses', () => {
    const result = filterGameSummaries(GAMES, 'lost', null);
    expect(result).toHaveLength(2);
    expect(result.every(g => g.result === 'lost')).toBe(true);
  });

  it('"draw" returns only draws', () => {
    const result = filterGameSummaries(GAMES, 'draw', null);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('4');
  });
});

describe('filterGameSummaries — opponent filter', () => {
  it('null opponent returns all games', () => {
    expect(filterGameSummaries(GAMES, 'all', null)).toHaveLength(5);
  });

  it('empty string opponent returns all games', () => {
    // empty string is treated as "no filter" (falsy)
    expect(filterGameSummaries(GAMES, 'all', '')).toHaveLength(5);
  });

  it('filters to a specific opponent', () => {
    const result = filterGameSummaries(GAMES, 'all', 'Beta');
    expect(result).toHaveLength(2);
    expect(result.every(g => g.opponent === 'Beta')).toBe(true);
  });

  it('returns empty array when no games match opponent', () => {
    expect(filterGameSummaries(GAMES, 'all', 'NoOne')).toHaveLength(0);
  });
});

describe('filterGameSummaries — combined filters', () => {
  it('losses against Beta', () => {
    const result = filterGameSummaries(GAMES, 'lost', 'Beta');
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('2');
  });

  it('wins against Alpha', () => {
    const result = filterGameSummaries(GAMES, 'win', 'Alpha');
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('1');
  });

  it('draws against Gamma returns empty', () => {
    expect(filterGameSummaries(GAMES, 'draw', 'Gamma')).toHaveLength(0);
  });
});
