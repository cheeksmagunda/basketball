import type { SlateData, PicksData } from '../types/slate';
import { etToday } from './formatDate';

export function transformSlateData(raw: any): SlateData {
  if (!raw || typeof raw !== 'object') raw = {};
  return {
    date: typeof raw.date === 'string' ? raw.date : etToday(),
    locked: Boolean(raw.locked),
    all_complete: Boolean(raw.all_complete),
    lock_time: typeof raw.lock_time === 'string' ? raw.lock_time : null,
    error: typeof raw.error === 'string' ? raw.error : null,
    no_games: Boolean(raw.no_games),
    next_slate_date: typeof raw.next_slate_date === 'string' ? raw.next_slate_date : undefined,
    draftable_count: typeof raw.draftable_count === 'number' ? raw.draftable_count : undefined,
    games: Array.isArray(raw.games) ? raw.games : [],
    lineups: {
      chalk: Array.isArray(raw.lineups?.chalk) ? raw.lineups.chalk : [],
      upside: Array.isArray(raw.lineups?.upside) ? raw.lineups.upside : [],
    },
  };
}

export function transformPicksData(raw: any): PicksData {
  if (!raw || typeof raw !== 'object') raw = {};
  return {
    date: typeof raw.date === 'string' ? raw.date : etToday(),
    locked: Boolean(raw.locked),
    gameScript: typeof raw.gameScript === 'string' ? raw.gameScript : '',
    strategy: {
      label: typeof raw.strategy?.label === 'string' ? raw.strategy.label : '',
      description: typeof raw.strategy?.description === 'string' ? raw.strategy.description : '',
      type: typeof raw.strategy?.type === 'string' ? raw.strategy.type : 'standard',
      total_mult: typeof raw.strategy?.total_mult === 'number' ? raw.strategy.total_mult : 0,
      favored_team: typeof raw.strategy?.favored_team === 'string' ? raw.strategy.favored_team : '',
    },
    lineups: {
      the_lineup: Array.isArray(raw.lineups?.the_lineup) ? raw.lineups.the_lineup : [],
    },
    game: raw.game ?? {
      gameId: '', label: '', home: { id: '', name: '', abbreviation: '' },
      away: { id: '', name: '', abbreviation: '' },
      spread: null, total: null, startTime: '',
    },
  };
}
