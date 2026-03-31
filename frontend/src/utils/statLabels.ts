export const STAT_LABEL: Record<string, string> = {
  points: 'PTS',
  rebounds: 'REB',
  assists: 'AST',
  steals: 'STL',
  blocks: 'BLK',
  threes: '3PM',
};

export function getStatLabel(statType: string): string {
  return STAT_LABEL[statType] || statType.toUpperCase();
}
