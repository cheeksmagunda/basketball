export function fmtDateShort(d: Date, upper?: boolean): string {
  const s = d.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  return upper ? s.toUpperCase() : s;
}

export function etToday(): string {
  return new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
}
