export function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "";
  }
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) {
    return `${total}s`;
  }
  const minutes = Math.floor(total / 60);
  const rest = total % 60;
  if (minutes < 60) {
    return rest === 0 ? `${minutes} min` : `${minutes} min ${rest}s`;
  }
  const hours = Math.floor(minutes / 60);
  const leftoverMin = minutes % 60;
  if (leftoverMin === 0) {
    return `${hours}h`;
  }
  return `${hours}h ${leftoverMin} min`;
}
