/**
 * WeChat App Data Entry (default data + runtime derived)
 *
 * Description:
 * - Default data comes from `defaults.json` (replaceable to build different test environments)
 * - Timestamp supports 4 formats of `resolveDataTimestamp` (see TimeService):
 *   - `"-1h"` / `"-2d30m"` etc. human-readable relative offsets
 *   - negative ms offset, absolute timestamp, date string
 * - `content` of time‑type messages is regenerated here based on timestamp (ensuring "Yesterday/xx/xx" logic is consistent)
 */

import { resolveDataTimestamp, now, fromTimestamp } from '../../../os/TimeService';
import type { AppData } from '../types';
import defaults from './defaults.json';
const asset = (r: unknown) => { const s = String(r ?? '').trim(); return (!s || s.startsWith('http')) ? s : `/@app-assets/Wechat/${s}`; };

const NOW = now();
const ts = (v: unknown) => resolveDataTimestamp(v as string | number);

const formatChatTimeline = (timestamp: number): string => {
  const date = fromTimestamp(timestamp);
  const nowDate = fromTimestamp(NOW);
  const isSameDay = (d1: Date, d2: Date) =>
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate();

  const hours = date.getHours().toString().padStart(2, '0');
  const minutes = date.getMinutes().toString().padStart(2, '0');
  const timeStr = `${hours}:${minutes}`;

  if (isSameDay(date, nowDate)) return timeStr;
  const yesterday = fromTimestamp(nowDate.getTime());
  yesterday.setDate(nowDate.getDate() - 1);
  if (isSameDay(date, yesterday)) return `Yesterday ${timeStr}`;

  if (date.getFullYear() === nowDate.getFullYear()) {
    // English format: "MM/DD HH:MM"
    return `${date.getMonth() + 1}/${date.getDate()} ${timeStr}`;
  }
  // English format: "YYYY/MM/DD HH:MM"
  return `${date.getFullYear()}/${date.getMonth() + 1}/${date.getDate()} ${timeStr}`;
};

const withResolvedTimestamps = (data: any): AppData => {
  const chats = Array.isArray(data?.chats)
    ? data.chats.map((chat: any) => {
        const messages = Array.isArray(chat?.messages)
          ? chat.messages.map((msg: any) => {
              const timestamp = ts(msg?.timestamp);
              if (msg?.type === 'time') {
                return { ...msg, timestamp, content: formatChatTimeline(timestamp) };
              }
              return { ...msg, timestamp };
            })
          : [];
        return { ...chat, messages };
      })
    : [];

  const moments = Array.isArray(data?.moments)
    ? data.moments.map((mo: any) => ({ ...mo, timestamp: ts(mo?.timestamp) }))
    : [];

  return {
    ...(data as AppData),
    chats,
    moments,
  };
};

const resolveAssetsDeep = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(resolveAssetsDeep);
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(obj)) {
      out[k] = resolveAssetsDeep(v);
    }
    return out;
  }
  if (typeof value === 'string' && value.includes('/')) return asset(value);
  return value;
};

let cachedWechatConfig: AppData | null = null;

/**
 * data-mode uses the full initial data directly (no longer maintain a separate "navigation projection layer/interface").
 */
export function getWechatConfig(): AppData {
  if (cachedWechatConfig) return cachedWechatConfig;
  const resolvedDefaults = resolveAssetsDeep(defaults) as typeof defaults;
  cachedWechatConfig = withResolvedTimestamps(resolvedDefaults);
  return cachedWechatConfig;
}

export const WECHAT_CONFIG = getWechatConfig();