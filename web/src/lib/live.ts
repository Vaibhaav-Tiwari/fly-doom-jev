import type {ControllerMode, LiveSnapshot, Population, ReplayStep, ScalarMap} from './types';

const configuredBase = import.meta.env.VITE_LIVE_API as string | undefined;
export const LIVE_API_BASE = configuredBase ?? (import.meta.env.DEV ? '/live-api' : 'http://127.0.0.1:8420');

async function liveRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`${LIVE_API_BASE}${path}`, {...init, signal: controller.signal});
    if (!response.ok) throw new Error(`Live broadcaster returned ${response.status}`);
    return await response.json() as T;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const getLiveHealth = () => liveRequest<{status: string; jev?: {ok?: boolean}}>(`/health`);
export const getLiveState = () => liveRequest<LiveSnapshot>(`/state`);
export const startFreshLiveRun = (scenario: 'fly_arena' | 'e1m1', controller: ControllerMode) => liveRequest<{status: string}>(`/new`, {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({scenario, controller}),
});

export function liveSnapshotToStep(snapshot: LiveSnapshot): ReplayStep {
  const scores: ScalarMap = snapshot.motor?.scores ?? {};
  const selected = snapshot.motor?.selected ?? 'noop';
  const total = Object.values(scores).reduce((sum, value) => sum + Math.max(0, value), 0);
  const populations: Record<string, Population> = {};
  Object.entries(snapshot.populations ?? {}).forEach(([name, mean_rate_hz]) => {
    populations[name] = {mean_rate_hz: Number.isFinite(mean_rate_hz) ? mean_rate_hz : 0};
  });
  const retinalInput = snapshot.retinal_input ?? snapshot.retina ?? snapshot.visual_input
    ?? snapshot.activity?.retinal_input ?? snapshot.activity?.retina ?? snapshot.activity?.visual_input;
  return {
    kind: 'step',
    t_ms: (snapshot.game?.alive_s ?? 0) * 1000,
    controller_step: snapshot.sequence,
    frame: snapshot.frame,
    state: {...snapshot.game, scenario: snapshot.scenario, controller: snapshot.controller},
    activity: retinalInput == null ? snapshot.activity : {...snapshot.activity, retinal_input: retinalInput},
    populations,
    motor: {
      selected,
      selected_action: selected,
      scores,
      channels: snapshot.motor?.channels,
      readout_rates: snapshot.motor?.readout_rates,
      confidence: total > 0 ? Math.max(0, scores[selected] ?? 0) / total : 0,
    },
    jev: snapshot.jev ?? undefined,
  };
}
