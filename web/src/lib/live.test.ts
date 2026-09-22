import {describe, expect, it} from 'vitest';
import {liveSnapshotToStep} from './live';

describe('liveSnapshotToStep', () => {
  it('adapts broadcaster telemetry to the replay step shape', () => {
    const step = liveSnapshotToStep({
      status: 'running',
      sequence: 42,
      frame: 'data:image/jpeg;base64,abc',
      game: {health: 73, ammo: 11, kills: 2, alive_s: 4.5, episode: 8},
      motor: {selected: 'attack', scores: {attack: .75, forward: .25}},
      activity: {top: [[123, 64.2]], motor_rates: [3.1]},
      populations: {ol_sensory: 12.5},
      jev: {model: 'jev-1.13.0', is_mock: false, probabilities: {ATTACK: .91}},
    });

    expect(step.t_ms).toBe(4500);
    expect(step.controller_step).toBe(42);
    expect(step.frame).toContain('image/jpeg');
    expect(step.state?.health).toBe(73);
    expect(step.motor?.selected_action).toBe('attack');
    expect(step.motor?.confidence).toBe(.75);
    expect(step.activity?.top?.[0]).toEqual([123, 64.2]);
    expect(step.populations?.ol_sensory.mean_rate_hz).toBe(12.5);
    expect(step.jev?.model).toBe('jev-1.13.0');
  });

  it('tolerates a starting snapshot with no telemetry', () => {
    const step = liveSnapshotToStep({status: 'starting'});
    expect(step.motor?.selected).toBe('noop');
    expect(step.motor?.confidence).toBe(0);
    expect(step.populations).toEqual({});
  });
});
