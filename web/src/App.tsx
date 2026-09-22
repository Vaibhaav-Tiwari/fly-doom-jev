import {useEffect, useMemo, useRef, useState} from 'react';
import {BrainCircuit, ChevronDown, Clock3, Database, Heart, Pause, Play, Radio, RotateCcw, Skull, Volume2, Zap} from 'lucide-react';
import type {CatalogItem, ControllerMode, LiveSnapshot, Recording} from './lib/types';
import {loadCatalog, loadRecording, nearestStep, pct} from './lib/recording';
import {getLiveHealth, getLiveState, liveSnapshotToStep, startFreshLiveRun} from './lib/live';
import BrainView from './components/BrainView';
import DoomViewport from './components/DoomViewport';

type Mode = 'live' | 'recorded';
type LivePhase = 'checking' | 'ready' | 'connecting' | 'playing' | 'offline';

const actionColor: Record<string, string> = {
  attack: '#c77979', forward: '#a8b9aa', turn_left: '#a391c5',
  turn_right: '#8fa9bb', noop: '#667075', retreat: '#c8a674',
};

export default function App() {
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [runs, setRuns] = useState<Record<string, Recording>>({});
  const [selected, setSelected] = useState('');
  const [mode, setMode] = useState<Mode>('live');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [scenario, setScenario] = useState<'fly_arena' | 'e1m1'>('fly_arena');
  const [controller, setController] = useState<ControllerMode>(() => {
    try { return window.localStorage.getItem('fly-doom-controller') === 'brain' ? 'brain' : 'jev'; }
    catch { return 'jev'; }
  });
  const [error, setError] = useState('');
  const [livePhase, setLivePhase] = useState<LivePhase>('checking');
  const [liveSnapshot, setLiveSnapshot] = useState<LiveSnapshot | null>(null);
  const [pollLive, setPollLive] = useState(false);
  const [liveNotice, setLiveNotice] = useState('Checking the broadcaster…');
  const raf = useRef(0);
  const last = useRef(0);

  useEffect(() => {
    loadCatalog().then(async items => {
      setCatalog(items);
      setSelected(items.find(item => item.featured)?.id ?? items[0]?.id ?? '');
      const loaded = await Promise.all(items.map(loadRecording));
      setRuns(Object.fromEntries(loaded.map(recording => [recording.id, recording])));
    }).catch(reason => setError(reason.message));
  }, []);

  useEffect(() => {
    let active = true;
    getLiveHealth().then(health => {
      if (!active) return;
      setLivePhase('ready');
      setLiveNotice(health.jev?.ok === false ? 'Broadcaster online · Jev degraded' : 'Broadcaster online · ready for a fresh run');
    }).catch(() => {
      if (!active) return;
      setLivePhase('offline');
      setLiveNotice('Broadcaster offline · showing a recorded run');
      setMode('recorded');
    });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!pollLive || mode !== 'live') return;
    let active = true;
    let inFlight = false;
    let failures = 0;
    const poll = async () => {
      if (!active || inFlight) return;
      inFlight = true;
      try {
        const snapshot = await getLiveState();
        if (!active) return;
        failures = 0;
        setLiveSnapshot(snapshot);
        setLivePhase(snapshot.game?.health === 0 ? 'connecting' : 'playing');
        setLiveNotice(snapshot.game?.health === 0 ? 'The fly fell · preparing a fresh arena…' : 'Live from the connectome');
      } catch {
        failures += 1;
        if (failures >= 2 && active) {
          setPollLive(false);
          setLivePhase('offline');
          setLiveNotice('Broadcaster offline · showing a recorded run');
          setMode('recorded');
        }
      } finally {
        inFlight = false;
      }
    };
    void poll();
    const interval = window.setInterval(poll, 200);
    return () => { active = false; window.clearInterval(interval); };
  }, [pollLive, mode]);

  const recording = runs[selected];
  const recordedStep = recording ? nearestStep(recording.steps, time) : null;
  const liveStep = useMemo(() => liveSnapshot ? liveSnapshotToStep(liveSnapshot) : null, [liveSnapshot]);
  const step = mode === 'live' ? (pollLive ? liveStep : null) : recordedStep;

  useEffect(() => {
    if (!playing || !recording || mode !== 'recorded') return;
    last.current = performance.now();
    let timer = 0;
    const schedule = () => {
      if (document.hidden) timer = window.setTimeout(() => loop(performance.now()), 100);
      else raf.current = requestAnimationFrame(loop);
    };
    const loop = (now: number) => {
      const delta = now - last.current;
      last.current = now;
      setTime(current => {
        if (current + delta >= recording.duration) {
          setPlaying(false);
          return recording.duration;
        }
        return current + delta;
      });
      schedule();
    };
    schedule();
    return () => { cancelAnimationFrame(raf.current); clearTimeout(timer); };
  }, [playing, recording, mode]);

  useEffect(() => { setTime(0); setPlaying(false); }, [selected]);

  const switchMode = (next: Mode) => {
    setMode(next);
    setPlaying(false);
    if (next === 'recorded') setPollLive(false);
    if (next === 'live' && livePhase === 'offline') setLiveNotice('Press play to retry the broadcaster');
  };

  const startLive = async () => {
    setMode('live');
    setPlaying(false);
    setLiveSnapshot(null);
    setLivePhase('connecting');
    setLiveNotice('Opening a fresh DOOM arena…');
    try {
      await startFreshLiveRun(scenario, controller);
      setPollLive(true);
    } catch {
      setPollLive(false);
      setLivePhase('offline');
      setLiveNotice('Broadcaster offline · showing a recorded run');
      setMode('recorded');
    }
  };

  const toggleRecording = () => {
    if (playing) { setPlaying(false); return; }
    if (time >= (recording?.duration ?? 0) - 1) setTime(0);
    setPlaying(true);
  };

  const action = step?.motor?.selected_action ?? step?.motor?.selected ?? '—';
  const scores = Object.entries(step?.motor?.scores ?? {}).sort((a, b) => b[1] - a[1]);
  const thinking = Object.entries(step?.jev?.probabilities ?? {}).sort((a, b) => b[1] - a[1]);
  const state = step?.state ?? {};
  const choices = Object.entries(step?.jev?.choices ?? {});
  const usage = step?.jev?.usage;
  const questionConfidence = typeof step?.jev?.confidence === 'object' ? step.jev.confidence : {};
  const liveRunning = mode === 'live' && livePhase === 'playing';
  const activeController: ControllerMode = mode === 'live'
    ? (liveSnapshot?.controller ?? controller)
    : (state.controller === 'brain' ? 'brain' : 'jev');
  const jevBypassed = mode === 'live' && activeController === 'brain';
  const neuronTotal = recording?.connectomeData?.count
    ?? recording?.header.connectome?.n_neurons
    ?? recording?.staticNeurons.length
    ?? 0;

  const selectController = (next: ControllerMode) => {
    setController(next);
    try { window.localStorage.setItem('fly-doom-controller', next); } catch { /* storage is optional */ }
  };

  if (error) return <div className="load-screen"><Skull/><h1>Replay cartridge failed to load.</h1><p>{error}</p></div>;
  if (!recording) return <div className="load-screen"><BrainCircuit className="pulse"/><p>Waking up the fly brain…</p></div>;

  return <div className="experience">
    <header>
      <a className="logo" href="#"><BrainCircuit/><div><b>DOOM, PLAYED BY A FRUIT FLY CONNECTOME</b><small>{neuronTotal.toLocaleString()} neurons · MaleCNS v1.0 · synchronized game, brain and decisions</small></div></a>
      <div className="header-controls">
        <div className="mode-switch" aria-label="Experience mode">
          <button className={mode === 'live' ? 'active' : ''} onClick={() => switchMode('live')}><Radio/> LIVE</button>
          <button className={mode === 'recorded' ? 'active' : ''} onClick={() => switchMode('recorded')}><Database/> RECORDED</button>
        </div>
        {mode === 'recorded' && <div className="run-select"><Database/><select value={selected} onChange={event => setSelected(event.target.value)}>{catalog.map(item => <option value={item.id} key={item.id}>{item.title}</option>)}</select><ChevronDown/></div>}
        <div className="launch-controls">
          {mode === 'live' && <label>MAP<select aria-label="DOOM scenario" value={scenario} onChange={event => setScenario(event.target.value as 'fly_arena' | 'e1m1')}><option value="fly_arena">FLY ARENA</option><option value="e1m1">E1M1</option></select></label>}
          {mode === 'live' && <div className="controller-switch" role="group" aria-label="Controller architecture">
            <button className={controller === 'jev' ? 'active' : ''} onClick={() => selectController('jev')} aria-pressed={controller === 'jev'}>JEV + BRAIN</button>
            <button className={controller === 'brain' ? 'active' : ''} onClick={() => selectController('brain')} aria-pressed={controller === 'brain'}>BRAIN ONLY</button>
          </div>}
          <button className={liveRunning ? 'live-cta' : ''} onClick={mode === 'live' ? startLive : toggleRecording}>{mode === 'live' ? (liveRunning ? <RotateCcw/> : <Play/>) : (playing ? <Pause/> : <Play/>)} {mode === 'live' ? (liveRunning ? 'NEW GAME' : 'PLAY') : (playing ? 'PAUSE' : 'PLAY')}</button>
        </div>
      </div>
    </header>
    <main>
      <section className="hero">
        <div className="game-side">
          <div className="game-head"><span>DOOM · {mode === 'live' ? (liveSnapshot?.scenario ?? scenario).replaceAll('_', ' ').toUpperCase() : 'RECORDED'}</span><div><i/> {mode === 'live' ? liveNotice : 'SYNCED RECORDING'}</div></div>
          <DoomViewport recording={recording} step={step} mode={mode}
            showMinimap={mode === 'live' && (liveSnapshot?.scenario ?? scenario) === 'e1m1'}
            minimapX={state.position_x as number | undefined}
            minimapY={state.position_y as number | undefined}
            goalDist={(state.goal as {dist?: number} | null | undefined)?.dist}/>
          <div className="game-stats">
            <div><Heart/><span>HEALTH</span><b>{Number(state.health ?? 0).toFixed(0)}</b></div>
            <div><Zap/><span>AMMO</span><b>{Number(state.ammo ?? 0).toFixed(0)}</b></div>
            <div><Skull/><span>KILLS</span><b>{Number(state.kills ?? 0).toFixed(0)}</b></div>
            <div><Clock3/><span>ALIVE</span><b>{Number(state.alive_s ?? (step?.t_ms ? step.t_ms / 1000 : 0)).toFixed(1)}s</b></div>
            <span className={`controller-badge ${activeController}`} title="Active controller architecture">{activeController === 'brain' ? 'BRAIN ONLY' : 'JEV + BRAIN'}</span>
          </div>
        </div>
        <div className="right-rail">
          <div className="brain-hero"><BrainView recording={recording} step={step}/></div>
          <section className="signal-strip">
            <div className="current-action"><small>MOTOR OUTPUT</small><div style={{'--accent': actionColor[action] ?? '#a8b9aa'} as React.CSSProperties}><i/><strong>{action.replaceAll('_', ' ')}</strong><span>{pct(step?.motor?.confidence ?? 0)} confidence</span></div></div>
            <div className={`jev-glance${jevBypassed ? ' is-bypassed' : ''}`}><small><Volume2/> JEV · {jevBypassed ? 'BYPASSED' : (step?.jev?.model ?? 'WAITING')} <b>{jevBypassed ? 'NO API' : step?.jev?.is_mock === false ? 'LIVE' : step?.jev ? 'MOCK' : '—'}</b></small>{jevBypassed ? <div className="jev-bypassed">Brain controller is driving the fly directly.</div> : <><div className="jev-meta">{choices.map(([question, value]) => <span key={question}>{question.replaceAll('_', ' ')}: <b>{value.replaceAll('_', ' ')}</b></span>)}<em>{step?.jev?.latency_ms?.toFixed(0) ?? '—'} ms · {usage ? `${(usage.input_tokens ?? 0) + (usage.output_tokens ?? 0)} tokens` : ''}</em></div>{thinking.slice(0, 3).map(([question, probability]) => <div key={question}><span>{question.replaceAll('_', ' ')}</span><i><b style={{width: pct(probability)}}/></i><em>{pct(probability)}{questionConfidence[question] != null ? <small> · c{pct(questionConfidence[question])}</small> : null}</em></div>)}</>}</div>
            <div className="scores"><small>ACTION SCORES</small>{scores.slice(0, 5).map(([name, score]) => <div className={name === action ? 'active' : ''} key={name}><span>{name.replaceAll('_', ' ')}</span><b>{score.toFixed(2)}</b></div>)}</div>
          </section>
        </div>
      </section>
    </main>
  </div>;
}
