import {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {BrainCircuit, ChevronDown, Clock3, Database, ExternalLink, Heart, Info, Pause, Play, Radio, RotateCcw, Skull, Volume2, Zap} from 'lucide-react';
import type {CatalogItem, LiveSnapshot, Recording} from './lib/types';
import {loadCatalog, loadRecording, nearestStep, pct} from './lib/recording';
import {getLiveHealth, getLiveState, liveSnapshotToStep, startFreshLiveRun} from './lib/live';
import BrainView from './components/BrainView';
import DoomViewport from './components/DoomViewport';
import ActionTimeline from './components/ActionTimeline';

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
  const [speed, setSpeed] = useState(1);
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

  const seek = useCallback((next: number) => {
    setTime(Math.max(0, Math.min(recording?.duration ?? 0, next)));
  }, [recording]);

  useEffect(() => {
    if (!playing || !recording || mode !== 'recorded') return;
    last.current = performance.now();
    let timer = 0;
    const schedule = () => {
      if (document.hidden) timer = window.setTimeout(() => loop(performance.now()), 100);
      else raf.current = requestAnimationFrame(loop);
    };
    const loop = (now: number) => {
      const delta = (now - last.current) * speed;
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
  }, [playing, speed, recording, mode]);

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
      await startFreshLiveRun();
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
  const sourcePopulation = typeof recording?.header.motor?.source_population === 'string' ? recording.header.motor.source_population : undefined;
  const contributionIndices: number[] = recording?.header.motor?.contributing?.[action]?.indices ?? [];
  const exactPopulations = [...new Set(contributionIndices.map(index => {
    const data = recording?.connectomeData;
    return data ? data.populationNames[data.population[index]] : undefined;
  }).filter((value): value is string => Boolean(value)))];
  const contributors = exactPopulations.length ? exactPopulations : (step?.motor?.contributing_populations?.[action] ?? (sourcePopulation ? [sourcePopulation] : []));
  const state = step?.state ?? {};
  const choices = Object.entries(step?.jev?.choices ?? {});
  const usage = step?.jev?.usage;
  const questionConfidence = typeof step?.jev?.confidence === 'object' ? step.jev.confidence : {};
  const liveRunning = mode === 'live' && livePhase === 'playing';

  if (error) return <div className="load-screen"><Skull/><h1>Replay cartridge failed to load.</h1><p>{error}</p></div>;
  if (!recording) return <div className="load-screen"><BrainCircuit className="pulse"/><p>Waking up the fly brain…</p></div>;

  return <div className="experience">
    <header>
      <a className="logo" href="#"><BrainCircuit/><div><b>FLY//DOOM</b><small>JEV DRIVES A CONNECTOME</small></div></a>
      <div className={`recording-chip ${mode === 'live' ? 'is-live' : ''}`}><i/> {mode === 'live' ? 'LIVE BROADCAST' : 'PLAYING A RECORDING'} <span>{mode === 'live' ? liveNotice : 'NO API · NO SETUP'}</span></div>
      <div className="mode-switch" aria-label="Experience mode">
        <button className={mode === 'live' ? 'active' : ''} onClick={() => switchMode('live')}><Radio/> LIVE</button>
        <button className={mode === 'recorded' ? 'active' : ''} onClick={() => switchMode('recorded')}><Database/> RECORDED</button>
      </div>
      {mode === 'recorded' && <div className="run-select"><Database/><select value={selected} onChange={event => setSelected(event.target.value)}>{catalog.map(item => <option value={item.id} key={item.id}>{item.title}</option>)}</select><ChevronDown/></div>}
      <a className="source" href="https://github.com/Vaibhaav-Tiwari/fly-doom-jev" target="_blank">SOURCE <ExternalLink/></a>
    </header>
    <main>
      <div className="headline">
        <div><span>{mode === 'live' ? 'LIVE FROM THE CONNECTOME' : 'MEET THE PILOT'}</span><h1>211K neurons.<br/><em>One tiny gamer.</em></h1></div>
        <p>{mode === 'live' ? 'Start a fresh arena and watch the fly see, think, and act in real time. Every run becomes a replay.' : 'Recorded signals ripple through a fruit fly connectome and become moves in DOOM. Hit play, then click an action to see the neurons behind it.'}</p>
        <button className={liveRunning ? 'live-cta' : ''} onClick={mode === 'live' ? startLive : toggleRecording}>{mode === 'live' ? (liveRunning ? <RotateCcw/> : <Play/>) : (playing ? <Pause/> : <Play/>)} {mode === 'live' ? (liveRunning ? 'START A NEW RUN' : 'PLAY LIVE') : (playing ? 'PAUSE THE FLY' : 'WATCH THE FLY THINK')}</button>
      </div>
      {mode === 'recorded' && livePhase === 'offline' && <div className="offline-banner"><Radio/> {liveNotice} <button onClick={() => switchMode('live')}>TRY LIVE AGAIN</button></div>}
      <section className="hero">
        <div className="brain-hero"><BrainView recording={recording} step={step}/></div>
        <div className="game-side">
          <div className="game-head"><span>DOOM // {mode === 'live' ? 'LIVE FEED' : 'RECORDED FEED'}</span><div><i/> {mode === 'live' ? livePhase.toUpperCase() : 'SYNCED'}{mode === 'live' && liveSnapshot?.game?.enemies != null ? ` · ${liveSnapshot.game.enemies} ENEMIES` : ''}</div></div>
          <DoomViewport recording={recording} step={step} mode={mode}/>
          <div className="game-stats">
            <div><Heart/><span>HEALTH</span><b>{Number(state.health ?? 0).toFixed(0)}</b></div>
            <div><Zap/><span>AMMO</span><b>{Number(state.ammo ?? 0).toFixed(0)}</b></div>
            <div><Skull/><span>KILLS</span><b>{Number(state.kills ?? 0).toFixed(0)}</b></div>
            <div><Clock3/><span>ALIVE</span><b>{Number(state.alive_s ?? (step?.t_ms ? step.t_ms / 1000 : 0)).toFixed(1)}s</b></div>
          </div>
        </div>
      </section>
      <section className="signal-strip">
        <div className="current-action"><small>MOTOR OUTPUT</small><div style={{'--accent': actionColor[action] ?? '#a8b9aa'} as React.CSSProperties}><i/><strong>{action.replaceAll('_', ' ')}</strong><span>{pct(step?.motor?.confidence ?? 0)} confidence</span></div></div>
        <div className="caused-by"><small>NEURAL DRIVE</small><div>{contributors.length ? contributors.map((population, index) => <span className="active" key={population}><i style={{opacity: 1 - index * .25}}/>{population.replaceAll('_', ' ')} <b>{contributionIndices.length ? `${contributionIndices.length} exact neuron${contributionIndices.length === 1 ? '' : 's'}` : `${Math.round(step?.populations?.[population]?.mean_rate_hz ?? 0)} Hz`}</b></span>) : <span>Waiting for a motor action</span>}</div><p><Info/> Recorded decoder inputs for this action.</p></div>
        <div className="jev-glance"><small><Volume2/> JEV · {step?.jev?.model ?? 'WAITING'} <b>{step?.jev?.is_mock === false ? 'LIVE' : step?.jev ? 'MOCK' : '—'}</b></small><div className="jev-meta">{choices.map(([question, value]) => <span key={question}>{question.replaceAll('_', ' ')}: <b>{value.replaceAll('_', ' ')}</b></span>)}<em>{step?.jev?.latency_ms?.toFixed(0) ?? '—'} ms · {usage ? `${(usage.input_tokens ?? 0) + (usage.output_tokens ?? 0)} tokens` : ''}</em></div>{thinking.slice(0, 3).map(([question, probability]) => <div key={question}><span>{question.replaceAll('_', ' ')}</span><i><b style={{width: pct(probability)}}/></i><em>{pct(probability)}{questionConfidence[question] != null ? <small> · c{pct(questionConfidence[question])}</small> : null}</em></div>)}</div>
        <div className="scores"><small>ACTION SCORES</small>{scores.slice(0, 5).map(([name, score]) => <div className={name === action ? 'active' : ''} key={name}><span>{name.replaceAll('_', ' ')}</span><b>{score.toFixed(2)}</b></div>)}</div>
      </section>
      {mode === 'recorded' ? <ActionTimeline recording={recording} time={time} playing={playing} speed={speed} onPlay={toggleRecording} onSeek={seek} onSpeed={setSpeed}/> : <section className="live-bar"><div><i className={liveRunning ? 'on' : ''}/><span>{liveNotice}</span></div><b>{liveSnapshot?.run_id ?? 'NO RUN YET'}</b><span>EP {liveSnapshot?.game?.episode ?? '—'} · FRAME {liveSnapshot?.sequence ?? '—'} · {Number(liveSnapshot?.game?.alive_s ?? 0).toFixed(1)}s</span><button onClick={startLive}><RotateCcw/> FRESH GAME</button></section>}
      <div className="data-note">{mode === 'live' ? `LIVE CLOSED LOOP · ${liveSnapshot?.run_id ?? 'PRESS PLAY TO BEGIN'} · EVERY EPISODE IS RECORDED` : `REAL RECORDING · ${recording.header.recording_format_version ? 'FORMAT ' + recording.header.recording_format_version : 'FORMAT UNKNOWN'} · ${recording.metrics.steps} FRAMES · ${recording.subtitle}`}</div>
    </main>
  </div>;
}
