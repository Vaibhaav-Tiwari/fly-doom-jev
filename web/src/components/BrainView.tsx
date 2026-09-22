import {useEffect, useMemo, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import {Maximize2, MousePointer2, Sparkles} from 'lucide-react';
import type {NeuronSample, Recording, ReplayStep} from '../lib/types';

const palette: Record<string, string> = {
  ol_sensory: '#8fb8ca', visual_projection: '#a9c9da', visual_centrifugal: '#789aaa',
  ol_intrinsic: '#9184bc', cx_intrinsic: '#c9a568', cb_intrinsic: '#b79589',
  cb_sensory: '#839d89', ascending_neuron: '#9994ae', descending_neuron: '#d7cfc3',
  vnc_sensory: '#7c929b', vnc_intrinsic: '#8b828f', vnc_motor: '#c4a16d',
  other: '#687278', recorded: '#b8bec1',
};
const getId = (n: NeuronSample, i = 0) => String(n.body_id ?? n.neuron ?? i);
const getPos = (n: NeuronSample): [number, number, number] | null =>
  n.position ?? (n.x != null ? [n.x, n.y ?? 0, n.z ?? 0] : null);

function annotatedPosition(id: string, index: number, group: string): [number, number, number] {
  const seed = Number(id.replace(/\D/g, '').slice(-7)) || index + 1;
  const r1 = Math.abs(Math.sin(seed * 12.9898));
  const r2 = Math.abs(Math.sin(seed * 78.233));
  const side = index % 2 ? 1 : -1;
  if (group === 'descending_neuron') return [(r1 - .5) * 1.25, -1.2 - r2 * 2.2, Math.sin(seed) * .5];
  if (group === 'cx_intrinsic') return [(r1 - .5) * 1.55, (r2 - .5) * 2.1, Math.sin(seed * .31) * .65];
  const spread = group === 'visual_projection' ? 2.25 : 1.75;
  const angle = r1 * Math.PI * 2;
  const radius = .25 + Math.sqrt(r2) * 1.2;
  return [side * spread + Math.cos(angle) * radius, Math.sin(angle) * radius * .72 + .45, Math.sin(seed * .77) * .75];
}

type DrawNeuron = {index: number; position: [number, number, number]; group: string};
type ActivitySummary = {count: number; peak: number};

export default function BrainView({recording, step}: {recording: Recording; step: ReplayStep | null}) {
  const host = useRef<HTMLDivElement>(null);
  const mesh = useRef<THREE.InstancedMesh | null>(null);
  const targets = useRef<Float32Array | null>(null);
  const levels = useRef<Float32Array | null>(null);
  const contributors = useRef<Uint8Array | null>(null);
  const baseColors = useRef<Float32Array | null>(null);
  const normalized = useRef<Float32Array | null>(null);
  const indexToInstance = useRef<Int32Array | null>(null);
  const firingGeometry = useRef<THREE.BufferGeometry | null>(null);
  const actionGeometry = useRef<THREE.BufferGeometry | null>(null);
  const firingHalo = useRef<THREE.PointsMaterial | null>(null);
  const firingCore = useRef<THREE.PointsMaterial | null>(null);
  const visualMaterial = useRef<THREE.PointsMaterial | null>(null);
  const visualLevel = useRef(0);
  const [drawCount, setDrawCount] = useState(0);
  const [sceneVersion, setSceneVersion] = useState(0);
  const [activity, setActivity] = useState<ActivitySummary>({count: 0, peak: 0});
  const bundle = recording.connectomeData;

  const neurons = useMemo<DrawNeuron[]>(() => {
    if (bundle) {
      const out: DrawNeuron[] = [];
      for (let i = 0; i < bundle.count; i++) {
        const j = i * 3;
        const x = bundle.positions[j], y = bundle.positions[j + 1], z = bundle.positions[j + 2];
        if ((bundle.flags[i] & 16) && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
          out.push({index: i, position: [x, y, z], group: bundle.populationNames[bundle.population[i]] ?? 'other'});
        }
      }
      return out;
    }
    if (recording.staticNeurons.length) {
      return recording.staticNeurons.map((n, i) => ({
        index: i,
        position: getPos(n) ?? annotatedPosition(getId(n, i), i, n.region ?? n.cell_type ?? 'recorded'),
        group: n.region ?? n.cell_type ?? 'recorded',
      }));
    }
    const out: DrawNeuron[] = [];
    Object.entries(step?.populations ?? {}).forEach(([group, p]) => (p.sampled ?? []).forEach((n, i) => {
      out.push({index: n.neuron ?? out.length, position: getPos(n) ?? annotatedPosition(getId(n, i), out.length, group), group});
    }));
    return out;
  }, [recording, bundle, bundle ? undefined : step?.populations]);

  useEffect(() => {
    const el = host.current;
    if (!el || !neurons.length) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, .1, 100);
    camera.position.set(0, .2, 11);
    const renderer = new THREE.WebGLRenderer({antialias: false, alpha: true, powerPreference: 'high-performance'});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = true;
    controls.autoRotateSpeed = .45;
    controls.minDistance = 3;
    controls.maxDistance = 22;

    const geometry = new THREE.OctahedronGeometry(bundle?.count ? .025 : .07, 0);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff, vertexColors: true, transparent: true, opacity: .9,
      blending: THREE.NormalBlending, depthWrite: false, toneMapped: false, fog: false,
    });
    const cloud = new THREE.InstancedMesh(geometry, material, neurons.length);
    cloud.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    cloud.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(neurons.length * 3), 3);
    cloud.instanceColor.setUsage(THREE.DynamicDrawUsage);
    mesh.current = cloud;
    scene.add(cloud);
    setDrawCount(neurons.length);

    targets.current = new Float32Array(neurons.length);
    levels.current = new Float32Array(neurons.length);
    contributors.current = new Uint8Array(neurons.length);
    const bases = new Float32Array(neurons.length * 3);
    const color = new THREE.Color();
    neurons.forEach((n, i) => {
      color.set(palette[n.group] ?? palette.other);
      bases[i * 3] = color.r; bases[i * 3 + 1] = color.g; bases[i * 3 + 2] = color.b;
    });
    baseColors.current = bases;

    const box = new THREE.Box3();
    neurons.forEach(n => box.expandByPoint(new THREE.Vector3(...n.position)));
    const center = new THREE.Vector3(), size = new THREE.Vector3();
    box.getCenter(center); box.getSize(size);
    const scale = 6 / Math.max(size.x, size.y, size.z, 1);
    const positions = new Float32Array(neurons.length * 3);
    const dummy = new THREE.Object3D();
    const map = new Int32Array(bundle?.count ?? Math.max(...neurons.map(n => n.index), 0) + 1);
    map.fill(-1);
    neurons.forEach((n, i) => {
      const j = i * 3;
      positions[j] = (n.position[0] - center.x) * scale;
      positions[j + 1] = (n.position[1] - center.y) * scale;
      positions[j + 2] = (n.position[2] - center.z) * scale;
      dummy.position.set(positions[j], positions[j + 1], positions[j + 2]);
      dummy.updateMatrix();
      cloud.setMatrixAt(i, dummy.matrix);
      if (n.index < map.length) map[n.index] = i;
    });
    normalized.current = positions;
    indexToInstance.current = map;
    cloud.instanceMatrix.needsUpdate = true;

    // A separate billboard layer makes the sparse, real firing set legible over 141k resting instances.
    const activeCapacity = Math.max(4096, (recording.header.motor_population?.indices?.length ?? 0) + 512);
    const activeGeometry = new THREE.BufferGeometry();
    activeGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(activeCapacity * 3), 3));
    activeGeometry.setDrawRange(0, 0);
    firingGeometry.current = activeGeometry;
    const haloMaterial = new THREE.PointsMaterial({color: 0xff9f2f, size: .18, transparent: true, opacity: .8, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    const coreMaterial = new THREE.PointsMaterial({color: 0xfff5cf, size: .075, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    firingHalo.current = haloMaterial;
    firingCore.current = coreMaterial;
    scene.add(new THREE.Points(activeGeometry, haloMaterial), new THREE.Points(activeGeometry, coreMaterial));

    const actionSets = recording.header.motor?.contributing as Record<string, {indices?: number[]}> | undefined;
    const actionCapacity = Math.max(64, ...Object.values(actionSets ?? {}).map(entry => entry.indices?.length ?? 0));
    const actionGeom = new THREE.BufferGeometry();
    actionGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(actionCapacity * 3), 3));
    actionGeom.setDrawRange(0, 0);
    actionGeometry.current = actionGeom;
    const actionMaterial = new THREE.PointsMaterial({color: 0xff426d, size: .24, transparent: true, opacity: 1, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    scene.add(new THREE.Points(actionGeom, actionMaterial));

    const visualPositions: number[] = [];
    neurons.forEach((n, i) => {
      if (n.group === 'ol_sensory' || n.group === 'visual_projection') {
        const j = i * 3;
        visualPositions.push(positions[j], positions[j + 1], positions[j + 2]);
      }
    });
    const visualGeometry = new THREE.BufferGeometry();
    visualGeometry.setAttribute('position', new THREE.Float32BufferAttribute(visualPositions, 3));
    const visualMat = new THREE.PointsMaterial({color: 0x8edcff, size: .045, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false, fog: false});
    visualMaterial.current = visualMat;
    scene.add(new THREE.Points(visualGeometry, visualMat));
    // The activity effect can run before Three has finished building these
    // buffers. Re-run it once the scene is ready so the paused first frame is
    // illuminated too, rather than waiting for playback to advance a step.
    setSceneVersion(version => version + 1);

    let frame = 0, lastColorUpdate = 0;
    const resize = () => {const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();};
    const observer = new ResizeObserver(resize); observer.observe(el); resize();
    const tick = (now: number) => {
      frame = requestAnimationFrame(tick);
      if (now - lastColorUpdate > 50) {
        lastColorUpdate = now;
        const out = cloud.instanceColor!.array as Float32Array;
        const t = targets.current!, l = levels.current!, flags = contributors.current!, base = baseColors.current!;
        for (let i = 0; i < neurons.length; i++) {
          const pulse = .76 + .24 * Math.sin(now * .01 + i * .31);
          const desired = t[i] * pulse;
          l[i] = desired > l[i] ? l[i] + (desired - l[i]) * .7 : l[i] * .965;
          const glow = Math.max(l[i], flags[i] ? 1 : 0), j = i * 3;
          out[j] = base[j] * (.1 + glow * .22) + glow * .95;
          out[j + 1] = base[j + 1] * (.1 + glow * .22) + glow * .58;
          out[j + 2] = base[j + 2] * (.1 + glow * .22) + glow * .16;
        }
        cloud.instanceColor!.needsUpdate = true;
      }
      haloMaterial.size = .17 + .045 * (.5 + .5 * Math.sin(now * .012));
      coreMaterial.size = .07 + .018 * (.5 + .5 * Math.sin(now * .015));
      visualMat.opacity = visualLevel.current * (.24 + .12 * (.5 + .5 * Math.sin(now * .01)));
      controls.update(); renderer.render(scene, camera);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
      geometry.dispose(); material.dispose(); activeGeometry.dispose(); haloMaterial.dispose(); coreMaterial.dispose();
      actionGeom.dispose(); actionMaterial.dispose(); visualGeometry.dispose(); visualMat.dispose(); renderer.dispose(); renderer.domElement.remove();
      mesh.current = null; firingGeometry.current = null; actionGeometry.current = null;
    };
  }, [neurons, bundle, recording]);

  useEffect(() => {
    const t = targets.current, flags = contributors.current, map = indexToInstance.current, positions = normalized.current;
    const activeGeom = firingGeometry.current, actionGeom = actionGeometry.current;
    if (!t || !flags || !map || !positions || !activeGeom || !actionGeom) return;
    t.fill(0); flags.fill(0);
    const top = step?.activity?.top ?? [];
    const motorIndices = recording.header.motor_population?.indices ?? [];
    const motorRates = step?.activity?.motor_rates ?? [];
    const peak = Math.max(1, ...top.map(x => x[1]), ...motorRates);
    const populationRates = new Map(Object.entries(step?.populations ?? {}).map(([g, p]) => [g, p.mean_rate_hz ?? 0]));
    const maxPopulation = Math.max(1, ...populationRates.values());
    neurons.forEach((n, i) => {
      const rate = (populationRates.get(n.group) ?? 0) / maxPopulation;
      // This is a recorded population mean, used only as a low-level regional wash.
      t[i] = rate * (n.group === 'ol_sensory' || n.group === 'visual_projection' ? .34 : .05);
    });

    const activeRates = new Map<number, number>();
    top.forEach(([idx, rate]) => activeRates.set(idx, Math.max(activeRates.get(idx) ?? 0, rate)));
    motorIndices.forEach((idx, i) => {const rate = motorRates[i] ?? 0; if (rate > 0) activeRates.set(idx, Math.max(activeRates.get(idx) ?? 0, rate));});
    const activePositions = activeGeom.getAttribute('position') as THREE.BufferAttribute;
    let activeCount = 0;
    activeRates.forEach((rate, idx) => {
      const instance = idx < map.length ? map[idx] : -1;
      if (instance < 0 || activeCount >= activePositions.count) return;
      t[instance] = Math.max(t[instance], Math.sqrt(rate / peak));
      const src = instance * 3, dst = activeCount * 3;
      activePositions.setXYZ(activeCount, positions[src], positions[src + 1], positions[src + 2]);
      activeCount++;
    });
    activePositions.needsUpdate = true;
    activeGeom.setDrawRange(0, activeCount);

    const selected = step?.motor?.selected_action ?? step?.motor?.selected ?? '';
    const exact: number[] = recording.header.motor?.contributing?.[selected]?.indices ?? [];
    const actionPositions = actionGeom.getAttribute('position') as THREE.BufferAttribute;
    let actionCount = 0;
    exact.forEach(idx => {
      const instance = idx < map.length ? map[idx] : -1;
      if (instance < 0 || actionCount >= actionPositions.count) return;
      flags[instance] = 1;
      const src = instance * 3;
      actionPositions.setXYZ(actionCount, positions[src], positions[src + 1], positions[src + 2]);
      actionCount++;
    });
    actionPositions.needsUpdate = true;
    actionGeom.setDrawRange(0, actionCount);
    visualLevel.current = Math.min(1, ((populationRates.get('ol_sensory') ?? 0) + (populationRates.get('visual_projection') ?? 0)) / Math.max(1, maxPopulation));
    setActivity({count: activeCount, peak});
  }, [neurons, step, recording, bundle, sceneVersion]);

  return <div className="brain-stage">
    <div ref={host} className="brain-canvas"/>
    <div className="brain-title"><span><Sparkles/> MALECNS // LIVE FIRING</span><strong>{(bundle?.count ?? drawCount).toLocaleString()}</strong><small>{bundle ? `${drawCount.toLocaleString()} positioned neurons · activity synced to frame` : `${drawCount} sampled neurons · activity synced to frame`}</small></div>
    <div className="activity-meter" role="status" aria-label={`${activity.count} firing neurons, ${activity.peak.toFixed(1)} hertz peak`}><b>{activity.count.toLocaleString()}</b> FIRING <span>{activity.peak.toFixed(1)} Hz PEAK</span></div>
    <div className="reset-view" title="Drag to rotate · scroll to zoom"><MousePointer2/> DRAG TO ORBIT <Maximize2/></div>
    <div className="activity-legend"><span><i/> FIRING</span><span><i/> DECODER INPUT</span><span><i/> RESTING</span></div>
    <div className="position-note">{bundle ? '● MALECNS v1.0 · RECORDED COORDINATES' : '◇ V1 HAS NO COORDINATES — DISPLAY LAYOUT IS ANNOTATED'}</div>
  </div>;
}
