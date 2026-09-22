import {useEffect, useMemo, useRef, useState} from 'react';
import * as THREE from 'three';
import {OrbitControls} from 'three/examples/jsm/controls/OrbitControls.js';
import type {NeuronSample, Recording, ReplayStep, RetinalInput} from '../lib/types';

const palette: Record<string, string> = {
  ol_sensory: '#2f82bd', visual_projection: '#2f82bd', visual_centrifugal: '#2f82bd',
  ol_intrinsic: '#2f82bd', cx_intrinsic: '#2f82bd', cb_intrinsic: '#2f82bd',
  cb_sensory: '#2f82bd', ascending_neuron: '#2f82bd', descending_neuron: '#2f82bd',
  vnc_sensory: '#2f82bd', vnc_intrinsic: '#2f82bd', vnc_motor: '#2f82bd',
  other: '#2f82bd', recorded: '#2f82bd',
};
const denseOpticGroups = new Set(['ol_sensory', 'visual_projection', 'visual_centrifugal', 'ol_intrinsic']);
const vncGroups = new Set(['vnc_sensory', 'vnc_intrinsic', 'vnc_motor']);
// MaleCNS soma Y coordinates above this plane are the ventral cord/neck tail.
// This is a display crop only; source recordings and connectome arrays remain intact.
const VNC_DISPLAY_CUTOFF_Y = 50_000;
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
type RecentRates = Map<string, number[]>;

function rememberRate(history: RecentRates, group: string, value: number) {
  const samples = history.get(group) ?? [];
  samples.push(value);
  if (samples.length > 50) samples.shift();
  history.set(group, samples);
}

function normalizeRecent(value: number, samples: number[] | undefined) {
  if (value <= 0 || !samples?.length) return 0;
  const low = Math.min(...samples), high = Math.max(...samples);
  if (high - low < Math.max(.05, high * .08)) return .62;
  return Math.max(0, Math.min(1, (value - low) / (high - low)));
}

function retinalSamples(input: RetinalInput | undefined) {
  if (!input || typeof input === 'number' || Array.isArray(input) || !input.indices?.length) return null;
  const drive = Array.isArray(input.drive) ? input.drive : input.values ?? input.photoreceptor_drive;
  if (!drive?.length) return null;
  return {indices: input.indices, drive};
}

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
  const retinaGeometry = useRef<THREE.BufferGeometry | null>(null);
  const populationHistory = useRef<RecentRates>(new Map());
  const neuronPeakHistory = useRef<RecentRates>(new Map());
  const [drawCount, setDrawCount] = useState(0);
  const [sceneVersion, setSceneVersion] = useState(0);
  const [hasRetinalInput, setHasRetinalInput] = useState(false);
  const regionNormalized = true;
  const bundle = recording.connectomeData;

  useEffect(() => {
    populationHistory.current.clear();
    neuronPeakHistory.current.clear();
    setHasRetinalInput(false);
  }, [recording.id]);

  const neurons = useMemo<DrawNeuron[]>(() => {
    if (bundle) {
      const out: DrawNeuron[] = [];
      for (let i = 0; i < bundle.count; i++) {
        const j = i * 3;
        const x = bundle.positions[j], y = bundle.positions[j + 1], z = bundle.positions[j + 2];
        const group = bundle.populationNames[bundle.population[i]] ?? 'other';
        if ((bundle.flags[i] & 16) && Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)
          && !vncGroups.has(group) && group !== 'ascending_neuron' && group !== 'descending_neuron'
          && y < VNC_DISPLAY_CUTOFF_Y) {
          out.push({index: i, position: [x, y, z], group});
        }
      }
      return out;
    }
    if (recording.staticNeurons.length) {
      return recording.staticNeurons.map((n, i) => ({
        index: i,
        position: getPos(n) ?? annotatedPosition(getId(n, i), i, n.region ?? n.cell_type ?? 'recorded'),
        group: n.region ?? n.cell_type ?? 'recorded',
      })).filter(n => !vncGroups.has(n.group) && n.group !== 'ascending_neuron' && n.group !== 'descending_neuron');
    }
    const out: DrawNeuron[] = [];
    Object.entries(step?.populations ?? {}).forEach(([group, p]) => {
      if (vncGroups.has(group) || group === 'ascending_neuron' || group === 'descending_neuron') return;
      (p.sampled ?? []).forEach((n, i) => {
      out.push({index: n.neuron ?? out.length, position: getPos(n) ?? annotatedPosition(getId(n, i), out.length, group), group});
      });
    });
    return out;
  }, [recording, bundle, bundle ? undefined : step?.populations]);

  useEffect(() => {
    const el = host.current;
    if (!el || !neurons.length) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, .1, 100);
    // Start from the standard frontal anatomical view shown in the reference.
    // Orbit remains available, but the brain does not move until the visitor drags.
    camera.position.set(0, 0, 11);
    const renderer = new THREE.WebGLRenderer({antialias: false, alpha: true, powerPreference: 'high-performance'});
    renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    el.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = false;
    controls.minDistance = 3;
    controls.maxDistance = 22;

    const geometry = new THREE.OctahedronGeometry(bundle?.count ? .009 : .06, 0);
    const material = new THREE.MeshBasicMaterial({
      color: 0xffffff, vertexColors: false, transparent: true, opacity: .68,
      blending: THREE.NormalBlending, depthWrite: true, toneMapped: false, fog: false,
    });
    const cloud = new THREE.InstancedMesh(geometry, material, neurons.length);
    cloud.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    cloud.frustumCulled = false;
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
    // Instanced colors default to black. Seed the resting palette immediately
    // so the connectome remains visible even before the first animation frame.
    const initialColors = cloud.instanceColor.array as Float32Array;
    for (let i = 0; i < initialColors.length; i++) initialColors[i] = bases[i];
    cloud.instanceColor.needsUpdate = true;

    const box = new THREE.Box3();
    neurons.forEach(n => box.expandByPoint(new THREE.Vector3(...n.position)));
    const center = new THREE.Vector3(), size = new THREE.Vector3();
    box.getCenter(center); box.getSize(size);
    const scale = 6 / Math.max(size.x, size.y, size.z, 1);
    const orbitCenter = new THREE.Vector3(0, 0, 0);
    const orbitSphere = new THREE.Sphere(orbitCenter, 0);
    const positions = new Float32Array(neurons.length * 3);
    const dummy = new THREE.Object3D();
    const map = new Int32Array(bundle?.count ?? Math.max(...neurons.map(n => n.index), 0) + 1);
    map.fill(-1);
    let maxRadiusSq = 0;
    neurons.forEach((n, i) => {
      const j = i * 3;
      // MaleCNS source coordinates arrive inverted relative to the familiar
      // frontal anatomical view. Rotate the display 180° in-plane so the
      // central brain sits above the optic lobes, matching the reference view.
      positions[j] = -(n.position[0] - center.x) * scale;
      positions[j + 1] = -(n.position[1] - center.y) * scale;
      positions[j + 2] = (n.position[2] - center.z) * scale;
      maxRadiusSq = Math.max(maxRadiusSq, positions[j] ** 2 + positions[j + 1] ** 2 + positions[j + 2] ** 2);
      dummy.position.set(positions[j], positions[j + 1], positions[j + 2]);
      // The optic lobes contain most positioned neurons. Thin only their
      // resting layer so the complete, unsampled activity overlay stays clear.
      dummy.scale.setScalar(!bundle || !denseOpticGroups.has(n.group) || n.index % 5 === 0 ? 1 : 0);
      dummy.updateMatrix();
      cloud.setMatrixAt(i, dummy.matrix);
      if (n.index < map.length) map[n.index] = i;
    });
    normalized.current = positions;
    indexToInstance.current = map;
    cloud.instanceMatrix.needsUpdate = true;
    orbitSphere.radius = Math.sqrt(maxRadiusSq) + geometry.parameters.radius;
    controls.target.copy(orbitCenter);
    controls.minDistance = Math.max(3, orbitSphere.radius * 1.15);
    controls.maxDistance = Math.max(22, orbitSphere.radius * 6);

    // A separate billboard layer makes the sparse, real firing set legible over 141k resting instances.
    const activeCapacity = Math.max(4096, (recording.header.motor_population?.indices?.length ?? 0) + 512);
    const activeGeometry = new THREE.BufferGeometry();
    activeGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(activeCapacity * 3), 3));
    activeGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(activeCapacity * 3), 3));
    activeGeometry.setDrawRange(0, 0);
    firingGeometry.current = activeGeometry;
    const glowCanvas = document.createElement('canvas');
    glowCanvas.width = glowCanvas.height = 64;
    const glowContext = glowCanvas.getContext('2d')!;
    const glowGradient = glowContext.createRadialGradient(32, 32, 0, 32, 32, 32);
    glowGradient.addColorStop(0, 'rgba(255,255,255,1)');
    glowGradient.addColorStop(.18, 'rgba(255,255,255,.96)');
    glowGradient.addColorStop(.5, 'rgba(255,255,255,.34)');
    glowGradient.addColorStop(1, 'rgba(255,255,255,0)');
    glowContext.fillStyle = glowGradient;
    glowContext.fillRect(0, 0, 64, 64);
    const glowMap = new THREE.CanvasTexture(glowCanvas);
    const haloMaterial = new THREE.PointsMaterial({color: 0xd98220, map: glowMap, size: .34, vertexColors: true, transparent: true, opacity: .22, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    const coreMaterial = new THREE.PointsMaterial({color: 0xe8b45f, map: glowMap, size: .12, vertexColors: true, transparent: true, opacity: .72, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    firingHalo.current = haloMaterial;
    firingCore.current = coreMaterial;
    const firingHaloPoints = new THREE.Points(activeGeometry, haloMaterial);
    const firingCorePoints = new THREE.Points(activeGeometry, coreMaterial);
    firingHaloPoints.renderOrder = 20;
    firingCorePoints.renderOrder = 21;
    firingHaloPoints.frustumCulled = firingCorePoints.frustumCulled = false;
    scene.add(firingHaloPoints, firingCorePoints);

    const actionSets = recording.header.motor?.contributing as Record<string, {indices?: number[]}> | undefined;
    const actionCapacity = Math.max(64, ...Object.values(actionSets ?? {}).map(entry => entry.indices?.length ?? 0));
    const actionGeom = new THREE.BufferGeometry();
    actionGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(actionCapacity * 3), 3));
    actionGeom.setDrawRange(0, 0);
    actionGeometry.current = actionGeom;
    const actionMaterial = new THREE.PointsMaterial({color: 0xe45768, map: glowMap, size: .34, transparent: true, opacity: .6, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    const actionPoints = new THREE.Points(actionGeom, actionMaterial);
    actionPoints.renderOrder = 22;
    actionPoints.frustumCulled = false;
    scene.add(actionPoints);

    // The retina layer is populated only from explicit backend drive samples.
    // Unpositioned photoreceptors are skipped rather than assigned fake points.
    const retinaCapacity = 4096;
    const retinaGeom = new THREE.BufferGeometry();
    retinaGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(retinaCapacity * 3), 3));
    retinaGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(retinaCapacity * 3), 3));
    retinaGeom.setDrawRange(0, 0);
    retinaGeometry.current = retinaGeom;
    const retinaMat = new THREE.PointsMaterial({color: 0xd9a94f, map: glowMap, size: .24, vertexColors: true, transparent: true, opacity: .42, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, toneMapped: false, fog: false});
    const retinaPoints = new THREE.Points(retinaGeom, retinaMat);
    retinaPoints.renderOrder = 18;
    retinaPoints.frustumCulled = false;
    scene.add(retinaPoints);
    // The activity effect can run before Three has finished building these
    // buffers. Re-run it once the scene is ready so the paused first frame is
    // illuminated too, rather than waiting for playback to advance a step.
    setSceneVersion(version => version + 1);

    let frame = 0, lastColorUpdate = 0;
    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      // Keep the canvas' CSS viewport in lockstep with its drawing buffer.
      // With updateStyle=false, DPR made the canvas 1.5x larger than its
      // clipped host, so the correctly-centered scene appeared down-right.
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      if (w > 0 && h > 0) {
        const verticalFov = THREE.MathUtils.degToRad(camera.fov);
        const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
        const limitingFov = Math.min(verticalFov, horizontalFov);
        // The VNC-trimmed brain occupies only the useful anatomy now. Fit its
        // sphere tightly so the optic lobes and central brain dominate the rail.
        const distance = orbitSphere.radius * .72 / Math.sin(limitingFov / 2);
        const direction = camera.position.clone().sub(orbitCenter).normalize();
        camera.position.copy(orbitCenter).add(direction.multiplyScalar(distance));
        camera.lookAt(orbitCenter);
        controls.target.copy(orbitCenter);
        controls.minDistance = distance * .62;
        controls.maxDistance = distance * 4;
        controls.update();
      }
    };
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
          const brightness = .88 + glow * .12;
          out[j] = base[j] * brightness;
          out[j + 1] = base[j + 1] * brightness;
          out[j + 2] = base[j + 2] * brightness;
        }
        cloud.instanceColor!.needsUpdate = true;
      }
      haloMaterial.size = .28 + .08 * (.5 + .5 * Math.sin(now * .012));
      coreMaterial.size = .1 + .035 * (.5 + .5 * Math.sin(now * .015));
      retinaMat.size = .2 + .06 * (.5 + .5 * Math.sin(now * .012));
      controls.update(); renderer.render(scene, camera);
    };
    frame = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); controls.dispose();
      geometry.dispose(); material.dispose(); activeGeometry.dispose(); haloMaterial.dispose(); coreMaterial.dispose();
      actionGeom.dispose(); actionMaterial.dispose(); retinaGeom.dispose(); retinaMat.dispose(); glowMap.dispose(); renderer.dispose(); renderer.domElement.remove();
      mesh.current = null; firingGeometry.current = null; actionGeometry.current = null; retinaGeometry.current = null;
    };
  }, [neurons, bundle, recording]);

  useEffect(() => {
    const t = targets.current, flags = contributors.current, map = indexToInstance.current, positions = normalized.current;
    const activeGeom = firingGeometry.current, actionGeom = actionGeometry.current, retinaGeom = retinaGeometry.current;
    if (!t || !flags || !map || !positions || !activeGeom || !actionGeom || !retinaGeom) return;
    t.fill(0); flags.fill(0);
    // v2.1 recordings currently carry top-256; newer live/recorded payloads
    // carry top-1000. Render up to 1000 without assuming either payload size.
    const top = (step?.activity?.top ?? []).slice(0, 1000);
    const motorIndices = recording.header.motor_population?.indices ?? [];
    const motorRates = step?.activity?.motor_rates ?? [];
    const peak = Math.max(0, ...top.map(x => x[1]), ...motorRates);
    const normalizedPeak = Math.max(1, peak);
    const populationRates = new Map(Object.entries(step?.populations ?? {}).map(([g, p]) => [g, p.mean_rate_hz ?? 0]));
    const maxPopulation = Math.max(1, ...populationRates.values());
    populationRates.forEach((rate, group) => rememberRate(populationHistory.current, group, rate));
    neurons.forEach((n, i) => {
      const rate = populationRates.get(n.group) ?? 0;
      // Population means are recorded data. Region mode only changes display
      // gain so low-rate deep-brain populations remain visible beside retina.
      const gain = regionNormalized ? normalizeRecent(rate, populationHistory.current.get(n.group)) : rate / maxPopulation;
      t[i] = rate > 0 ? gain * (regionNormalized ? .34 : .22) : 0;
    });

    const activeRates = new Map<number, number>();
    top.forEach(([idx, rate]) => activeRates.set(idx, Math.max(activeRates.get(idx) ?? 0, rate)));
    motorIndices.forEach((idx, i) => {const rate = motorRates[i] ?? 0; if (rate > 0) activeRates.set(idx, Math.max(activeRates.get(idx) ?? 0, rate));});
    const currentGroupPeaks = new Map<string, number>();
    activeRates.forEach((rate, idx) => {
      const instance = idx < map.length ? map[idx] : -1;
      if (instance >= 0) currentGroupPeaks.set(neurons[instance].group, Math.max(currentGroupPeaks.get(neurons[instance].group) ?? 0, rate));
    });
    currentGroupPeaks.forEach((rate, group) => rememberRate(neuronPeakHistory.current, group, rate));
    const activePositions = activeGeom.getAttribute('position') as THREE.BufferAttribute;
    const activeColors = activeGeom.getAttribute('color') as THREE.BufferAttribute;
    let activeCount = 0;
    activeRates.forEach((rate, idx) => {
      const instance = idx < map.length ? map[idx] : -1;
      if (instance < 0 || activeCount >= activePositions.count) return;
      const group = neurons[instance].group;
      const groupHigh = Math.max(1, ...(neuronPeakHistory.current.get(group) ?? [1]));
      const displayedRate = Math.sqrt(rate / (regionNormalized ? groupHigh : normalizedPeak));
      t[instance] = Math.max(t[instance], displayedRate);
      const src = instance * 3, dst = activeCount * 3;
      activePositions.setXYZ(activeCount, positions[src], positions[src + 1], positions[src + 2]);
      const visibleIntensity = .55 + displayedRate * .45;
      activeColors.setXYZ(activeCount, visibleIntensity, visibleIntensity, visibleIntensity);
      activeCount++;
    });
    activePositions.needsUpdate = true;
    activeColors.needsUpdate = true;
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

    const retinalInput = step?.activity?.retinal_input ?? step?.activity?.retina ?? step?.activity?.visual_input;
    const retinal = retinalSamples(retinalInput);
    const retinaPositions = retinaGeom.getAttribute('position') as THREE.BufferAttribute;
    const retinaColors = retinaGeom.getAttribute('color') as THREE.BufferAttribute;
    let retinaCount = 0;
    if (retinal) {
      const maxDrive = Math.max(1, ...retinal.drive);
      retinal.indices.forEach((idx, i) => {
        const instance = idx < map.length ? map[idx] : -1;
        const drive = retinal.drive[i] ?? 0;
        if (instance < 0 || drive <= 0 || retinaCount >= retinaPositions.count) return;
        const src = instance * 3;
        const intensity = .5 + .5 * Math.sqrt(drive / maxDrive);
        retinaPositions.setXYZ(retinaCount, positions[src], positions[src + 1], positions[src + 2]);
        retinaColors.setXYZ(retinaCount, intensity, intensity, intensity);
        retinaCount++;
      });
    }
    retinaPositions.needsUpdate = true;
    retinaColors.needsUpdate = true;
    retinaGeom.setDrawRange(0, retinaCount);
    setHasRetinalInput(Boolean(retinal));
  }, [neurons, step, recording, bundle, sceneVersion, regionNormalized]);

  return <div className="brain-stage">
    <div ref={host} className="brain-canvas"/>
    <div className="brain-title"><span>BRAIN · MALECNS V1.0 · {drawCount.toLocaleString()} POSITIONED · VNC HIDDEN — DISPLAY ONLY</span></div>
    <div className="activity-legend"><span className="firing-key"><i/> FIRING</span>{hasRetinalInput && <span className="visual-key"><i/> RETINAL INPUT · DRIVE</span>}<span className="decoder-key"><i/> ACTION INPUT</span><span className="resting-key"><i/> RESTING</span></div>
  </div>;
}
