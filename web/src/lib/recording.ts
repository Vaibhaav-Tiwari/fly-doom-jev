import type {CatalogItem,ConnectomeData,Recording,ReplayStep,RunMetrics} from './types';
const asNumber=(v:unknown,f=0)=>typeof v==='number'&&Number.isFinite(v)?v:f;
const connectomeCache = new Map<string, Promise<ConnectomeData>>();
export async function loadCatalog():Promise<CatalogItem[]> { const r=await fetch('./recordings/index.json');if(!r.ok)throw new Error('Recording catalog unavailable');return (await r.json()).recordings??[]; }
async function fetchBuffer(url:string){const r=await fetch(url);if(!r.ok)throw new Error(`Missing connectome asset: ${url}`);return r.arrayBuffer()}
async function loadConnectome(path:string):Promise<ConnectomeData>{
 const root=new URL(`./${path.replace(/^\.\//,'').replace(/\/?$/,'/')}`,location.href);const meta=await fetch(new URL('meta.json',root)).then(r=>{if(!r.ok)throw new Error('Missing connectome metadata');return r.json()});
 const [pb,ib,lb,fb]=await Promise.all(['positions.f32','ids.i64','population.i16','flags.u8'].map(f=>fetchBuffer(new URL(f,root).href)));const n=Number(meta.n_neurons);const positions=new Float32Array(n*3),ids=new BigInt64Array(n),population=new Int16Array(n),flags=new Uint8Array(fb);
 const pd=new DataView(pb),idv=new DataView(ib),ld=new DataView(lb);for(let i=0;i<n;i++){positions[i*3]=pd.getFloat32(i*12,true);positions[i*3+1]=pd.getFloat32(i*12+4,true);positions[i*3+2]=pd.getFloat32(i*12+8,true);ids[i]=idv.getBigInt64(i*8,true);population[i]=ld.getInt16(i*2,true)}
 return {count:n,positions,ids,population,flags,populationNames:meta.populations??[],withPosition:Number(meta.with_position??n),coordinateSpace:meta.coordinate_space};
}
function loadSharedConnectome(path:string):Promise<ConnectomeData>{
 const cached=connectomeCache.get(path);if(cached)return cached;
 const pending=loadConnectome(path).catch(error=>{connectomeCache.delete(path);throw error});connectomeCache.set(path,pending);return pending;
}
export async function loadRecording(item:CatalogItem):Promise<Recording>{
 const recordingUrl=new URL('./'+item.path,location.href);const response=await fetch(recordingUrl);if(!response.ok)throw new Error(`Could not load ${item.title}`);const entries=(await response.text()).split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));const header=entries.find(x=>x.kind==='header')??{kind:'header'};
 const steps:ReplayStep[]=entries.filter(x=>x.kind==='step'||x.kind==='telemetry').map((x:any,i:number)=>({...x,t_ms:asNumber(x.t_ms,x.timestamp_ms??i*100),frame_index:x.frame_index??x.controller_step??i,frame:x.frame??(x.frame_ref?new URL(x.frame_ref,recordingUrl).href:undefined)}));const frames=entries.find(x=>x.kind==='frames_meta'||x.kind==='frames');const duration=steps.at(-1)?.t_ms??0;const last=steps.at(-1)?.state??{};const actions:Record<string,number>={};let conf=0,latency=0,reward=0;for(const s of steps){const a=s.motor?.selected_action??s.motor?.selected??'unknown';actions[a]=(actions[a]??0)+1;conf+=asNumber(s.motor?.confidence);latency+=asNumber(s.jev?.latency_ms);reward+=asNumber(s.reward)}
 const metrics:RunMetrics={duration,steps:steps.length,kills:asNumber(last.kills),health:asNumber(last.health),ammo:asNumber(last.ammo),meanConfidence:steps.length?conf/steps.length:0,meanLatency:steps.length?latency/steps.length:0,reward,actions};const positionEntry=entries.find(x=>['neuron_positions','neurons_meta','annotations'].includes(x.kind));const embedded=(positionEntry?.neurons??header?.connectome?.neurons??header?.annotations?.neurons??[]) as any[];const packed=positionEntry?.positions as number[][]|undefined;const ids=positionEntry?.body_ids as Array<number|string>|undefined;const staticNeurons=embedded.length?embedded:packed?.map((position,i)=>({body_id:ids?.[i]??i,position}))??[];
 const recording:Recording={...item,header,steps,frames,duration,metrics,staticNeurons,positionSource:positionEntry?.source??(staticNeurons.length?'recording':undefined)};
 if(item.assetsPath&&header.connectome_assets)recording.connectomeData=await loadSharedConnectome(item.assetsPath);
 if(frames?.file){const fr=await fetch(new URL(frames.file,recordingUrl));if(fr.ok)recording.frameData=new Uint8Array(await fr.arrayBuffer())}return recording;
}
export function nearestStep(steps:ReplayStep[],t:number){let lo=0,hi=steps.length-1;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(steps[m].t_ms<=t)lo=m;else hi=m-1}return steps[lo]??null}
export function pct(v:number){return `${Math.round(v*100)}%`}
export function formatTime(ms:number){const s=ms/1000;return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`}
