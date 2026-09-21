import type {CatalogItem, Recording, ReplayStep, RunMetrics} from './types';

const asNumber=(v:unknown,f=0)=>typeof v==='number'&&Number.isFinite(v)?v:f;
export async function loadCatalog():Promise<CatalogItem[]> { const r=await fetch('./recordings/index.json'); if(!r.ok) throw new Error('Recording catalog unavailable'); return (await r.json()).recordings ?? []; }
export async function loadRecording(item:CatalogItem):Promise<Recording>{
 const response=await fetch('./'+item.path); if(!response.ok) throw new Error(`Could not load ${item.title}`);
 const entries=(await response.text()).split(/\r?\n/).filter(Boolean).map(line=>JSON.parse(line));
 const header=entries.find(x=>x.kind==='header') ?? {kind:'header'};
 const steps:ReplayStep[]=entries.filter(x=>x.kind==='step'||x.kind==='telemetry').map((x:any,i:number)=>({...x,t_ms:asNumber(x.t_ms,x.timestamp_ms??i*100),frame_index:x.frame_index??i}));
 const frames=entries.find(x=>x.kind==='frames_meta'||x.kind==='frames');
 const duration=steps.at(-1)?.t_ms??0;
 const last=steps.at(-1)?.state??{};
 const actions:Record<string,number>={}; let conf=0, latency=0, reward=0;
 for(const s of steps){const a=s.motor?.selected_action??s.motor?.selected??'unknown';actions[a]=(actions[a]??0)+1;conf+=asNumber(s.motor?.confidence);latency+=asNumber(s.jev?.latency_ms);reward+=asNumber(s.reward)}
 const metrics:RunMetrics={duration,steps:steps.length,kills:asNumber(last.kills),health:asNumber(last.health),ammo:asNumber(last.ammo),meanConfidence:steps.length?conf/steps.length:0,meanLatency:steps.length?latency/steps.length:0,reward,actions};
 const recording:Recording={...item,header,steps,frames,duration,metrics};
 if(frames?.file){const url=new URL(item.path,location.href);url.pathname=url.pathname.replace(/[^/]+$/,frames.file);const fr=await fetch(url);if(fr.ok)recording.frameData=new Uint8Array(await fr.arrayBuffer())}
 return recording;
}
export function nearestStep(steps:ReplayStep[],t:number){let lo=0,hi=steps.length-1;while(lo<hi){const m=Math.ceil((lo+hi)/2);if(steps[m].t_ms<=t)lo=m;else hi=m-1}return steps[lo]??null}
export function pct(v:number){return `${Math.round(v*100)}%`}
export function formatTime(ms:number){const s=ms/1000;return `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`}
