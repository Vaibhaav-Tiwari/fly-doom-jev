export type ScalarMap = Record<string, number>;
export type ControllerMode = 'jev' | 'brain';
export interface NeuronSample { neuron?: number; body_id?: number | string; rate_hz?: number; activity?: number; position?: [number,number,number]; x?:number;y?:number;z?:number; cell_type?:string; region?:string }
export interface Population { size?: number; mean_rate_hz?: number; activity?:number; sampled?: NeuronSample[]; contributing_actions?: string[] }
export type RetinalInput = number | number[] | {indices?:number[];values?:number[];drive?:number|number[];photoreceptor_drive?:number[];mean_drive?:number;peak_drive?:number;intensity?:number};
export interface NeuronActivity {top?:Array<[number,number]>;motor_rates?:number[];retinal_input?:RetinalInput;retina?:RetinalInput;visual_input?:RetinalInput}
export interface ReplayStep { kind:string;t_ms:number;controller_step?:number;episode_tic?:number;frame_index?:number;frame?:string;frame_ref?:string;state?:Record<string,unknown>;jev?:{request_id?:string;probabilities?:ScalarMap;questions?:Array<{question:string;probability?:number;score?:number}>;latency_ms?:number;model?:string;is_mock?:boolean;confidence?:number|ScalarMap;decision?:string;choices?:Record<string,string>;usage?:{input_tokens?:number;output_tokens?:number;cost_usd?:number}};populations?:Record<string,Population>;neurons?:NeuronSample[];activity?:NeuronActivity;neural?:{time_ms?:number;steps?:number;total_spikes?:number};motor?:{scores?:ScalarMap;selected?:string;selected_action?:string;confidence?:number;raw_rates?:ScalarMap;channels?:ScalarMap;readout_rates?:Record<string,{positive?:number;negative?:number}>;contributing_populations?:Record<string,string[]>};reward?:number;controller_latency_ms?:number }
export interface FramesMeta {kind:string;file?:string;count?:number;shape?:number[];dtype?:string;downsample?:number;format?:string;files?:string[];timestamps_ms?:number[];pattern?:string;codec?:string}
export interface ConnectomeData {count:number;positions:Float32Array;ids:BigInt64Array;population:Int16Array;flags:Uint8Array;populationNames:string[];withPosition:number;coordinateSpace?:string}
export interface RecordingHeader {kind:string;recording_format_version?:string;created_at?:number|string;seed?:number;software_version?:string;environment_backend?:string;warnings?:string[];config?:Record<string,unknown>;connectome?:{provenance?:any;populations?:Record<string,unknown>;n_neurons?:number;n_edges?:number;neurons?:NeuronSample[]};jev?:Record<string,unknown>;bridge?:any;motor?:any;motor_population?:{indices?:number[];body_ids?:Array<number|string>};connectome_assets?:any;annotations?:any}
export interface Recording {id:string;title:string;subtitle:string;path:string;header:RecordingHeader;steps:ReplayStep[];frames?:FramesMeta;duration:number;frameData?:Uint8Array;metrics:RunMetrics;staticNeurons:NeuronSample[];positionSource?:string;connectomeData?:ConnectomeData}
export interface RunMetrics {duration:number;steps:number;kills:number;health:number;ammo:number;meanConfidence:number;meanLatency:number;reward:number;actions:Record<string,number>}
export interface CatalogItem {id:string;title:string;subtitle:string;path:string;featured?:boolean;assetsPath?:string}

export interface LiveSnapshot {
  status: string;
  scenario?: 'fly_arena' | 'e1m1';
  controller?: ControllerMode;
  run_id?: string;
  sequence?: number;
  generated_at_ms?: number;
  frame?: string;
  game?: {health?: number; kills?: number; ammo?: number; enemies?: number | null; alive_s?: number; episode?: number};
  motor?: {selected?: string; combo?: string[]; scores?: ScalarMap; channels?: ScalarMap; readout_rates?: Record<string, {positive?: number; negative?: number}>};
  activity?: NeuronActivity;
  retinal_input?: RetinalInput;
  retina?: RetinalInput;
  visual_input?: RetinalInput;
  populations?: Record<string, number>;
  jev?: ReplayStep['jev'] | null;
}
