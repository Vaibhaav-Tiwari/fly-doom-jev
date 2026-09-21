import {describe,expect,it} from 'vitest';
import {formatTime,nearestStep,pct} from './recording';
import type {ReplayStep} from './types';
const step=(t_ms:number):ReplayStep=>({kind:'step',t_ms});
describe('recording helpers',()=>{
 it('selects the latest synchronized step at a replay time',()=>{const steps=[step(0),step(100),step(250)];expect(nearestStep(steps,175)?.t_ms).toBe(100);expect(nearestStep(steps,999)?.t_ms).toBe(250)});
 it('handles an empty recording',()=>expect(nearestStep([],10)).toBeNull());
 it('formats telemetry values',()=>{expect(formatTime(62550)).toBe('01:02.5');expect(pct(.816)).toBe('82%')});
});
