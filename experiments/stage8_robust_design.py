"""Frozen robust design study, with a separate held-out recovery phase."""
import argparse
import copy
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.joint_inverse import predict_joint_observations
from src.design import temporal_schedules
from src.robust_design import (INITIAL_MODES, PARAMETERS, continuum_predict, initial_field,
    scenario_ensemble, frozen_hash, score_candidates)

N=32
DT=2/256
STEPS=256
TIMES=jnp.arange(0,257,32)
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'docs/stage8_results.json'
PRIMARY_SEEDS=list(range(500,520))
DIAGNOSTIC_SEEDS=list(range(530,540))


def save(r):
    RESULT.write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')


def evaluate_tensor(scenario, modes=None):
    x=jnp.arange(N)*2*jnp.pi/N
    modes=INITIAL_MODES[scenario['initial']] if modes is None else modes
    theta=jnp.array([scenario['v'],jnp.log(scenario['nu4'])])
    if scenario['model']=='fd':
        psi0=continuum_predict(theta,x,jnp.array([0.]),modes)[0]
        prediction=lambda t:predict_joint_observations(t,psi0,DT,STEPS,2*jnp.pi/N,TIMES)
    else:
        prediction=lambda t:continuum_predict(t,x,TIMES*DT,modes)
    values=prediction(theta)
    jac=jax.jit(jax.jacfwd(prediction))(theta)
    return np.asarray(values),np.asarray(jac)


def design_at(index, schedules, layouts):
    ti,li=divmod(index,len(layouts))
    return {'candidate_index':index,'schedule_index':ti,'layout_index':li,
            'time_indices':list(schedules[ti]),'times':[i*DT for i in schedules[ti]],
            'sensor_indices':layouts[li]['sensors'],'layout_id':layouts[li]['id']}


def score_at(scores, index):
    return {key:value[index] for key,value in scores.items() if key!='ranking'}


def score_one(jac, values, d, scaling, rms, weighted=True, floor=.01, relative=.03):
    scores,refs=score_candidates(jac,values,[d['time_indices']],[{'sensors':d['sensor_indices']}],scaling,rms,floor,relative,weighted)
    return score_at(scores,0)


def build_designs():
    previous=json.loads((ROOT/'docs/stage7_results.json').read_text())
    scenarios=scenario_ensemble(); schedules=temporal_schedules()
    scaling=previous['configuration']['normalized_scaling']
    values=[]; jac=[]
    for scenario in scenarios:
        y,j=evaluate_tensor(scenario);values.append(y);jac.append(j)
        print('Sensitivity scenario',scenario['id'],flush=True)
    values=np.array(values);jac=np.array(jac)
    nominal=next(i for i,a in enumerate(scenarios) if a['id']=='fd_A_P2')
    rms=float(np.sqrt(np.mean(values[nominal,1:]**2)))
    frozen={'scenario_definitions':scenarios,'initial_conditions':INITIAL_MODES,
            'parameter_points':PARAMETERS,'scaling':scaling,
            'noise_model':{'reference':'nominal FD A, all 8 positive times and all 32 sensors',
                           'reference_rms':rms,'floor_fraction':.01,'relative':.03},
            'primary_metric':'maximin scenario-relative weighted E efficiency',
            'schedules':schedules,'candidate_pools':{},'selected':{},
            'primary_seeds':PRIMARY_SEEDS,'diagnostic_seeds':DIAGNOSTIC_SEEDS,
            'diagnostic_policy':'four sensors; weighted ranks top 3, middle 3, bottom 3; matched A and continuum A; 5% noise',
            'diagnostic_selection':{},
            'recovery_noise':'Stage 7 homoscedastic independent Gaussian; 2%/5% selected-clean-positive RMS; t=0 exact',
            'seed_count_rationale':'20 selected before recovery to limit runtime of 48 primary comparison groups'}
    r={'frozen':frozen,'candidate_pool_hashes':{},'scores':{},'references':{},
       'important_scores':{},'scenario_values':values.tolist(),'scenario_jacobians':jac.tolist(),
       'recovery':{},'diagnostic_recovery':{},'conclusion':'DESIGNS FROZEN; RECOVERY PENDING'}
    for count in ('8','4'):
        layouts=copy.deepcopy(previous['spatial'][count]['layouts'])
        frozen['candidate_pools'][count]=layouts
        r['candidate_pool_hashes'][count]={'stage7':frozen_hash(previous['spatial'][count]['layouts']),
                                          'stage8':frozen_hash(layouts)}
        assert r['candidate_pool_hashes'][count]['stage7']==r['candidate_pool_hashes'][count]['stage8']
        r['scores'][count]={}; r['references'][count]={}
        for label,weighted in (('unweighted',False),('weighted',True)):
            scores,refs=score_candidates(jac,values,schedules,layouts,scaling,rms,weighted=weighted)
            assert len(scores['ranking'])==14056
            r['scores'][count][label]=scores; r['references'][count][label]=refs
        selected={name:copy.deepcopy(d) for name,d in previous['selected'][count].items()}
        for label in ('unweighted','weighted'):
            name='robust_noise_weighted' if label=='weighted' else 'robust_unweighted'
            selected[name]=design_at(r['scores'][count][label]['ranking'][0],schedules,layouts)
        frozen['selected'][count]=selected
        r['important_scores'][count]={name:{label:score_one(jac,values,d,scaling,rms,weighted=label=='weighted')
                    for label in ('unweighted','weighted')} for name,d in selected.items()}
        if count=='4':
            order=r['scores'][count]['weighted']['ranking']; middle=len(order)//2
            frozen['diagnostic_selection']={'top':order[:3],'middle':order[middle-1:middle+2],'bottom':order[-3:]}
        print('Selected',count,{k:(v['times'],v['sensor_indices']) for k,v in selected.items()},flush=True)
    r['noise_assumption_checks']={}
    for count in ('8','4'):
        r['noise_assumption_checks'][count]={}
        for label,weighted,floor,relative in (('A_homoscedastic',False,.01,.03),('B_primary',True,.01,.03),('C',True,.02,.01),('D',True,.005,.05)):
            r['noise_assumption_checks'][count][label]={name:score_one(jac,values,frozen['selected'][count][name],scaling,rms,weighted,floor,relative)
                     for name in ('joint_E','robust_noise_weighted')}
    # Verify continuum derivatives at two points and both design initial fields.
    r['continuum_derivative_checks']=[]
    x=jnp.arange(N)*2*jnp.pi/N; sensor=jnp.array([1,7,18,26]); times=jnp.array([.25,1.75,2.])
    for ic in ('A','B'):
        for v,nu in (PARAMETERS[0],PARAMETERS[2]):
            theta=jnp.array([v,jnp.log(nu)])
            f=lambda t:continuum_predict(t,x,times,INITIAL_MODES[ic])[:,sensor].reshape(-1)
            J=jax.jacfwd(f)(theta)
            for col,eps in ((0,1e-6),(1,1e-5)):
                step=jnp.zeros(2).at[col].set(eps); fd=(f(theta+step)-f(theta-step))/(2*eps)
                r['continuum_derivative_checks'].append({'initial':ic,'v':v,'nu4':nu,'coordinate':['v','q'][col],
                    'epsilon':eps,'relative_l2_error':float(jnp.linalg.norm(J[:,col]-fd)/jnp.linalg.norm(fd)),
                    'max_absolute_error':float(jnp.max(jnp.abs(J[:,col]-fd)))})
    r['model_sensitivity_comparison']=[];r['mechanisms']={}
    modal={}
    for si,s in enumerate(scenarios):
        modal[si]=[evaluate_tensor(s,[mode])[1] for mode in INITIAL_MODES[s['initial']]]
    for count in ('8','4'):
        r['mechanisms'][count]={}
        for name in ('joint_E','robust_noise_weighted'):
            d=frozen['selected'][count][name];ti=np.array(d['time_indices'][1:])//32;mask=d['sensor_indices']
            entries=[]
            for si,s in enumerate(scenarios):
                J=jac[si,ti][:,mask,:]; y=values[si,ti][:,mask];flat=J.reshape(-1,2)
                weights=1/((.01*rms)**2+(.03*np.abs(y))**2)
                norm=np.linalg.norm(flat,axis=0)
                entries.append({'scenario':s['id'],'jacobian':J.tolist(),'column_norms':norm.tolist(),
                    'column_cosine':float(np.dot(flat[:,0],flat[:,1])/np.prod(norm)),
                    'squared_strength_by_time':np.sum(J**2,axis=1).tolist(),
                    'weighted_squared_strength_by_time':np.sum(J**2*weights[:,:,None],axis=1).tolist(),
                    'weight_min':float(weights.min()),'weight_max':float(weights.max()),
                    'modal_column_norms':{str(mode[0]):np.linalg.norm(modal[si][mi][ti][:,mask,:].reshape(-1,2),axis=0).tolist()
                                          for mi,mode in enumerate(INITIAL_MODES[s['initial']])}})
            r['mechanisms'][count][name]=entries
            for si,s in enumerate(scenarios[:6]):
                a=jac[si,ti][:,mask,:].reshape(-1,2);b=jac[si+6,ti][:,mask,:].reshape(-1,2)
                r['model_sensitivity_comparison'].append({'sensors':int(count),'design':name,'initial':s['initial'],
                    'parameter_id':s['parameter_id'],'relative_column_difference':(np.linalg.norm(a-b,axis=0)/np.linalg.norm(b,axis=0)).tolist(),
                    'column_cosines':(np.sum(a*b,axis=0)/(np.linalg.norm(a,axis=0)*np.linalg.norm(b,axis=0))).tolist()})
    r['frozen_hash']=frozen_hash(frozen)
    r['rankings_hash']=frozen_hash(r['scores'])
    save(r); print('Frozen hash:',r['frozen_hash'],flush=True)
    return r


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--design-only',action='store_true')
    parser.add_argument('--recover-frozen',action='store_true')
    args=parser.parse_args()
    r=json.loads(RESULT.read_text()) if args.recover_frozen else build_designs()
    if not args.design_only:
        from experiments.stage8_recovery import run_recovery
        run_recovery(r)
