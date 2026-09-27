"""Uniform Stage 7 stationarity continuation; historical results stay intact."""

import copy
import hashlib
import json
import math
import statistics

import jax
import jax.numpy as jnp

from src.joint_inverse import projected_adam_kernel, joint_inverse_loss, predict_joint_observations
from src.observations import add_measurement_noise
from src.design import rank_correlation

INITIAL_UPDATES = 2000
MAX_UPDATES = 4000
THRESHOLD = 1e-7


def kkt_residual(theta, gradient, lower, upper):
    """The original Stage 7 projected/KKT criterion, unchanged."""
    residual = jnp.where((theta <= lower + 1e-8) & (gradient >= 0), 0., gradient)
    residual = jnp.where((theta >= upper - 1e-8) & (gradient <= 0), 0., residual)
    return jnp.linalg.norm(residual)


def continue_if_required(loss, state, gradient, lower, upper,
                         maximum_updates=MAX_UPDATES):
    """Continue to the fixed cap iff the existing residual fails its cutoff."""
    residual = float(kkt_residual(state['theta'], gradient, lower, upper))
    if not math.isfinite(residual):
        raise FloatingPointError('Nonfinite stationarity residual')
    if residual < THRESHOLD:
        return None
    remaining = maximum_updates - int(state['update_count'])
    if remaining <= 0:
        return None
    return jax.jit(lambda: projected_adam_kernel(
        loss, state['theta'], lower, upper, remaining, .05,
        optimizer_state=state))()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def harden_results(results):
    """Apply the same residual-only policy to every stored production fit.

    Legacy files lack optimizer moments. Replaying the identical original
    initialization reconstructs those moments; terminal parameters and gradients
    must agree with the historical record before any continuation is permitted.
    Historical recovery records, summaries, selections, and scores are immutable.
    """
    from experiments.stage7_observation_design import (
        setup, DT, STEPS, N, CANDIDATE_TIMES, summary, freeze_hash)
    from experiments.stage3_parameter_inference import continuum_observations

    protected_keys = ('configuration', 'temporal', 'temporal_rankings', 'spatial',
                      'selected', 'frozen_selection_hash', 'association_selection',
                      'recovery', 'association_pairs', 'association_rank_correlation',
                      'checks')
    before = digest({key: results[key] for key in protected_keys})
    x, psi0, nominal = setup()
    lower = jnp.array([.5, jnp.log(1e-4)])
    upper = jnp.array([1.5, jnp.log(.02)])
    initial = jnp.array([.7, jnp.log(.0005)])
    truth = predict_joint_observations(nominal, psi0, DT, STEPS, 2*jnp.pi/N, CANDIDATE_TIMES)
    continuum = continuum_observations(x, CANDIDATE_TIMES*DT, 1., .002)
    hardened = {'policy': {
        'initial_update_budget': INITIAL_UPDATES, 'maximum_total_updates': MAX_UPDATES,
        'stationarity_threshold': THRESHOLD, 'learning_rate': .05,
        'beta1': .9, 'beta2': .999, 'epsilon': 1e-8,
        'decision': 'Continue iff residual >= threshold; continue same state to 4000 total updates.',
        'legacy_state_reconstruction': 'Replay identical first 2000 updates only when saved moments are absent; verify against historical endpoint before continuation.'},
        'historical_records_hash': before, 'recovery': {}}
    continued = 0
    for key, group in results['recovery'].items():
        rows = []
        for original in group['runs']:
            row = copy.deepcopy(original)
            required = original['kkt_residual'] >= THRESHOLD
            metadata = {'initial_update_budget': INITIAL_UPDATES,
                        'continuation_required': required, 'final_total_updates': INITIAL_UPDATES,
                        'final_kkt_residual': original['kkt_residual'],
                        'parameter_changes': {'v': 0., 'q': 0., 'nu4': 0.},
                        'state_source': 'not_needed'}
            if required:
                times = jnp.array(original['time_indices'])
                sensors = jnp.array(original['sensor_indices'])
                clean_full = truth if original['truth_model'] == 'matched' else continuum
                clean = clean_full[times//32][:, sensors]
                data, _ = add_measurement_noise(clean, times, original['sigma_rel'], original['noise_seed'])
                loss = lambda theta: joint_inverse_loss(theta, psi0, DT, STEPS, 2*jnp.pi/N, times, data, sensors)
                if 'optimizer_state' in original:
                    state = {k: jnp.asarray(v) for k, v in original['optimizer_state'].items()}
                    gradient = jnp.asarray(original['gradient'])
                    metadata['state_source'] = 'saved_optimizer_state'
                else:
                    replay = jax.jit(lambda: projected_adam_kernel(
                        loss, initial, lower, upper, INITIAL_UPDATES, .05))()
                    state = replay['optimizer_state']
                    gradient = replay['gradient_history'][-1]
                    metadata['state_source'] = 'verified_original_history_replay'
                    theta_difference = float(jnp.max(jnp.abs(state['theta']-jnp.array([original['v'], original['q']]))))
                    gradient_difference = float(jnp.max(jnp.abs(gradient-jnp.array(original['gradient']))))
                    metadata['replay_max_theta_difference'] = theta_difference
                    metadata['replay_max_gradient_difference'] = gradient_difference
                    if theta_difference > 1e-11 or gradient_difference > 1e-11:
                        raise RuntimeError('Reconstructed history does not match original endpoint')
                if int(state['update_count']) != INITIAL_UPDATES:
                    raise ValueError('Production state must contain exactly 2000 initial updates')
                metadata['state_at_continuation'] = {k: v.tolist() for k, v in state.items()}
                fit = continue_if_required(loss, state, gradient, lower, upper)
                if fit is None:
                    raise RuntimeError('Reconstructed continuation decision differs from historical criterion')
                if not all(bool(jnp.all(jnp.isfinite(fit[k]))) for k in
                           ('theta_history', 'gradient_history', 'loss_history')):
                    raise FloatingPointError('Nonfinite continuation history')
                v, q = fit['theta'].tolist(); nu = math.exp(q)
                gradient = fit['gradient_history'][-1]
                residual = float(kkt_residual(fit['theta'], gradient, lower, upper))
                final = predict_joint_observations(fit['theta'], psi0, DT, STEPS, 2*jnp.pi/N, jnp.array([256]))[0]
                row.update(v=v, q=q, nu4=nu, relative_v_error=abs(v-1),
                           relative_nu4_error=abs(nu-.002)/.002, signed_v_error=v-1,
                           signed_nu4_error=nu-.002, final_loss=float(fit['loss_history'][-1]),
                           gradient=gradient.tolist(), kkt_residual=residual,
                           clean_field_l2=float(jnp.linalg.norm(final-clean_full[-1])/jnp.linalg.norm(clean_full[-1])),
                           at_bound=bool(v<=.5+1e-8 or v>=1.5-1e-8 or nu<=1e-4*(1+1e-8) or nu>=.02*(1-1e-8)),
                           optimizer_state={k: a.tolist() for k, a in fit['optimizer_state'].items()})
                row['proposal_hits'] = (jnp.array(original['proposal_hits'])+fit['proposal_boundary_hits']).tolist()
                # The shared endpoint belongs to the original history, not twice.
                shared_contact = (state['theta']<=lower)|(state['theta']>=upper)
                row['bound_contacts'] = (jnp.array(original['bound_contacts'])+fit['parameter_bound_contacts']-shared_contact).tolist()
                row['parameter_min'] = jnp.minimum(jnp.array(original['parameter_min']),jnp.min(fit['theta_history'],axis=0)).tolist()
                row['parameter_max'] = jnp.maximum(jnp.array(original['parameter_max']),jnp.max(fit['theta_history'],axis=0)).tolist()
                if original['truth_model'] == 'continuum':
                    row['observation_shift_v'] = v-results['continuum_reference']['v']
                    row['observation_shift_nu4'] = nu-results['continuum_reference']['nu4']
                metadata.update(final_total_updates=int(fit['optimizer_state']['update_count']),
                                final_kkt_residual=residual,
                                parameter_changes={'v':v-original['v'],'q':q-original['q'],'nu4':nu-original['nu4']})
                continued += 1
                print('Continued', key, original['design'], original['noise_seed'], metadata, flush=True)
            row['continuation'] = metadata
            rows.append(row)
        hardened['recovery'][key] = {'runs': rows}
    # Original summary objects are retained separately, never overwritten.
    for key, group in hardened['recovery'].items():
        original = results['recovery'][key]
        group['by_design'] = ({name:summary([a for a in group['runs'] if a['design']==name])
                               for name in original['by_design']} if continued else copy.deepcopy(original['by_design']))
        if key.startswith('continuum_'):
            for parameter in ('v','nu4'):
                group['mean_observation_shift_'+parameter] = statistics.mean(a['observation_shift_'+parameter] for a in group['runs'])
    hardened['association_pairs'] = []
    for pair in results['association_pairs']:
        item = dict(pair)
        item['median_nu4_error'] = hardened['recovery']['association_'+pair['tier']]['by_design'][f"candidate_{pair['candidate_index']}"]['median_relative_nu4_error']
        hardened['association_pairs'].append(item)
    hardened['association_rank_correlation'] = rank_correlation(
        [a['score'] for a in hardened['association_pairs']],
        [a['median_nu4_error'] for a in hardened['association_pairs']])
    rows = [a for g in hardened['recovery'].values() for a in g['runs']]
    after = digest({key: results[key] for key in protected_keys})
    assert before == after
    checks = dict(results['checks'])
    checks.update(stationary=all(a['kkt_residual']<THRESHOLD for a in rows),
                  within_update_cap=all(a['continuation']['final_total_updates']<=MAX_UPDATES for a in rows),
                  historical_results_and_protocol_preserved=before==after,
                  frozen_designs=freeze_hash(results['selected'])==results['frozen_selection_hash'],
                  stable_bounds=all(a['parameter_min'][0]>=.5-1e-12 and a['parameter_max'][0]<=1.5+1e-12 and a['parameter_min'][1]>=math.log(1e-4)-1e-12 and a['parameter_max'][1]<=math.log(.02)+1e-12 for a in rows))
    # Exact equality is a conservative sufficient check for unchanged scientific
    # comparisons here. Other datasets require review rather than an automatic
    # claim that a changed comparison is immaterial.
    comparisons_unchanged = all(
        group['by_design'] == hardened['recovery'][key]['by_design']
        for key, group in results['recovery'].items() if not key.startswith('association_'))
    association_unchanged = results['association_pairs'] == hardened['association_pairs']
    checks['scientific_conclusions_unchanged'] = comparisons_unchanged and association_unchanged
    hardened['scientific_comparison'] = {
        'primary_and_continuum_summaries_identical': comparisons_unchanged,
        'all_association_layout_medians_identical': association_unchanged,
        'original_rank_correlation': results['association_rank_correlation'],
        'hardened_rank_correlation': hardened['association_rank_correlation'],
        'rank_correlation_change': hardened['association_rank_correlation']-results['association_rank_correlation']}
    hardened['fits_requiring_continuation'] = continued
    hardened['checks'] = checks
    hardened['experiment_conclusion'] = 'PASS' if all(checks.values()) else 'NEEDS ATTENTION'
    results.setdefault('original_experiment_conclusion', results['experiment_conclusion'])
    results['hardening'] = hardened
    results['experiment_conclusion'] = hardened['experiment_conclusion']
    return hardened
