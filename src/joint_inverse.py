"""Joint (velocity, log hyperdiffusion) inference using the validated solver."""

import math

import jax
import jax.numpy as jnp

from src.inverse import inverse_loss, predict_observations


def unpack_joint_parameters(theta):
    """Return signed velocity and positive hyperdiffusion from [v, log(nu4)]."""
    return theta[0], jnp.exp(theta[1])


def predict_joint_observations(theta, psi0, dt, num_steps, dx,
                               observation_time_indices, sensor_indices=None):
    return predict_observations(theta[1], psi0, dt, num_steps, theta[0], dx,
                                observation_time_indices, sensor_indices)


def joint_inverse_loss(theta, psi0, dt, num_steps, dx, observation_time_indices,
                       observations, sensor_indices=None):
    """Normalized MSE over observed entries only, identical to the scalar loss."""
    return inverse_loss(theta[1], psi0, dt, num_steps, theta[0], dx,
                        observation_time_indices, observations, sensor_indices)


def projected_adam_kernel(loss_fn, theta, lower, upper, iterations, learning_rate,
                          optimizer_state=None):
    """Pure single-fit kernel for JIT/vmap; callers validate inputs and outputs.

    Extracted unchanged from the Stage 6 optimizer so batched design validation
    reuses exactly the same updates, projection, and history convention.
    A supplied optimizer_state retains both moments and the absolute Adam step
    for bias correction; iterations is the number of additional updates.
    """
    if optimizer_state is None:
        m0, s0, start = jnp.zeros_like(theta), jnp.zeros_like(theta), 0
    else:
        theta = optimizer_state['theta']
        m0, s0 = optimizer_state['first_moment'], optimizer_state['second_moment']
        start = optimizer_state['update_count']
    value_and_grad = jax.value_and_grad(loss_fn)
    def update(state, iteration):
        theta,m,s = state
        value,gradient = value_and_grad(theta)
        m = 0.9*m+0.1*gradient
        s = 0.999*s+0.001*gradient**2
        proposal = theta-learning_rate*(m/(1-0.9**iteration))/(jnp.sqrt(s/(1-0.999**iteration))+1e-8)
        hits = (proposal<lower)|(proposal>upper)
        next_theta = jnp.clip(proposal,lower,upper)
        return (next_theta,m,s),(theta,value,gradient,hits)
    (final,final_m,final_s), (states,losses,gradients,hits) = jax.lax.scan(
        update,(theta,m0,s0),start+jnp.arange(1,iterations+1))
    final_loss,final_gradient = value_and_grad(final)
    history = jnp.concatenate((states,final[None,:]),axis=0)
    return {'optimizer_state': {'theta': final, 'first_moment': final_m,
                                'second_moment': final_s, 'update_count': jnp.asarray(start+iterations)},
            'theta': final, 'theta_history': history,
            'v_history': history[:,0], 'q_history': history[:,1],
            'nu4_history': jnp.exp(history[:,1]),
            'loss_history': jnp.concatenate((losses,final_loss[None])),
            'gradient_history': jnp.concatenate((gradients,final_gradient[None,:]),axis=0),
            'proposal_boundary_hits': jnp.sum(hits,axis=0),
            'parameter_bound_contacts': jnp.sum((history<=lower)|(history>=upper),axis=0)}


def optimize_joint_parameters(loss_fn, initial, v_bounds=(0.5,1.5),
                              nu4_bounds=(1e-4,0.02), iterations=800,
                              learning_rate=0.05):
    """Vector projected Adam; initial is the physical pair (v, nu4).

    Return all initial/pre-update states plus the final state and gradient.
    Bounds and optimization settings contain no reference to physical truth.
    """
    vlo,vhi = v_bounds
    nlo,nhi = nu4_bounds
    if not (vlo < vhi and 0 < nlo < nhi and vlo <= initial[0] <= vhi
            and nlo <= initial[1] <= nhi):
        raise ValueError('Initial physical pair must lie in ordered bounds')
    if iterations < 1 or learning_rate <= 0:
        raise ValueError('Require positive iterations and learning rate')
    lower = jnp.array([vlo,math.log(nlo)])
    upper = jnp.array([vhi,math.log(nhi)])
    theta0 = jnp.array([initial[0],math.log(initial[1])])
    run = jax.jit(lambda theta: projected_adam_kernel(
        loss_fn, theta, lower, upper, iterations, learning_rate))
    result = run(theta0)
    if not all(bool(jnp.all(jnp.isfinite(result[key]))) for key in
               ('theta_history','loss_history','gradient_history','nu4_history')):
        raise FloatingPointError('Nonfinite joint optimizer history')
    return result


def hessian_diagnostics(loss_fn, theta):
    """Local geometry in (v,q), not covariance or statistical correlation.

    Eigenvectors are columns, ordered from smallest to largest eigenvalue.
    Null condition number indicates no numerically resolved positive definiteness.
    """
    raw = jax.jit(jax.hessian(loss_fn))(theta)
    hessian = (raw+raw.T)/2
    if not bool(jnp.all(jnp.isfinite(hessian))):
        raise FloatingPointError('Nonfinite Hessian')
    eigenvalues,eigenvectors = jnp.linalg.eigh(hessian)
    tolerance = 100*jnp.finfo(hessian.dtype).eps*max(1.,float(jnp.max(jnp.abs(eigenvalues))))
    positive = bool(jnp.all(eigenvalues>tolerance))
    coupling = (float(hessian[0,1]/jnp.sqrt(hessian[0,0]*hessian[1,1]))
                if float(hessian[0,0])>0 and float(hessian[1,1])>0 else None)
    return {'matrix': hessian.tolist(), 'eigenvalues': eigenvalues.tolist(),
            'eigenvectors_columns': eigenvectors.tolist(),
            'positive_definite': positive, 'eigenvalue_tolerance': tolerance,
            'condition_number': float(eigenvalues[-1]/eigenvalues[0]) if positive else None,
            'normalized_hessian_coupling': coupling}
