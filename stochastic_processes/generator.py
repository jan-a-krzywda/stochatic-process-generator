"""
1/f noise generation: IFFT, Over_f_noise (NumPy), and JAX-accelerated OU sum.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

from stochastic_processes.over_f import Over_f_noise, OU_noise


class NoiseGenerator:
    def __init__(self, dt: float):
        self.dt = dt

    def generate_1f_ifft(self, n_steps: int) -> np.ndarray:
        """
        Generate 1/f-like noise via IFFT (frequency-domain amplitudes ~ 1/sqrt(f)).
        Fast, grid-based; useful for comparison with OU-based generation.
        """
        frequencies = np.fft.rfftfreq(n_steps, d=self.dt)
        amplitudes = np.zeros_like(frequencies)
        amplitudes[1:] = 1.0 / np.sqrt(frequencies[1:])
        phases = np.random.uniform(0, 2 * np.pi, size=len(frequencies))
        spectrum = amplitudes * np.exp(1j * phases)
        noise = np.fft.irfft(spectrum, n=n_steps)
        return noise / np.std(noise)

    @staticmethod
    @jax.jit
    def _ou_scan(carry, noise_row):
        x_prev, decay, noise_std = carry
        x_next = x_prev * decay + noise_std * noise_row
        return (x_next, decay, noise_std), x_next

    def generate_ou_jax(
        self,
        n_steps: int,
        gammas: np.ndarray,
        sigmas: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generate 1/f-like noise as sum of OU processes using JAX (lax.scan).
        Use same gammas/sigmas as an Over_f_noise instance for comparable spectrum.
        gammas: relaxation rates (1/tc), sigmas: per-fluctuator std.
        """
        g = jnp.array(gammas)
        s = jnp.array(sigmas)
        tc = 1.0 / g
        decay = jnp.exp(-self.dt / tc)
        noise_std = s * jnp.sqrt(1 - jnp.exp(-2 * self.dt / tc))
        x0 = jnp.zeros(len(gammas))
        key = random.PRNGKey(seed)
        noise_matrix = random.normal(key, shape=(n_steps, len(gammas)))
        _, x_all = jax.lax.scan(
            self._ou_scan, (x0, decay, noise_std), noise_matrix
        )
        total = jnp.sum(x_all, axis=1)
        return np.asarray(total / jnp.std(total))

    def generate_over_f(
        self,
        n_steps: int,
        S1: float,
        ommax: float,
        ommin: float,
        n_fluctuators: int = 20,
        sigma_couplings: float = 0.0,
        fluctuator_class=OU_noise,
        x0=None,
        equally_dist: bool = False,
    ) -> np.ndarray:
        """
        Generate 1/f noise via Over_f_noise.gen_trajectory with constant dt.
        """
        over_f = Over_f_noise(
            n_fluctuators=n_fluctuators,
            S1=S1,
            sigma_couplings=sigma_couplings,
            ommax=ommax,
            ommin=ommin,
            fluctuator_class=fluctuator_class,
            x0=x0,
            equally_dist=equally_dist,
        )
        times = np.full(n_steps, self.dt)
        trajectory = np.array(over_f.gen_trajectory(times))
        return trajectory / np.std(trajectory)
