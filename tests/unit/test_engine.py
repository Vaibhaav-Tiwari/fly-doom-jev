import numpy as np

from flydoom.neural import LIFEngine


def test_engine_interface_and_spiking(fixture_connectome):
    eng = LIFEngine(fixture_connectome, timestep_ms=2.0)
    eng.reset()
    assert eng.get_activity().shape == (fixture_connectome.n_neurons,)
    sensory = fixture_connectome.population_indices("visual_projection")
    eng.inject_input(sensory, np.full(len(sensory), 30.0, dtype=np.float32))
    eng.step(20)
    pop = eng.get_population_activity()
    assert set(pop) >= {"visual_projection", "descending_neuron"}
    assert pop["visual_projection"]["mean_rate_hz"] > 0  # driven population fires
    motor = eng.get_motor_output()
    assert motor.shape == (fixture_connectome.population_sizes()["descending_neuron"],)


def test_engine_reset_clears_state(fixture_connectome):
    eng = LIFEngine(fixture_connectome)
    eng.inject_input(np.array([0]), np.array([50.0], dtype=np.float32))
    eng.step(10)
    eng.reset()
    assert eng.time_ms == 0.0
    assert np.all(eng.get_activity() == 0.0)


def test_engine_deterministic_given_seed(fixture_connectome):
    a = LIFEngine(fixture_connectome, noise=1.0, seed=7)
    b = LIFEngine(fixture_connectome, noise=1.0, seed=7)
    a.step(50)
    b.step(50)
    np.testing.assert_array_equal(a.get_activity(), b.get_activity())
