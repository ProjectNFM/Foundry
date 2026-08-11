import numpy as np

from foundry.models.poyo_eeg import create_linspace_latent_tokens


class TestCreateLinspaceLatentTokens:
    def test_basic_shape(self):
        idx, ts = create_linspace_latent_tokens(
            0.0, 1.0, step=0.1, num_latents_per_step=1
        )
        expected_steps = len(np.arange(0, 1.0, 0.1))
        assert idx.shape == (expected_steps,)
        assert ts.shape == (expected_steps,)

    def test_multiple_latents_per_step(self):
        n_latents = 3
        idx, ts = create_linspace_latent_tokens(
            0.0, 1.0, step=0.25, num_latents_per_step=n_latents
        )
        num_steps = len(np.arange(0, 1.0, 0.25))
        assert len(idx) == num_steps * n_latents
        assert len(ts) == num_steps * n_latents

    def test_timestamps_at_bin_centres(self):
        step = 0.2
        idx, ts = create_linspace_latent_tokens(
            0.0, 1.0, step=step, num_latents_per_step=1
        )
        expected = np.arange(0, 1.0, step) + step / 2
        np.testing.assert_allclose(ts, expected)

    def test_nonzero_start(self):
        start, end, step = 2.0, 3.0, 0.5
        idx, ts = create_linspace_latent_tokens(
            start, end, step=step, num_latents_per_step=1
        )
        expected = np.arange(0, end - start, step) + step / 2 + start
        np.testing.assert_allclose(ts, expected)

    def test_latent_index_tiling(self):
        n_latents = 4
        idx, ts = create_linspace_latent_tokens(
            0.0, 1.0, step=0.5, num_latents_per_step=n_latents
        )
        num_steps = len(np.arange(0, 1.0, 0.5))
        for s in range(num_steps):
            group = idx[s * n_latents : (s + 1) * n_latents]
            np.testing.assert_array_equal(group, np.arange(n_latents))

    def test_timestamps_repeated_within_step(self):
        n_latents = 3
        step = 0.25
        idx, ts = create_linspace_latent_tokens(
            0.0, 1.0, step=step, num_latents_per_step=n_latents
        )
        num_steps = len(np.arange(0, 1.0, step))
        for s in range(num_steps):
            group_ts = ts[s * n_latents : (s + 1) * n_latents]
            assert np.all(group_ts == group_ts[0])

    def test_dtype(self):
        idx, ts = create_linspace_latent_tokens(
            0.0, 1.0, step=0.1, num_latents_per_step=2
        )
        assert idx.dtype == np.int64

    def test_single_step(self):
        idx, ts = create_linspace_latent_tokens(
            0.0, 0.1, step=0.1, num_latents_per_step=1
        )
        assert len(idx) == 1
        assert len(ts) == 1
        np.testing.assert_allclose(ts, [0.05])

    def test_step_larger_than_range_produces_single_token(self):
        idx, ts = create_linspace_latent_tokens(
            0.0, 0.5, step=1.0, num_latents_per_step=1
        )
        assert len(idx) == 1
        assert len(ts) == 1
        np.testing.assert_allclose(ts, [0.5])


class TestDynamicLatentGrid:
    """Verify that using _num_latent_bins + effective_step produces a
    fixed token count across different window durations."""

    def test_fixed_token_count_across_durations(self):
        """All durations produce the same number of latent tokens."""
        base_seq_len = 2.0
        latent_step = 0.1
        num_latents_per_step = 16
        num_bins = round(base_seq_len / latent_step)

        durations = [1.0, 2.0, 5.0, 10.0, 30.0]
        expected_count = num_bins * num_latents_per_step

        for dur in durations:
            effective_step = dur / num_bins
            idx, ts = create_linspace_latent_tokens(
                0, dur, step=effective_step,
                num_latents_per_step=num_latents_per_step,
            )
            assert len(idx) == expected_count, (
                f"Duration {dur}s produced {len(idx)} tokens, "
                f"expected {expected_count}"
            )
            assert len(ts) == expected_count

    def test_baseline_duration_matches_original(self):
        """For the baseline duration, effective_step == latent_step."""
        base_seq_len = 2.0
        latent_step = 0.1
        num_latents_per_step = 16
        num_bins = round(base_seq_len / latent_step)

        effective_step = base_seq_len / num_bins
        np.testing.assert_allclose(effective_step, latent_step)

        idx_dyn, ts_dyn = create_linspace_latent_tokens(
            0, base_seq_len, step=effective_step,
            num_latents_per_step=num_latents_per_step,
        )
        idx_orig, ts_orig = create_linspace_latent_tokens(
            0, base_seq_len, step=latent_step,
            num_latents_per_step=num_latents_per_step,
        )
        np.testing.assert_array_equal(idx_dyn, idx_orig)
        np.testing.assert_allclose(ts_dyn, ts_orig)

    def test_timestamps_span_full_duration(self):
        """Latent timestamps should cover [0, duration) evenly."""
        num_bins = 20
        num_latents_per_step = 16

        for dur in [1.0, 5.0, 30.0]:
            effective_step = dur / num_bins
            _, ts = create_linspace_latent_tokens(
                0, dur, step=effective_step,
                num_latents_per_step=num_latents_per_step,
            )
            unique_ts = np.unique(ts)
            assert len(unique_ts) == num_bins
            assert unique_ts[0] > 0
            assert unique_ts[-1] < dur
