"""The public engine: a frozen timing package plus a JAX residual pipeline.

One pulsar, built once, immutable afterwards. The object satisfies nltiming's
timing-engine protocols structurally without importing nltiming;
:class:`~vela_jax.backend.VelaJaxTimingEngine` supplies the twenty renames.

Two independent choices, both static:

``timing_package``
    who reads the files. :meth:`Engine.from_files` (default
    ``timing_package="pint"``) and :meth:`Engine.from_pint` use PINT.
    :meth:`Engine.from_tempo2` (also ``from_files(..., timing_package="tempo2")``)
    lets tempo2 do the reading -- clocks, ``TRACK -2`` pulse numbers, INCLUDE
    trees, site and ephemeris vectors -- and then freezes *its* arrays. The
    delay physics is identical either way.

``binary_conventions``
    whose binary-input conventions to apply. ``"pint"`` is Vela's own.
    ``"tempo2"`` uses tempo2's ELL1 Roemer truncation
    (:mod:`vela_jax.binary.ell1`). It is a separate flag from the timing
    package on purpose: a tempo2-read ELL1 file is usually wanted with
    ``"tempo2"``, but the two are testable in isolation.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from psrdata import ResidualCentering, TOARows

from .binary import binary_facts, uses_fbx
from .binary.facts import BinaryFacts
from .config import require_x64
from .errors import UnsupportedModelError
from .freeze import (
    account_for_parameters,
    build_delta_sequences,
    build_sequences,
    freeze,
    load_pint,
)
from .params import build_layout
from .pipeline import (
    binary_family,
    build_chain,
    form_residuals,
    run_chain,
    validate_conventions,
)

#: Vela's ``form_residuals`` removes no mean. ``standard_output`` describes
#: the PINT-family convention a consumer would see if it *did* reduce these
#: residuals. The type is psrdata's because the value is serialized in the
#: record.
RESIDUAL_CENTERING = ResidualCentering(
    stored_residuals="none",
    standard_output="mean_removed",
    standard_weighted=True,
)


#: Parameters whose absolute value cannot be reconstructed from a float64 fit
#: coordinate at the precision the engine works to (SPEC §4.7).
#: ``TGEO`` is here even though freeze refuses a free one: every epoch is
#: unsafe as an absolute float64 coordinate.
_PRECISION_CRITICAL = frozenset(
    {"F0", "PEPOCH", "POSEPOCH", "DMEPOCH", "T0", "TASC", "TGEO", "PB", "FB0"}
)

#: Parameters whose delay is affine in the parameter (up to the second-order
#: feedback of the delay into later stages, which is far below a ps here).
_IDENTICALLY_LINEAR_NAMES = frozenset({"PHOFF", "DM"})
_IDENTICALLY_LINEAR_PREFIXES = ("DM", "DMX_", "JUMP", "FD")


class Engine:
    """Vela's delay chain over one pulsar, as a differentiable JAX function."""

    #: tempo2's raw per-TOA state, when :meth:`from_tempo2` built this engine.
    #: Diagnostics only -- the chain reads :attr:`frozen`.
    tempo2_columns = None

    def __init__(
        self,
        model,
        toas,
        *,
        binary_conventions: str = "pint",
        timing_package: str = "pint",
        source_units: str = "TDB",
        par_text: str | None = None,
        pulse_number_source: str = "model",
        obliquity: float | None = None,
        jit: bool = True,
    ):
        require_x64()
        import jax

        if timing_package not in ("pint", "tempo2"):
            raise ValueError(
                f"timing_package must be 'pint' or 'tempo2'; got {timing_package!r}"
            )
        self.pint_model = model
        self.pint_toas = toas
        self.binary_conventions = validate_conventions(binary_conventions)
        self.timing_package = timing_package
        self.pulse_number_source = pulse_number_source
        self._source_units = source_units
        self._par_text = par_text
        self._fixed_obliquity = obliquity

        sequences = build_sequences(model)
        delta_sequences = build_delta_sequences(model)
        free_names = tuple(model.free_params)
        self.layout = build_layout(
            model,
            free_names=free_names,
            sequences=sequences,
            delta_sequences=delta_sequences,
        )

        family = binary_family(model)
        self.frozen, self.toa_columns = freeze(
            model,
            toas,
            self.layout,
            family=family,
            use_fbx=uses_fbx(model) if family else False,
        )
        self.chain = build_chain(
            model,
            sequences,
            delta_sequences,
            self.frozen,
            conventions=self.binary_conventions,
            obliquity=obliquity,
        )

        unconsumed = sorted(set(free_names) - self.chain.consumed)
        if unconsumed:
            raise UnsupportedModelError(
                f"free parameters no component evaluates: {unconsumed}"
            )
        account_for_parameters(model, toas, self.chain.consumed)

        self.param_names = self.layout.names
        self.param_units = dict(self.layout.units)
        self.toa_count = self.frozen.n_toas
        self.stages = self.chain.names

        self._binary_facts = (
            binary_facts(
                model,
                family,
                self.chain.use_fbx,
                conventions=self.binary_conventions,
            )
            if family
            else None
        )
        self._residuals_jax = (
            jax.jit(self._residuals_raw) if jit else self._residuals_raw
        )
        self._reference = np.asarray(
            self._residuals_jax(np.zeros(len(self.param_names))), dtype=float
        )
        self._pint_design = None
        self._jacobian = None
        self._pulsar_data = None
        self._reference_pass_cache = None
        self._reference_pass_count = 0

    # --- construction ------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        par,
        tim,
        *,
        timing_package: str = "pint",
        binary_conventions: str = "pint",
        jit: bool = True,
        **kwargs,
    ) -> "Engine":
        """Read par/tim. ``timing_package`` chooses PINT (default) or tempo2."""
        if timing_package == "tempo2":
            return cls.from_tempo2(
                par, tim, binary_conventions=binary_conventions, jit=jit, **kwargs
            )
        if timing_package != "pint":
            raise ValueError(
                f"timing_package must be 'pint' or 'tempo2'; got {timing_package!r}"
            )
        model, toas = load_pint(par, tim, **kwargs)
        return cls(model, toas, binary_conventions=binary_conventions, jit=jit)

    @classmethod
    def from_tempo2(
        cls,
        par,
        tim,
        *,
        binary_conventions: str = "pint",
        jit: bool = True,
        pulsar=None,
        **pulsar_kwargs,
    ) -> "Engine":
        """Alias: tempo2 reads the files; the physics is unchanged.

        Use this when the file needs tempo2 to open it at all -- an INCLUDE
        tree, a clock chain PINT does not carry, ``TRACK -2`` pulse numbers.
        Pair it with ``binary_conventions="tempo2"`` to also adopt tempo2's
        ELL1 Roemer truncation. Equivalent to :meth:`from_files` with
        ``timing_package="tempo2"``.

        ``pulsar`` accepts an already-built ``libstempo.sandbox.tempopulsar``
        whose par must be the TDB one; otherwise one is built, with
        ``dofit=False``. There is no native-libstempo path.
        """
        from . import read_tempo2

        model, toas, package, columns = read_tempo2.load(
            par, tim, pulsar=pulsar, **pulsar_kwargs
        )
        engine = cls(
            model,
            toas,
            binary_conventions=binary_conventions,
            timing_package=package["timing_package"],
            source_units=package["source_units"],
            par_text=package["par_text"],
            pulse_number_source=package["pulse_number_source"],
            obliquity=package["obliquity"],
            jit=jit,
        )
        engine.tempo2_columns = columns
        return engine

    @classmethod
    def from_pint(
        cls, model, toas, *, binary_conventions: str = "pint", jit: bool = True
    ) -> "Engine":
        """Freeze a caller-built PINT pair; the same invariants are checked.

        ``prepare_model`` is idempotent, so a pair that has already been
        through pyvela's ``fix_params`` is accepted unchanged. Mutates
        ``model`` (``prepare_model``): PHOFF is freed and the zeroable
        parameters are set. No copy is taken.
        """
        from .freeze import prepare_model

        prepare_model(model, toas)
        if toas.get_pulse_numbers() is None:
            toas.compute_pulse_numbers(model)
        return cls(model, toas, binary_conventions=binary_conventions, jit=jit)

    # --- residuals ---------------------------------------------------------

    def _residuals_raw(self, delta):
        params = self.layout.build(delta)
        return form_residuals(self.frozen, run_chain(self.frozen, self.chain, params))

    def _reference_pass(self):
        """One snapshotting chain run at ``theta*``, cached for the engine's life.

        Everything the reference channel is asked for -- the full correction,
        the barycentric cutoff, the topocentric spin frequency -- comes out of
        this single pass. Before it was cached, building one pulsar record ran
        the chain three times over identical inputs.

        ``theta*`` does not move, so the cache never invalidates.
        """
        if self._reference_pass_cache is None:
            params = self.layout.build(np.zeros(len(self.param_names)))
            self._reference_pass_cache = run_chain(
                self.frozen, self.chain, params, snapshot=True
            )
            self._reference_pass_count += 1
        return self._reference_pass_cache

    def reference_correction(self):
        """The full :class:`Correction` at ``theta*``.

        This is the reference channel the perturbative engine expands around,
        and source **(E)** for every geometric field of the pulsar record.
        """
        return self._reference_pass()[0]

    #: Kept as a private alias: the perturbative engine and older callers use it.
    _reference_correction = reference_correction

    def toa_rows(self) -> TOARows:
        """``(stoas, freqs, toaerrs)`` -- the three columns that identify a row.

        The pulsar record carries the same three, and a consumer that builds an
        engine from the same par/tim compares them to check it got the same
        rows. Deliberately does *not* touch the Jacobian: identifying a row is
        not worth an ``N x n_par`` forward-mode pass (R9.6).
        """
        n = self.toa_count
        _, doppler = self.reference_barycentric()
        freq_hz = np.asarray(self.frozen.freq_hz, dtype=float)[:n]
        finite = np.asarray(self.frozen.finite_freq)[:n]
        # The freeze substitutes a finite placeholder for an infinite observing
        # frequency so the dispersive derivative stays defined (R3.7). That
        # placeholder is an artefact of the graph, not a measurement: the row
        # really had no frequency, which is what PINT's
        # `barycentric_radio_freq` reports, so the mask puts `inf` back.
        freqs = np.where(finite, freq_hz * (1.0 - doppler) / 1.0e6, np.inf)
        return TOARows(
            stoas=np.asarray(self.toa_columns.sat_seconds, dtype=float),
            freqs=freqs,
            toaerrs=np.asarray(self.toa_columns.toaerrs, dtype=float),
        )

    def gauge_direction(self) -> np.ndarray:
        """The residual direction an unmeasurable phase offset moves.

        ``1/F_i`` over the pulsar-frame spin frequency -- exactly the ``PHOFF``
        column of :meth:`design_matrix` (R9.3), both coming from the one
        divisor (§8). That frequency drifts only through ``F1`` and up, so the
        direction is very nearly the constant vector nltiming's gauge check
        assumes; while the divisor carried the annual doppler it was not.
        Lives on the engine so a consumer can ask without constructing an
        nltiming adapter.
        """
        return 1.0 / self.reference_spin_frequency()

    def reference_barycentric(self):
        """``(delay, doppler)`` at PINT's barycentric cutoff, per TOA (R-B1.4).

        The accumulated delay and Doppler shift after the dispersion stages
        and before the binary (after *every* delay stage for an isolated
        pulsar), taken during the reference pass. ``PulsarData.toas`` and
        ``PulsarData.freqs`` are built from these; the fully corrected TOA is
        engine-internal and is never published as ``toas``.
        """
        bary = self._reference_pass()[1]
        return (
            np.asarray(bary.delay, dtype=float)[:-1],
            np.asarray(bary.doppler, dtype=float)[:-1],
        )

    def residual_delta_jax(self, delta_theta):
        """Traced ``r(theta*+delta) - r(theta*)``; the only traced input is delta."""
        return self._residuals_jax(delta_theta) - self._reference

    def residual_delta(self, delta_theta) -> np.ndarray:
        delta = np.asarray(delta_theta, dtype=float).reshape(-1)
        if delta.shape != (len(self.param_names),):
            raise ValueError(
                f"delta_theta has shape {delta.shape}, "
                f"expected ({len(self.param_names)},)"
            )
        if not delta.any():
            return np.zeros(self.toa_count)
        return np.asarray(self.residual_delta_jax(delta), dtype=float)

    def residuals(self, theta=None) -> np.ndarray:
        """Absolute ``r(theta)`` in seconds, ``theta`` in PINT units.

        The delta is formed in ``Decimal`` against the exact reference strings,
        never in float64 inside the trace.
        """
        if theta is None:
            return self._reference.copy()
        return self._reference + self.residual_delta(
            self.layout.delta_from_theta(theta)
        )

    # --- derivatives -------------------------------------------------------

    def residual_jacobian(self) -> np.ndarray:
        """``J`` with ``r(theta*+delta) ~= r(theta*) + J delta`` (forward-mode).

        An ``N x n_par`` forward-mode evaluation with its own compile -- a real
        cost on J1713-class data, which is why it is computed once and cached
        here (R9.6). A consumer that wants a frozen-parameter analysis without
        paying it uses a previously written feather plus nltiming's
        ``LinearTimingEngine``, never a rebuilt engine and never PINT's matrix
        substituted for this one.
        """
        if self._jacobian is None:
            import jax

            zero = np.zeros(len(self.param_names))
            self._jacobian = np.asarray(
                jax.jacfwd(self._residuals_jax)(zero), dtype=float
            )
        return self._jacobian

    def design_matrix(self, *, source: str = "jacobian") -> np.ndarray:
        """The design matrix in fitter sign, ``param_names`` order (R9.3).

        ``source="jacobian"`` (the default, and the *product*) returns

            ``M = -residual_jacobian()``

        so that ``r(theta*+delta) ~= r(theta*) - M delta`` with ``J = -M``
        exactly, by construction rather than by gate. This is the matrix the
        pulsar record carries, the matrix Enterprise and Discovery marginalize,
        and the matrix nltiming's ``"analytic"`` route reads -- so the analytic
        and autodiff routes coincide for this pulsar. It is TZR-aware and
        delay-feedback-aware because the residual it differentiates is.

        ``source="pint"`` returns PINT's analytic matrix on the same frozen
        pair, converted to ``param_names`` order (R9.4). Its rows are the
        TOA table's, which are also this engine's -- nothing reorders TOAs here
        (:data:`vela_jax.freeze.NO_REORDERING`). It is an **oracle**,
        not a product: identically-linear columns must agree with the default
        to ~1e-6 relative, while the remaining columns differ in phase frame
        (PINT ignores the TZR row's parameter dependence and the delay
        feedback, and divides by the constant ``F0`` rather than the spin
        Taylor series). Substituting it into ``Mmat`` would
        reintroduce the pulsar/engine disagreement this package exists to
        remove, and is a spec violation rather than an optimization.
        """
        if source == "jacobian":
            return -self.residual_jacobian()
        if source != "pint":
            raise ValueError(
                f"design_matrix source must be 'jacobian' or 'pint'; got {source!r}"
            )
        if self._pint_design is None:
            matrix, names, _ = self.pint_model.designmatrix(
                self.pint_toas, incoffset=False
            )
            index = {name: i for i, name in enumerate(names)}
            missing = [n for n in self.param_names if n not in index]
            if missing:
                raise UnsupportedModelError(f"no PINT design column for {missing}")
            self._pint_design = np.asarray(
                matrix[:, [index[n] for n in self.param_names]], dtype=float
            )
        return self._pint_design

    # --- facts -------------------------------------------------------------

    def reference_theta_exact(self) -> Mapping[str, str]:
        return dict(self.layout.theta_exact)

    def reference_theta(self) -> np.ndarray:
        return self.layout.reference_theta()

    def reference_spin_frequency(self) -> np.ndarray:
        """Pulsar-frame spin frequency at ``theta*``, one per TOA.

        The divisor :func:`~vela_jax.pipeline.form_residuals` uses, and hence
        the period a one-turn pulse-number shift moves a residual by. This is
        PINT's ``calctype="taylor"`` frequency -- the spin Taylor series with
        no doppler factor, drifting only through ``F1`` and up. Vela's own
        topocentric divisor is
        :func:`~vela_jax.correction.topo_spin_frequency`, which this
        deliberately is not (§8, G2).
        """
        reference = self._reference_correction()
        return np.asarray(reference.spin_frequency)[:-1]

    def precision_critical_params(self) -> frozenset[str]:
        return frozenset(_PRECISION_CRITICAL & set(self.param_names))

    def identically_linear_params(self) -> frozenset[str]:
        def linear(name: str) -> bool:
            if name in _IDENTICALLY_LINEAR_NAMES:
                return True
            prefix = getattr(self.pint_model[name], "prefix", None)
            return prefix in _IDENTICALLY_LINEAR_PREFIXES

        return frozenset(n for n in self.param_names if linear(n))

    def binary_chart_facts(self) -> BinaryFacts | None:
        return self._binary_facts

    @property
    def residual_centering(self) -> ResidualCentering:
        return RESIDUAL_CENTERING

    @property
    def obliquity(self) -> float:
        """Obliquity of the ecliptic this engine rotates the line of sight with.

        The par's ``ECL`` when PINT read the files; tempo2's own constant when
        tempo2 read them (tempo2 ignores ``ECL`` and has already rotated every
        ephemeris vector with it). Radians.
        """
        from .constants import OBL, obliquity_radians

        if self._fixed_obliquity is not None:
            return self._fixed_obliquity
        model = self.pint_model
        return obliquity_radians(model["ECL"].value) if "ECL" in model else OBL

    @property
    def source_units(self) -> str:
        """Timescale of the par as the user wrote it (``"TDB"`` / ``"TCB"``)."""
        return self._source_units

    @property
    def par_text(self) -> str | None:
        """The TDB par text PINT parsed, when tempo2 produced one."""
        return self._par_text

    def pulsar_data(self):
        """The frozen pulsar record this engine produces (Addendum B).

        Building it triggers :meth:`residual_jacobian` (R9.6): a
        ``PulsarData`` without ``Mmat`` does not exist.
        """
        if self._pulsar_data is None:
            from .pulsar_data import build_pulsar_data

            self._pulsar_data = build_pulsar_data(self)
        return self._pulsar_data

    def perturbative(self, live_nonlinear, *, dtype=None):
        """A delta-formulated engine over a restricted live set (SPEC §11)."""
        from .perturbative import PerturbativeEngine

        return PerturbativeEngine(self, live_nonlinear, dtype=dtype)

    def __repr__(self) -> str:
        name = self.pint_model["PSR"].value or "?"
        return (
            f"<vela_jax.Engine {name}: {self.toa_count} TOAs, "
            f"{len(self.param_names)} free params, "
            f"timing_package={self.timing_package}, "
            f"conventions={self.binary_conventions}, stages={list(self.stages)}>"
        )


__all__ = ["Engine", "RESIDUAL_CENTERING", "BinaryFacts"]
