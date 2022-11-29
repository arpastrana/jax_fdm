import numpy as np
import jax.numpy as jnp

from jax_fdm import DTYPE_JAX

from jax_fdm.parameters import split
from jax_fdm.parameters import combine

from jax_fdm.parameters import EdgeParameter
from jax_fdm.parameters import NodeAnchorParameter
from jax_fdm.parameters import NodeLoadParameter

from jax_fdm.parameters import EdgeForceDensityParameter
from jax_fdm.parameters import NodeAnchorXParameter
from jax_fdm.parameters import NodeAnchorYParameter
from jax_fdm.parameters import NodeAnchorZParameter
from jax_fdm.parameters import NodeLoadXParameter
from jax_fdm.parameters import NodeLoadYParameter
from jax_fdm.parameters import NodeLoadZParameter


class ParameterManager:
    """
    A parameter manager for optimization purposes.

    Parameters
    ----------
    model : `jax_fdm.equilibrium.EquilibriumModel`
        An equilibrium model.
    parameters : `List[jax_fdm.parameters.OptimizationParameter]`
        A list of optimization parameters.
    """
    parameter_types = [EdgeForceDensityParameter,
                       NodeAnchorXParameter,
                       NodeAnchorYParameter,
                       NodeAnchorZParameter,
                       NodeLoadXParameter,
                       NodeLoadYParameter,
                       NodeLoadZParameter]

    def __init__(self, model, parameters):
        """
        Initialize the manager.
        """
        self.model = model  # model.structure.network holds an FD network object.
        self.structure = model.structure
        self.network = model.structure.network
        self.parameters = parameters

        self._indices_opt = None
        self._indices_frozen = None
        self._indices_optfrozen = None

        self._indices_fdm = None
        self._indices_fd = None
        self._indices_xyzfixed = None
        self._indices_loads = None

        self._parameters_ordered = None
        self._parameters_model = None
        self._parameters_opt = None
        self._parameters_frozen = None

        self._bounds_low = None
        self._bounds_up = None

        self._startindex_fd = None
        self._startindex_xyzfixed = None
        self._startindex_loads = None

        self.init()

# ==========================================================================
# Warm start
# ==========================================================================

    def init(self):
        """
        Initialiaze the properties of this object so that it is static after this call.
        """
        self.indices_fdm
        self.indices_optfrozen
        self.parameters_model
        self.parameters_opt
        self.parameters_frozen

# ==========================================================================
# Starting indices
# ==========================================================================

    @property
    def startindex_fd(self):
        """
        The starting index of the force density of the edges of a network.
        """
        if not self._startindex_fd:
            self._startindex_fd = 0
        return self._startindex_fd

    @property
    def startindex_xyzfixed(self):
        """
        The starting index of the xyz coordinates of the anchor nodes of a network.
        """
        if not self._startindex_xyzfixed:
            self._startindex_xyzfixed = self.network.number_of_edges()
        return self._startindex_xyzfixed

    @property
    def startindex_loads(self):
        """
        The starting index of the xyz coordinates of the loads at the nodes of a network.
        """
        if not self._startindex_loads:
            self._startindex_loads = self.network.number_of_anchors() * 3 + self.startindex_xyzfixed
        return self._startindex_loads

# ==========================================================================
# Indices
# ==========================================================================

    @property
    def indices_fd(self):
        """
        The ordered indices of the force density of the edges of a network.
        """
        if self._indices_fd is None:
            start = self.startindex_fd
            stop = self.network.number_of_edges()
            self._indices_fd = np.array(range(start, start + stop))
        return self._indices_fd

    @property
    def indices_xyzfixed(self):
        """
        The ordered indices of the .
        """
        if self._indices_xyzfixed is None:
            start = self.startindex_xyzfixed
            stop = self.network.number_of_anchors() * 3 + start
            self._indices_xyzfixed = np.array(range(start, stop))
        return self._indices_xyzfixed

    @property
    def indices_loads(self):
        """
        The ordered indices of the xyz coordinates of the anchor nodes of a network.
        """
        if self._indices_loads is None:
            start = self.startindex_loads
            stop = self.network.number_of_nodes() * 3 + start
            self._indices_loads = np.array(range(start, stop))
        return self._indices_loads

    @property
    def indices_opt(self):
        """
        The ordered indices of the optimization parameters.
        """
        if self._indices_opt is None:
            indices = []
            pshift = 0
            for ptype in self.parameter_types:
                itype = self._indices_type(ptype)
                indices.extend(self._indices_shift(itype, pshift))
                pshift += self._shift_type(ptype)

            self._indices_opt = indices

        return self._indices_opt

    @property
    def indices_optfrozen(self):
        if self._indices_optfrozen is None:
            _, indices = split(self.parameters_model, func=self.mask_optimizable)
            self._indices_optfrozen = indices
        return self._indices_optfrozen

    @property
    def indices_fdm(self):
        if self._indices_fdm is None:
            self._indices_fdm = [self.startindex_xyzfixed, self.startindex_loads]
        return self._indices_fdm

# ==========================================================================
# Helpers
# ==========================================================================

    def _indices_type(self, cls):
        """
        Compute the parameter index if a parameter is an instance of a given type.
        """
        indices = []
        for parameter in self.parameters:
            if isinstance(parameter, cls):
                indices.append(parameter.index(self.model))

        return np.array(indices)

    def _indices_shift(self, indices, shift):
        """
        Shift a collection of parameter indices by a given integer.
        """
        return [idx + shift for idx in indices]

    def _shift_type(self, ptype):
        """
        The number of indices to shift of a collection of parameters of a given type.
        """
        if issubclass(ptype, EdgeParameter):
            shift = self.network.number_of_edges()
        elif issubclass(ptype, NodeAnchorParameter):
            shift = self.network.number_of_anchors()
        elif issubclass(ptype, NodeLoadParameter):
            shift = self.network.number_of_nodes()

        return shift

# ==========================================================================
# Bounds
# ==========================================================================

    @property
    def bounds_low(self):
        """
        Return an array with the lower bound of the optimization parameters.
        """
        if self._bounds_low is None:
            bounds = []
            for parameter in self.parameters_ordered:
                bounds.append(parameter.bound_low)

            self._bounds_low = np.array(bounds)

        return self._bounds_low

    @property
    def bounds_up(self):
        """
        Return an array with the upper bound of the optimization parameters.
        """
        if self._bounds_up is None:
            bounds = []
            for parameter in self.parameters_ordered:
                bounds.append(parameter.bound_up)

            self._bounds_up = np.array(bounds)

        return self._bounds_up

# ==========================================================================
# Parameters
# ==========================================================================

    @property
    def parameters_ordered(self):
        """
        The optimization parameter objects, sorted by type.
        """
        if self._parameters_ordered is None:
            parameters = []
            for ptype in self.parameter_types:
                for parameter in self.parameters:
                    if isinstance(parameter, ptype):
                        parameters.append(parameter)
            self._parameters_ordered = parameters

        return self._parameters_ordered

    @property
    def parameters_model(self):
        """
        The model parameters.
        """
        if self._parameters_model is None:
            param_arrays = []
            for param in self.network.parameters():
                param_arrays.append(jnp.asarray(param, dtype=DTYPE_JAX))

            q, xyz_fixed, loads = param_arrays
            self._parameters_model = jnp.concatenate((q,
                                                      jnp.ravel(xyz_fixed, order="F"),
                                                      jnp.ravel(loads, order="F")))
        return self._parameters_model

    @property
    def parameters_opt(self):
        """
        The optimizable model parameters.
        """
        if self._parameters_opt is None:
            parameters, _ = split(self.parameters_model, func=self.mask_optimizable)
            opt, _ = parameters
            self._parameters_opt = opt
        return self._parameters_opt

    @property
    def parameters_frozen(self):
        """
        The non-optimizable model parameters.
        """
        if self._parameters_frozen is None:
            parameters, _ = split(self.parameters_model, func=self.mask_optimizable)
            _, frozen = parameters
            self._parameters_frozen = frozen
        return self._parameters_frozen

    def parameters_fdm(self, params_opt):
        """
        Reshape optimizable model parameters into fdm parameters.
        """
        params = combine(params_opt, self.parameters_frozen, adef=self.indices_optfrozen)
        q, xyz_fixed, loads = jnp.split(params, self.indices_fdm)

        return q, jnp.reshape(xyz_fixed, (-1, 3), order="F"), jnp.reshape(loads, (-1, 3), order="F")

# ==========================================================================
# Masks
# ==========================================================================

    def mask_optimizable(self, array):
        """
        """
        mask = np.zeros_like(array, dtype=np.int64)
        mask[self.indices_opt] = 1

        return mask, np.logical_not(mask)

    def mask_fdm(self, array):
        """
        """
        for indices in (self.indices_fd,
                        self.indices_xyzfixed,
                        self.indices_loads):
            mask = np.zeros_like(array, dtype=np.int64)
            mask[indices] = 1
            yield mask
