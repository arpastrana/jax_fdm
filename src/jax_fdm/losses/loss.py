from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.equilibrium import equilibrium_state_from_datastructure
from jax_fdm.losses.errors import Error
from jax_fdm.losses.regularizers import Regularizer

# ==========================================================================
# Loss
# ==========================================================================

__all__ = ["Loss"]


class Loss:
    """
    A scalar objective summing error and regularization terms.

    Parameters
    ----------
    args :
        The error and regularization terms to sum; they are sorted into the two
        groups by type.
    name :
        The name of the loss. If None, defaults to the class name.
    """

    def __init__(self, *args: Error | Regularizer, name: str | None = None) -> None:
        self._error_terms: list[Error] = []
        self._regularization_terms: list[Regularizer] = []

        self.terms_error = args
        self.terms_regularization = args
        self.name = name or self.__class__.__name__

    def __call__(
        self,
        params: EquilibriumParametersState,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, ""]:
        """
        Evaluate the loss by solving for equilibrium and summing all terms.

        Parameters
        ----------
        params :
            The parameters defining the equilibrium problem.
        model :
            The equilibrium model that computes the equilibrium state.
        structure :
            The structure that provides the connectivity.

        Returns
        -------
        loss :
            The scalar loss, the sum of the error terms evaluated on the
            equilibrium state and the regularization terms evaluated on the
            parameters.
        """
        eq_state = model(params, structure)

        loss = jnp.asarray(0.0)
        for error_term in self.terms_error:
            loss = loss + error_term(eq_state, structure)

        for reg_term in self.terms_regularization:
            loss = loss + reg_term(params)

        return loss

    def evaluate(
        self,
        datastructure: FDMesh | FDNetwork,
        sparse: bool = True,
    ) -> Float[Array, ""]:
        """
        Evaluate the loss directly on a datastructure, without an optimization.

        Parameters
        ----------
        datastructure :
            The network or mesh to read the equilibrium state from. Its geometry
            is used as-is; no form-finding is run.
        sparse :
            If True, assemble the equilibrium state with the sparse model.

        Returns
        -------
        loss :
            The scalar loss, the sum of the error terms evaluated on the
            datastructure's equilibrium state and the regularization terms
            evaluated on its parameters.

        Notes
        -----
        Builds the equilibrium state once and reuses it across every error term
        and regularizer. Error terms evaluate their raw goals as singletons, so
        the loss works before ``constrained_fdm`` has grouped them into
        collections.
        """
        equilibrium = equilibrium_state_from_datastructure(datastructure, sparse)

        loss = jnp.asarray(0.0)
        for error_term in self.terms_error:
            loss = loss + error_term.evaluate_state(
                equilibrium.eq_state,
                equilibrium.structure,
            )

        for reg_term in self.terms_regularization:
            loss = loss + reg_term(equilibrium.parameters)

        return loss

    @property
    def terms_error(self) -> list[Error]:
        """
        The error terms in the loss function.
        """
        return self._error_terms

    @terms_error.setter
    def terms_error(self, terms: Sequence[Error | Regularizer]) -> None:
        self._error_terms = [term for term in terms if isinstance(term, Error)]

    @property
    def terms_regularization(self) -> list[Regularizer]:
        """
        The regularization terms in the loss function.
        """
        return self._regularization_terms

    @terms_regularization.setter
    def terms_regularization(self, terms: Sequence[Error | Regularizer]) -> None:
        self._regularization_terms = [
            term for term in terms if isinstance(term, Regularizer)
        ]

    @property
    def terms(self) -> list[Error | Regularizer]:
        """
        The error and regularization terms of the loss function.
        """
        return self.terms_error + self.terms_regularization

    def number_of_goals(self) -> int:
        """
        The total number of individual goals for all error terms in the loss.
        """
        return sum(term.number_of_goals() for term in self.terms_error)

    def number_of_regularizers(self) -> int:
        """
        The total number of regularization terms in the loss.
        """
        return len(self.terms_regularization)

    def number_of_collections(self) -> int:
        """
        The total number of goal collections for all error terms in the loss.
        """
        return sum(term.number_of_collections() for term in self.terms_error)
