from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumParametersState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.losses import Error
from jax_fdm.losses import Regularizer

# ==========================================================================
# Loss
# ==========================================================================

class Loss:
    """
    A function composed of error and regularization terms.
    """
    def __init__(self, *args: Error | Regularizer, name: str | None = None):
        self._terms_error = None
        self._terms_regularization = None

        self.terms_error = args
        self.terms_regularization = args
        self.name = name or self.__class__.__name__

    def __call__(
        self,
        params: EquilibriumParametersState,
        model: EquilibriumModel,
        structure: EquilibriumStructure,
    ) -> Float[Array, ""] | float:
        """
        Compute the scalar output of the loss function.
        """
        eq_state = model(params, structure)

        loss = 0.0
        for error_term in self.terms_error:
            loss = loss + error_term(eq_state)

        for reg_term in self.terms_regularization:
            loss = loss + reg_term(params)  # pyright: ignore[reportCallIssue]  # Regularizer defines __call__ only on its subclasses, not on the base class

        return loss

    @property
    def terms_error(self) -> list[Error]:
        """
        The error terms in the loss function.
        """
        return self._error_terms

    @terms_error.setter
    def terms_error(self, terms: tuple[Error | Regularizer, ...]) -> None:
        self._error_terms = [term for term in terms if isinstance(term, Error)]

    @property
    def terms_regularization(self) -> list[Regularizer]:
        """
        The regularization terms in the loss function.
        """
        return self._regularization_terms

    @terms_regularization.setter
    def terms_regularization(self, terms: tuple[Error | Regularizer, ...]) -> None:
        self._regularization_terms = [term for term in terms if isinstance(term, Regularizer)]

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
