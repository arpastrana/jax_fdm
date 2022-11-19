from functools import partial

from jax import jit

from jax_fdm.losses import Error
from jax_fdm.losses import Regularizer


# ==========================================================================
# Loss
# ==========================================================================

class Loss:
    """
    A function composed of error and regularization terms.
    """
    def __init__(self, *args, name=None):
        self._terms_error = None
        self._terms_regularization = None

        self.terms_error = args
        self.terms_regularization = args
        self.name = name or self.__class__.__name__

    @partial(jit, static_argnums=(0, 4))
    def __call__(self, q, xyz_fixed, loads, model):
        """
        Compute the scalar output of the loss function.
        """
        eqstate = model(q, xyz_fixed, loads)

        loss = 0.0
        for error_term in self.terms:
            loss = loss + error_term(eqstate)

        return loss

    @property
    def terms_error(self):
        """
        The error terms in the loss function.
        """
        return self._error_terms

    @terms_error.setter
    def terms_error(self, terms):
        self._error_terms = [term for term in terms if isinstance(term, Error)]

    @property
    def terms_regularization(self):
        """
        The regularization terms in the loss function.
        """
        return self._regularization_terms

    @terms_regularization.setter
    def terms_regularization(self, terms):
        self._regularization_terms = [term for term in terms if isinstance(term, Regularizer)]

    @property
    def terms(self):
        """
        The error and regularization terms of the loss function.
        """
        return self.terms_error + self.terms_regularization

    def number_of_goals(self):
        """
        The total number of individual goals for all error terms in the loss.
        """
        return sum(term.number_of_goals() for term in self.terms_error)

    def number_of_regularizers(self):
        """
        The total number of regularization terms in the loss.
        """
        return len(self.terms_regularization)

    def number_of_collections(self):
        """
        The total number of goal collections for all error terms in the loss.
        """
        return sum(term.number_of_collections() for term in self.terms_error)
