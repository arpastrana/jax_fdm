class Constraint:
    def __init__(self, bound_low, bound_up, **kwargs):
        self.bound_low = bound_low  # dict / array of shape (m,) or scalar
        self.bound_up = bound_up  # dict / array of shape (m,) or scalar

    def __call__(self, q, model):
        """
        The constraint function.
        """
        eqstate = model(q)
        return self.constraint(eqstate, model)

    def constraint(self, eqstate, model, **kwargs):
        raise NotImplementedError
