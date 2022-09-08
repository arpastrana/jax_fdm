class Constraint:
    def __init__(self, bound_low, bound_up):
        self.bound_low = bound_low  # dict / array of shape (m,) or scalar
        self.bound_up = bound_up  # dict / array of shape (m,) or scalar

    def __call__(self, q, model):
        """
        The constraint function.
        """
        eqstate = model(q)
        return self.constraint(eqstate)

    def constraint(self, eqstate, **kwargs):
        raise NotImplementedError


class LengthConstraint(Constraint):
    """
    Set constraint bounds to the length of the edges of a network in equilibrium.
    """
    def constraint(self, eqstate, **kwargs):
        """
        The constraint function relative to a equilibrium state.
        """
        return eqstate.lengths


class ForceConstraint(Constraint):
    """
    Set constraint bounds to the length of the edges of a network in equilibrium.
    """
    def constraint(self, eqstate, **kwargs):
        """
        The constraint function relative to a equilibrium state.
        """
        return eqstate.forces



if __name__ == "__main__":
    from statistics import mean

    import autograd.numpy as np
    # compas
    from compas.colors import Color
    from compas.geometry import Line
    from compas.geometry import Point
    from compas.geometry import Polyline
    from compas.geometry import add_vectors
    from compas.geometry import length_vector

    # visualization
    from compas_view2.app import App

    # static equilibrium
    from dfdm.datastructures import FDNetwork
    from dfdm.equilibrium import fdm
    from dfdm.equilibrium import constrained_fdm
    from dfdm.equilibrium import EquilibriumModel
    from dfdm.optimization import SLSQP
    from dfdm.optimization import TrustRegionConstrained
    from dfdm.goals import NetworkLoadPathGoal
    from dfdm.losses import Loss
    from dfdm.losses import PredictionError

    # ==========================================================================
    # Initial parameters
    # ==========================================================================

    arch_length = 5.0
    num_segments = 10
    q_init = -1.0
    pz = -0.1
    length_min = 0.5
    length_max = 1.0
    optimizer = SLSQP

    # ==========================================================================
    # Create the geometry of an arch
    # ==========================================================================

    start = [0.0, 0.0, 0.0]
    end = add_vectors(start, [arch_length, 0.0, 0.0])
    curve = Polyline([start, end])
    points = curve.divide_polyline(num_segments)
    lines = Polyline(points).lines

    # ==========================================================================
    # Create arch
    # ==========================================================================

    network = FDNetwork.from_lines(lines)

    # ==========================================================================
    # Define structural system
    # ==========================================================================

    # assign supports
    network.node_support(key=0)
    network.node_support(key=len(points) - 1)

    # set initial q to all edges
    network.edges_forcedensities(q_init, keys=network.edges())

    # set initial point loads to all nodes of the network
    network.nodes_loads([0.0, 0.0, pz], keys=network.nodes_free())

    # ==========================================================================
    # Run the force density method
    # ==========================================================================

    print("\nForm found network")
    network = fdm(network)

    # ==========================================================================
    # Report stats
    # ==========================================================================

    q = list(network.edges_forcedensities())
    f = list(network.edges_forces())
    l = list(network.edges_lengths())

    print(f"Load path: {round(network.loadpath(), 3)}")
    for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    # ==========================================================================
    # Create loss funtion with total load path as the only goal
    # ==========================================================================

    goal = NetworkLoadPathGoal()
    loss = Loss(PredictionError(goals=[goal]))

    # ==========================================================================
    # Constrained form-finding
    # ==========================================================================

    print("\nInverse form found network. No constraints")
    ceq_network = constrained_fdm(network,
                                  optimizer=optimizer(),
                                  loss=loss)

    # ==========================================================================
    # Report stats
    # ==========================================================================

    q = list(ceq_network.edges_forcedensities())
    f = list(ceq_network.edges_forces())
    l = list(ceq_network.edges_lengths())

    print(f"Load path: {round(ceq_network.loadpath(), 3)}")
    for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    # ==========================================================================
    # Create constraint
    # ==========================================================================

    constraint = LengthConstraint(bound_low=length_min, bound_up=length_max)
    constraints = [constraint]

    # ==========================================================================
    # Constrained form-finding with constraint
    # ==========================================================================

    print("\nInverse form found network. With constraints")
    ceq_network = constrained_fdm(network,
                                  optimizer=optimizer(),
                                  loss=loss,
                                  constraints=constraints,
                                  maxiter=1000,
                                  tol=1e-6)

    # ==========================================================================
    # Report stats
    # ==========================================================================

    q = list(ceq_network.edges_forcedensities())
    f = list(ceq_network.edges_forces())
    l = list(ceq_network.edges_lengths())

    print(f"Load path: {round(ceq_network.loadpath(), 3)}")
    for name, vals in zip(("FDs", "Forces", "Lengths"), (q, f, l)):

        minv = round(min(vals), 3)
        maxv = round(max(vals), 3)
        meanv = round(sum(vals) / len(vals), 3)
        print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    # ==========================================================================
    # Visualization
    # ==========================================================================

    viewer = App(width=1600, height=900, show_grid=True)

    viewer.add(network, linewidth=1.0, linecolor=Color.grey().darkened())

    eq_network = ceq_network

    # equilibrated arch
    viewer.add(eq_network,
               show_vertices=True,
               pointsize=12.0,
               show_edges=True,
               linecolor=Color.teal(),
               linewidth=5.0)

    # reference arch
    viewer.add(network, show_points=False, linewidth=4.0)

    for node in eq_network.nodes():

        pt = eq_network.node_coordinates(node)

        # draw lines betwen subject and target nodes
        target_pt = network.node_coordinates(node)
        viewer.add(Line(target_pt, pt))

        # draw residual forces
        residual = eq_network.node_residual(node)

        if length_vector(residual) < 0.001:
            continue

        residual_line = Line(pt, add_vectors(pt, residual))
        viewer.add(residual_line,
                   linewidth=4.0,
                   color=Color.pink())

    # draw applied loads
    for node in eq_network.nodes():
        pt = eq_network.node_coordinates(node)
        load = network.node_load(node)
        viewer.add(Line(pt, add_vectors(pt, load)),
                   linewidth=4.0,
                   color=Color.green().darkened())

    # draw supports
    for node in eq_network.nodes_supports():
        x, y, z = eq_network.node_coordinates(node)
        viewer.add(Point(x, y, z), color=Color.green(), size=20)

    # show le crème
    viewer.show()
