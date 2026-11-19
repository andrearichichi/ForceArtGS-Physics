"""ForceArtGS-Physics: physical parameter identification of 1-DoF articulated objects from estimated kinematics
(ForceArtGS) and observed forces. Four validated estimators (PINN, Backprop, MLP, AKF) behind one interface.

    from forceartgs_physics import api
    res = api.identify(interaction, "Backprop")
    pred = api.predict(res.parameters, new_forces, joint, t_query)
"""
__version__ = "0.1.0"
