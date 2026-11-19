"""Dataset adapters. Each returns schemas.Interaction objects with explicit frames and SI units.

forcesapien  ForceSAPIEN benchmark inputs (estimator_input.json, simulation.json force records, bundled A3/B1-B10)
forceartgs   Samuele Mara's ForceArtGS outputs (joint authority / joint_candidates.json) + LeRobot robot data
hoi          Hoi! (Leica joint, gripper wrist F/T force CSVs produced by our freeartgs-hoi force pipeline)
real         REAL robot dataset (LeRobot parquet: EE pose + wrench) - thin wrapper over forceartgs.robot
"""
