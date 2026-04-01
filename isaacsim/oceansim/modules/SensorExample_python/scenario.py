# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from isaacsim.core.utils.xforms import get_world_pose

from ...utils.hydrodynamics import UnderwaterHydrodynamics


class MHL_Sensor_Example_Scenario():
    def __init__(self):
        self._rob = None
        self._rob_prim_path = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None

        self._ctrl_mode = None

        self._running_scenario = False
        self._time = 0.0
        self._rigid_prim = None
        self._hydrodynamics = None
        self._force_cmd = None
        self._torque_cmd = None
        self._manual_debug_elapsed = 0.0
        self._manual_debug_interval = 0.5
        self._manual_backend_fallback_active = False
        self._manual_backend_disabled = False
        self._last_world_position = None
        self._last_world_orientation = None

    def setup_scenario(self, rob, sonar, cam, DVL, baro, ctrl_mode):
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._ctrl_mode = ctrl_mode
        self._rob_prim_path = get_prim_path(self._rob)
        self._rigid_prim = self._get_rigid_prim(refresh=True)
        self._manual_backend_fallback_active = False
        self._manual_backend_disabled = False
        self._last_world_position = None
        self._last_world_orientation = None
        if self._sonar is not None:
            self._sonar.sonar_initialize(include_unlabelled=True)
        if self._cam is not None:
            self._cam.initialize()
        if self._DVL is not None:
            self._DVL_reading = [0.0, 0.0, 0.0]
        if self._baro is not None:
            self._baro_reading = 101325.0 # atmospheric pressure (Pa)
        
        
        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control":
            from ...utils.keyboard_cmd import keyboard_cmd

            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
            # The controller computes world-frame wrenches, so configure the PhysX force API to match.
            self._rob_forceAPI.CreateWorldFrameEnabledAttr().Set(True)
            self._rob_forceAPI.CreateModeAttr().Set("force")
            # Let the custom underwater model handle damping instead of the isotropic PhysX damping.
            rob_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(self._rob)
            rob_rigid_body_api.GetLinearDampingAttr().Set(0.0)
            rob_rigid_body_api.GetAngularDampingAttr().Set(0.0)
            self._hydrodynamics = UnderwaterHydrodynamics(
                angular_command_tau=0.08,
                angular_drag=np.array([4.0, 4.0, 3.0]),
                quadratic_angular_drag=np.array([1.2, 1.2, 1.0]),
                attitude_stiffness=np.array([0.0, 0.0, 0.0]),
                attitude_damping=np.array([0.8, 0.8, 0.0]),
                max_angular_velocity=np.array([2.0, 2.0, 1.8]),
                max_smoothed_torque_cmd=np.array([10.0, 10.0, 10.0]),
                max_total_torque=np.array([12.0, 12.0, 10.0]),
            )
            self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # forward command
                                        "W": [10.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-10.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 10.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -10.0, 0.0],
                                         # rise command
                                        "UP": [0.0, 0.0, 10.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -10.0],
                                      })
            self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                      input_keyboard_mapping={
                                        # yaw command (left)
                                        "J": [0.0, 0.0, 8.0],
                                        # yaw command (right)
                                        "L": [0.0, 0.0, -8.0],
                                        # pitch command (up)
                                        "I": [0.0, -8.0, 0.0],
                                        # pitch command (down)
                                        "K": [0.0, 8.0, 0.0],
                                        # row command (left)
                                        "LEFT": [-8.0, 0.0, 0.0],
                                        # row command (negative)
                                        "RIGHT": [8.0, 0.0, 0.0],
                                      })
            self.set_manual_control_enabled(False)
            
        self._running_scenario = True

    def set_manual_control_enabled(self, enabled: bool):
        if self._ctrl_mode != "Manual control":
            return
        if self._force_cmd is not None:
            self._force_cmd.set_enabled(enabled)
        if self._torque_cmd is not None:
            self._torque_cmd.set_enabled(enabled)

    def _get_rigid_prim(self, refresh: bool = False):
        if self._rob_prim_path is None:
            return None
        if refresh or self._rigid_prim is None:
            self._rigid_prim = SingleRigidPrim(prim_path=self._rob_prim_path)
            try:
                self._rigid_prim.initialize()
            except Exception:
                pass
        return self._rigid_prim

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

    @staticmethod
    def _quat_multiply(quat_a: np.ndarray, quat_b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = quat_a
        bw, bx, by, bz = quat_b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float64,
        )

    def _estimate_manual_velocities(self, world_position: np.ndarray, world_orientation: np.ndarray, step: float):
        if step <= 1e-6 or self._last_world_position is None or self._last_world_orientation is None:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        linear_velocity = (world_position - self._last_world_position) / step

        prev_orientation = np.array(self._last_world_orientation, dtype=np.float64)
        curr_orientation = np.array(world_orientation, dtype=np.float64)
        prev_orientation /= max(np.linalg.norm(prev_orientation), 1e-9)
        curr_orientation /= max(np.linalg.norm(curr_orientation), 1e-9)
        if np.dot(prev_orientation, curr_orientation) < 0.0:
            curr_orientation = -curr_orientation

        delta_quat = self._quat_multiply(curr_orientation, self._quat_conjugate(prev_orientation))
        delta_quat /= max(np.linalg.norm(delta_quat), 1e-9)

        sin_half_angle = np.linalg.norm(delta_quat[1:])
        if sin_half_angle <= 1e-9:
            angular_velocity = np.zeros(3, dtype=np.float64)
        else:
            angle = 2.0 * np.arctan2(sin_half_angle, max(abs(delta_quat[0]), 1e-9))
            axis = delta_quat[1:] / sin_half_angle
            angular_velocity = axis * (angle / step)

        return linear_velocity, angular_velocity

    def _get_manual_state_from_usd(self, step: float):
        if self._rob_prim_path is None:
            raise RuntimeError("Robot prim path is not set")

        world_position, world_orientation = get_world_pose(self._rob_prim_path)
        world_position = np.array(world_position, dtype=np.float64)
        world_orientation = np.array(world_orientation, dtype=np.float64)
        world_orientation /= max(np.linalg.norm(world_orientation), 1e-9)

        linear_velocity = None
        angular_velocity = None
        state_source = "fallback"
        backend_error = None

        if not self._manual_backend_disabled:
            for attempt_index in range(2):
                try:
                    rigid_prim = self._get_rigid_prim(refresh=attempt_index > 0)
                except Exception as exc:
                    backend_error = exc
                    self._rigid_prim = None
                    continue
                if rigid_prim is None:
                    continue
                try:
                    linear_velocity = np.array(rigid_prim.get_linear_velocity(), dtype=np.float64)
                    angular_velocity = np.array(rigid_prim.get_angular_velocity(), dtype=np.float64)
                    state_source = "backend" if attempt_index == 0 else "backend_rebuilt"
                    if self._manual_backend_fallback_active:
                        print("[OceanSim ManualCtrl] rigid body backend restored.")
                        self._manual_backend_fallback_active = False
                    break
                except Exception as exc:
                    backend_error = exc
                    self._rigid_prim = None

        if linear_velocity is None or angular_velocity is None:
            linear_velocity, angular_velocity = self._estimate_manual_velocities(
                world_position=world_position,
                world_orientation=world_orientation,
                step=step,
            )
            if backend_error is not None:
                self._manual_backend_disabled = True
            if backend_error is not None and not self._manual_backend_fallback_active:
                print(
                    "[OceanSim ManualCtrl] "
                    f"backend state unavailable, using pose-delta fallback: {backend_error}"
                )
                self._manual_backend_fallback_active = True

        self._last_world_position = world_position
        self._last_world_orientation = world_orientation
        return world_orientation, linear_velocity, angular_velocity, state_source

    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        def read_data_from_file(file_path):
            # Initialize an empty list to store the floats
            data = []
            
            # Open the file in read mode
            with open(file_path, 'r') as file:
                # Read each line in the file
                for line in file:
                    # Strip any leading/trailing whitespace and split the line by spaces
                    float_strings = line.strip().split()
                    
                    # Convert the list of strings to a list of floats
                    floats = [float(x) for x in float_strings]
                    
                    # Append the list of floats to the data list
                    data.append(floats)
            
            return data
        try:
            self.waypoints = read_data_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except:
            self.waypoints = read_data_from_file(default_waypoint_path)
            print('Fail to load this waypoints. Back to default waypoints.')

        
    def teardown_scenario(self):

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control":
            self.set_manual_control_enabled(False)
            if self._force_cmd is not None:
                self._force_cmd.cleanup()
            if self._torque_cmd is not None:
                self._torque_cmd.cleanup()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._rob_prim_path = None
        self._rigid_prim = None
        self._hydrodynamics = None
        self._force_cmd = None
        self._torque_cmd = None
        self._running_scenario = False
        self._time = 0.0
        self._manual_debug_elapsed = 0.0
        self._manual_backend_fallback_active = False
        self._manual_backend_disabled = False
        self._last_world_position = None
        self._last_world_orientation = None


    def update_scenario(self, step: float):

        
        if not self._running_scenario:
            return
        
        self._time += step
        
        if self._sonar is not None:
            self._sonar.make_sonar_data()
        if self._cam is not None:
            self._cam.render()
        if self._DVL is not None:
            self._DVL_reading = self._DVL.get_linear_vel()
        if self._baro is not None:
            self._baro_reading = self._baro.get_pressure()

        if self._ctrl_mode=="Manual control":
            if self._force_cmd is not None:
                self._force_cmd.update()
            if self._torque_cmd is not None:
                self._torque_cmd.update()
            try:
                world_orientation, linear_velocity, angular_velocity, state_source = self._get_manual_state_from_usd(step)
            except Exception as exc:
                print(f"[OceanSim ManualCtrl] rigid body state unavailable: {exc}")
                return
            force_cmd, torque_cmd = self._hydrodynamics.compute_wrench(
                step=step,
                world_orientation=world_orientation,
                world_linear_velocity=linear_velocity,
                world_angular_velocity=angular_velocity,
                desired_force_cmd_body=self._force_cmd._base_command,
                desired_torque_cmd_body=self._torque_cmd._base_command,
            )
            self._rob_forceAPI.CreateForceAttr().Set(Gf.Vec3f(*force_cmd))
            self._rob_forceAPI.CreateTorqueAttr().Set(Gf.Vec3f(*torque_cmd))
            self._manual_debug_elapsed += step
            if self._manual_debug_elapsed >= self._manual_debug_interval:
                self._manual_debug_elapsed = 0.0
                print(
                    "[OceanSim ManualCtrl] "
                    f"state={state_source} "
                    f"force_keys={getattr(self._force_cmd, '_active_keys', [])} "
                    f"torque_keys={getattr(self._torque_cmd, '_active_keys', [])} "
                    f"force_body={np.round(self._force_cmd._base_command, 3).tolist()} "
                    f"torque_body={np.round(self._torque_cmd._base_command, 3).tolist()} "
                    f"force_world={np.round(force_cmd, 3).tolist()} "
                    f"torque_world={np.round(torque_cmd, 3).tolist()}"
                )
        elif self._ctrl_mode=="Waypoints":
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')                
        elif self._ctrl_mode=="Straight line":
            rigid_prim = self._get_rigid_prim()
            if rigid_prim is not None:
                rigid_prim.set_linear_velocity(np.array([0.5,0,0])) 




        

        
