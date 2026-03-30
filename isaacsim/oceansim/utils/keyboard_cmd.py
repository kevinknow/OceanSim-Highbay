# Copyright (c) 2020-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import carb
import numpy as np
import omni
import omni.appwindow  # Contains handle to keyboard


KEYBOARD_INPUT_MAP = {
    "W": carb.input.KeyboardInput.W,
    "A": carb.input.KeyboardInput.A,
    "S": carb.input.KeyboardInput.S,
    "D": carb.input.KeyboardInput.D,
    "I": carb.input.KeyboardInput.I,
    "J": carb.input.KeyboardInput.J,
    "K": carb.input.KeyboardInput.K,
    "L": carb.input.KeyboardInput.L,
    "UP": carb.input.KeyboardInput.UP,
    "DOWN": carb.input.KeyboardInput.DOWN,
    "LEFT": carb.input.KeyboardInput.LEFT,
    "RIGHT": carb.input.KeyboardInput.RIGHT,
}


# THis can only be used after the scene is loaded 
class keyboard_cmd:
    def __init__(self,
                 base_command: np.array = np.array([0.0, 0.0, 0.0]),
                 input_keyboard_mapping: dict = {
                                        # forward command
                                        "W": [1.0, 0.0, 0.0],
                                        # backward command
                                        "S": [-1.0, 0.0, 0.0],
                                        # leftward command
                                        "A": [0.0, 1.0, 0.0],
                                        # rightward command
                                        "D": [0.0, -1.0, 0.0],
                                        # rise command
                                        "UP": [0.0, 0.0, 1.0],
                                        # sink command
                                        "DOWN": [0.0, 0.0, -1.0],
                                        }
                ) -> None:
        self._base_command = base_command
        self._input_keyboard_mapping = input_keyboard_mapping
        self._enabled = False
        self._active_keys = []

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = None

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if not enabled:
            self.reset()

    def reset(self):
        self._base_command[:] = 0.0

    def update(self):
        if not self._enabled:
            return self._base_command

        self._base_command[:] = 0.0
        self._active_keys = []

        if self._input is None or self._keyboard is None:
            return self._base_command

        for key_name, command in self._input_keyboard_mapping.items():
            keyboard_input = KEYBOARD_INPUT_MAP.get(key_name)
            if keyboard_input is None:
                continue
            if self._input.get_keyboard_value(self._keyboard, keyboard_input):
                self._active_keys.append(key_name)
                self._base_command += np.array(command)

        return self._base_command


    def cleanup(self):
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
