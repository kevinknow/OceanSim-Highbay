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
    def __init__(
        self,
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
        },
    ) -> None:
        self._base_command = base_command
        self._input_keyboard_mapping = input_keyboard_mapping
        self._enabled = False
        self._active_keys = []

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard() if self._appwindow is not None else None
        self._sub_keyboard = None
        self._pressed_inputs = set()
        self._pulse_inputs = set()
        self._tracked_inputs = {
            keyboard_input
            for key_name in self._input_keyboard_mapping
            for keyboard_input in [KEYBOARD_INPUT_MAP.get(key_name)]
            if keyboard_input is not None
        }

        if self._input is not None and self._keyboard is not None:
            self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.input not in self._tracked_inputs:
            return True

        event_type = event.type
        keyboard_event_type = carb.input.KeyboardEventType
        if event_type == keyboard_event_type.KEY_PRESS:
            self._pressed_inputs.add(event.input)
            self._pulse_inputs.add(event.input)
        elif event_type == keyboard_event_type.KEY_REPEAT:
            self._pressed_inputs.add(event.input)
        elif event_type == keyboard_event_type.KEY_RELEASE:
            self._pressed_inputs.discard(event.input)

        return True

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if not enabled:
            self.reset()

    def reset(self):
        self._base_command[:] = 0.0
        self._active_keys = []
        self._pressed_inputs.clear()
        self._pulse_inputs.clear()

    def update(self):
        if not self._enabled:
            return self._base_command

        self._base_command[:] = 0.0
        self._active_keys = []

        if self._input is None or self._keyboard is None:
            return self._base_command

        active_inputs = set(self._pressed_inputs)
        active_inputs.update(self._pulse_inputs)
        self._pulse_inputs.clear()

        for key_name, command in self._input_keyboard_mapping.items():
            keyboard_input = KEYBOARD_INPUT_MAP.get(key_name)
            if keyboard_input is None:
                continue
            if keyboard_input in active_inputs:
                self._active_keys.append(key_name)
                self._base_command += np.array(command)

        return self._base_command

    def cleanup(self):
        if self._input is not None and self._keyboard is not None and self._sub_keyboard is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub_keyboard)
        self._appwindow = None
        self._input = None
        self._keyboard = None
        self._sub_keyboard = None
        self._pressed_inputs.clear()
        self._pulse_inputs.clear()
