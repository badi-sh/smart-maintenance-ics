# Install required libraries
!pip install gradio numpy matplotlib pandas tqdm --quiet

# Import necessary libraries
import numpy as np
import gradio as gr
import os
import tempfile
import shutil
import logging
import pandas as pd
from tqdm.notebook import tqdm
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify ffmpeg installation
try:
    FFMpegWriter()
except Exception as e:
    logger.error(f"FFmpeg initialization failed: {e}. Installing ffmpeg...")
    !apt-get install -y ffmpeg

# Enums for better state management
class PumpState(Enum):
    FAIL = auto()
    NON_FAIL = auto()

class FailureType(Enum):
    BEARING = auto()
    CAVITATION = auto()
    OVERLOAD = auto()
    SEAL_LEAK = auto()
    IMPELLER_DAMAGE = auto()
    NONE = auto()

# EnvironmentParams and SensorReadings dataclasses
@dataclass
class EnvironmentParams:
    ambient_temp: float = 25.0
    humidity: float = 50.0
    fluid_viscosity: float = 1.0
    external_vibration: float = 0.0

    def validate(self) -> None:
        self.ambient_temp = np.clip(self.ambient_temp, 10, 40)
        self.humidity = np.clip(self.humidity, 30, 60)
        self.fluid_viscosity = max(0.1, self.fluid_viscosity)
        self.external_vibration = np.clip(self.external_vibration, 7, 10)

@dataclass
class SensorReadings:
    time: float
    rpm: float
    pressure: float
    flow_rate: float
    temperature: float
    vibration: float
    current: float
    noise: float
    ambient_temp: float
    humidity: float
    fluid_viscosity: float
    external_vibration: float
    failure_triggered: bool
    failure_type: FailureType
    failure_transition: str = ''

# FailureManager class
class FailureManager:
    def __init__(self):
        self.active_failures: List[Tuple[FailureType, float]] = []
        self.last_failure_type = FailureType.NONE

    def trigger_failure(self, pump_time: float) -> FailureType:
        if not self.active_failures:
            failure_type = random.choice(list(FailureType)[:-1])  # Exclude NONE
            self.last_failure_type = failure_type
            self.active_failures.append((failure_type, pump_time))
            logger.info(f"Triggered new failure: {failure_type.name} at time {pump_time:.1f}")
            return failure_type
        return FailureType.NONE

    def propagate_failures(self, pump_time: float) -> List[FailureType]:
        new_failures = []
        propagated = []
        for failure_type, start_time in self.active_failures:
            time_elapsed = pump_time - start_time
            if time_elapsed > 60:
                continue
            if failure_type == FailureType.BEARING and time_elapsed > 10 and random.random() < 0.3:
                new_failure = FailureType.SEAL_LEAK
                self.last_failure_type = new_failure
                new_failures.append((new_failure, pump_time))
                propagated.append(new_failure)
                logger.info(f"Propagated {failure_type.name} to {new_failure.name}")
            elif failure_type == FailureType.CAVITATION and time_elapsed > 15 and random.random() < 0.25:
                new_failure = FailureType.IMPELLER_DAMAGE
                self.last_failure_type = new_failure
                new_failures.append((new_failure, pump_time))
                propagated.append(new_failure)
                logger.info(f"Propagated {failure_type.name} to {new_failure.name}")
            elif failure_type == FailureType.SEAL_LEAK and time_elapsed > 20 and random.random() < 0.35:
                new_failure = FailureType.OVERLOAD
                self.last_failure_type = new_failure
                new_failures.append((new_failure, pump_time))
                propagated.append(new_failure)
                logger.info(f"Propagated {failure_type.name} to {new_failure.name}")
            new_failures.append((failure_type, start_time))
        self.active_failures = new_failures
        return propagated

# VirtualPump class
class VirtualPump:
    def __init__(self):
        self.state = PumpState.NON_FAIL
        self.time = 0.0
        self.environment = EnvironmentParams()
        self.failure_manager = FailureManager()
        self.data_log = []
        self.custom_thresholds = {'temperature': 90, 'vibration': 10, 'current': 30, 'pressure': 3.0}
        self.shutdown_triggered = False
        self._initialize_sensors()

    def _initialize_sensors(self) -> None:
        self._rpm = 1635.0
        self._pressure = 4.0
        self._flow_rate = 124.0
        self._temperature = 39.0
        self._vibration = 4.7
        self._current = 19.2
        self._noise = 100.0

    def update_environment(self, params: EnvironmentParams) -> None:
        self.environment = params
        self.environment.validate()

    def set_state(self, state: PumpState) -> None:
        self.state = state
        if state == PumpState.NON_FAIL:
            self.shutdown_triggered = False
            self.failure_manager.active_failures.clear()
            self.failure_manager.last_failure_type = FailureType.NONE

    def step(self, enforce_state: Optional[PumpState] = None) -> SensorReadings:
        self.time += 0.5
        self._update_environment_dynamically()
        self._update_sensor_values()
        self._check_failures(enforce_state)
        if self._temperature > 95 and self.state == PumpState.FAIL:
            self._temperature = self._temperature + 0.1 * (120 - self._temperature)
        if self._rpm < 1000 and not self.shutdown_triggered:
            self._rpm = max(100, self._rpm * 0.95)
        readings = self._get_current_readings()
        self.data_log.append(readings)
        return readings

    def _update_environment_dynamically(self) -> None:
        self.environment.ambient_temp += np.random.uniform(-0.2, 0.2)
        self.environment.humidity += np.random.uniform(-0.5, 0.5)
        self.environment.external_vibration += np.random.uniform(-0.2, 0.2)
        self.environment.validate()

    def _update_sensor_values(self) -> None:
        load_factor = 1 + 0.05 * self.environment.fluid_viscosity
        failure_factor = 1.5 if self.state == PumpState.FAIL else 1.0
        base_values = np.array([1635.0, 4.0, 124.0, 39.0, 4.7, 19.2, 100.0])
        noise = np.random.normal(0, [12, 0.5, 6, 2, 0.4, 1, 5]) * failure_factor
        scaled_values = base_values + noise * (0.9 if self.state == PumpState.FAIL else 1.0)

        sensors = np.array([scaled_values[0], scaled_values[1], scaled_values[2],
                           self.environment.ambient_temp + 10 * load_factor + 2 * (self.environment.humidity - 50) / 10 + scaled_values[3],
                           scaled_values[4], scaled_values[5], scaled_values[6]])
        limits = np.array([[100, 1700], [2.5, 4.25], [90, 136], [35, 120], [3.8, 15.0], [16.7, 33.0], [88, 120]])

        sensors = np.clip(sensors, limits[:, 0], limits[:, 1])
        self._rpm, self._pressure, self._flow_rate, self._temperature, self._vibration, self._current, self._noise = sensors

        for failure, _ in self.failure_manager.active_failures:
            self._apply_failure_effects(failure)
            self._rpm = np.clip(self._rpm, 100, 1700)
            self._pressure = np.clip(self._pressure, 2.5, 4.25)
            self._flow_rate = np.clip(self._flow_rate, 90, 136)
            self._temperature = np.clip(self._temperature, 35, 120)
            self._vibration = np.clip(self._vibration, 3.8, 15.0)
            self._current = np.clip(self._current + np.random.uniform(-1, 1), 16.7, 33.0)
            self._noise = np.clip(self._noise, 88, 120)

    def _apply_failure_effects(self, failure: FailureType) -> None:
        severity_factor = 1.0 + 0.05 * (self.time - 450.0) / 1000
        if failure == FailureType.BEARING:
            self._vibration += 1.0 * severity_factor
            self._noise += 5
        elif failure == FailureType.CAVITATION:
            self._pressure *= 0.9
            self._noise += 5
            self._vibration += 0.5 * severity_factor
            self._temperature -= 2
        elif failure == FailureType.OVERLOAD:
            self._current += 1.5 * severity_factor
            self._temperature += 3 * severity_factor
            self._rpm *= 0.95
            self._pressure *= 0.9
        elif failure == FailureType.SEAL_LEAK:
            self._pressure *= 0.95
            self._flow_rate *= 0.95
            self._noise += 3
        elif failure == FailureType.IMPELLER_DAMAGE:
            self._vibration += 1.0 * severity_factor
            self._flow_rate *= 0.95
            self._pressure *= 0.97

    def _check_failures(self, enforce_state: Optional[PumpState] = None) -> None:
        current_state = enforce_state if enforce_state is not None else self.state
        custom_thresholds = self.custom_thresholds
        temp_threshold = custom_thresholds['temperature'] - 5 * (self.environment.fluid_viscosity - 1)
        vib_threshold = custom_thresholds['vibration'] - 2 * (self.environment.fluid_viscosity - 1) + 0.5 * (self.environment.humidity - 50) / 10
        curr_threshold = custom_thresholds['current']
        press_threshold = custom_thresholds['pressure']
        flow_threshold = 100.0

        severity = 0
        if self._pressure < press_threshold: severity += 40 * (press_threshold - self._pressure)
        if self._vibration > vib_threshold: severity += 40 * (self._vibration - vib_threshold)
        if self._temperature > temp_threshold: severity += 40 * (self._temperature - temp_threshold)
        if self._current > curr_threshold: severity += 40 * (self._current - curr_threshold)
        if self._flow_rate < flow_threshold: severity += 40 * (flow_threshold - self._flow_rate)

        critical_thresholds = {'pressure': 4.0, 'vibration': 5.0, 'temperature': 45.0, 'current': 20.0, 'flow_rate': 110.0}
        conditions_exceeded = any(getattr(self, f'_{k}') < v if k in ['pressure', 'flow_rate']
                                else getattr(self, f'_{k}') > v
                                for k, v in critical_thresholds.items())

        if current_state == PumpState.FAIL:
            failure_prob = 0.4 if conditions_exceeded or severity > 50 else 0.1
        elif current_state == PumpState.NON_FAIL:
            failure_prob = 0.0 if not conditions_exceeded and severity <= 20 else 0.0
        else:
            failure_prob = 0.01 if conditions_exceeded else 0.002

        if current_state == PumpState.FAIL and random.random() < failure_prob and conditions_exceeded:
            failure_type = self.failure_manager.trigger_failure(self.time)
            if severity > 50 or conditions_exceeded:
                self._shutdown(failure_type)
        elif severity > 20 and current_state != PumpState.FAIL:
            self._alert(severity)
        self.failure_manager.propagate_failures(self.time)

    def _shutdown(self, failure_type) -> None:
        if not self.shutdown_triggered and failure_type != FailureType.NONE:
            self.state = PumpState.FAIL
            logger.warning(f"Time {self.time:.1f}: Shutdown due to {failure_type.name}")
            if self.data_log:
                self.data_log[-1] = SensorReadings(
                    time=self.time,
                    rpm=self._rpm,
                    pressure=self._pressure,
                    flow_rate=self._flow_rate,
                    temperature=self._temperature,
                    vibration=self._vibration,
                    current=self._current,
                    noise=self._noise,
                    ambient_temp=self.environment.ambient_temp,
                    humidity=self.environment.humidity,
                    fluid_viscosity=self.environment.fluid_viscosity,
                    external_vibration=self.environment.external_vibration,
                    failure_triggered=True,
                    failure_type=failure_type,
                    failure_transition=self.data_log[-1].failure_transition
                )
            self.shutdown_triggered = True

    def _alert(self, severity) -> None:
        if severity > 20:
            logger.info(f"Time {self.time:.1f}: Alert - Severity {severity:.1f}, Maintenance recommended")

    def _get_current_readings(self) -> SensorReadings:
        return SensorReadings(
            time=self.time,
            rpm=self._rpm,
            pressure=self._pressure,
            flow_rate=self._flow_rate,
            temperature=self._temperature,
            vibration=self._vibration,
            current=self._current,
            noise=self._noise,
            ambient_temp=self.environment.ambient_temp,
            humidity=self.environment.humidity,
            fluid_viscosity=self.environment.fluid_viscosity,
            external_vibration=self.environment.external_vibration,
            failure_triggered=bool(self.failure_manager.active_failures),
            failure_type=self.failure_manager.last_failure_type,
            failure_transition=','.join(f.name for f, _ in self.failure_manager.active_failures)
        )

# PumpVisualizer class
class PumpVisualizer:
    def __init__(self, pump: VirtualPump):
        self.pump = pump
        self.sensor_config = {
            'Temperature (°C)': {'color': 'r', 'ylim': (30, 120)},
            'Vibration (mm/s)': {'color': 'm', 'ylim': (0, 15)},
            'Current (A)': {'color': 'y', 'ylim': (10, 40)},
            'RPM': {'color': 'b', 'ylim': (0, 2000)},
            'Pressure (bar)': {'color': 'g', 'ylim': (0, 6)},
            'Flow Rate (L/min)': {'color': 'c', 'ylim': (0, 250)},
            'Noise (dB)': {'color': 'k', 'ylim': (50, 120)},
            'Ambient Temp (°C)': {'color': 'orange', 'ylim': (10, 40)},
            'Humidity (%)': {'color': 'purple', 'ylim': (0, 100)}
        }

    def save_animation(self, sensor_name: str, filename: str) -> Optional[str]:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig, ax = plt.subplots(figsize=(4, 2.5), dpi=100)  # Smaller figure, lower DPI
            config = self.sensor_config[sensor_name]
            line, = ax.plot([], [], config['color'])
            fail_text = ax.text(0.02, 0.92, '', transform=ax.transAxes, va='top')

            ax.set_xlim(0, 10)
            ax.set_ylim(config['ylim'])
            ax.set_title(f"{sensor_name} - Virtual Pump (Time: {self.pump.time:.1f}s)")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(sensor_name)
            ax.grid(True)

            time_data, sensor_data = [], []

            def init():
                line.set_data([], [])
                fail_text.set_text('')
                return line, fail_text

            def update(frame):
                readings = self.pump.step()
                time_data.append(readings.time)
                attr_map = {
                    'Temperature (°C)': 'temperature', 'Vibration (mm/s)': 'vibration',
                    'Current (A)': 'current', 'RPM': 'rpm', 'Pressure (bar)': 'pressure',
                    'Flow Rate (L/min)': 'flow_rate', 'Noise (dB)': 'noise',
                    'Ambient Temp (°C)': 'ambient_temp', 'Humidity (%)': 'humidity'
                }
                value = getattr(readings, attr_map.get(sensor_name, 'temperature'))
                sensor_data.append(value)

                line.set_data(time_data, sensor_data)
                ax.set_xlim(max(0, readings.time - 10), readings.time)
                ax.set_title(f"{sensor_name} - Virtual Pump (Time: {readings.time:.1f}s)")
                fail_text.set_text(f'Failure: {readings.failure_type.name}' if readings.failure_triggered else '')
                ax.set_facecolor('#ffcccc' if readings.failure_triggered else 'white')
                return line, fail_text

            frames = int(10 * 10)
            ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=100, repeat=False)

            writer = FFMpegWriter(fps=10, metadata=dict(artist='VirtualPump'), bitrate=1200)  # Lower bitrate
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                ani.save(tmp_file.name, writer=writer, dpi=100)  # Match DPI
                plt.close(fig)
                shutil.move(tmp_file.name, filename)
            logger.info(f"Successfully saved animation for {sensor_name} to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving animation for {sensor_name}: {str(e)}")
            return None

# PumpDataLogger class
class PumpDataLogger:
    def __init__(self, pump: VirtualPump):
        self.pump = pump
        self.batch_size = 1000

    def generate_data(self, num_samples: int = 10000, output_file: str = 'pump_data.csv', state: PumpState = None) -> None:
        output_dir = '/content/output/data'
        os.makedirs(output_dir, exist_ok=True)
        if state == PumpState.FAIL:
            output_file = os.path.join(output_dir, 'fail.csv')
        elif state == PumpState.NON_FAIL:
            output_file = os.path.join(output_dir, 'non_fail.csv')
        else:
            output_file = os.path.join(output_dir, 'custom.csv')
        output_path = os.path.abspath(output_file)

        self.pump.data_log = []
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)

        try:
            with tqdm(total=num_samples, desc="Generating data") as pbar:
                for i in range(0, num_samples, self.batch_size):
                    current_batch = min(self.batch_size, num_samples - i)
                    self._generate_batch(current_batch, state)
                    df = self._convert_batch_to_df()
                    df = df.round({'time': 1, 'rpm': 1, 'pressure': 2, 'flow_rate': 1, 'temperature': 1,
                                 'vibration': 1, 'current': 1, 'noise': 1, 'ambient_temp': 1,
                                 'humidity': 1, 'fluid_viscosity': 2, 'external_vibration': 1})
                    failure_map = {ft.name: i + 1 for i, ft in enumerate(FailureType)}
                    failure_map['NONE'] = 0
                    df['failure_type'] = df['failure_type'].map(failure_map).astype('int8')
                    df['failure_transition'] = df['failure_transition'].apply(lambda x: x.split(',')[0] if x else '')
                    df.to_csv(temp_file.name, mode='a' if i > 0 else 'w', header=(i == 0), index=False, float_format='%.3f')
                    pbar.update(current_batch)
            shutil.move(temp_file.name, output_path)
            logger.info(f"Successfully generated {num_samples} samples to {output_path}")
        except Exception as e:
            logger.error(f"Error generating data: {str(e)}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise
        finally:
            if not temp_file.closed:
                temp_file.close()

    def _generate_batch(self, batch_size: int, state: Optional[PumpState] = None) -> None:
        for _ in range(batch_size):
            self.pump.step(enforce_state=state)

    def _convert_batch_to_df(self) -> pd.DataFrame:
        batch_data = self.pump.data_log[-self.batch_size:]
        return pd.DataFrame([{
            'time': r.time, 'rpm': r.rpm, 'pressure': r.pressure, 'flow_rate': r.flow_rate,
            'temperature': r.temperature, 'vibration': r.vibration, 'current': r.current,
            'noise': r.noise, 'ambient_temp': r.ambient_temp, 'humidity': r.humidity,
            'fluid_viscosity': r.fluid_viscosity, 'external_vibration': r.external_vibration,
            'failure': int(r.failure_triggered), 'failure_type': r.failure_type.name,
            'failure_transition': r.failure_transition
        } for r in batch_data])

# PumpWebApp class
class PumpWebApp:
    def __init__(self):
        self.pump = VirtualPump()
        self.visualizer = PumpVisualizer(self.pump)
        self.logger = PumpDataLogger(self.pump)
        self.selected_state = None

        os.makedirs('/content/output/animations', exist_ok=True)
        os.makedirs('/content/output/data', exist_ok=True)
        logger.info("Initialized PumpWebApp with directories created.")

    def update_environment(self, temp, viscosity, vibration, humidity):
        if not (10 <= temp <= 40):
            logger.warning(f"Invalid temperature {temp}, clipping to [10, 40]")
        if not (0.5 <= viscosity <= 5):
            logger.warning(f"Invalid viscosity {viscosity}, clipping to [0.5, 5]")
        if not (7 <= vibration <= 10):
            logger.warning(f"Invalid vibration {vibration}, clipping to [7, 10]")
        if not (30 <= humidity <= 60):
            logger.warning(f"Invalid humidity {humidity}, clipping to [30, 60]")

        params = EnvironmentParams(
            ambient_temp=temp,
            fluid_viscosity=viscosity,
            external_vibration=vibration,
            humidity=humidity
        )
        self.pump.update_environment(params)
        logger.info(f"Environment updated: Temp={temp}°C, Viscosity={viscosity}cP, Vibration={vibration}mm/s, Humidity={humidity}%")
        return f"Environment updated: Temp={temp}°C, Viscosity={viscosity}cP, Vibration={vibration}mm/s, Humidity={humidity}%"

    def update_thresholds(self, temp_threshold, vib_threshold, curr_threshold, press_threshold):
        self.pump.custom_thresholds = {
            'temperature': temp_threshold,
            'vibration': vib_threshold,
            'current': curr_threshold,
            'pressure': press_threshold
        }
        logger.info(f"Thresholds updated: Temp={temp_threshold}°C, Vib={vib_threshold}mm/s, Current={curr_threshold}A, Pressure={press_threshold}bar")
        return f"Thresholds updated: Temp={temp_threshold}°C, Vib={vib_threshold}mm/s, Current={curr_threshold}A, Pressure={press_threshold}bar"

    def set_state(self, state):
        if state == "Fail":
            self.selected_state = PumpState.FAIL
            self.pump.set_state(PumpState.FAIL)
            logger.info("State set to Fail")
            return "State set to Fail"
        elif state == "Non-Fail":
            self.selected_state = PumpState.NON_FAIL
            self.pump.set_state(PumpState.NON_FAIL)
            logger.info("State set to Non-Fail")
            return "State set to Non-Fail"
        elif state == "Custom":
            self.selected_state = None
            self.pump.set_state(None)
            logger.info("State set to Custom")
            return "State set to Custom (using current settings)"

    def reset_pump(self):
        self.pump = VirtualPump()
        self.visualizer = PumpVisualizer(self.pump)
        self.logger = PumpDataLogger(self.pump)
        self.selected_state = None
        logger.info("Pump reset to initial state")
        return "Pump reset to initial state"

    def set_log_level(self, enabled):
        logging.getLogger().setLevel(logging.INFO if enabled else logging.WARNING)
        logger.info(f"Logging set to {'INFO' if enabled else 'WARNING'}")
        return f"Logging set to {'INFO' if enabled else 'WARNING'}"

    def start_simulation(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_text = []
        videos = {sensor: None for sensor in self.visualizer.sensor_config.keys()}
        files = []

        try:
            if self.selected_state:
                state_name = self.selected_state.name.lower()
                output_text.append(f"\nStarting {state_name} state simulation...")
                logger.info(f"Starting simulation for {state_name} state")

                output_text.append("\nGenerating sensor animations...")
                for sensor in self.visualizer.sensor_config:
                    safe_name = sensor.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                    filename = f"/content/output/animations/pump_{safe_name}_{state_name}_animation_{timestamp}.mp4"
                    logger.info(f"Attempting to save animation for {sensor} to {filename}")
                    video_path = self.visualizer.save_animation(sensor, filename)
                    if video_path and os.path.exists(video_path):
                        videos[sensor] = video_path
                        files.append(video_path)
                        output_text.append(f"Saved {sensor} animation for {state_name} to {filename}")
                        logger.info(f"Successfully saved {sensor} animation")
                    else:
                        output_text.append(f"Failed to save {sensor} animation for {state_name}")
                        logger.error(f"Failed to save animation for {sensor}")

                output_text.append("\nGenerating dataset...")
                output_file = f"/content/output/data/{state_name}.csv"
                logger.info(f"Generating dataset to {output_file}")
                self.logger.generate_data(10000, output_file, self.selected_state)
                if os.path.exists(output_file):
                    files.append(output_file)
                    output_text.append(f"Dataset saved to {output_file}")
                    logger.info(f"Dataset generated successfully")
                else:
                    output_text.append(f"Failed to generate dataset")
                    logger.error(f"Dataset generation failed for {output_file}")
                output_text.append("Recommendation: Dataset generation set to 10000 samples to capture sufficient failure cycles.")

                try:
                    df = pd.read_csv(output_file)
                    output_text.append("\nSample data:")
                    output_text.append(str(df.head(5).to_string()))
                    logger.info("Sample data displayed successfully")
                except Exception as e:
                    logger.error(f"Error reading sample data: {e}")
                    output_text.append(f"Error reading sample data: {e}")

            elif self.selected_state is None:
                output_text.append("\nStarting Custom State simulation with current settings...")
                logger.info("Starting Custom State simulation")
                output_file = f"/content/output/data/custom.csv"
                self.logger.generate_data(10000, output_file)
                if os.path.exists(output_file):
                    files.append(output_file)
                    output_text.append(f"Dataset saved to {output_file}")
                    logger.info(f"Custom dataset generated successfully")
                else:
                    output_text.append(f"Failed to generate dataset")
                    logger.error(f"Custom dataset generation failed for {output_file}")
                output_text.append("Recommendation: Dataset generation set to 10000 samples to capture sufficient failure cycles.")

                try:
                    df = pd.read_csv(output_file)
                    output_text.append("\nSample data:")
                    output_text.append(str(df.head(5).to_string()))
                    logger.info("Sample data displayed successfully")
                except Exception as e:
                    logger.error(f"Error reading sample data: {e}")
                    output_text.append(f"Error reading sample data: {e}")

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}", exc_info=True)
            output_text.append(f"\nError: Simulation failed due to {str(e)}. Check logs for details.")

        video_list = [videos.get(sensor) for sensor in self.visualizer.sensor_config.keys()]
        return [output_text] + video_list + [files]

    def build_interface(self):
        with gr.Blocks(title="Virtual Pump Simulation Web App") as demo:
            gr.Markdown("# Virtual Pump Simulation Web App")
            gr.Markdown("Adjust settings, select a state, and generate datasets/animations. View animations or download files.")

            with gr.Row():
                log_level_toggle = gr.Checkbox(label="Enable Detailed Logging", value=False)

            with gr.Row():
                temp_slider = gr.Slider(minimum=10, maximum=40, step=1, value=25, label="Ambient Temp (°C)")
                humidity_slider = gr.Slider(minimum=30, maximum=60, step=5, value=50, label="Humidity (%)")
            with gr.Row():
                viscosity_slider = gr.Slider(minimum=0.5, maximum=5, step=0.1, value=1, label="Viscosity (cP)")
                vibration_slider = gr.Slider(minimum=7, maximum=10, step=0.1, value=8, label="Ext. Vibration (mm/s)")

            with gr.Row():
                temp_threshold = gr.Slider(minimum=40, maximum=100, step=5, value=90, label="Temp Threshold (°C)")
                vib_threshold = gr.Slider(minimum=5, maximum=15, step=0.5, value=10, label="Vib Threshold (mm/s)")
            with gr.Row():
                curr_threshold = gr.Slider(minimum=20, maximum=40, step=1, value=30, label="Current Threshold (A)")
                press_threshold = gr.Slider(minimum=2, maximum=5, step=0.1, value=3.0, label="Pressure Threshold (bar)")

            with gr.Row():
                fail_button = gr.Button("Fail State", variant="stop")
                non_fail_button = gr.Button("Non-Fail State", variant="primary")
                custom_button = gr.Button("Custom State", variant="secondary")
                reset_button = gr.Button("Reset Pump", variant="secondary")

            output_text = gr.HTML()
            video_outputs = {}
            with gr.Tab("Animations"):
                for sensor in self.visualizer.sensor_config.keys():
                    with gr.Row():
                        video_outputs[sensor] = gr.Video(label=f"{sensor} Animation", height=300)  # Smaller video player
            file_outputs = gr.Files()

            log_level_toggle.change(fn=self.set_log_level, inputs=log_level_toggle, outputs=output_text)
            temp_slider.change(fn=self.update_environment, inputs=[temp_slider, viscosity_slider, vibration_slider, humidity_slider], outputs=output_text)
            humidity_slider.change(fn=self.update_environment, inputs=[temp_slider, viscosity_slider, vibration_slider, humidity_slider], outputs=output_text)
            viscosity_slider.change(fn=self.update_environment, inputs=[temp_slider, viscosity_slider, vibration_slider, humidity_slider], outputs=output_text)
            vibration_slider.change(fn=self.update_environment, inputs=[temp_slider, viscosity_slider, vibration_slider, humidity_slider], outputs=output_text)
            temp_threshold.change(fn=self.update_thresholds, inputs=[temp_threshold, vib_threshold, curr_threshold, press_threshold], outputs=output_text)
            vib_threshold.change(fn=self.update_thresholds, inputs=[temp_threshold, vib_threshold, curr_threshold, press_threshold], outputs=output_text)
            curr_threshold.change(fn=self.update_thresholds, inputs=[temp_threshold, vib_threshold, curr_threshold, press_threshold], outputs=output_text)
            press_threshold.change(fn=self.update_thresholds, inputs=[temp_threshold, vib_threshold, curr_threshold, press_threshold], outputs=output_text)

            fail_button.click(fn=self.set_state, inputs=gr.State(value="Fail"), outputs=output_text)
            non_fail_button.click(fn=self.set_state, inputs=gr.State(value="Non-Fail"), outputs=output_text)
            custom_button.click(fn=self.set_state, inputs=gr.State(value="Custom"), outputs=output_text)
            reset_button.click(fn=self.reset_pump, inputs=[], outputs=output_text)

            gr.Button("Generate").click(
                fn=self.start_simulation,
                inputs=[],
                outputs=[output_text] + [video_outputs[sensor] for sensor in video_outputs.keys()] + [file_outputs]
            )

        return demo

# Run the web app
if __name__ == "__main__":
    app = PumpWebApp()
    interface = app.build_interface()
    interface.launch(share=True, debug=True)
