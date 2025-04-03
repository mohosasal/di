from typing import Tuple, Optional
import numpy as np
import pandas as pd

from di.environment.mobility.base_wd import IWirelessDevice
from di.environment.task import Task


class WirelessDevice(IWirelessDevice):
    vehicles_df = None
    vehicles = []
    time = 1000.00

    def __init__(self):
        # Remove these calls from __init__ as they should be called explicitly
        pass

    @staticmethod
    def load_data(file_path="./TAVF-Hamburg/simulation_results.txt"):
        WirelessDevice.vehicles_df = pd.read_csv(file_path, sep="\t")

    @staticmethod
    def populate():
        if WirelessDevice.vehicles_df is None:
            raise ValueError("Dataframe is not loaded. Call WirelessDevice.load_data() first.")

        WirelessDevice.vehicles.clear()

        for row in WirelessDevice.vehicles_df.itertuples(index=False):
            if float(row.SimulationTime) == WirelessDevice.time:
                wd = WirelessDevice()
                wd.VehicleID = row.VehicleID
                wd.x = float(row.x)
                wd.y = float(row.y)
                wd.speed = float(row.speed)
                wd.angle = float(row.angle)
                wd.cc = 0

                WirelessDevice.vehicles.append(wd)

    def move(self):
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y += self.speed * np.sin(np.radians(self.angle))

    @staticmethod
    def move_all():
        for vehicle in WirelessDevice.vehicles:
            vehicle.move()
        WirelessDevice.time +=1

    def process_local_task(self, task: Task, queue_delay: float) -> float:
        processing_time = task.k_i / self.cc
        return processing_time + queue_delay
#
    @staticmethod
    def get_vehicle_by_id(wd_id):
        for vehicle in WirelessDevice.vehicles:
            if vehicle.wd_id == wd_id:
                return vehicle