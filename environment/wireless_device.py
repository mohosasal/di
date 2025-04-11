# implementation.py
from typing import Tuple, List
import pandas as pd
from di.environment.base_object import IWirelessDevice
from di.environment.task import Task


class WirelessDevice(IWirelessDevice):
    vehicles_df = None
    vehicles: List["WirelessDevice"] = []
    time = 1000.00

    def __init__(self, id, x, y, s, a, cc):
        self.id = id
        self.x = x
        self.y = y
        self.s = s
        self.a = a
        self.cc = cc

    @staticmethod
    def load_data(file_path="./TAVF-Hamburg/simulation_results.txt"):
        try:
            WirelessDevice.vehicles_df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            print(f"Error loading data: {e}")

    @staticmethod
    def move_all(time):
        WirelessDevice.time += time
        if not WirelessDevice.vehicles:
            for row in WirelessDevice.vehicles_df.itertuples(index=False):
                vehicle = WirelessDevice(
                    id=row.VehicleID,
                    x=float(row.x),
                    y=float(row.y),
                    s=float(row.speed),
                    a=float(row.angle),
                    cc=10
                )
                WirelessDevice.vehicles.append(vehicle)
        else:
            for vehicle in WirelessDevice.vehicles:
                vehicle.move()

    def move(self):
        row = WirelessDevice.vehicles_df[
            (WirelessDevice.vehicles_df['VehicleID'] == self.id) &
            (WirelessDevice.vehicles_df['SimulationTime'] == WirelessDevice.time)
        ]
        if not row.empty:
            row = row.iloc[0]
            self.x = float(row.x)
            self.y = float(row.y)
            self.s = float(row.speed)
            self.a = float(row.angle)

    @staticmethod
    def get_wd_by_id(id):
        for vehicle in WirelessDevice.vehicles:
            if vehicle.id == id:
                return vehicle
        return None



    def get_position(self) -> Tuple[float, float]:
        return self.x, self.y

    def get_id(self) -> str:
        return self.id
