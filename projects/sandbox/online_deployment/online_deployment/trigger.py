import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Union

import numpy as np
from ligo.gracedb.rest import GraceDb

from aframe.analysis.ledger.events import EventSet

Gdb = Literal["playground", "test", "production"]
SECONDS_PER_YEAR = 31556952


@dataclass
class Event:
    time: float
    detection_statistic: float
    far: float

    def __str__(self):
        return (
            "Event("
            f"time={self.time:0.3f}, "
            f"detection_statistic={self.detection_statistic:0.2f}, "
            f"far={self.far:0.3e} Hz"
            ")"
        )


class Searcher:
    def __init__(
        self,
        outdir: Path,
        fars_per_day: List[float],
        inference_sampling_rate: float,
        refractory_period: float,
    ) -> None:
        logging.debug("Loading background measurements")
        background_file = outdir / "infer" / "background.h5"
        self.background = EventSet.read(background_file)

        fars_per_day = np.sort(fars_per_day)
        num_events = np.floor(fars_per_day * self.background.Tb / 3600 / 24)
        num_events = num_events.astype(int)
        idx = np.where(num_events == 0)[0]
        if idx:
            raise ValueError(
                "Background livetime {}s not enough to detect "
                "events with daily false alarm rate of {}".format(
                    self.background.Tb, ", ".join(fars_per_day[idx])
                )
            )

        events = np.sort(self.background.detection_statistic)
        self.thresholds = events[-num_events]
        for threshold, far in zip(self.thresholds, fars_per_day):
            logging.info(
                "FAR {}/day threshold is {:0.3f}".format(far, threshold)
            )

        self.inference_sampling_rate = inference_sampling_rate
        self.refractory_period = refractory_period
        self.last_detection_time = time.time()
        self.detecting = False

    def check_refractory(self, value):
        time_since_last = time.time() - self.last_detection_time
        if time_since_last < self.refractory_period:
            logging.warning(
                "Detected event with detection statistic {:0.3f} "
                "but it has only been {:0.2f}s since last detection, "
                "so skipping".format(value, time_since_last)
            )
            return True
        return False

    def build_event(self, value: float, t0: float, idx: int):
        if self.check_refractory(value):
            return None

        timestamp = t0 + idx / self.inference_sampling_rate
        far = self.background.far(value)
        far /= SECONDS_PER_YEAR

        logging.info(
            "Event coalescence time found to be {:0.3f} "
            "with FAR {:0.3e} Hz".format(timestamp, far)
        )
        self.last_detection_time = time.time()
        return Event(timestamp, value, far)

    def search(self, y: np.ndarray, t0: float):
        """
        Search for above-threshold events in the
        timeseries of integrated network outputs
        `y`. `t0` should represent the timestamp
        of the last sample of *input* to the
        *neural network* that represents the
        *first sample* of the integration window.
        """

        # if we're already mid-detection, take as
        # the event the max in the current window
        max_val = y.max()
        if self.detecting:
            idx = np.argmax(y)
            self.detecting = False
            return self.build_event(max_val, t0, idx)

        # otherwise check all of our thresholds to
        # see if we have an event relative to any of them
        if not (max_val >= self.thresholds).any():
            # if not, nothing to do here
            return None

        logging.info(
            f"Detected event with detection statistic>={max_val:0.3f}"
        )

        # check if the integrated output is still
        # ramping as we get to the end of the frame
        idx = np.argmax(y)
        if idx < (len(y) - 1):
            # if not, assume the event is in this
            # frame and build an event around it
            return self.build_event(max_val, t0, idx)
        else:
            # otherwise, note that we're mid-event but
            # wait until the next frame to mark it
            self.detecting = True
            return None


@dataclass
class LocalGdb:
    write_dir: Path

    def __post_init__(self):
        self.write_dir.mkdir(exist_ok=True, parents=True)

    def createEvent(self, filecontents: str, filename: str, **_):
        filecontents = json.loads(filecontents.replace("'", '"'))
        filename = self.write_dir / filename
        logging.info(f"Submitting trigger to file {filename}")
        with open(filename, "w") as f:
            json.dump(filecontents, f)
        return filename


class Trigger:
    def __init__(self, server: Union[Gdb, Path]) -> None:
        if isinstance(server, Path):
            self.gdb = LocalGdb(server)
            return

        if server in ["playground", "test"]:
            server = f"https://gracedb-{server}.ligo.org/api/"
        elif server == "production":
            server = "https://gracedb.ligo.org/api/"
        else:
            raise ValueError(f"Unknown server {server}")
        self.gdb = GraceDb(service_url=server)

    def submit(self, event: Event, ifos: List[str]):
        filename = f"event-{int(event.time)}.json"
        event = asdict(event)
        event["IFOs"] = ifos
        filecontents = str(event)

        # alternatively we can write a file to disk,
        # pass that path to the filename argument,
        # and set filecontents=None
        response = self.gdb.createEvent(
            group="CBC",
            pipeline="BBHNet",
            filename=filename,
            search="BBH",
            filecontents=filecontents,
        )
        return response