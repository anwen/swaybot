import time

from swaybot.automation import Automation


def test_automation_runs_interval_job():
    counter = {"value": 0}

    def increment():
        counter["value"] += 1

    auto = Automation()
    auto.add_interval(0.05, increment)
    auto.start()
    time.sleep(0.18)
    auto.stop()

    assert counter["value"] >= 2
