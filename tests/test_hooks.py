from swaybot.agent import Agent
from swaybot.hook import AgentHook


class RecordingHook(AgentHook):
    def __init__(self) -> None:
        self.before = []
        self.after = []
        self.runs = []

    def before_iteration(self, task, step, perception):
        self.before.append((task, step, perception["task"]))

    def after_iteration(self, task, step, action, result, metadata=None):
        self.after.append((task, step, action["name"], result))

    def after_run(self, task, env, reflections):
        self.runs.append((task, env.done, reflections))


def test_hook_records_iterations_and_run():
    hook = RecordingHook()
    agent = Agent(hooks=[hook])
    env = agent.run("demo", max_steps=2)
    assert env.done
    assert len(hook.before) == 2
    assert len(hook.after) == 2
    assert len(hook.runs) == 1
    assert hook.runs[0][0] == "demo"


def test_multiple_hooks_run():
    hook1 = RecordingHook()
    hook2 = RecordingHook()
    agent = Agent(hooks=[hook1, hook2])
    agent.run("demo", max_steps=1)
    assert len(hook1.before) == 1
    assert len(hook2.before) == 1
