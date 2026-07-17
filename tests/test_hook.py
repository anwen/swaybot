from swaybot.agent import Agent
from swaybot.hook import AgentHook, CompositeHook


class CapturingHook:
    def __init__(self):
        self.tokens = []
        self.reasoning = []
        self.iterations = []

    def before_iteration(self, task, step, perception):
        pass

    def after_iteration(self, task, step, action, result, metadata=None):
        self.iterations.append((step, action, result))

    def after_run(self, task, env, reflections):
        pass

    def on_token(self, token):
        self.tokens.append(token)

    def on_reasoning(self, reasoning):
        self.reasoning.append(reasoning)


def test_composite_hook_dispatches_optional_callbacks():
    hook = CapturingHook()
    composite = CompositeHook([hook])
    composite.on_token("hello")
    composite.on_reasoning("because")
    assert hook.tokens == ["hello"]
    assert hook.reasoning == ["because"]


def test_composite_hook_skips_hooks_without_optional_methods():
    class MinimalHook:
        def before_iteration(self, task, step, perception):
            pass

        def after_iteration(self, task, step, action, result, metadata=None):
            pass

        def after_run(self, task, env, reflections):
            pass

    composite = CompositeHook([MinimalHook()])
    composite.on_token("hello")
    composite.on_reasoning("because")


def test_agent_runs_with_streaming_hook():
    class TokenBrain:
        def think(self, perception, available_tools, metadata=None, stream_callback=None, reasoning_callback=None):
            if stream_callback:
                stream_callback("tok")
            if reasoning_callback:
                reasoning_callback("reason")
            return {"name": "done", "args": {}}

    hook = CapturingHook()
    agent = Agent(brain=TokenBrain(), hooks=[hook])
    env = agent.run("test", max_steps=2)
    assert env.done
    assert hook.tokens == ["tok"]
    assert hook.reasoning == ["reason"]


def test_agent_runs_when_brain_does_not_support_callbacks():
    class OldBrain:
        def think(self, perception, available_tools):
            return {"name": "done", "args": {}}

    hook = CapturingHook()
    agent = Agent(brain=OldBrain(), hooks=[hook])
    env = agent.run("test", max_steps=2)
    assert env.done
    assert hook.tokens == []
