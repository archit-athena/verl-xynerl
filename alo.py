# run_single_turn_demo.py
import asyncio
import types
from types import SimpleNamespace

# Minimal monkeypatch so the class import doesn't trigger heavy behavior.
# (Only necessary if importing the real package causes extra imports; skip if import already works.)
# from verl.experimental.agent_loop.agent_loop import AgentLoopBase  # not required here

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

# Dummy tokenizer and server_manager to exercise .run()
class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        # return simple token ids
        return [1,2,3]

class DummyServerManager:
    class Output:
        def __init__(self, token_ids, log_probs=None):
            self.token_ids = token_ids
            self.log_probs = log_probs
    async def generate(self, request_id, prompt_ids, sampling_params):
        await asyncio.sleep(0.01)
        return DummyServerManager.Output(token_ids=list(range(200, 200 + sampling_params.get("max_tokens", 8))),
                                         log_probs=[-0.1]*sampling_params.get("max_tokens", 8))

async def main():
    fake_config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(prompt_length=64, response_length=16)),
        data={"apply_chat_template_kwargs": {}},
    )

    tokenizer = DummyTokenizer()
    server_manager = DummyServerManager()

    # Instantiate with expected attrs: config, tokenizer, server_manager
    loop = SingleTurnAgentLoop(config=fake_config, tokenizer=tokenizer, server_manager=server_manager)

    sampling_params = {"temperature": 0.7, "top_p": 0.9, "max_tokens": 12}
    raw_prompt = [{"role": "user", "content": "Hello, this is a demo."}]

    out = await loop.run(sampling_params=sampling_params, raw_prompt=raw_prompt)
    print("prompt_ids:", out.prompt_ids[:50])
    print("response_ids:", out.response_ids)
    print("response_mask:", out.response_mask)
    print("response_logprobs:", out.response_logprobs)
    print("metrics:", out.metrics)
    print("num_turns:", out.num_turns)

if __name__ == "__main__":
    asyncio.run(main())
