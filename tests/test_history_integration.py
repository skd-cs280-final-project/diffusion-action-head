"""Exercise history across offline evaluation and online observation preparation."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.eval_diffusion_head import evaluate_diffusion
from src.model_wrapper.aerovla_diffusion import AeroVLADiffusionModel
from src.model_wrapper.aerovla_wrapper_ui import AerialVLAWrapper
from src.model_wrapper.diffusion_action_head import DiffusionActionHead, DiffusionActionHeadConfig
from src.uav_history import HISTORY_SCHEMA, STATE_DIM, encode_state_history
from test_aerovla_diffusion import make_model
from test_eval_diffusion_head import TinyDiffusionPolicy, records


@pytest.fixture(autouse=True, scope="module")
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(index, *, offset=0.0, velocity=1.0):
    return {
        "sensors": {"state": {
            "position": [offset + index, index * 0.5, -10.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity": [velocity, 0.0, 0.0],
            "timestamp": 1_000_000_000 + index * 100_000_000,
        }},
        "rgb": [np.full((8, 8, 3), index, dtype=np.uint8) for _ in range(5)],
    }


def test_evaluation_delivers_history_to_loss_and_sampling_reproducibly():
    torch.manual_seed(37)
    model = TinyDiffusionPolicy()
    model.diffusion_head = DiffusionActionHead(DiffusionActionHeadConfig(
        condition_dim=4, hidden_dim=8, num_layers=1, num_heads=2,
        num_train_timesteps=8, num_inference_steps=2, dropout=0.5,
        state_dim=STATE_DIM, history_length=4,
    ))
    data = records()
    for index, record in enumerate(data):
        history, mask = encode_state_history([observation(i, velocity=index + 1) for i in range(index + 1)], 4)
        record.update(state_history=history, state_history_mask=mask)
    loader = DataLoader(data, batch_size=2, generator=torch.Generator().manual_seed(11))
    rng_before = torch.get_rng_state().clone()
    loader_before = loader.generator.get_state().clone()
    modes_before = [module.training for module in model.modules()]
    with patch.object(model, "encode_condition", wraps=model.encode_condition) as encode, \
         patch.object(model.diffusion_head, "compute_loss", wraps=model.diffusion_head.compute_loss) as loss, \
         patch.object(model.diffusion_head, "sample", wraps=model.diffusion_head.sample) as sample:
        first = evaluate_diffusion(model, loader, device="cpu", seed=19)
    assert encode.call_count == loss.call_count == sample.call_count == 2
    for start, encode_call, loss_call, sample_call in zip((0, 2), encode.call_args_list,
                                                        loss.call_args_list, sample.call_args_list):
        assert set(encode_call.kwargs) == {"input_ids", "attention_mask", "pixel_values"}
        expected = data[start:start + 2]
        for call in (loss_call, sample_call):
            torch.testing.assert_close(call.kwargs["state_history"], torch.stack([row["state_history"] for row in expected]))
            torch.testing.assert_close(call.kwargs["state_history_mask"], torch.stack([row["state_history_mask"] for row in expected]))
    assert first == evaluate_diffusion(model, loader, device="cpu", seed=19)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(loader.generator.get_state(), loader_before)
    assert [module.training for module in model.modules()] == modes_before
    assert all(parameter.grad is None for parameter in model.parameters())
    changed = copy.deepcopy(data)
    for row in changed:
        row["state_history"][row["state_history_mask"], 9] = -1.0
    altered = evaluate_diffusion(model, DataLoader(changed, batch_size=2), device="cpu", seed=19)
    assert altered["noise_mse"] != first["noise_mse"]
    assert altered["normalized_continuous_mse"] != first["normalized_continuous_mse"]


class TinyTokenizer:
    def __init__(self):
        self.padding_side = "left"
        self.add_eos_token = True

    def __len__(self):
        return 32

    def __call__(self, prompts, *, return_tensors, padding):
        assert return_tensors == "pt" and padding
        sequences = [[1] + [2 + len(word) % 29 for word in prompt.split()] for prompt in prompts]
        length = max(map(len, sequences))
        ids = torch.zeros(len(prompts), length, dtype=torch.long)
        for index, tokens in enumerate(sequences):
            if self.padding_side == "right":
                ids[index, :len(tokens)] = torch.tensor(tokens)
            else:
                ids[index, length - len(tokens):] = torch.tensor(tokens)
        return {"input_ids": ids, "attention_mask": ids.ne(0)}


class TinyImageProcessor:
    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        assert images.size == (224, 448)
        pixels = torch.from_numpy(np.asarray(images.resize((4, 4))).copy())
        return {"pixel_values": pixels.permute(2, 0, 1).float().unsqueeze(0) / 255.0}


def online_wrapper(history_length=4):
    wrapper = AerialVLAWrapper.__new__(AerialVLAWrapper)
    wrapper.device = torch.device("cpu")
    wrapper.diffusion_mode = True
    wrapper.tokenizer = TinyTokenizer()
    wrapper.tokenizer.padding_side = "right"
    wrapper.image_processor = TinyImageProcessor()
    wrapper.model = SimpleNamespace(head_config=SimpleNamespace(history_length=history_length), dtype=torch.float32)
    return wrapper


def prepare(wrapper, episodes):
    targets = [[10.0, 0.0, -10.0] for _ in episodes]
    instructions = ["The target is 90 degrees from you. A red building. Please control the UAV."
                    for _ in episodes]
    return wrapper.prepare_inputs(episodes, targets, instructions)[0]


def test_online_history_matches_encoder_for_independent_episodes_and_first_frame():
    wrapper = online_wrapper()
    episodes = [[observation(i) for i in range(3)], [observation(0, offset=100, velocity=-2)]]
    batch = prepare(wrapper, episodes)
    assert batch["state_history"].shape == (2, 4, STATE_DIM)
    assert batch["pixel_values"].shape == (2, 3, 4, 4)
    for index, episode in enumerate(episodes):
        expected, mask = encode_state_history(episode, 4)
        torch.testing.assert_close(batch["state_history"][index], expected, rtol=0, atol=0)
        torch.testing.assert_close(batch["state_history_mask"][index], mask)
        alone = prepare(wrapper, [episode])
        torch.testing.assert_close(batch["state_history"][index], alone["state_history"][0], rtol=0, atol=0)
    assert batch["state_history_mask"].tolist() == [[False, True, True, True], [False, False, False, True]]
    assert torch.count_nonzero(batch["state_history"][1, :-1]) == 0
    # Replacing a neighbouring episode cannot affect the second UAV's history.
    replaced = prepare(wrapper, [[observation(0, offset=-300)], episodes[1]])
    torch.testing.assert_close(batch["state_history"][1], replaced["state_history"][1], rtol=0, atol=0)


def test_online_history_deduplicates_five_terminal_observations():
    wrapper = online_wrapper()
    episode = [observation(i) for i in range(3)]
    base = prepare(wrapper, [episode])
    repeated = prepare(wrapper, [episode + [copy.deepcopy(episode[-1]) for _ in range(5)]])
    torch.testing.assert_close(base["state_history"], repeated["state_history"], rtol=0, atol=0)
    torch.testing.assert_close(base["state_history_mask"], repeated["state_history_mask"])
    assert repeated["state_history_mask"].sum().item() == 3


def test_legacy_online_diffusion_omits_history_even_without_valid_state_timestamps():
    wrapper = online_wrapper(history_length=0)
    episode = [observation(0)]
    episode[0]["sensors"]["state"].pop("timestamp")
    episode[0]["sensors"]["state"].pop("linear_velocity")
    batch = prepare(wrapper, [episode])
    assert set(batch) == {"input_ids", "attention_mask", "pixel_values"}


def test_online_constructor_enforces_saved_diffusion_tokenizer_settings(tmp_path):
    (tmp_path / "diffusion_config.json").write_text("{}")
    tokenizer, processor = TinyTokenizer(), TinyImageProcessor()
    model = make_model(state_dim=STATE_DIM, history_length=4)
    config = {"training_metadata": {"history_schema": HISTORY_SCHEMA}}
    with patch("torch.cuda.is_available", return_value=False), \
         patch("src.model_wrapper.aerovla_wrapper_ui.AutoTokenizer.from_pretrained", return_value=tokenizer), \
         patch("src.model_wrapper.aerovla_wrapper_ui.AutoImageProcessor.from_pretrained", return_value=processor), \
         patch.object(AeroVLADiffusionModel, "from_pretrained", return_value=model) as load, \
         patch.object(AeroVLADiffusionModel, "read_checkpoint_config", return_value=config):
        wrapper = AerialVLAWrapper(SimpleNamespace(model_path=str(tmp_path), model_base=None), None)
    assert wrapper.model is model and not model.training
    assert tokenizer.padding_side == "right" and tokenizer.add_eos_token is False
    assert load.call_args.kwargs["tokenizer_length"] == 32
    batch = prepare(wrapper, [[observation(0)], [observation(0, velocity=-2)]])
    result = wrapper.model.predict_action(**batch, generator=torch.Generator().manual_seed(41))
    assert result.shape == (2, 2, 4) and torch.isfinite(result).all()
