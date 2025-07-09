import torch
import os
import logging


def restore_checkpoint(ckpt_path, state, device):
    if not os.path.exists(ckpt_path):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        logging.warning(f"No checkpoint found at {ckpt_path}. "
                        f"Returning the same state as input.")
        return state
    else:
        loaded_state = torch.load(ckpt_path, map_location=device)
        state['optimizer'].load_state_dict(loaded_state.get('optimizer', {}))
        state['model'].load_state_dict(loaded_state.get('model', {}), strict=False)
        if 'ema' in loaded_state and 'ema' in state:
            state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state.get('step', 0)
        return state


def save_checkpoint(ckpt_path, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict() if 'ema' in state else {},
        'step': state.get('step', 0)
    }
    torch.save(saved_state, ckpt_path)
