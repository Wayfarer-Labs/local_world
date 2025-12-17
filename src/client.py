from typing import AsyncIterable, AsyncIterator

import asyncio
import functools
import sys

import pygame
import torch
import torch.nn.functional as F
import torchvision

from world_engine import WorldEngine, CtrlInput


MODEL_URI = "OpenWorldLabs/CoD-V3-30K-SF"
MAX_FRAMES = 4096
MOUSE_SENSITIVITY = 1.5


@functools.lru_cache
def load_seed_frame(
    target_size: tuple[int, int] = (360, 640),
    device: str = "cuda",
) -> torch.Tensor | None:
    """Cached seed frame as (H,W,3) uint8 on GPU, or None if unavailable."""
    import urllib.request

    url = "https://gist.github.com/user-attachments/assets/5d91c49a-2ae9-418f-99c0-e93ae387e1de"
    urllib.request.urlretrieve(url, "img.png")

    img = torchvision.io.read_image("img.png")  # torch.uint8, shape (C,H,W)
    if img is None or img.numel() == 0:
        return None

    img = img[:3]  # ensure RGB (drop alpha if present)
    frame = img.unsqueeze(0).float()  # (1,3,H,W)
    frame = F.interpolate(frame, size=target_size, mode="bilinear", align_corners=False)[0]
    return frame.to(device=device, dtype=torch.uint8).permute(1, 2, 0)  # (H,W,3)


async def render(frames: AsyncIterable[torch.Tensor]) -> None:
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Y=new seed, U=restart, ESC=exit")

    async for t in frames:
        img = t.detach()
        if img.dtype != torch.uint8:
            img = img.clamp(0, 255).to(torch.uint8)

        frame = img.cpu().numpy()  # (H,W,3)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # (W,H,3)
        surf = pygame.transform.scale(surf, screen.get_size())

        screen.blit(surf, (0, 0))
        pygame.display.flip()
        await asyncio.sleep(0)

    pygame.quit()


async def frame_stream(
    engine: WorldEngine,
    ctrls: AsyncIterable[CtrlInput],
    seed: torch.Tensor | None,
    max_frames: int = MAX_FRAMES - 2,
) -> AsyncIterator[torch.Tensor]:
    current_seed = seed
    n = 0

    async def reset_with_seed() -> None:
        await asyncio.to_thread(engine.reset)
        if current_seed is not None:
            await asyncio.to_thread(engine.append_frame, current_seed)

    if current_seed is not None:
        await asyncio.to_thread(engine.append_frame, current_seed)

    yield await asyncio.to_thread(engine.gen_frame)
    n = 1

    async for ctrl in ctrls:
        cmd = getattr(ctrl, "reset_command", None)

        # Y/U: reset and immediately yield a frame with NO ctrl applied
        if cmd == "new_seed":
            current_seed = await asyncio.to_thread(load_seed_frame)
            await reset_with_seed()
            yield await asyncio.to_thread(engine.gen_frame)
            n = 1
            continue

        if cmd == "restart_seed":
            await reset_with_seed()
            yield await asyncio.to_thread(engine.gen_frame)
            n = 1
            continue

        # Periodic reset: reset context but still apply the current ctrl on the next gen_frame
        if n >= max_frames:
            await reset_with_seed()
            n = 0

        yield await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
        n += 1


async def ctrl_stream(*, dtype: torch.dtype, device: str = "cuda") -> AsyncIterator[CtrlInput]:
    # Mouse button bitmasks (kept as-is for engine compatibility)
    MOUSE_MAP = {1: 0x01, 2: 0x04, 3: 0x02}  # LMB, MMB, RMB
    SHIFT_CODE = 0xA0  # VK_LSHIFT

    CMD_BY_KEY = {pygame.K_y: "new_seed", pygame.K_u: "restart_seed"}

    # Hardcoded legal keys -> engine button codes
    LEGAL_KEYS = {
        pygame.K_w: ord("W"),
        pygame.K_a: ord("A"),
        pygame.K_s: ord("S"),
        pygame.K_d: ord("D"),
        pygame.K_SPACE: ord(" "),
    }


    while True:
        reset_command = None

        # Only consume events needed for edge-trigger actions / quit
        for ev in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
            if ev.type == pygame.QUIT:
                return
            if ev.key == pygame.K_ESCAPE:
                return
            reset_command = CMD_BY_KEY.get(ev.key) or reset_command

        # Snapshot current held state (no bookkeeping)
        pressed = pygame.key.get_pressed()
        buttons: set[int] = {code for k, code in LEGAL_KEYS.items() if pressed[k]}

        if pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT]:
            buttons.add(SHIFT_CODE)

        lmb, mmb, rmb = pygame.mouse.get_pressed(3)
        if lmb:
            buttons.add(MOUSE_MAP[1])
        if mmb:
            buttons.add(MOUSE_MAP[2])
        if rmb:
            buttons.add(MOUSE_MAP[3])

        dx, dy = pygame.mouse.get_rel()
        mouse = torch.tensor([dx, dy], dtype=dtype, device=device) * MOUSE_SENSITIVITY

        ctrl = CtrlInput(button=buttons, mouse=mouse)
        ctrl.reset_command = reset_command
        yield ctrl

        await asyncio.sleep(0)


async def main() -> None:
    uri = sys.argv[1] if len(sys.argv) > 1 else MODEL_URI

    seed = await asyncio.to_thread(load_seed_frame)

    engine = WorldEngine(
        uri,
        device="cuda",
        model_config_overrides={"n_frames": MAX_FRAMES},
        apply_patches=True,
        quant=None,
    )

    ctrls = ctrl_stream(dtype=engine.dtype, device="cuda")
    frames = frame_stream(engine, ctrls, seed)
    await render(frames)


if __name__ == "__main__":
    asyncio.run(main())
