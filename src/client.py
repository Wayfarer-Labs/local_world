from typing import AsyncIterator

import asyncio
import functools
import os
import urllib.request

import pygame
import torch
import torch.nn.functional as F
import torchvision

from world_engine import WorldEngine, CtrlInput, QUANTS


def fetch_model_uris(org_uri: str = "OverWorld") -> list[str]:
    """Models from an author/org on the Hub, most recent first."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    try:
        # Treat the old `collection_uri` default as an author/org name.
        # e.g. "OverWorld-Beta" (org) or "OpenWorldLabs" (user/org)
        api = HfApi()
        models = [m.modelId for m in api.list_models(author=org_uri, sort="lastModified", direction=-1)]
    except HfHubHTTPError as e:
        if getattr(e, "response", None) is not None and e.response.status_code in (401, 403):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
            from huggingface_hub import logout
            try:
                logout()  # clear saved token(s) on disk (if any)
            except OSError:
                pass
        raise
    if not models:
        raise RuntimeError(f"No models found for Hugging Face author/org {org_uri}")
    return models


def launch_form(*, title: str = "World Engine") -> tuple[str, str | None] | None:
    import tkinter as tk
    from tkinter import ttk, messagebox

    try:
        models = fetch_model_uris()
    except Exception as e:
        messagebox.showerror(title, f"Failed to load models:\n{e}")
        return None

    qvals = list(map(str, QUANTS))  # keep your existing semantics

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    # Modal-ish behavior + nicer focus
    root.lift()
    root.focus_force()

    pad = {"padx": 10, "pady": 6}
    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")
    frm.columnconfigure(1, weight=1)

    selected: dict[str, str | None] = {"uri": None, "quant": None}

    var = tk.StringVar(value="select model")
    cvar = tk.StringVar(value="")
    qvar = tk.StringVar(value=qvals[0])

    ttk.Label(frm, text="Model").grid(row=0, column=0, sticky="w", **pad)
    cmb = ttk.Combobox(frm, textvariable=var, values=["select model", "custom model", *models], state="readonly", width=60)
    cmb.grid(row=0, column=1, sticky="ew", **pad)

    clbl = ttk.Label(frm, text="Custom model")
    cent = ttk.Entry(frm, textvariable=cvar, width=60)
    clbl.grid(row=1, column=0, sticky="w", **pad)
    cent.grid(row=1, column=1, sticky="ew", **pad)
    clbl.grid_remove(); cent.grid_remove()

    def _on_model(_e=None) -> None:
        show = var.get() == "custom model"
        (clbl.grid if show else clbl.grid_remove)()
        (cent.grid if show else cent.grid_remove)()
        if show:
            cent.focus_set()

    cmb.bind("<<ComboboxSelected>>", _on_model)

    ttk.Label(frm, text="Quantization").grid(row=2, column=0, sticky="w", **pad)
    qcmb = ttk.Combobox(frm, textvariable=qvar, values=qvals, state="readonly", width=60)
    qcmb.grid(row=2, column=1, sticky="ew", **pad)

    ttk.Separator(frm).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 8))

    def close(*, cancel: bool) -> None:
        if not cancel:
            v = var.get()
            uri = cvar.get().strip() if v == "custom model" else v
            if uri in ("", "select model"):
                messagebox.showwarning(title, "Please select a model.")
                return
            selected["uri"] = uri
            q = qvar.get()
            selected["quant"] = None if q == qvals[0] else q
        root.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=4, column=0, columnspan=2, sticky="e")
    ttk.Button(btns, text="Cancel", command=lambda: close(cancel=True)).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btns, text="Run", command=lambda: close(cancel=False)).grid(row=0, column=1)

    root.protocol("WM_DELETE_WINDOW", lambda: close(cancel=True))
    root.bind("<Escape>", lambda _e: close(cancel=True))
    root.bind("<Return>", lambda _e: close(cancel=False))

    # make it behave like a dialog
    root.grab_set()
    cmb.focus_set()
    root.wait_window()

    return (selected["uri"], selected["quant"]) if selected["uri"] else None


def seed_form(*, title: str = "Seed") -> str | None:
    import tkinter as tk
    from tkinter import ttk, filedialog

    root = tk.Tk()
    root.title(title)
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=12)
    frm.grid(sticky="nsew")

    chosen: dict[str, str | None] = {"path": None}
    shown = tk.StringVar(value="(default)")

    def pick() -> None:
        p = filedialog.askopenfilename(
            title="Select seed image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"), ("All files", "*.*")],
        )
        if p:
            chosen["path"] = p
            shown.set(os.path.basename(p))

    ttk.Label(frm, text="Image").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ttk.Button(frm, text="Choose…", command=pick).grid(row=0, column=1, sticky="w", pady=(0, 8))
    ttk.Label(frm, textvariable=shown).grid(row=0, column=2, sticky="w", pady=(0, 8))

    ttk.Label(frm, text="Prompt").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(0, 8))
    ent = ttk.Entry(frm, width=52)
    ent.insert(0, "Sample prompt")
    ent.configure(state="disabled")
    ent.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(0, 8))

    def ok() -> None:
        root.destroy()

    ttk.Button(frm, text="Start", command=ok).grid(row=2, column=2, sticky="e")
    root.protocol("WM_DELETE_WINDOW", ok)
    root.mainloop()
    return chosen["path"]


SEED_FRAME_URLS: list[str] = [
    "https://gist.github.com/user-attachments/assets/5d91c49a-2ae9-418f-99c0-e93ae387e1de",
    "https://gist.github.com/user-attachments/assets/4adc5a3d-6980-4d1e-b6e8-9033cdf61c66",
    "https://gist.github.com/user-attachments/assets/ae398747-de4c-4d43-bac4-54fe61ab0ca8",
    "https://gist.github.com/user-attachments/assets/9d7336fa-5cec-4c7d-bb65-eaebac0a6336",
    "https://gist.github.com/user-attachments/assets/55dae2d3-00e3-4d03-bb6c-2c7e0ac70f5f",
]


@functools.lru_cache
def _load_seed_frame_from_url(
    url: str,
    target_size: tuple[int, int] = (360, 640),
) -> torch.Tensor:
    """Cached seed frame as (H,W,3) uint8 on CPU"""
    # Use a per-URL filename to avoid stomping img.png
    fn = f"seed_{abs(hash(url))}.png"
    urllib.request.urlretrieve(url, fn)

    img = torchvision.io.read_image(fn)          # (C,H,W) uint8
    img = img[:3].unsqueeze(0).float()           # (1,3,H,W)
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    return frame.to(torch.uint8).permute(1, 2, 0).contiguous()  # (H,W,3)


def load_seed_frame(target_size: tuple[int, int] = (360, 640)) -> torch.Tensor:
    import random
    url = random.choice(SEED_FRAME_URLS)
    return _load_seed_frame_from_url(url, target_size).clone()


def load_seed_frame_from_file(path: str, target_size: tuple[int, int] = (360, 640)) -> torch.Tensor:
    from torchvision.io import read_file, decode_image, ImageReadMode
    data = read_file(path)  # bytes -> uint8 1D tensor
    img = decode_image(data, mode=ImageReadMode.RGB)  # (3,H,W) uint8 (includes WEBP if supported)
    img = img.unsqueeze(0).float()  # (1,3,H,W)
    frame = F.interpolate(img, size=target_size, mode="bilinear", align_corners=False)[0]
    return frame.to(dtype=torch.uint8).permute(1, 2, 0).contiguous()


# pygame keycode -> Windows VK int (main ANSI rows only)
PYGAME_TO_VK = (
    {pygame.key.key_code(ch): ord(ch) for ch in "1234567890"}  # 1..0
    | {pygame.K_MINUS: 0xBD, pygame.K_EQUALS: 0xBB}            # - =
    | {pygame.key.key_code(ch): ord(ch.upper()) for ch in "qwertyuiop"}
    | {pygame.K_LEFTBRACKET: 0xDB, pygame.K_RIGHTBRACKET: 0xDD, pygame.K_BACKSLASH: 0xDC}  # [ ] \|
    | {pygame.key.key_code(ch): ord(ch.upper()) for ch in "asdfghjkl"}
    | {pygame.K_SEMICOLON: 0xBA, pygame.K_QUOTE: 0xDE}         # ;: '"
    | {pygame.key.key_code(ch): ord(ch.upper()) for ch in "zxcvbnm"}
    | {pygame.K_COMMA: 0xBC, pygame.K_PERIOD: 0xBE, pygame.K_SLASH: 0xBF}  # ,< .> /?
    | {pygame.K_SPACE: 0x20, pygame.K_LSHIFT: 0x10, pygame.K_RSHIFT: 0x10}
)


# enable all
WHITELIST_KEYS = frozenset(PYGAME_TO_VK.values()) | frozenset({0x01, 0x02, 0x04})


async def ctrl_stream(
    restart_event: asyncio.Event,
    seed_event: asyncio.Event,
    mouse_sensitivity: float = 1.5,
    whitelisted_keys=None,
) -> AsyncIterator[CtrlInput]:
    whitelisted_keys = WHITELIST_KEYS if whitelisted_keys is None else whitelisted_keys

    codes = (
        {("k", k): v for k, v in PYGAME_TO_VK.items()} |
        {("m", 1): 0x01, ("m", 2): 0x04, ("m", 3): 0x02}  # note: pygame has middle wheel as m2
    )
    codes = {k: v for k, v in codes.items() if v in whitelisted_keys}

    while True:
        btn: set[int] = set()

        for e in pygame.event.get():  # edge presses + drain
            if e.type == pygame.QUIT:
                return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return
                if e.key == pygame.K_u:
                    restart_event.set()
                if e.key == pygame.K_DELETE:
                    seed_event.set()
                if (c := codes.get(("k", e.key))) is not None:
                    btn.add(c)
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if (c := codes.get(("m", e.button))) is not None:
                    btn.add(c)

        pressed = pygame.key.get_pressed()
        btn.update(c for (kind, raw), c in codes.items() if kind == "k" and pressed[raw])

        mb = pygame.mouse.get_pressed(3)
        btn.update(
            c for i, down in enumerate(mb, 1)
            if down and (c := codes.get(("m", i))) is not None
        )

        dx, dy = pygame.mouse.get_rel()
        yield CtrlInput(button=btn, mouse=(dx * mouse_sensitivity, dy * mouse_sensitivity))
        await asyncio.sleep(0)


async def run_loop(
    *,
    engine: WorldEngine,
    seed: torch.Tensor | None,
    n_frames: int,
    mouse_sensitivity: float = 1.5,
) -> None:
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
    pygame.event.set_grab(True)
    pygame.display.set_caption("U=restart, DEL=seed, ESC=menu")

    restart = asyncio.Event()
    seed_req = asyncio.Event()
    ctrls = ctrl_stream(restart_event=restart, seed_event=seed_req, mouse_sensitivity=mouse_sensitivity)
    limit = max(1, n_frames - 2)

    async def reset(*, reload_seed: bool = False) -> None:
        nonlocal seed
        await asyncio.to_thread(engine.reset)
        if reload_seed or seed is None:
            seed = await asyncio.to_thread(load_seed_frame)
        if seed is not None:
            await asyncio.to_thread(engine.append_frame, seed)

    def draw(img: torch.Tensor) -> None:
        img = img.detach()
        if img.dtype != torch.uint8:
            img = img.clamp(0, 255).to(torch.uint8)

        frame = img.cpu().numpy()  # (H,W,3)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))  # (W,H,3)
        surf = pygame.transform.scale(surf, screen.get_size())
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    try:
        await reset(reload_seed=True)

        frames = 0
        async for ctrl in ctrls:
            if seed_req.is_set():
                seed_req.clear()
                await reset(reload_seed=True)
                frames = 0
                continue
            if restart.is_set() or frames >= limit:
                restart.clear()
                await reset(reload_seed=False)
                frames = 0

            img = await asyncio.to_thread(engine.gen_frame, ctrl=ctrl)
            frames += 1
            draw(img)

            await asyncio.sleep(0)
    finally:
        pygame.event.set_grab(False)
        pygame.quit()


async def main(
    *,
    model_uri: str,
    n_frames: int = 4096,
    device: str = "cuda",
    quant: str | None = None,
) -> None:
    seed = None

    engine = WorldEngine(
        model_uri,
        device=device,
        model_config_overrides={
            "n_frames": n_frames,
            "ae_uri": "OpenWorldLabs/owl_vae_f16_c16_distill_v0_nogan"
        },
        quant=quant,
    )
    await run_loop(engine=engine, seed=seed, n_frames=n_frames)


def ensure_hf_token() -> None:
    t = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if t:
        return
    from huggingface_hub import get_token, login
    t = get_token()
    import tkinter as tk
    from tkinter import simpledialog
    if not t:
        r = tk.Tk()
        r.withdraw()
        t = simpledialog.askstring("HF token", "Enter Hugging Face token:", show="*")
        r.destroy()
    if not t:
        raise SystemExit("No Hugging Face token provided.")
    login(token=t, add_to_git_credential=False)
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_HUB_TOKEN"] = t


if __name__ == "__main__":
    ensure_hf_token()
    while True:
        sel = launch_form()
        if not sel:
            break
        uri, quant = sel
        asyncio.run(main(model_uri=uri, quant=quant))
