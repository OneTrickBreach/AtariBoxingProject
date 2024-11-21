from pettingzoo.atari import boxing_v2

rom_path = "roms"
env = boxing_v2.env(render_mode="human", auto_rom_install_path=rom_path)
env.reset(seed=42)
print("Environment initialized successfully!")
env.close()
